//! n-dim `ldl` on the BLAS device: the pivoted LDL factorization of every
//! matrix slice of the stack (`?SYTRF` / `?HETRF`), in SciPy's `ldl` form.
//!
//! Matrix slices: the two matrix axes are the last two shape axes under a
//! row-major device and the first two under a column-major one; the rest is
//! the batch.
//!
//! The result is [`LdlResult`]: `lu` of shape `(..., N, N)` is the permuted
//! unit lower (or upper) triangular outer factor, `d` of shape `(..., N, N)`
//! is the dense block-diagonal matrix with 1x1 and 2x2 blocks (Hermitian, or
//! symmetric for `hermitian = false`), and `perm` of shape `(..., N)` holds
//! the 0-based row indices with `lu[perm, :]` triangular and
//! `A = lu @ d @ luᴴ` (or `lu @ d @ luᵀ` for the symmetric form).
//!
//! `uplo` selects the triangle of `a` that is read and the outer factor; the
//! default is `Lower` under both device orders (the SciPy convention). The
//! other triangle of `a` is never referenced. `hermitian = false` factorizes
//! a complex symmetric matrix (`?SYTRF`, `A = L D Lᵀ`); it has no effect on
//! real input.
//!
//! The input is consumed as scratch: an immutable `a` is factored from a
//! copy; an owned or mutable `a` whose layout is the device order is
//! factored in place and destroyed; an off-order mutable `a` is factored
//! from a work buffer and left untouched (the outputs are fresh).
//!
//! A singular slice is not an error: the factorization completes with an
//! exactly-zero pivot on the diagonal of `d`. For the Hermitian form the
//! imaginary parts of a complex diagonal are ignored (LAPACK reads and
//! stores the diagonal real); SciPy warns, rstsr states it here. Zero-size
//! slices are synthesized without a driver call. An input with fewer than
//! two axes or a non-square slice is an `InvalidLayout` error.
//!
//! See also [`solve_symmetric`](crate::linalg_auto_impl::solve_symmetric).

use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/// Device-facing entry of the n-dim `ldl`; see the module documentation for
/// the shapes, the `uplo` / `hermitian` options and the input policy.
#[allow(clippy::type_complexity)]
pub fn blas_impl_ldl_f<T>(
    a: TensorReference<'_, T, DeviceBLAS, IxD>,
    hermitian: bool,
    uplo: Option<FlagUpLo>,
) -> Result<(Tensor<T, DeviceBLAS, IxD>, Tensor<T, DeviceBLAS, IxD>, Tensor<usize, DeviceBLAS, IxD>)>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceAPI<usize, Raw = Vec<usize>>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationNumAPI<usize>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
{
    let device = a.device().clone();
    let order = device.default_order();
    let stack = StackLayout::new(a.layout(), 2, order)?;
    let inner_shape = stack.inner_shape();
    rstsr_assert_eq!(
        inner_shape[0],
        inner_shape[1],
        InvalidLayout,
        "ldl: the matrix must be square, got {}x{}",
        inner_shape[0],
        inner_shape[1]
    )?;
    let n = inner_shape[0];
    let uplo = uplo.unwrap_or(Lower);
    let batch_shape = stack.batch_shape().to_vec();

    let mut lu = zeros_f((stack_shape(&batch_shape, &[n, n], order), &device))?;
    let mut d = zeros_f((stack_shape(&batch_shape, &[n, n], order), &device))?;
    let mut perm = zeros_f((stack_shape(&batch_shape, &[n], order), &device))?;
    let lu_stack = StackLayout::new(lu.layout(), 2, order)?;
    let d_stack = StackLayout::new(d.layout(), 2, order)?;
    let perm_stack = StackLayout::new(perm.layout(), 1, order)?;

    let mut converted = overwritable_convert_with_order(a, order)?;
    let converted_stack = StackLayout::new(converted.view().layout(), 2, order)?;

    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let task = || -> Result<()> {
        for i in 0..stack.len() {
            let a_slice = stack_slice_at_mut(converted.view_mut(), &converted_stack, i)?.into_dim::<Ix2>();
            let lu_slice = stack_slice_at_mut(lu.view_mut(), &lu_stack, i)?.into_dim::<Ix2>();
            let d_slice = stack_slice_at_mut(d.view_mut(), &d_stack, i)?.into_dim::<Ix2>();
            let perm_slice = stack_slice_at_mut(perm.view_mut(), &perm_stack, i)?.into_dim::<Ix1>();
            ref_impl_ldl_into_f(a_slice.into(), hermitian, uplo, lu_slice, d_slice, perm_slice)?;
        }
        Ok(())
    };
    device.with_blas_num_threads(nthreads, task)?;
    Ok((lu, d, perm))
}

/* #region full-args */

#[duplicate_item(
    ImplType                              ToRef                                    Tr                                ;
   [T, D, R: DataAPI<Data = Vec<T>>] [a.view().to_dyn().into()              ] [&TensorAny<R, T, DeviceBLAS, D>   ];
   ['a, T, D                          ] [a.to_dyn().into()                     ] [TensorView<'a, T, DeviceBLAS, D>  ];
   [    T, D                          ] [a.view_mut().into_dim::<IxD>().into() ] [Tensor<T, DeviceBLAS, D>         ];
   ['a, T, D                          ] [a.view_mut().into_dim::<IxD>().into() ] [TensorMut<'a, T, DeviceBLAS, D>  ];
)]
impl<ImplType> LdlAPI<DeviceBLAS> for (Tr, bool, Option<FlagUpLo>)
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceAPI<usize, Raw = Vec<usize>>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationNumAPI<usize>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
{
    type Out = LdlResult<Tensor<T, DeviceBLAS, D>, Tensor<T, DeviceBLAS, D>, Tensor<usize, DeviceBLAS, D::SmallerOne>>;
    // the mutable rows hand the scratch to the factorization in place
    #[allow(unused_mut)]
    fn ldl_f(self) -> Result<Self::Out> {
        let (mut a, hermitian, uplo) = self;
        let (lu, d, perm) = blas_impl_ldl_f(ToRef, hermitian, uplo)?;
        Ok(LdlResult { lu: lu.into_dim::<D>(), d: d.into_dim::<D>(), perm: perm.into_dim::<D::SmallerOne>() })
    }
}

/* #endregion */

/* #region sub-args */

#[duplicate_item(
    ImplStruct              args_tuple             internal_tuple                    ;
   [(Tr, bool, FlagUpLo)] [(a, hermitian, uplo)] [(a, hermitian, Some(uplo))];
   [(Tr, bool,         )] [(a, hermitian,     )] [(a, hermitian, None      )];
   [(Tr,       FlagUpLo)] [(a,            uplo)] [(a, true,      Some(uplo))];
   [(Tr,               )] [(a,                )] [(a, true,      None      )];
)]
impl<Tr> LdlAPI<DeviceBLAS> for ImplStruct
where
    (Tr, bool, Option<FlagUpLo>): LdlAPI<DeviceBLAS>,
{
    type Out = <(Tr, bool, Option<FlagUpLo>) as LdlAPI<DeviceBLAS>>::Out;
    fn ldl_f(self) -> Result<Self::Out> {
        let args_tuple = self;
        LdlAPI::<DeviceBLAS>::ldl_f(internal_tuple)
    }
}

/* #endregion */

/* #region bare */

#[duplicate_item(
    ImplType                              Tr;
   ['a, T, D, R: DataAPI<Data = Vec<T>>] [&'a TensorAny<R, T, DeviceBLAS, D>];
   ['a, T, D,                          ] [TensorView<'a, T, DeviceBLAS, D>  ];
   [    T, D                           ] [Tensor<T, DeviceBLAS, D>          ];
   ['a, T, D                           ] [TensorMut<'a, T, DeviceBLAS, D>   ];
)]
impl<ImplType> LdlAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    (Tr, bool, Option<FlagUpLo>): LdlAPI<DeviceBLAS>,
{
    type Out = <(Tr, bool, Option<FlagUpLo>) as LdlAPI<DeviceBLAS>>::Out;
    fn ldl_f(self) -> Result<Self::Out> {
        let a = self;
        LdlAPI::<DeviceBLAS>::ldl_f((a, true, None))
    }
}

/* #endregion */
