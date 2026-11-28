//! n-dim `lu` on the BLAS device: the LU decomposition `A = P L U` of every
//! matrix slice of the stack (`?GETRF`), in SciPy's `p_indices=True` form.
//!
//! Matrix slices: the two matrix axes are the last two shape axes under a
//! row-major device and the first two under a column-major one; the rest is
//! the batch.
//!
//! With `K = min(M, N)` for a slice, the result is [`LuResult`] with `p` of
//! shape `(..., M)`, `l` `(..., M, K)` and `u` `(..., K, N)`: `l` is unit
//! lower trapezoidal, `u` upper trapezoidal, and `p` lists row indices such
//! that `A = L[P] @ U` (equivalently `A = P L U` with the permutation matrix
//! `P[i, p[i]] = 1`). Rows `K..M` of `l` hold the multipliers of the
//! eliminated rows. With `permute_l` set, `p` is `None` and `l` is the
//! already permuted `P L`, so that `A = L @ U`.
//!
//! Overloads: `lu(a)` or `lu((a, permute_l))`, where `a` is `&TensorAny`,
//! `TensorView`, `Tensor` or `TensorMut` and `permute_l` a `bool`.
//!
//! The input is consumed as scratch: an owned or mutable `a` whose layout is
//! the device order is destroyed by the factorization; every other `a` is
//! only read (an off-order mutable `a` is factored from a work buffer and
//! left untouched).
//!
//! A singular slice is not an error: its factors are returned, with the
//! exactly-zero pivot on the diagonal of `u`. Zero-size slices are
//! synthesized without a driver call (`p` the identity, zero `l` and `u`).
//! An input with fewer than two axes is an `InvalidLayout` error.
//!
//! See also [`lu_factor`](crate::linalg_auto_impl::lu_factor) and
//! [`lu_solve`](crate::linalg_auto_impl::lu_solve).

use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/// Device-facing entry of the n-dim `lu`; see the module documentation for
/// the shapes and the input policy.
#[allow(clippy::type_complexity)]
pub fn blas_impl_lu_f<T>(
    a: TensorReference<'_, T, DeviceBLAS, IxD>,
    permute_l: bool,
) -> Result<LuResult<Option<Tensor<usize, DeviceBLAS, IxD>>, Tensor<T, DeviceBLAS, IxD>, Tensor<T, DeviceBLAS, IxD>>>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>,
{
    let device = a.device().clone();
    let order = device.default_order();
    let stack = StackLayout::new(a.layout(), 2, order)?;
    let inner_shape = stack.inner_shape();
    let (m, n) = (inner_shape[0], inner_shape[1]);
    let k = m.min(n);
    let batch_shape = stack.batch_shape().to_vec();

    let mut p: Tensor<usize, DeviceBLAS, IxD> = zeros_f((stack_shape(&batch_shape, &[m], order), &device))?;
    let mut l: Tensor<T, DeviceBLAS, IxD> = zeros_f((stack_shape(&batch_shape, &[m, k], order), &device))?;
    let mut u: Tensor<T, DeviceBLAS, IxD> = zeros_f((stack_shape(&batch_shape, &[k, n], order), &device))?;
    let p_stack = StackLayout::new(p.layout(), 1, order)?;
    let l_stack = StackLayout::new(l.layout(), 2, order)?;
    let u_stack = StackLayout::new(u.layout(), 2, order)?;

    let mut converted = overwritable_convert_with_order(a, order)?;
    let converted_stack = StackLayout::new(converted.view().layout(), 2, order)?;

    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let task = || -> Result<()> {
        for i in 0..stack.len() {
            let a_slice = stack_slice_at_mut(converted.view_mut(), &converted_stack, i)?.into_dim::<Ix2>();
            let p_slice = stack_slice_at_mut(p.view_mut(), &p_stack, i)?.into_dim::<Ix1>();
            let l_slice = stack_slice_at_mut(l.view_mut(), &l_stack, i)?.into_dim::<Ix2>();
            let u_slice = stack_slice_at_mut(u.view_mut(), &u_stack, i)?.into_dim::<Ix2>();
            ref_impl_lu_into_f(a_slice.into(), p_slice, l_slice, u_slice, permute_l)?;
        }
        Ok(())
    };
    device.with_blas_num_threads(nthreads, task)?;

    let p = (!permute_l).then_some(p);
    Ok(LuResult { p, l, u })
}

/* #region full-args */

#[duplicate_item(
    ImplType                              ToRef                                       Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [a.view().to_dyn().into()               ] [&TensorAny<R, T, DeviceBLAS, D>   ];
   ['a, T, D                          ] [a.to_dyn().into()                      ] [TensorView<'a, T, DeviceBLAS, D>  ];
   [    T, D                          ] [a.view_mut().into_dim::<IxD>().into()  ] [Tensor<T, DeviceBLAS, D>          ];
   ['a, T, D                          ] [a.view_mut().into_dim::<IxD>().into()  ] [TensorMut<'a, T, DeviceBLAS, D>   ];
)]
impl<ImplType> LuAPI<DeviceBLAS> for (Tr, bool)
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out =
        LuResult<Option<Tensor<usize, DeviceBLAS, D::SmallerOne>>, Tensor<T, DeviceBLAS, D>, Tensor<T, DeviceBLAS, D>>;
    // the view rows never mutate `a`
    #[allow(unused_mut)]
    fn lu_f(self) -> Result<Self::Out> {
        let (mut a, permute_l) = self;
        let LuResult { p, l, u } = blas_impl_lu_f(ToRef, permute_l)?;
        Ok(LuResult { p: p.map(|p| p.into_dim::<D::SmallerOne>()), l: l.into_dim::<D>(), u: u.into_dim::<D>() })
    }
}

/* #endregion */

/* #region sub-args */

#[duplicate_item(
    ImplType                              Tr                               ;
   ['a, T, D, R: DataAPI<Data = Vec<T>>] [&'a TensorAny<R, T, DeviceBLAS, D> ];
   ['a, T, D                          ] [TensorView<'a, T, DeviceBLAS, D>  ];
   [    T, D                          ] [Tensor<T, DeviceBLAS, D>          ];
   ['a, T, D                          ] [TensorMut<'a, T, DeviceBLAS, D>   ];
)]
impl<ImplType> LuAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    (Tr, bool): LuAPI<DeviceBLAS>,
{
    type Out = <(Tr, bool) as LuAPI<DeviceBLAS>>::Out;
    fn lu_f(self) -> Result<Self::Out> {
        let a = self;
        LuAPI::<DeviceBLAS>::lu_f((a, false))
    }
}

/* #endregion */
