//! n-dim `lu_factor` on the BLAS device: the packed LU factorization of
//! every matrix slice of the stack (`?GETRF`), in SciPy's `lu_factor` form.
//!
//! Matrix slices: the two matrix axes are the last two shape axes under a
//! row-major device and the first two under a column-major one; the rest is
//! the batch.
//!
//! The result is [`LuFactorResult`]: `lu` of shape `(..., M, N)` holds `U`
//! in its upper triangle (diagonal included) and the multipliers of the unit
//! lower trapezoidal `L` below it; `piv` of shape `(..., min(M, N))` lists,
//! 0-based, the row that was interchanged with row `i` at step `i`. Together
//! they are the packed factor form that [`lu_solve`] takes.
//!
//! The input is consumed as scratch: an immutable `a` is factored into a
//! fresh buffer; an owned or mutable `a` whose layout is the device order is
//! factored in place and destroyed (strided is fine, the leading dimension
//! covers the gap); an off-order mutable `a` is factored from a work buffer
//! that is written back into `a`.
//!
//! A singular slice is not an error: its factors are returned, with the
//! exactly-zero pivot on the diagonal of `lu`. Zero-size slices are
//! synthesized without a driver call (`lu` all zeros, empty `piv`). An input
//! with fewer than two axes is an `InvalidLayout` error.
//!
//! See also [`lu`](crate::linalg_auto_impl::lu) and
//! [`lu_solve`](crate::linalg_auto_impl::lu_solve).

use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/// Device-facing entry of the n-dim `lu_factor`; see the module
/// documentation for the shapes and the input policy.
#[allow(clippy::type_complexity)]
pub fn blas_impl_lu_factor_f<T>(
    a: TensorReference<'_, T, DeviceBLAS, IxD>,
) -> Result<(TensorMutable<'_, T, DeviceBLAS, IxD>, Tensor<usize, DeviceBLAS, IxD>)>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>,
{
    let device = a.device().clone();
    let order = device.default_order();
    let stack = StackLayout::new(a.layout(), 2, order)?;
    let inner_shape = stack.inner_shape();
    let k = inner_shape[0].min(inner_shape[1]);
    let batch_shape = stack.batch_shape().to_vec();

    let mut piv: Tensor<usize, DeviceBLAS, IxD> = zeros_f((stack_shape(&batch_shape, &[k], order), &device))?;
    let piv_stack = StackLayout::new(piv.layout(), 1, order)?;

    let mut converted = overwritable_convert_with_order(a, order)?;
    let converted_stack = StackLayout::new(converted.view().layout(), 2, order)?;

    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let task = || -> Result<()> {
        for i in 0..stack.len() {
            let slice = stack_slice_at_mut(converted.view_mut(), &converted_stack, i)?.into_dim::<Ix2>();
            let (result, ipiv, _info) = ref_impl_lu_factor_f(slice.into())?;
            result.clone_to_mut();
            let mut piv_slice = stack_slice_at_mut(piv.view_mut(), &piv_stack, i)?.into_dim::<Ix1>();
            let offsets: Vec<usize> = IterLayout::new(piv_slice.layout(), TensorIterOrder::C)?.collect();
            for (dst, &src) in offsets.iter().zip(ipiv.iter()) {
                piv_slice.raw_mut()[*dst] = src;
            }
        }
        Ok(())
    };
    device.with_blas_num_threads(nthreads, task)?;
    Ok((converted, piv))
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> LuFactorAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = LuFactorResult<Tensor<T, DeviceBLAS, D>, Tensor<usize, DeviceBLAS, D::SmallerOne>>;
    fn lu_factor_f(self) -> Result<Self::Out> {
        let view = self.view();
        let (lu, piv) = blas_impl_lu_factor_f(view.to_dyn().into())?;
        Ok(LuFactorResult { lu: lu.into_owned().into_dim::<D>(), piv: piv.into_dim::<D::SmallerOne>() })
    }
}

#[duplicate_item(
    ImplType   Tr                              ;
   ['a, T, D] [TensorMut<'a, T, DeviceBLAS, D>];
   [    T, D] [Tensor<T, DeviceBLAS, D>       ];
)]
impl<ImplType> LuFactorAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = LuFactorResult<Tr, Tensor<usize, DeviceBLAS, D::SmallerOne>>;
    fn lu_factor_f(self) -> Result<Self::Out> {
        let mut a = self;
        let (result, piv) = blas_impl_lu_factor_f(a.view_mut().into_dim::<IxD>().into())?;
        result.clone_to_mut();
        Ok(LuFactorResult { lu: a, piv: piv.into_dim::<D::SmallerOne>() })
    }
}
