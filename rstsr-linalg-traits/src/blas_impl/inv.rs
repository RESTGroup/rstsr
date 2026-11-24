use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/// n-dim `inv`: invert every matrix slice of the stack (the two matrix
/// axes are the last two shape axes under a row-major device, the first two
/// under a column-major one; the rest is the batch).
///
/// The whole stack goes through `overwritable_convert` once: an immutable
/// input is inverted into a fresh contiguous buffer; a mutable input whose
/// layout matches the device order is inverted in place, even when strided
/// (the leading dimension covers the gap); a mutable input whose layout
/// disagrees with the device order is inverted in a contiguous work buffer,
/// written back by `clone_to_mut` at the call site.
pub fn blas_impl_inv_f<T>(a: TensorReference<'_, T, DeviceBLAS, IxD>) -> Result<TensorMutable<'_, T, DeviceBLAS, IxD>>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, IxD> + OpAssignAPI<T, IxD>,
{
    let device = a.device().clone();
    let order = device.default_order();
    let stack = StackLayout::new(a.layout(), 2, order)?;
    let inner_shape = stack.inner_shape();
    rstsr_assert_eq!(
        inner_shape[0],
        inner_shape[1],
        InvalidLayout,
        "inv: the matrix must be square, got {}x{}",
        inner_shape[0],
        inner_shape[1]
    )?;

    let mut converted = overwritable_convert_with_order(a, order)?;
    let converted_stack = StackLayout::new(converted.view().layout(), 2, order)?;

    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let task = || -> Result<()> {
        for i in 0..stack.len() {
            let slice = stack_slice_at_mut(converted.view_mut(), &converted_stack, i)?.into_dim::<Ix2>();
            let result = ref_impl_inv_f(slice.into())?;
            result.clone_to_mut();
        }
        Ok(())
    };
    device.with_blas_num_threads(nthreads, task)?;
    Ok(converted)
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> InvAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, IxD> + OpAssignAPI<T, IxD>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn inv_f(self) -> Result<Self::Out> {
        Ok(blas_impl_inv_f(self.view().to_dyn().into())?.into_owned().into_dim::<D>())
    }
}

#[duplicate_item(
    ImplType   Tr                              ;
   ['a, T, D] [TensorMut<'a, T, DeviceBLAS, D>];
   [    T, D] [Tensor<T, DeviceBLAS, D>       ];
)]
impl<ImplType> InvAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, IxD> + OpAssignAPI<T, IxD>,
{
    type Out = Tr;
    fn inv_f(self) -> Result<Self::Out> {
        let mut a = self;
        let result = blas_impl_inv_f(a.view_mut().into_dim::<IxD>().into())?;
        result.clone_to_mut();
        Ok(a)
    }
}
