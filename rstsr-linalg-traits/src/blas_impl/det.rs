use crate::DeviceBLAS;
use num::complex::ComplexFloat;
use rstsr_blas_traits::prelude::*;
use rstsr_blas_traits::util::util_tensor::overwritable_convert_with_order;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/// n-dim `det`: the determinant of every matrix slice of the stack (the two
/// matrix axes are the last two shape axes under a row-major device, the
/// first two under a column-major one; the rest is the batch), with the batch
/// shape; a 2-D input yields a 0-d tensor.
///
/// The whole stack goes through `overwritable_convert` once: an immutable
/// input is factored from a fresh contiguous buffer; a mutable input is
/// destroyed by the per-slice LU factorization.
pub fn blas_impl_det_f<T>(a: TensorReference<'_, T, DeviceBLAS, IxD>) -> Result<Tensor<T, DeviceBLAS, IxD>>
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
        "det: the matrix must be square, got {}x{}",
        inner_shape[0],
        inner_shape[1]
    )?;

    let mut converted = overwritable_convert_with_order(a, order)?;
    let converted_stack = StackLayout::new(converted.view().layout(), 2, order)?;
    let mut out: Tensor<T, DeviceBLAS, IxD> = zeros_f((stack.batch_shape().to_vec(), &device))?;
    let mut out_offsets = IterLayout::new(out.layout(), TensorIterOrder::C)?;

    for i in 0..stack.len() {
        let slice = stack_slice_at_mut(converted.view_mut(), &converted_stack, i)?.into_dim::<Ix2>();
        let (sign, logabsdet) = ref_impl_slogdet_f(slice.into())?;
        out.raw_mut()[out_offsets.next().unwrap_or(i)] = sign * logabsdet.exp();
    }
    Ok(out)
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> DetAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, IxD> + OpAssignAPI<T, IxD>,
{
    type Out = Tensor<T, DeviceBLAS, IxD>;
    fn det_f(self) -> Result<Self::Out> {
        blas_impl_det_f(self.view().to_dyn().into())
    }
}

#[duplicate_item(
    ImplType   Tr                              ;
   ['a, T, D] [TensorMut<'a, T, DeviceBLAS, D>];
   [    T, D] [Tensor<T, DeviceBLAS, D>       ];
)]
impl<ImplType> DetAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, IxD> + OpAssignAPI<T, IxD>,
{
    type Out = Tensor<T, DeviceBLAS, IxD>;
    fn det_f(self) -> Result<Self::Out> {
        let mut a = self;
        blas_impl_det_f(a.view_mut().into_dim::<IxD>().into())
    }
}
