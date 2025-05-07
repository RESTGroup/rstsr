use crate::DeviceBLAS;
use duplicate::duplicate_item;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D                           ] [TensorCow<'_, T, DeviceBLAS, D> ];
)]
impl<ImplType> InvAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn inv_f(self) -> Result<Self::Out> {
        rstsr_assert_eq!(
            self.ndim(),
            2,
            InvalidLayout,
            "Currently we can only handle 2-D matrix."
        )?;
        let a = self.view().into_dim::<Ix2>();
        let result = ref_impl_inv_f(a.into())?.into_owned();
        Ok(result.into_dim::<IxD>().into_dim::<D>())
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
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tr;
    fn inv_f(self) -> Result<Self::Out> {
        rstsr_assert_eq!(
            self.ndim(),
            2,
            InvalidLayout,
            "Currently we can only handle 2-D matrix."
        )?;
        let mut a = self;
        let a_view = a.view_mut().into_dim::<Ix2>();
        let result = ref_impl_inv_f(a_view.into())?;
        result.clone_to_mut();
        Ok(a)
    }
}
