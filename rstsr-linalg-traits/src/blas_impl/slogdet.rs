use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> SLogDetAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = SLogDetResult<T>;
    fn slogdet_f(self) -> Result<Self::Out> {
        rstsr_assert_eq!(self.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a = self;
        let a_view = a.view().into_dim::<Ix2>();
        let (sign, logabsdet) = ref_impl_slogdet_f(a_view.into())?;
        Ok(SLogDetResult { sign, logabsdet })
    }
}

#[duplicate_item(
    ImplType   Tr                              ;
   ['a, T, D] [TensorMut<'a, T, DeviceBLAS, D>];
   [    T, D] [Tensor<T, DeviceBLAS, D>       ];
)]
impl<ImplType> SLogDetAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = SLogDetResult<T>;
    fn slogdet_f(self) -> Result<Self::Out> {
        rstsr_assert_eq!(self.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let mut a = self;
        let a_view = a.view_mut().into_dim::<Ix2>();
        let (sign, logabsdet) = ref_impl_slogdet_f(a_view.into())?;
        Ok(SLogDetResult { sign, logabsdet })
    }
}
