use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
   ['a, T, D                       ] [TensorMut<'a, T, DeviceBLAS, D> ];
   [T, D                           ] [Tensor<T, DeviceBLAS, D>        ];
)]
impl<ImplType> SLogDetAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = SLogDetResult<Tensor<T, DeviceBLAS, IxD>, Tensor<T::Real, DeviceBLAS, IxD>>;
    fn slogdet_f(self) -> Result<Self::Out> {
        let a = self;
        let a_view = a.to_dyn();
        let (sign, logabsdet) = ref_impl_slogdet_nd_f(a_view)?;
        Ok(SLogDetResult { sign, logabsdet })
    }
}
