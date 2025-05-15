use crate::DeviceBLAS;
use num::FromPrimitive;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

impl<T, D, R> PinvAPI<DeviceBLAS> for (&TensorAny<R, T, DeviceBLAS, D>, T::Real, T::Real)
where
    R: DataAPI<Data = Vec<T>>,
    T: BlasFloat,
    T::Real: FromPrimitive,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = PinvResult<Tensor<T, DeviceBLAS, D>>;
    fn pinv_f(self) -> Result<Self::Out> {
        let (a, atol, rtol) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a = a.view().into_dim::<Ix2>();
        let (b, rank) = ref_impl_pinv_f(a, Some(atol), Some(rtol))?.into();
        let b = b.into_dim::<IxD>().into_dim::<D>();
        return Ok(PinvResult { pinv: b, rank });
    }
}

impl<T, D, R> PinvAPI<DeviceBLAS> for &TensorAny<R, T, DeviceBLAS, D>
where
    R: DataAPI<Data = Vec<T>>,
    T: BlasFloat,
    T::Real: FromPrimitive,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = PinvResult<Tensor<T, DeviceBLAS, D>>;
    fn pinv_f(self) -> Result<Self::Out> {
        let a = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a = a.view().into_dim::<Ix2>();
        let (pinv, rank) = ref_impl_pinv_f(a, None, None)?.into();
        let pinv = pinv.into_dim::<IxD>().into_dim::<D>();
        return Ok(PinvResult { pinv, rank });
    }
}

#[duplicate_item(
    Tr                               ;
   [Tensor<T, DeviceBLAS, D>        ];
   [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<T, D> PinvAPI<DeviceBLAS> for (Tr, T::Real, T::Real)
where
    T: BlasFloat,
    T::Real: FromPrimitive,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = PinvResult<Tensor<T, DeviceBLAS, D>>;
    fn pinv_f(self) -> Result<Self::Out> {
        let (a, atol, rtol) = self;
        PinvAPI::<DeviceBLAS>::pinv_f((&a, atol, rtol))
    }
}

#[duplicate_item(
    Tr                               ;
   [Tensor<T, DeviceBLAS, D>        ];
   [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<T, D> PinvAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    T::Real: FromPrimitive,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = PinvResult<Tensor<T, DeviceBLAS, D>>;
    fn pinv_f(self) -> Result<Self::Out> {
        let a = self;
        PinvAPI::<DeviceBLAS>::pinv_f(&a)
    }
}
