use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/* #region full-args */

impl<T, D, R> SVDAPI<DeviceBLAS> for (&TensorAny<R, T, DeviceBLAS, D>, bool)
where
    R: DataAPI<Data = Vec<T>>,
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out =
        SVDResult<Tensor<T, DeviceBLAS, D>, Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tensor<T, DeviceBLAS, D>>;
    fn svd_f(self) -> Result<Self::Out> {
        let (a, full_matrices) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a = a.view().into_dim::<Ix2>();
        let svd_args = SVDArgs::default().a(a).full_matrices(full_matrices).build()?;
        let (u, s, vt) = ref_impl_svd_simple_f(svd_args)?;
        // convert dimensions
        let u = u.unwrap().into_dim::<IxD>().into_dim::<D>();
        let vt = vt.unwrap().into_dim::<IxD>().into_dim::<D>();
        let s = s.into_dim::<IxD>().into_dim::<D::SmallerOne>();
        Ok(SVDResult { u, s, vt })
    }
}

#[duplicate_item(
    Tr; [Tensor<T, DeviceBLAS, D>]; [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<T, D> SVDAPI<DeviceBLAS> for (Tr, bool)
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out =
        SVDResult<Tensor<T, DeviceBLAS, D>, Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tensor<T, DeviceBLAS, D>>;
    fn svd_f(self) -> Result<Self::Out> {
        let (a, full_matrices) = self;
        SVDAPI::<DeviceBLAS>::svd_f((&a, full_matrices))
    }
}

/* #endregion */

/* #region sub-args */

#[duplicate_item(
    ImplType                              Tr;
   ['a, T, D, R: DataAPI<Data = Vec<T>>] [&'a TensorAny<R, T, DeviceBLAS, D>];
   ['a, T, D,                          ] [TensorView<'a, T, DeviceBLAS, D>  ];
   [    T, D                           ] [Tensor<T, DeviceBLAS, D>          ];
)]
impl<ImplType> SVDAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI,
    (Tr, bool): SVDAPI<DeviceBLAS>,
{
    type Out = <(Tr, bool) as SVDAPI<DeviceBLAS>>::Out;
    fn svd_f(self) -> Result<Self::Out> {
        let a = self;
        SVDAPI::<DeviceBLAS>::svd_f((a, true))
    }
}

/* #endregion */

/* #region SVDArgs implementation */

impl<'a, T> SVDAPI<DeviceBLAS> for SVDArgs<'a, DeviceBLAS, T>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = SVDResult<Tensor<T, DeviceBLAS, Ix2>, Tensor<T::Real, DeviceBLAS, Ix1>, Tensor<T, DeviceBLAS, Ix2>>;
    fn svd_f(self) -> Result<Self::Out> {
        SVDAPI::<DeviceBLAS>::svd_f(self.build()?)
    }
}

impl<'a, T> SVDAPI<DeviceBLAS> for SVDArgs_<'a, DeviceBLAS, T>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = SVDResult<Tensor<T, DeviceBLAS, Ix2>, Tensor<T::Real, DeviceBLAS, Ix1>, Tensor<T, DeviceBLAS, Ix2>>;
    fn svd_f(self) -> Result<Self::Out> {
        let args = self;
        rstsr_assert!(
            args.full_matrices.is_some(),
            InvalidValue,
            "`svd` must compute UV. Refer to `svdvals` if UV is not required."
        )?;
        let (u, s, vt) = ref_impl_svd_simple_f(args)?;
        let u = u.unwrap().into_dim::<IxD>().into_dim::<Ix2>();
        let vt = vt.unwrap().into_dim::<IxD>().into_dim::<Ix2>();
        let s = s.into_dim::<IxD>().into_dim::<Ix1>();
        Ok(SVDResult { u, s, vt })
    }
}

/* #endregion */
