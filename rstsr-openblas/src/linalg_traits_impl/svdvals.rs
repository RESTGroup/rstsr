use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/* #region full-args */

impl<T, D, R> SVDvalsAPI<DeviceBLAS> for &TensorAny<R, T, DeviceBLAS, D>
where
    R: DataAPI<Data = Vec<T>>,
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T::Real, DeviceBLAS, D::SmallerOne>;
    fn svdvals_f(self) -> Result<Self::Out> {
        let a = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a = a.view().into_dim::<Ix2>();
        let svd_args = SVDArgs::default().a(a).full_matrices(None).build()?;
        let (_, s, _) = ref_impl_svd_simple_f(svd_args)?;
        // convert dimensions
        let s = s.into_dim::<IxD>().into_dim::<D::SmallerOne>();
        Ok(s)
    }
}

#[duplicate_item(
    Tr; [Tensor<T, DeviceBLAS, D>]; [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<T, D> SVDvalsAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T::Real, DeviceBLAS, D::SmallerOne>;
    fn svdvals_f(self) -> Result<Self::Out> {
        let a = self;
        SVDvalsAPI::<DeviceBLAS>::svdvals_f(&a)
    }
}

/* #endregion */

/* #region SVDArgs implementation */

impl<'a, T> SVDvalsAPI<DeviceBLAS> for SVDArgs<'a, DeviceBLAS, T>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T::Real, DeviceBLAS, Ix1>;
    fn svdvals_f(self) -> Result<Self::Out> {
        SVDvalsAPI::<DeviceBLAS>::svdvals_f(self.build()?)
    }
}

impl<'a, T> SVDvalsAPI<DeviceBLAS> for SVDArgs_<'a, DeviceBLAS, T>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T::Real, DeviceBLAS, Ix1>;
    fn svdvals_f(self) -> Result<Self::Out> {
        let args = self;
        rstsr_assert!(
            args.full_matrices.is_none(),
            InvalidValue,
            "`svdvals` must not compute UV. Refer to `svd` if UV is required."
        )?;
        let (_, s, _) = ref_impl_svd_simple_f(args)?;
        let s = s.into_dim::<IxD>().into_dim::<Ix1>();
        Ok(s)
    }
}

/* #endregion */
