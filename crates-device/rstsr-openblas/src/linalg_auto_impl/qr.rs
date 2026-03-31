use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/* #region full-args (tensor view with mode and pivoting) */

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> QRAPI<DeviceBLAS> for (Tr, &'static str, bool)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = QRResult<
        Tensor<T, DeviceBLAS, D>,
        Tensor<T, DeviceBLAS, D>,
        Tensor<T, DeviceBLAS, D>,
        Tensor<T, DeviceBLAS, Ix1>,
        Tensor<blas_int, DeviceBLAS, Ix1>,
    >;
    fn qr_f(self) -> Result<Self::Out> {
        let (a, mode, pivoting) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a = a.view().into_dim::<Ix2>();
        let result = ref_impl_qr_f(a.into(), mode, pivoting)?;

        // Convert results back to original dimensionality D
        match mode {
            "raw" => Ok(QRResult {
                q: None,
                r: None,
                h: result.h.map(|h| h.into_owned().into_dim::<IxD>().into_dim::<D>()),
                tau: result.tau,
                p: result.p,
            }),
            "r" => Ok(QRResult {
                q: None,
                r: result.r.map(|r| r.into_dim::<IxD>().into_dim::<D>()),
                h: None,
                tau: None,
                p: result.p,
            }),
            _ => Ok(QRResult {
                q: result.q.map(|q| q.into_dim::<IxD>().into_dim::<D>()),
                r: result.r.map(|r| r.into_dim::<IxD>().into_dim::<D>()),
                h: None,
                tau: None,
                p: result.p,
            }),
        }
    }
}

/* #endregion */

/* #region sub-args: tensor view with mode only (no pivoting) */

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> QRAPI<DeviceBLAS> for (Tr, &'static str)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = QRResult<
        Tensor<T, DeviceBLAS, D>,
        Tensor<T, DeviceBLAS, D>,
        Tensor<T, DeviceBLAS, D>,
        Tensor<T, DeviceBLAS, Ix1>,
        Tensor<blas_int, DeviceBLAS, Ix1>,
    >;
    fn qr_f(self) -> Result<Self::Out> {
        let (a, mode) = self;
        QRAPI::<DeviceBLAS>::qr_f((a, mode, false))
    }
}

/* #endregion */

/* #region default args: tensor view only (reduced mode, no pivoting) */

#[duplicate_item(
    ImplType                              Tr;
   ['a, T, D, R: DataAPI<Data = Vec<T>>] [&'a TensorAny<R, T, DeviceBLAS, D>];
   ['a, T, D,                          ] [TensorView<'a, T, DeviceBLAS, D>  ];
   [    T, D                           ] [Tensor<T, DeviceBLAS, D>          ];
)]
impl<ImplType> QRAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI,
    (Tr, &'static str, bool): QRAPI<DeviceBLAS>,
{
    type Out = <(Tr, &'static str, bool) as QRAPI<DeviceBLAS>>::Out;
    fn qr_f(self) -> Result<Self::Out> {
        QRAPI::<DeviceBLAS>::qr_f((self, "reduced", false))
    }
}

/* #endregion */

/* #region owned/mutable tensor: in-place when possible */

#[duplicate_item(
    ImplType   Tr                              ;
   ['a, T, D] [TensorMut<'a, T, DeviceBLAS, D>];
   [    T, D] [Tensor<T, DeviceBLAS, D>       ];
)]
impl<ImplType> QRAPI<DeviceBLAS> for (Tr, &'static str, bool)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = QRResult<
        Tensor<T, DeviceBLAS, D>,
        Tensor<T, DeviceBLAS, D>,
        Tensor<T, DeviceBLAS, D>,
        Tensor<T, DeviceBLAS, Ix1>,
        Tensor<blas_int, DeviceBLAS, Ix1>,
    >;
    fn qr_f(self) -> Result<Self::Out> {
        let (a, mode, pivoting) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_ix2 = a.view().into_dim::<Ix2>();
        let result = ref_impl_qr_f(a_ix2.into(), mode, pivoting)?;

        // Convert results back to original dimensionality D
        match mode {
            "raw" => Ok(QRResult {
                q: None,
                r: None,
                h: result.h.map(|h| h.into_owned().into_dim::<IxD>().into_dim::<D>()),
                tau: result.tau,
                p: result.p,
            }),
            "r" => Ok(QRResult {
                q: None,
                r: result.r.map(|r| r.into_dim::<IxD>().into_dim::<D>()),
                h: None,
                tau: None,
                p: result.p,
            }),
            _ => Ok(QRResult {
                q: result.q.map(|q| q.into_dim::<IxD>().into_dim::<D>()),
                r: result.r.map(|r| r.into_dim::<IxD>().into_dim::<D>()),
                h: None,
                tau: None,
                p: result.p,
            }),
        }
    }
}

#[duplicate_item(
    ImplType   Tr                              ;
   ['a, T, D] [TensorMut<'a, T, DeviceBLAS, D>];
   [    T, D] [Tensor<T, DeviceBLAS, D>       ];
)]
impl<ImplType> QRAPI<DeviceBLAS> for (Tr, &'static str)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = QRResult<
        Tensor<T, DeviceBLAS, D>,
        Tensor<T, DeviceBLAS, D>,
        Tensor<T, DeviceBLAS, D>,
        Tensor<T, DeviceBLAS, Ix1>,
        Tensor<blas_int, DeviceBLAS, Ix1>,
    >;
    fn qr_f(self) -> Result<Self::Out> {
        let (a, mode) = self;
        QRAPI::<DeviceBLAS>::qr_f((a, mode, false))
    }
}

/* #endregion */
