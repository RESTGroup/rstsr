use crate::DeviceBLAS;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/* #region eig */

/// Implementation of EigAPI for tensors that return complex eigenvalues
///
/// For real input matrices, eigenvalues are always returned as complex numbers
/// since non-symmetric matrices can have complex eigenvalues.

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> EigAPI<DeviceBLAS> for (Tr, bool, bool)
where
    T: BlasFloat,
    T::Real: num::Zero + PartialEq + Clone,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
    DeviceBLAS: DeviceAPI<T::Real>,
    DeviceBLAS: DeviceAPI<Complex<T::Real>>,
    DeviceBLAS: DeviceCreationAnyAPI<Complex<T::Real>>,
    DeviceBLAS: DeviceRawAPI<T::Real, Raw = Vec<T::Real>>,
    DeviceBLAS: DeviceRawAPI<T, Raw = Vec<T>>,
    DeviceBLAS: DeviceRawAPI<Complex<T::Real>, Raw = Vec<Complex<T::Real>>>,
    (): EigenvectorConvertAPI<DeviceBLAS, T, ComplexType = Tensor<Complex<T::Real>, DeviceBLAS, Ix2>>,
{
    type Out = EigResult<
        Tensor<Complex<T::Real>, DeviceBLAS, D::SmallerOne>,
        Tensor<Complex<T::Real>, DeviceBLAS, D>,
        Tensor<Complex<T::Real>, DeviceBLAS, D>,
    >;

    fn eig_f(self) -> Result<Self::Out> {
        let (a, left, right) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let result = ref_impl_eig_f(a_view.into(), left, right)?;

        // Convert dimensions
        let eigenvalues = result.eigenvalues.into_dim::<IxD>().into_dim::<D::SmallerOne>();
        let left_eigenvectors = result.left_eigenvectors.map(|v| v.into_dim::<IxD>().into_dim::<D>());
        let right_eigenvectors = result.right_eigenvectors.map(|v| v.into_dim::<IxD>().into_dim::<D>());

        Ok(EigResult { eigenvalues, left_eigenvectors, right_eigenvectors })
    }
}

/// Default implementation: right eigenvectors only
#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> EigAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    T::Real: num::Zero + PartialEq + Clone,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
    DeviceBLAS: DeviceAPI<T::Real>,
    DeviceBLAS: DeviceAPI<Complex<T::Real>>,
    DeviceBLAS: DeviceCreationAnyAPI<Complex<T::Real>>,
    DeviceBLAS: DeviceRawAPI<T::Real, Raw = Vec<T::Real>>,
    DeviceBLAS: DeviceRawAPI<T, Raw = Vec<T>>,
    DeviceBLAS: DeviceRawAPI<Complex<T::Real>, Raw = Vec<Complex<T::Real>>>,
    (): EigenvectorConvertAPI<DeviceBLAS, T, ComplexType = Tensor<Complex<T::Real>, DeviceBLAS, Ix2>>,
{
    type Out = EigResult<
        Tensor<Complex<T::Real>, DeviceBLAS, D::SmallerOne>,
        Tensor<Complex<T::Real>, DeviceBLAS, D>,
        Tensor<Complex<T::Real>, DeviceBLAS, D>,
    >;

    fn eig_f(self) -> Result<Self::Out> {
        EigAPI::<DeviceBLAS>::eig_f((self, false, true))
    }
}

/* #endregion */

/* #region generalized eig */

/// Implementation of generalized EigAPI for (A, B) matrix pair
#[duplicate_item(
    ImplType                                                       TrA                                TrB                              ;
   [T, D, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceBLAS, D>] [&TensorAny<Rb, T, DeviceBLAS, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceBLAS, D> ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceBLAS, D>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D,                                                       ] [TensorView<'_, T, DeviceBLAS, D>] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> EigAPI<DeviceBLAS> for (TrA, TrB, bool, bool)
where
    T: BlasFloat,
    T::Real: num::Zero + PartialEq + Clone,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
    DeviceBLAS: GGEVDriverAPI<T>,
    DeviceBLAS: DeviceAPI<T::Real>,
    DeviceBLAS: DeviceAPI<Complex<T::Real>>,
    DeviceBLAS: DeviceCreationAnyAPI<Complex<T::Real>>,
    DeviceBLAS: DeviceRawAPI<T::Real, Raw = Vec<T::Real>>,
    DeviceBLAS: DeviceRawAPI<T, Raw = Vec<T>>,
    DeviceBLAS: DeviceRawAPI<Complex<T::Real>, Raw = Vec<Complex<T::Real>>>,
    (): EigenvectorConvertAPI<DeviceBLAS, T, ComplexType = Tensor<Complex<T::Real>, DeviceBLAS, Ix2>>,
{
    type Out = EigResult<
        Tensor<Complex<T::Real>, DeviceBLAS, D::SmallerOne>,
        Tensor<Complex<T::Real>, DeviceBLAS, D>,
        Tensor<Complex<T::Real>, DeviceBLAS, D>,
    >;

    fn eig_f(self) -> Result<Self::Out> {
        let (a, b, left, right) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = b.view().into_dim::<Ix2>();
        let result = ref_impl_eig_generalized_f(a_view.into(), b_view.into(), left, right)?;

        // Convert dimensions
        let eigenvalues = result.eigenvalues.into_dim::<IxD>().into_dim::<D::SmallerOne>();
        let left_eigenvectors = result.left_eigenvectors.map(|v| v.into_dim::<IxD>().into_dim::<D>());
        let right_eigenvectors = result.right_eigenvectors.map(|v| v.into_dim::<IxD>().into_dim::<D>());

        Ok(EigResult { eigenvalues, left_eigenvectors, right_eigenvectors })
    }
}

/// Implementation of generalized EigAPI for (A, B) with default eigenvector options
#[duplicate_item(
    ImplType                                                       TrA                                TrB                              ;
   [T, D, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceBLAS, D>] [&TensorAny<Rb, T, DeviceBLAS, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceBLAS, D> ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceBLAS, D>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D,                                                       ] [TensorView<'_, T, DeviceBLAS, D>] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> EigAPI<DeviceBLAS> for (TrA, TrB)
where
    T: BlasFloat,
    T::Real: num::Zero + PartialEq + Clone,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
    DeviceBLAS: GGEVDriverAPI<T>,
    DeviceBLAS: DeviceAPI<T::Real>,
    DeviceBLAS: DeviceAPI<Complex<T::Real>>,
    DeviceBLAS: DeviceCreationAnyAPI<Complex<T::Real>>,
    DeviceBLAS: DeviceRawAPI<T::Real, Raw = Vec<T::Real>>,
    DeviceBLAS: DeviceRawAPI<T, Raw = Vec<T>>,
    DeviceBLAS: DeviceRawAPI<Complex<T::Real>, Raw = Vec<Complex<T::Real>>>,
    (): EigenvectorConvertAPI<DeviceBLAS, T, ComplexType = Tensor<Complex<T::Real>, DeviceBLAS, Ix2>>,
{
    type Out = EigResult<
        Tensor<Complex<T::Real>, DeviceBLAS, D::SmallerOne>,
        Tensor<Complex<T::Real>, DeviceBLAS, D>,
        Tensor<Complex<T::Real>, DeviceBLAS, D>,
    >;

    fn eig_f(self) -> Result<Self::Out> {
        EigAPI::<DeviceBLAS>::eig_f((self.0, self.1, false, true))
    }
}

/* #endregion */

/* #region eigvals */

/// Implementation of EigvalsAPI - returns only eigenvalues
#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> EigvalsAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    T::Real: num::Zero + PartialEq + Clone,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
    DeviceBLAS: DeviceAPI<T::Real>,
    DeviceBLAS: DeviceAPI<Complex<T::Real>>,
    DeviceBLAS: DeviceCreationAnyAPI<Complex<T::Real>>,
    DeviceBLAS: DeviceRawAPI<T::Real, Raw = Vec<T::Real>>,
    DeviceBLAS: DeviceRawAPI<T, Raw = Vec<T>>,
    DeviceBLAS: DeviceRawAPI<Complex<T::Real>, Raw = Vec<Complex<T::Real>>>,
    (): EigenvectorConvertAPI<DeviceBLAS, T, ComplexType = Tensor<Complex<T::Real>, DeviceBLAS, Ix2>>,
{
    type Out = Tensor<Complex<T::Real>, DeviceBLAS, D::SmallerOne>;

    fn eigvals_f(self) -> Result<Self::Out> {
        let result = EigAPI::<DeviceBLAS>::eig_f((self, false, false))?;
        Ok(result.eigenvalues)
    }
}

/* #endregion */
