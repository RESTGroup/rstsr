use crate::DeviceBLAS;
use duplicate::duplicate_item;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/* #region simple eigh */

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D                           ] [TensorCow<'_, T, DeviceBLAS, D> ];
)]
impl<ImplType> EighAPI<DeviceBLAS> for (Tr, FlagUpLo)
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tensor<T, DeviceBLAS, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let eigh_args = EighArgs::default().a(a_view).uplo(uplo).build()?;
        let (vals, vecs) = ref_impl_eigh_simple_f(eigh_args)?;
        let vals = vals.into_dim::<IxD>().into_dim::<D::SmallerOne>();
        let vecs = vecs.unwrap().into_owned().into_dim::<IxD>().into_dim::<D>();
        return Ok(EighResult { eigenvalues: vals, eigenvectors: vecs });
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D                           ] [TensorCow<'_, T, DeviceBLAS, D> ];
)]
impl<ImplType> EighAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tensor<T, DeviceBLAS, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let a = self;
        let uplo = match a.device().default_order() {
            RowMajor => Lower,
            ColMajor => Upper,
        };
        EighAPI::<DeviceBLAS>::eigh_f((a, uplo))
    }
}

#[duplicate_item(
    ImplType   Tr                              ;
   ['a, T, D] [TensorMut<'a, T, DeviceBLAS, D>];
   [    T, D] [Tensor<T, DeviceBLAS, D>       ];
)]
impl<ImplType> EighAPI<DeviceBLAS> for (Tr, FlagUpLo)
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tr>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (mut a, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view_mut().into_dim::<Ix2>();
        let eigh_args = EighArgs::default().a(a_view).uplo(uplo).build()?;
        let (vals, vecs) = ref_impl_eigh_simple_f(eigh_args)?;
        let vals = vals.into_dim::<IxD>().into_dim::<D::SmallerOne>();
        vecs.unwrap().clone_to_mut();
        return Ok(EighResult { eigenvalues: vals, eigenvectors: a });
    }
}

#[duplicate_item(
    ImplType   Tr                              ;
   ['a, T, D] [TensorMut<'a, T, DeviceBLAS, D>];
   [    T, D] [Tensor<T, DeviceBLAS, D>       ];
)]
impl<ImplType> EighAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tr>;
    fn eigh_f(self) -> Result<Self::Out> {
        let a = self;
        let uplo = match a.device().default_order() {
            RowMajor => Lower,
            ColMajor => Upper,
        };
        EighAPI::<DeviceBLAS>::eigh_f((a, uplo))
    }
}

/* #endregion */

/* #region general eigh */

#[duplicate_item(
    ImplType                                                       TrA                                TrB                              ;
   [T, D, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceBLAS, D>] [&TensorAny<Rb, T, DeviceBLAS, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceBLAS, D> ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceBLAS, D> ] [TensorCow<'_, T, DeviceBLAS, D> ];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceBLAS, D>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [TensorCow<'_, T, DeviceBLAS, D> ] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D,                                                       ] [TensorView<'_, T, DeviceBLAS, D>] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D,                                                       ] [TensorView<'_, T, DeviceBLAS, D>] [TensorCow<'_, T, DeviceBLAS, D> ];
   [T, D,                                                       ] [TensorCow<'_, T, DeviceBLAS, D> ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D,                                                       ] [TensorCow<'_, T, DeviceBLAS, D> ] [TensorCow<'_, T, DeviceBLAS, D> ];
)]
impl<ImplType> EighAPI<DeviceBLAS> for (TrA, TrB, FlagUpLo, i32)
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tensor<T, DeviceBLAS, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, b, uplo, eig_type) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_pattern!(eig_type, 1..=3, InvalidLayout, "Only eig_type = 1, 2, or 3 allowed.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = b.view().into_dim::<Ix2>();
        let eigh_args =
            EighArgs::default().a(a_view).b(b_view).uplo(uplo).eig_type(eig_type).build()?;
        let (vals, vecs) = ref_impl_eigh_simple_f(eigh_args)?;
        let vals = vals.into_dim::<IxD>().into_dim::<D::SmallerOne>();
        let vecs = vecs.unwrap().into_owned().into_dim::<IxD>().into_dim::<D>();
        return Ok(EighResult { eigenvalues: vals, eigenvectors: vecs });
    }
}

#[duplicate_item(
    ImplType                                                       TrA                                TrB                              ;
   [T, D, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceBLAS, D>] [&TensorAny<Rb, T, DeviceBLAS, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceBLAS, D> ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceBLAS, D> ] [TensorCow<'_, T, DeviceBLAS, D> ];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceBLAS, D>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [TensorCow<'_, T, DeviceBLAS, D> ] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D,                                                       ] [TensorView<'_, T, DeviceBLAS, D>] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D,                                                       ] [TensorView<'_, T, DeviceBLAS, D>] [TensorCow<'_, T, DeviceBLAS, D> ];
   [T, D,                                                       ] [TensorCow<'_, T, DeviceBLAS, D> ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D,                                                       ] [TensorCow<'_, T, DeviceBLAS, D> ] [TensorCow<'_, T, DeviceBLAS, D> ];
)]
impl<ImplType> EighAPI<DeviceBLAS> for (TrA, TrB, FlagUpLo)
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tensor<T, DeviceBLAS, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, b, uplo) = self;
        EighAPI::<DeviceBLAS>::eigh_f((a, b, uplo, 1))
    }
}

#[duplicate_item(
    ImplType                                                       TrA                                TrB                              ;
   [T, D, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceBLAS, D>] [&TensorAny<Rb, T, DeviceBLAS, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceBLAS, D> ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceBLAS, D> ] [TensorCow<'_, T, DeviceBLAS, D> ];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceBLAS, D>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [TensorCow<'_, T, DeviceBLAS, D> ] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D,                                                       ] [TensorView<'_, T, DeviceBLAS, D>] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D,                                                       ] [TensorView<'_, T, DeviceBLAS, D>] [TensorCow<'_, T, DeviceBLAS, D> ];
   [T, D,                                                       ] [TensorCow<'_, T, DeviceBLAS, D> ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D,                                                       ] [TensorCow<'_, T, DeviceBLAS, D> ] [TensorCow<'_, T, DeviceBLAS, D> ];
)]
impl<ImplType> EighAPI<DeviceBLAS> for (TrA, TrB)
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tensor<T, DeviceBLAS, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, b) = self;
        let uplo = match a.device().default_order() {
            RowMajor => Lower,
            ColMajor => Upper,
        };
        EighAPI::<DeviceBLAS>::eigh_f((a, b, uplo, 1))
    }
}

/* #endregion */

/* #region EighArgs implementation */

impl<'a, 'b, T> EighAPI<DeviceBLAS> for EighArgs<'a, 'b, DeviceBLAS, T>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, Ix1>, TensorMutable<'a, T, DeviceBLAS, Ix2>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let args = self.build()?;
        rstsr_assert!(
            !args.eigvals_only,
            InvalidValue,
            "Eigh only supports eigvals_only = false."
        )?;
        let (vals, vecs) = ref_impl_eigh_simple_f(args)?;
        Ok(EighResult { eigenvalues: vals, eigenvectors: vecs.unwrap() })
    }
}

impl<'a, 'b, T> EighAPI<DeviceBLAS> for EighArgs_<'a, 'b, DeviceBLAS, T>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, Ix1>, TensorMutable<'a, T, DeviceBLAS, Ix2>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let args = self;
        rstsr_assert!(
            !args.eigvals_only,
            InvalidValue,
            "Eigh only supports eigvals_only = false."
        )?;
        let (vals, vecs) = ref_impl_eigh_simple_f(args)?;
        Ok(EighResult { eigenvalues: vals, eigenvectors: vecs.unwrap() })
    }
}

/* #endregion */
