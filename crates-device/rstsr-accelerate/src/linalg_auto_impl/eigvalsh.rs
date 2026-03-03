use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/* #region simple eigh */

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> EigvalshAPI<DeviceBLAS> for (Tr, FlagUpLo)
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T::Real, DeviceBLAS, D::SmallerOne>;
    fn eigvalsh_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let eigh_args = EighArgs::default().a(a_view).uplo(uplo).eigvals_only(true).build()?;
        let (vals, _) = ref_impl_eigh_simple_f(eigh_args)?;
        let vals = vals.into_dim::<IxD>().into_dim::<D::SmallerOne>();
        return Ok(vals);
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> EigvalshAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T::Real, DeviceBLAS, D::SmallerOne>;
    fn eigvalsh_f(self) -> Result<Self::Out> {
        let a = self;
        let uplo = match a.device().default_order() {
            RowMajor => Lower,
            ColMajor => Upper,
        };
        EigvalshAPI::<DeviceBLAS>::eigvalsh_f((a, uplo))
    }
}

#[duplicate_item(
    ImplType   Tr                              ;
   ['a, T, D] [TensorMut<'a, T, DeviceBLAS, D>];
   [    T, D] [Tensor<T, DeviceBLAS, D>       ];
)]
impl<ImplType> EigvalshAPI<DeviceBLAS> for (Tr, FlagUpLo)
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T::Real, DeviceBLAS, D::SmallerOne>;
    fn eigvalsh_f(self) -> Result<Self::Out> {
        let (mut a, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view_mut().into_dim::<Ix2>();
        let eigh_args = EighArgs::default().a(a_view).uplo(uplo).eigvals_only(true).build()?;
        let (vals, _) = ref_impl_eigh_simple_f(eigh_args)?;
        let vals = vals.into_dim::<IxD>().into_dim::<D::SmallerOne>();
        return Ok(vals);
    }
}

#[duplicate_item(
    ImplType   Tr                              ;
   ['a, T, D] [TensorMut<'a, T, DeviceBLAS, D>];
   [    T, D] [Tensor<T, DeviceBLAS, D>       ];
)]
impl<ImplType> EigvalshAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T::Real, DeviceBLAS, D::SmallerOne>;
    fn eigvalsh_f(self) -> Result<Self::Out> {
        let a = self;
        let uplo = match a.device().default_order() {
            RowMajor => Lower,
            ColMajor => Upper,
        };
        EigvalshAPI::<DeviceBLAS>::eigvalsh_f((a, uplo))
    }
}

/* #endregion */

/* #region general eigh */

#[duplicate_item(
    ImplType                                                       TrA                                TrB                              ;
   [T, D, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceBLAS, D>] [&TensorAny<Rb, T, DeviceBLAS, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceBLAS, D> ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceBLAS, D>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D,                                                       ] [TensorView<'_, T, DeviceBLAS, D>] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> EigvalshAPI<DeviceBLAS> for (TrA, TrB, FlagUpLo, i32)
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T::Real, DeviceBLAS, D::SmallerOne>;
    fn eigvalsh_f(self) -> Result<Self::Out> {
        let (a, b, uplo, eig_type) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_pattern!(eig_type, 1..=3, InvalidLayout, "Only eig_type = 1, 2, or 3 allowed.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = b.view().into_dim::<Ix2>();
        let eigh_args =
            EighArgs::default().a(a_view).b(b_view).uplo(uplo).eig_type(eig_type).eigvals_only(true).build()?;
        let (vals, _) = ref_impl_eigh_simple_f(eigh_args)?;
        let vals = vals.into_dim::<IxD>().into_dim::<D::SmallerOne>();
        return Ok(vals);
    }
}

#[duplicate_item(
    ImplType                                                       TrA                                TrB                              ;
   [T, D, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceBLAS, D>] [&TensorAny<Rb, T, DeviceBLAS, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceBLAS, D> ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceBLAS, D>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D,                                                       ] [TensorView<'_, T, DeviceBLAS, D>] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> EigvalshAPI<DeviceBLAS> for (TrA, TrB, FlagUpLo)
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T::Real, DeviceBLAS, D::SmallerOne>;
    fn eigvalsh_f(self) -> Result<Self::Out> {
        let (a, b, uplo) = self;
        EigvalshAPI::<DeviceBLAS>::eigvalsh_f((a, b, uplo, 1))
    }
}

#[duplicate_item(
    ImplType                                                       TrA                                TrB                              ;
   [T, D, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceBLAS, D>] [&TensorAny<Rb, T, DeviceBLAS, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceBLAS, D> ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceBLAS, D>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D,                                                       ] [TensorView<'_, T, DeviceBLAS, D>] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> EigvalshAPI<DeviceBLAS> for (TrA, TrB)
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T::Real, DeviceBLAS, D::SmallerOne>;
    fn eigvalsh_f(self) -> Result<Self::Out> {
        let (a, b) = self;
        let uplo = match a.device().default_order() {
            RowMajor => Lower,
            ColMajor => Upper,
        };
        EigvalshAPI::<DeviceBLAS>::eigvalsh_f((a, b, uplo, 1))
    }
}

/* #endregion */

/* #region EighArgs implementation */

impl<'a, 'b, T> EigvalshAPI<DeviceBLAS> for EighArgs<'a, 'b, DeviceBLAS, T>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T::Real, DeviceBLAS, Ix1>;
    fn eigvalsh_f(self) -> Result<Self::Out> {
        let args = self.build()?;
        rstsr_assert!(args.eigvals_only, InvalidValue, "Eigvalsh only supports eigvals_only = true.")?;
        let (vals, _) = ref_impl_eigh_simple_f(args)?;
        Ok(vals)
    }
}

impl<'a, 'b, T> EigvalshAPI<DeviceBLAS> for EighArgs_<'a, 'b, DeviceBLAS, T>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T::Real, DeviceBLAS, Ix1>;
    fn eigvalsh_f(self) -> Result<Self::Out> {
        let args = self;
        rstsr_assert!(args.eigvals_only, InvalidValue, "Eigvalsh only supports eigvals_only = true.")?;
        let (vals, _) = ref_impl_eigh_simple_f(args)?;
        Ok(vals)
    }
}

/* #endregion */
