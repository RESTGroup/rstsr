use crate::DeviceBLAS;
use duplicate::duplicate_item;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

impl<R, T, D> EighAPI<DeviceBLAS> for (&TensorAny<R, T, DeviceBLAS, D>, FlagUpLo)
where
    T: BlasFloat,
    R: DataAPI<Data = Vec<T>>,
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

impl<R, T, D> EighAPI<DeviceBLAS> for &TensorAny<R, T, DeviceBLAS, D>
where
    T: BlasFloat,
    R: DataAPI<Data = Vec<T>>,
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
    TSR;
    [TensorView<'_, T, DeviceBLAS, D>];
    [TensorCow<'_, T, DeviceBLAS, D>]
)]
impl<T, D> EighAPI<DeviceBLAS> for (TSR, FlagUpLo)
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tensor<T, DeviceBLAS, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        EighAPI::<DeviceBLAS>::eigh_f((&a, uplo))
    }
}

#[duplicate_item(
    TSR;
    [TensorView<'_, T, DeviceBLAS, D>];
    [TensorCow<'_, T, DeviceBLAS, D>]
)]
impl<T, D> EighAPI<DeviceBLAS> for TSR
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tensor<T, DeviceBLAS, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        EighAPI::<DeviceBLAS>::eigh_f(&self)
    }
}

impl<Ra, Rb, T, D> EighAPI<DeviceBLAS>
    for (&TensorAny<Ra, T, DeviceBLAS, D>, &TensorAny<Rb, T, DeviceBLAS, D>, FlagUpLo, i32)
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    Ra: DataAPI<Data = Vec<T>>,
    Rb: DataAPI<Data = Vec<T>>,
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

impl<Ra, Rb, T, D> EighAPI<DeviceBLAS>
    for (&TensorAny<Ra, T, DeviceBLAS, D>, &TensorAny<Rb, T, DeviceBLAS, D>, FlagUpLo)
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    Ra: DataAPI<Data = Vec<T>>,
    Rb: DataAPI<Data = Vec<T>>,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tensor<T, DeviceBLAS, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, b, uplo) = self;
        EighAPI::<DeviceBLAS>::eigh_f((a, b, uplo, 1))
    }
}

impl<Ra, Rb, T, D> EighAPI<DeviceBLAS>
    for (&TensorAny<Ra, T, DeviceBLAS, D>, &TensorAny<Rb, T, DeviceBLAS, D>)
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    Ra: DataAPI<Data = Vec<T>>,
    Rb: DataAPI<Data = Vec<T>>,
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

#[duplicate_item(
    Inp;
    [(TensorView<'_, T, DeviceBLAS, D>, &TensorAny<R, T, DeviceBLAS, D>, FlagUpLo, i32)];
    [(TensorCow<'_, T, DeviceBLAS, D>, &TensorAny<R, T, DeviceBLAS, D>, FlagUpLo, i32)];
    [(&TensorAny<R, T, DeviceBLAS, D>, TensorView<'_, T, DeviceBLAS, D>, FlagUpLo, i32)];
    [(&TensorAny<R, T, DeviceBLAS, D>, TensorCow<'_, T, DeviceBLAS, D>, FlagUpLo, i32)];
)]
impl<R, T, D> EighAPI<DeviceBLAS> for Inp
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    R: DataAPI<Data = Vec<T>>,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tensor<T, DeviceBLAS, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, b, uplo, eig_type) = self;
        let a = a.view();
        let b = b.view();
        EighAPI::<DeviceBLAS>::eigh_f((&a, &b, uplo, eig_type))
    }
}

#[duplicate_item(
    Inp;
    [(TensorView<'_, T, DeviceBLAS, D>, &TensorAny<R, T, DeviceBLAS, D>, FlagUpLo)];
    [(TensorCow<'_, T, DeviceBLAS, D>, &TensorAny<R, T, DeviceBLAS, D>, FlagUpLo)];
    [(&TensorAny<R, T, DeviceBLAS, D>, TensorView<'_, T, DeviceBLAS, D>, FlagUpLo)];
    [(&TensorAny<R, T, DeviceBLAS, D>, TensorCow<'_, T, DeviceBLAS, D>, FlagUpLo)];
)]
impl<R, T, D> EighAPI<DeviceBLAS> for Inp
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    R: DataAPI<Data = Vec<T>>,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tensor<T, DeviceBLAS, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, b, uplo) = self;
        let a = a.view();
        let b = b.view();
        let eig_type = 1;
        EighAPI::<DeviceBLAS>::eigh_f((&a, &b, uplo, eig_type))
    }
}

#[duplicate_item(
    Inp;
    [(TensorView<'_, T, DeviceBLAS, D>, &TensorAny<R, T, DeviceBLAS, D>)];
    [(TensorCow<'_, T, DeviceBLAS, D>, &TensorAny<R, T, DeviceBLAS, D>)];
    [(&TensorAny<R, T, DeviceBLAS, D>, TensorView<'_, T, DeviceBLAS, D>)];
    [(&TensorAny<R, T, DeviceBLAS, D>, TensorCow<'_, T, DeviceBLAS, D>)];
)]
impl<R, T, D> EighAPI<DeviceBLAS> for Inp
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    R: DataAPI<Data = Vec<T>>,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tensor<T, DeviceBLAS, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, b) = self;
        let a = a.view();
        let b = b.view();
        let uplo = match a.device().default_order() {
            RowMajor => Lower,
            ColMajor => Upper,
        };
        let eig_type = 1;
        EighAPI::<DeviceBLAS>::eigh_f((&a, &b, uplo, eig_type))
    }
}

#[duplicate_item(
    Inp;
    [(TensorView<'_, T, DeviceBLAS, D>, TensorView<'_, T, DeviceBLAS, D>, FlagUpLo, i32)];
    [(TensorView<'_, T, DeviceBLAS, D>, TensorCow<'_, T, DeviceBLAS, D>, FlagUpLo, i32)];
    [(TensorCow<'_, T, DeviceBLAS, D>, TensorView<'_, T, DeviceBLAS, D>, FlagUpLo, i32)];
    [(TensorCow<'_, T, DeviceBLAS, D>, TensorCow<'_, T, DeviceBLAS, D>, FlagUpLo, i32)];
)]
impl<T, D> EighAPI<DeviceBLAS> for Inp
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tensor<T, DeviceBLAS, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, b, uplo, eig_type) = self;
        EighAPI::<DeviceBLAS>::eigh_f((&a, &b, uplo, eig_type))
    }
}

#[duplicate_item(
    Inp;
    [(TensorView<'_, T, DeviceBLAS, D>, TensorView<'_, T, DeviceBLAS, D>, FlagUpLo)];
    [(TensorView<'_, T, DeviceBLAS, D>, TensorCow<'_, T, DeviceBLAS, D>, FlagUpLo)];
    [(TensorCow<'_, T, DeviceBLAS, D>, TensorView<'_, T, DeviceBLAS, D>, FlagUpLo)];
    [(TensorCow<'_, T, DeviceBLAS, D>, TensorCow<'_, T, DeviceBLAS, D>, FlagUpLo)];
)]
impl<T, D> EighAPI<DeviceBLAS> for Inp
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tensor<T, DeviceBLAS, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, b, uplo) = self;
        EighAPI::<DeviceBLAS>::eigh_f((&a, &b, uplo))
    }
}

#[duplicate_item(
    Inp;
    [(TensorView<'_, T, DeviceBLAS, D>, TensorView<'_, T, DeviceBLAS, D>)];
    [(TensorView<'_, T, DeviceBLAS, D>, TensorCow<'_, T, DeviceBLAS, D>)];
    [(TensorCow<'_, T, DeviceBLAS, D>, TensorView<'_, T, DeviceBLAS, D>)];
    [(TensorCow<'_, T, DeviceBLAS, D>, TensorCow<'_, T, DeviceBLAS, D>)];
)]
impl<T, D> EighAPI<DeviceBLAS> for Inp
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tensor<T, DeviceBLAS, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, b) = self;
        EighAPI::<DeviceBLAS>::eigh_f((&a, &b))
    }
}
