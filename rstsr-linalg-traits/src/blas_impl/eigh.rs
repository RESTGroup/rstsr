use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_blas_traits::util::util_tensor::overwritable_convert_with_order;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/// n-dim `eigh` (standard form): the eigenvalues and eigenvectors of every
/// symmetric/Hermitian matrix slice of the stack (the two matrix axes are the
/// last two shape axes under a row-major device, the first two under a
/// column-major one; the rest is the batch).
///
/// The whole stack goes through `overwritable_convert` once: an immutable
/// input is factored into a fresh contiguous buffer (which becomes the
/// eigenvectors); a mutable input whose layout matches the device order is
/// overwritten with the eigenvectors in place; a disagreeing mutable layout
/// uses a contiguous work buffer written back by `clone_to_mut` at the call
/// site.
#[allow(clippy::type_complexity)]
pub fn blas_impl_eigh_f<T>(
    a: TensorReference<'_, T, DeviceBLAS, IxD>,
    uplo: FlagUpLo,
) -> Result<(Tensor<T::Real, DeviceBLAS, IxD>, TensorMutable<'_, T, DeviceBLAS, IxD>)>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T::Real>
        + OpAssignArbitaryAPI<T::Real, Ix1, Ix1>
        + OpAssignAPI<T::Real, Ix1>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
{
    let device = a.device().clone();
    let order = device.default_order();
    let stack = StackLayout::new(a.layout(), 2, order)?;
    let inner_shape = stack.inner_shape();
    rstsr_assert_eq!(
        inner_shape[0],
        inner_shape[1],
        InvalidLayout,
        "eigh: the matrix must be square, got {}x{}",
        inner_shape[0],
        inner_shape[1]
    )?;

    let mut converted = overwritable_convert_with_order(a, order)?;
    let converted_stack = StackLayout::new(converted.view().layout(), 2, order)?;
    let mut w: Tensor<T::Real, DeviceBLAS, IxD> =
        zeros_f((stack_shape(stack.batch_shape(), &[inner_shape[0]], order), &device))?;
    let w_stack = StackLayout::new(w.layout(), 1, order)?;

    for i in 0..stack.len() {
        let slice = stack_slice_at_mut(converted.view_mut(), &converted_stack, i)?.into_dim::<Ix2>();
        let eigh_args = EighArgs::default().a(TensorReference::from(slice)).uplo(uplo).build()?;
        let (w_i, v_i) = ref_impl_eigh_simple_f(eigh_args)?;
        let mut w_slice = stack_slice_at_mut(w.view_mut(), &w_stack, i)?.into_dim::<Ix1>();
        w_slice.assign_f(w_i)?;
        v_i.unwrap().clone_to_mut();
    }
    Ok((w, converted))
}

/* #region simple eigh */

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> EighAPI<DeviceBLAS> for (Tr, FlagUpLo)
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T::Real>
        + OpAssignArbitaryAPI<T::Real, Ix1, Ix1>
        + OpAssignAPI<T::Real, Ix1>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tensor<T, DeviceBLAS, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        let a_view = a.view();
        let a_dyn = a_view.to_dyn();
        let (w, v) = blas_impl_eigh_f(a_dyn.into(), uplo)?;
        Ok(EighResult { eigenvalues: w.into_dim::<D::SmallerOne>(), eigenvectors: v.into_owned().into_dim::<D>() })
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> EighAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T::Real>
        + OpAssignArbitaryAPI<T::Real, Ix1, Ix1>
        + OpAssignAPI<T::Real, Ix1>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
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
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T::Real>
        + OpAssignArbitaryAPI<T::Real, Ix1, Ix1>
        + OpAssignAPI<T::Real, Ix1>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tr>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (mut a, uplo) = self;
        let (w, v) = blas_impl_eigh_f(a.view_mut().into_dim::<IxD>().into(), uplo)?;
        v.clone_to_mut();
        return Ok(EighResult { eigenvalues: w.into_dim::<D::SmallerOne>(), eigenvectors: a });
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
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T::Real>
        + OpAssignArbitaryAPI<T::Real, Ix1, Ix1>
        + OpAssignAPI<T::Real, Ix1>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
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
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceBLAS, D>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D,                                                       ] [TensorView<'_, T, DeviceBLAS, D>] [TensorView<'_, T, DeviceBLAS, D>];
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
        let eigh_args = EighArgs::default().a(a_view).b(b_view).uplo(uplo).eig_type(eig_type).build()?;
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
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceBLAS, D>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D,                                                       ] [TensorView<'_, T, DeviceBLAS, D>] [TensorView<'_, T, DeviceBLAS, D>];
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
   [T, D,                                                       ] [TensorView<'_, T, DeviceBLAS, D>] [TensorView<'_, T, DeviceBLAS, D>];
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
        rstsr_assert!(!args.eigvals_only, InvalidValue, "Eigh only supports eigvals_only = false.")?;
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
        rstsr_assert!(!args.eigvals_only, InvalidValue, "Eigh only supports eigvals_only = false.")?;
        let (vals, vecs) = ref_impl_eigh_simple_f(args)?;
        Ok(EighResult { eigenvalues: vals, eigenvectors: vecs.unwrap() })
    }
}

/* #endregion */
