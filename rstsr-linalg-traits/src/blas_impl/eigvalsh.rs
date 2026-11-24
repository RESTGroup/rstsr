use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/// n-dim `eigvalsh`: the eigenvalues of every symmetric/Hermitian matrix slice
/// of the stack (the two matrix axes are the last two shape axes under a
/// row-major device, the first two under a column-major one; the rest is the
/// batch), with the batch shape.
///
/// The whole stack goes through `overwritable_convert` once: an immutable
/// input is factored from a fresh contiguous buffer; a mutable input whose
/// layout matches the device order is destroyed in place by the per-slice
/// eigendecomposition.
pub fn blas_impl_eigvalsh_f<T>(
    a: TensorReference<'_, T, DeviceBLAS, IxD>,
    uplo: FlagUpLo,
) -> Result<Tensor<T::Real, DeviceBLAS, IxD>>
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
        "eigvalsh: the matrix must be square, got {}x{}",
        inner_shape[0],
        inner_shape[1]
    )?;

    let mut converted = overwritable_convert_with_order(a, order)?;
    let converted_stack = StackLayout::new(converted.view().layout(), 2, order)?;
    let mut out: Tensor<T::Real, DeviceBLAS, IxD> =
        zeros_f((stack_shape(stack.batch_shape(), &[inner_shape[0]], order), &device))?;
    let out_stack = StackLayout::new(out.layout(), 1, order)?;

    for i in 0..stack.len() {
        let slice = stack_slice_at_mut(converted.view_mut(), &converted_stack, i)?.into_dim::<Ix2>();
        let eigh_args = EighArgs::default().a(TensorReference::from(slice)).uplo(uplo).eigvals_only(true).build()?;
        let (vals, _) = ref_impl_eigh_simple_f(eigh_args)?;
        let mut out_slice = stack_slice_at_mut(out.view_mut(), &out_stack, i)?.into_dim::<Ix1>();
        out_slice.assign_f(vals)?;
    }
    Ok(out)
}

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
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T::Real>
        + OpAssignArbitaryAPI<T::Real, Ix1, Ix1>
        + OpAssignAPI<T::Real, Ix1>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
{
    type Out = Tensor<T::Real, DeviceBLAS, D::SmallerOne>;
    fn eigvalsh_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        blas_impl_eigvalsh_f(a.view().to_dyn().into(), uplo).map(|t| t.into_dim::<D::SmallerOne>())
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
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T::Real>
        + OpAssignArbitaryAPI<T::Real, Ix1, Ix1>
        + OpAssignAPI<T::Real, Ix1>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
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
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T::Real>
        + OpAssignArbitaryAPI<T::Real, Ix1, Ix1>
        + OpAssignAPI<T::Real, Ix1>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
{
    type Out = Tensor<T::Real, DeviceBLAS, D::SmallerOne>;
    fn eigvalsh_f(self) -> Result<Self::Out> {
        let (mut a, uplo) = self;
        let result = blas_impl_eigvalsh_f(a.view_mut().into_dim::<IxD>().into(), uplo)?;
        let vals = result.into_dim::<D::SmallerOne>();
        Ok(vals)
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
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T::Real>
        + OpAssignArbitaryAPI<T::Real, Ix1, Ix1>
        + OpAssignAPI<T::Real, Ix1>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
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
