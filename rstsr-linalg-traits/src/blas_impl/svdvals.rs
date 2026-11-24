use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/// n-dim `svdvals`: the singular values of every matrix slice of the stack
/// (the two matrix axes are the last two shape axes under a row-major device,
/// the first two under a column-major one; the rest is the batch), with the
/// batch shape; slices may be rectangular `(M, N)`.
///
/// The whole stack goes through `overwritable_convert` once: an immutable
/// input is factored from a fresh contiguous buffer; a mutable input whose
/// layout matches the device order is destroyed by the per-slice SVD (the
/// `SVDvalsAPI` overloads take immutable inputs only).
pub fn blas_impl_svdvals_f<T>(a: TensorReference<'_, T, DeviceBLAS, IxD>) -> Result<Tensor<T::Real, DeviceBLAS, IxD>>
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
    let m = inner_shape[0];
    let n = inner_shape[1];

    let mut converted = overwritable_convert_with_order(a, order)?;
    let converted_stack = StackLayout::new(converted.view().layout(), 2, order)?;
    let mut out: Tensor<T::Real, DeviceBLAS, IxD> =
        zeros_f((stack_shape(stack.batch_shape(), &[Ord::min(m, n)], order), &device))?;
    let out_stack = StackLayout::new(out.layout(), 1, order)?;

    for i in 0..stack.len() {
        let slice = stack_slice_at_mut(converted.view_mut(), &converted_stack, i)?.into_dim::<Ix2>();
        let svd_args = SVDArgs::default().a(TensorReference::from(slice)).full_matrices(None).build()?;
        let (_, s, _) = ref_impl_svd_simple_f(svd_args)?;
        let mut out_slice = stack_slice_at_mut(out.view_mut(), &out_stack, i)?.into_dim::<Ix1>();
        out_slice.assign_f(s)?;
    }
    Ok(out)
}

/* #region full-args */

impl<T, D, R> SVDvalsAPI<DeviceBLAS> for &TensorAny<R, T, DeviceBLAS, D>
where
    R: DataAPI<Data = Vec<T>>,
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
    fn svdvals_f(self) -> Result<Self::Out> {
        blas_impl_svdvals_f(self.view().to_dyn().into()).map(|t| t.into_dim::<D::SmallerOne>())
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
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T::Real>
        + OpAssignArbitaryAPI<T::Real, Ix1, Ix1>
        + OpAssignAPI<T::Real, Ix1>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
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
