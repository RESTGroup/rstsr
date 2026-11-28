use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/// n-dim `svd`: the singular-value decomposition of every matrix slice of the
/// stack (the two matrix axes are the last two shape axes under a row-major
/// device, the first two under a column-major one; the rest is the batch);
/// slices may be rectangular `(M, N)`.
///
/// The input is never written through the `SVDAPI` overloads: the whole
/// stack goes through `overwritable_convert` once and is factored from a
/// fresh contiguous buffer in the device order. A mutable reference handed
/// to this function directly is factored in place when its layout matches
/// the device order.
#[allow(clippy::type_complexity)]
pub fn blas_impl_svd_f<T>(
    a: TensorReference<'_, T, DeviceBLAS, IxD>,
    full_matrices: bool,
) -> Result<(Tensor<T, DeviceBLAS, IxD>, Tensor<T::Real, DeviceBLAS, IxD>, Tensor<T, DeviceBLAS, IxD>)>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>,
{
    let device = a.device().clone();
    let order = device.default_order();
    let stack = StackLayout::new(a.layout(), 2, order)?;
    let inner_shape = stack.inner_shape();
    let [m, n] = [inner_shape[0], inner_shape[1]];
    let k = Ord::min(m, n);
    let (u_inner, vt_inner) = match full_matrices {
        true => ([m, m], [n, n]),
        false => ([m, k], [k, n]),
    };

    let mut converted = overwritable_convert_with_order(a, order)?;
    let converted_stack = StackLayout::new(converted.view().layout(), 2, order)?;
    let mut u: Tensor<T, DeviceBLAS, IxD> = zeros_f((stack_shape(stack.batch_shape(), &u_inner, order), &device))?;
    let mut s: Tensor<T::Real, DeviceBLAS, IxD> = zeros_f((stack_shape(stack.batch_shape(), &[k], order), &device))?;
    let mut vt: Tensor<T, DeviceBLAS, IxD> = zeros_f((stack_shape(stack.batch_shape(), &vt_inner, order), &device))?;
    let u_stack = StackLayout::new(u.layout(), 2, order)?;
    let s_stack = StackLayout::new(s.layout(), 1, order)?;
    let vt_stack = StackLayout::new(vt.layout(), 2, order)?;

    for i in 0..stack.len() {
        let slice = stack_slice_at_mut(converted.view_mut(), &converted_stack, i)?.into_dim::<Ix2>();
        let svd_args = SVDArgs::default().a(TensorReference::from(slice)).full_matrices(full_matrices).build()?;
        let (u_i, s_i, vt_i) = ref_impl_svd_simple_f(svd_args)?;
        let mut u_slice = stack_slice_at_mut(u.view_mut(), &u_stack, i)?.into_dim::<Ix2>();
        u_slice.assign_f(u_i.unwrap())?;
        let mut s_slice = stack_slice_at_mut(s.view_mut(), &s_stack, i)?.into_dim::<Ix1>();
        s_slice.assign_f(s_i)?;
        let mut vt_slice = stack_slice_at_mut(vt.view_mut(), &vt_stack, i)?.into_dim::<Ix2>();
        vt_slice.assign_f(vt_i.unwrap())?;
    }
    Ok((u, s, vt))
}

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
        let (u, s, vt) = blas_impl_svd_f(a.view().to_dyn().into(), full_matrices)?;
        Ok(SVDResult { u: u.into_dim::<D>(), s: s.into_dim::<D::SmallerOne>(), vt: vt.into_dim::<D>() })
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
