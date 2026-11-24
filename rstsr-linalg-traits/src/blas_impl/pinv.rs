use crate::DeviceBLAS;
use num::FromPrimitive;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/// n-dim `pinv`: the Moore-Penrose pseudoinverse of every matrix slice of the
/// stack (the two matrix axes are the last two shape axes under a row-major
/// device, the first two under a column-major one; the rest is the batch);
/// slices may be rectangular `(M, N)`. The `rank` scalar reports the largest
/// per-matrix rank in the stack.
///
/// The whole stack goes through `overwritable_convert` once: an immutable
/// input is factored from a fresh contiguous buffer; a mutable input is
/// destroyed by the per-slice SVD.
pub fn blas_impl_pinv_f<T>(
    a: TensorReference<'_, T, DeviceBLAS, IxD>,
    atol: Option<T::Real>,
    rtol: Option<T::Real>,
) -> Result<PinvResult<Tensor<T, DeviceBLAS, IxD>>>
where
    T: BlasFloat,
    T::Real: FromPrimitive,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, Ix2, Ix2>
        + OpAssignAPI<T, Ix2>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
{
    let device = a.device().clone();
    let order = device.default_order();
    let stack = StackLayout::new(a.layout(), 2, order)?;
    let inner_shape = stack.inner_shape();
    let m = inner_shape[0];
    let n = inner_shape[1];

    // a zero-size matrix axis yields a zero pinv of shape (n, m) and rank 0;
    // the SVD below would otherwise fail on an empty singular-value vector
    let mut rank = 0;
    if m == 0 || n == 0 || stack.is_empty() {
        let pinv = zeros_f((stack_shape(stack.batch_shape(), &[n, m], order), &device))?;
        return Ok(PinvResult { pinv, rank });
    }

    let mut converted = overwritable_convert_with_order(a, order)?;
    let converted_stack = StackLayout::new(converted.view().layout(), 2, order)?;
    let mut pinv: Tensor<T, DeviceBLAS, IxD> = zeros_f((stack_shape(stack.batch_shape(), &[n, m], order), &device))?;
    let pinv_stack = StackLayout::new(pinv.layout(), 2, order)?;

    for i in 0..stack.len() {
        let slice = stack_slice_at_mut(converted.view_mut(), &converted_stack, i)?.into_dim::<Ix2>();
        let result = ref_impl_pinv_f(slice.view(), atol, rtol)?;
        rank = Ord::max(rank, result.rank);
        let mut pinv_slice = stack_slice_at_mut(pinv.view_mut(), &pinv_stack, i)?.into_dim::<Ix2>();
        pinv_slice.assign_f(result.pinv.view())?;
    }
    Ok(PinvResult { pinv, rank })
}

impl<T, D, R> PinvAPI<DeviceBLAS> for (&TensorAny<R, T, DeviceBLAS, D>, T::Real, T::Real)
where
    R: DataAPI<Data = Vec<T>>,
    T: BlasFloat,
    T::Real: FromPrimitive,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, Ix2, Ix2>
        + OpAssignAPI<T, Ix2>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
{
    type Out = PinvResult<Tensor<T, DeviceBLAS, D>>;
    fn pinv_f(self) -> Result<Self::Out> {
        let (a, atol, rtol) = self;
        let result = blas_impl_pinv_f(a.view().to_dyn().into(), Some(atol), Some(rtol))?;
        Ok(PinvResult { pinv: result.pinv.into_dim::<D>(), rank: result.rank })
    }
}

impl<T, D, R> PinvAPI<DeviceBLAS> for &TensorAny<R, T, DeviceBLAS, D>
where
    R: DataAPI<Data = Vec<T>>,
    T: BlasFloat,
    T::Real: FromPrimitive,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, Ix2, Ix2>
        + OpAssignAPI<T, Ix2>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
{
    type Out = PinvResult<Tensor<T, DeviceBLAS, D>>;
    fn pinv_f(self) -> Result<Self::Out> {
        let a = self;
        let result = blas_impl_pinv_f(a.view().to_dyn().into(), None, None)?;
        Ok(PinvResult { pinv: result.pinv.into_dim::<D>(), rank: result.rank })
    }
}

#[duplicate_item(
    Tr                               ;
   [Tensor<T, DeviceBLAS, D>        ];
   [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<T, D> PinvAPI<DeviceBLAS> for (Tr, T::Real, T::Real)
where
    T: BlasFloat,
    T::Real: FromPrimitive,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, Ix2, Ix2>
        + OpAssignAPI<T, Ix2>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
{
    type Out = PinvResult<Tensor<T, DeviceBLAS, D>>;
    fn pinv_f(self) -> Result<Self::Out> {
        let (a, atol, rtol) = self;
        PinvAPI::<DeviceBLAS>::pinv_f((&a, atol, rtol))
    }
}

#[duplicate_item(
    Tr                               ;
   [Tensor<T, DeviceBLAS, D>        ];
   [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<T, D> PinvAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    T::Real: FromPrimitive,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, Ix2, Ix2>
        + OpAssignAPI<T, Ix2>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
{
    type Out = PinvResult<Tensor<T, DeviceBLAS, D>>;
    fn pinv_f(self) -> Result<Self::Out> {
        let a = self;
        PinvAPI::<DeviceBLAS>::pinv_f(&a)
    }
}
