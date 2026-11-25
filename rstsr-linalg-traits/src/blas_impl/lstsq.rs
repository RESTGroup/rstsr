use crate::DeviceBLAS;
use num::{Float, FromPrimitive};
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/// n-dim `lstsq`: the least-squares solution of every matrix slice of the
/// stack (the two matrix axes are the last two shape axes under a row-major
/// device, the first two under a column-major one; the rest is the batch).
/// Slices may be rectangular: tall slices minimize `||AX - B||`, wide slices
/// return the minimum-norm solution.
///
/// `rcond` is the LAPACK cutoff for small singular values (`None` uses
/// `eps * max(M, N)`): singular values at or below `rcond * s_max` are treated
/// as zero. `a` is destroyed by the per-slice factorization; `b` is copied
/// into the driver workspace and left untouched.
#[allow(clippy::type_complexity)]
pub fn blas_impl_lstsq_f<T>(
    a: TensorReference<'_, T, DeviceBLAS, IxD>,
    b: TensorReference<'_, T, DeviceBLAS, IxD>,
    rcond: Option<T::Real>,
) -> Result<
    LstsqResult<
        Tensor<T, DeviceBLAS, IxD>,
        Tensor<T::Real, DeviceBLAS, IxD>,
        Tensor<usize, DeviceBLAS, IxD>,
        Tensor<T::Real, DeviceBLAS, IxD>,
    >,
>
where
    T: BlasFloat,
    T::Real: FromPrimitive,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationAnyAPI<T::Real>
        + DeviceCreationNumAPI<usize>
        + OpAssignArbitaryAPI<T, Ix2, Ix2>
        + OpAssignAPI<T, Ix2>
        + OpAssignArbitaryAPI<T::Real, Ix1, Ix1>
        + OpAssignAPI<T::Real, Ix1>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>
        + OpAssignArbitaryAPI<T::Real, IxD, IxD>
        + OpAssignAPI<T::Real, IxD>,
{
    let device = a.device().clone();
    rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch, "lstsq: a and b are on different devices")?;
    let order = device.default_order();
    let plan = lstsq_plan(a.shape(), b.shape(), order, "lstsq")?;

    let rcond = rcond.unwrap_or_else(|| {
        let maxmn = T::Real::from_usize(plan.m.max(plan.n)).unwrap();
        maxmn * T::Real::epsilon()
    });
    if <T::Real as Float>::is_nan(rcond) {
        return Err(rstsr_error!(InvalidValue, "lstsq: rcond must not be NaN"));
    }

    // consume `a` in place when its batch already matches the broadcast plan;
    // a broadcast view cannot be destroyed, so it is copied
    let a_shape = plan.a_shape(order);
    let mut a_conv = if a.shape() == a_shape.as_slice() {
        overwritable_convert_with_order(a, order)?
    } else {
        overwritable_convert_with_order(TensorReference::from(a.broadcast_to(a_shape)), order)?
    };
    let b_bcast = b.broadcast_to(plan.b_shape(order));

    let mut x: Tensor<T, DeviceBLAS, IxD> = zeros_f((plan.x_shape(order), &device))?;
    let mut s: Tensor<T::Real, DeviceBLAS, IxD> = zeros_f((plan.s_shape(order), &device))?;
    let mut rank: Tensor<usize, DeviceBLAS, IxD> = zeros_f((plan.rank_shape(), &device))?;
    let mut residuals: Tensor<T::Real, DeviceBLAS, IxD> = zeros_f((plan.residual_shape(order), &device))?;

    let a_stack = StackLayout::new(a_conv.view().layout(), 2, order)?;
    let b_stack = StackLayout::new(b_bcast.layout(), if plan.is_vec { 1 } else { 2 }, order)?;
    let x_stack = StackLayout::new(x.layout(), if plan.is_vec { 1 } else { 2 }, order)?;
    let s_stack = StackLayout::new(s.layout(), 1, order)?;
    let mut rank_offsets = IterLayout::new(rank.layout(), TensorIterOrder::C)?;
    let res_stack = StackLayout::new(residuals.layout(), if plan.is_vec { 0 } else { 1 }, order)?;
    let mut res_offsets = IterLayout::new(residuals.layout(), TensorIterOrder::C)?;

    for i in 0..a_stack.len() {
        let a_i = stack_slice_at_mut(a_conv.view_mut(), &a_stack, i)?.into_dim::<Ix2>();
        let b_i = stack_slice_at(b_bcast.view(), &b_stack, i)?;
        let b_i = if plan.is_vec { b_i.i((.., None)).into_dim::<Ix2>() } else { b_i.into_dim::<Ix2>() };
        let (x_i, s_i, rank_i, res_i) =
            GELSD::default().a(TensorReference::from(a_i)).b(TensorReference::from(b_i)).rcond(rcond).build()?.run()?;

        let x_slice = stack_slice_at_mut(x.view_mut(), &x_stack, i)?;
        if plan.is_vec {
            x_slice.into_dim::<Ix1>().assign_f(x_i.i((.., 0)))?;
        } else {
            x_slice.into_dim::<Ix2>().assign_f(x_i)?;
        }
        let mut s_slice = stack_slice_at_mut(s.view_mut(), &s_stack, i)?.into_dim::<Ix1>();
        s_slice.assign_f(s_i)?;
        rank.raw_mut()[rank_offsets.next().unwrap_or(i)] = rank_i;

        if plan.is_vec {
            if plan.m > plan.n {
                let offset = res_offsets.next().unwrap_or(i);
                residuals.raw_mut()[offset] = match res_i {
                    Some(res_i) => res_i.raw()[0],
                    None => T::Real::nan(),
                };
            }
        } else {
            let mut res_slice = stack_slice_at_mut(residuals.view_mut(), &res_stack, i)?.into_dim::<Ix1>();
            match res_i {
                Some(res_i) => res_slice.assign_f(res_i)?,
                None => res_slice.fill_f(T::Real::nan())?,
            }
        }
    }
    Ok(LstsqResult { x, residuals, rank, s })
}

/* #region full-args */

#[duplicate_item(
    ImplTypes                                                                    Tr1                                     Tr2;
   [T, D1, D2, R1: DataAPI<Data = Vec<T>>, R2: DataAPI<Data = Vec<T>>] [&TensorAny<R1, T, DeviceBLAS, D1>] [&TensorAny<R2, T, DeviceBLAS, D2>];
   [T, D1, D2,                                                       ] [TensorView<'_, T, DeviceBLAS, D1>]   [TensorView<'_, T, DeviceBLAS, D2>]  ;
   [T, D1, D2,                                                       ] [TensorMut<'_, T, DeviceBLAS, D1>]    [TensorMut<'_, T, DeviceBLAS, D2>]   ;
   [T, D1, D2,                                                       ] [Tensor<T, DeviceBLAS, D1>]           [Tensor<T, DeviceBLAS, D2>]          ;
)]
impl<ImplTypes> LstsqAPI<DeviceBLAS> for (Tr1, Tr2)
where
    T: BlasFloat,
    T::Real: FromPrimitive,
    D1: DimAPI,
    D2: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationAnyAPI<T::Real>
        + DeviceCreationNumAPI<usize>
        + OpAssignArbitaryAPI<T, Ix2, Ix2>
        + OpAssignAPI<T, Ix2>
        + OpAssignArbitaryAPI<T::Real, Ix1, Ix1>
        + OpAssignAPI<T::Real, Ix1>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>
        + OpAssignArbitaryAPI<T::Real, IxD, IxD>
        + OpAssignAPI<T::Real, IxD>,
{
    type Out = LstsqResult<
        Tensor<T, DeviceBLAS, IxD>,
        Tensor<T::Real, DeviceBLAS, IxD>,
        Tensor<usize, DeviceBLAS, IxD>,
        Tensor<T::Real, DeviceBLAS, IxD>,
    >;
    fn lstsq_f(self) -> Result<Self::Out> {
        let (a, b) = self;
        blas_impl_lstsq_f(a.view().to_dyn().into(), b.view().to_dyn().into(), None)
    }
}

#[duplicate_item(
    ImplTypes                                                                    Tr1                                     Tr2;
   [T, D1, D2, R1: DataAPI<Data = Vec<T>>, R2: DataAPI<Data = Vec<T>>] [&TensorAny<R1, T, DeviceBLAS, D1>] [&TensorAny<R2, T, DeviceBLAS, D2>];
   [T, D1, D2,                                                       ] [TensorView<'_, T, DeviceBLAS, D1>]   [TensorView<'_, T, DeviceBLAS, D2>]  ;
   [T, D1, D2,                                                       ] [TensorMut<'_, T, DeviceBLAS, D1>]    [TensorMut<'_, T, DeviceBLAS, D2>]   ;
   [T, D1, D2,                                                       ] [Tensor<T, DeviceBLAS, D1>]           [Tensor<T, DeviceBLAS, D2>]          ;
)]
impl<ImplTypes> LstsqAPI<DeviceBLAS> for (Tr1, Tr2, T::Real)
where
    T: BlasFloat,
    T::Real: FromPrimitive,
    D1: DimAPI,
    D2: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationAnyAPI<T::Real>
        + DeviceCreationNumAPI<usize>
        + OpAssignArbitaryAPI<T, Ix2, Ix2>
        + OpAssignAPI<T, Ix2>
        + OpAssignArbitaryAPI<T::Real, Ix1, Ix1>
        + OpAssignAPI<T::Real, Ix1>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>
        + OpAssignArbitaryAPI<T::Real, IxD, IxD>
        + OpAssignAPI<T::Real, IxD>,
{
    type Out = LstsqResult<
        Tensor<T, DeviceBLAS, IxD>,
        Tensor<T::Real, DeviceBLAS, IxD>,
        Tensor<usize, DeviceBLAS, IxD>,
        Tensor<T::Real, DeviceBLAS, IxD>,
    >;
    fn lstsq_f(self) -> Result<Self::Out> {
        let (a, b, rcond) = self;
        blas_impl_lstsq_f(a.view().to_dyn().into(), b.view().to_dyn().into(), Some(rcond))
    }
}

/* #endregion */
