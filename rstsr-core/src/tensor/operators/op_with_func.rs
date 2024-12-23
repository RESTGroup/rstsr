use crate::prelude_dev::*;

/* #region op_func */

pub fn op_mutc_refa_refb_func<RA, RB, RC, DA, DB, DC, TA, TB, TC, B, F>(
    c: &mut TensorBase<RC, DC>,
    a: &TensorBase<RA, DA>,
    b: &TensorBase<RB, DB>,
    f: &mut F,
) -> Result<()>
where
    // lifetime and data constraints
    RA: DataAPI<Data = Storage<TA, B>>,
    RB: DataAPI<Data = Storage<TB, B>>,
    RC: DataMutAPI<Data = Storage<TC, B>>,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
    // broadcast constraints
    DC: DimMaxAPI<DA, Max = DC> + DimMaxAPI<DB, Max = DC>,
    // operation constraints
    B: DeviceOp_MutC_RefA_RefB_API<TA, TB, TC, DC, F>,
    F: FnMut(&mut TC, &TA, &TB),
{
    rstsr_assert!(c.device().same_device(a.device()), DeviceMismatch)?;
    rstsr_assert!(c.device().same_device(b.device()), DeviceMismatch)?;
    let lc = c.layout();
    let la = a.layout();
    let lb = b.layout();
    // all layouts should be broadcastable to lc
    // we can first generate broadcasted shape, then check this
    let (lc_b, la_b) = broadcast_layout_to_first(lc, la)?;
    rstsr_assert_eq!(lc_b, *lc, InvalidLayout)?;
    let (lc_b, lb_b) = broadcast_layout_to_first(lc, lb)?;
    rstsr_assert_eq!(lc_b, *lc, InvalidLayout)?;
    // op provided by device
    let device = c.device().clone();
    let storage_c = c.data_mut().storage_mut();
    let storage_a = a.data().storage();
    let storage_b = b.data().storage();
    device.op_mutc_refa_refb_func(storage_c, &lc_b, storage_a, &la_b, storage_b, &lb_b, f)
}

pub fn op_refa_refb_func<RA, RB, DA, DB, DC, TA, TB, TC, B, F>(
    a: &TensorBase<RA, DA>,
    b: &TensorBase<RB, DB>,
    f: &mut F,
) -> Result<Tensor<TC, DC, B>>
where
    // lifetime and data constraints
    RA: DataAPI<Data = Storage<TA, B>>,
    RB: DataAPI<Data = Storage<TB, B>>,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    B: DeviceAPI<TA> + DeviceAPI<TB>,
    // broadcast constraints
    DA: DimMaxAPI<DB, Max = DC>,
    // operation constraints
    B: DeviceOp_MutC_RefA_RefB_API<TA, TB, TC, DC, F>,
    B: DeviceCreationAnyAPI<TC>,
    F: FnMut(&mut TC, &TA, &TB),
{
    rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
    let la = a.layout();
    let lb = b.layout();
    let (la_b, lb_b) = broadcast_layout(la, lb)?;
    // generate output layout
    let lc_from_a = layout_for_array_copy(&la_b, TensorIterOrder::K)?;
    let lc_from_b = layout_for_array_copy(&lb_b, TensorIterOrder::K)?;
    let lc = if lc_from_a == lc_from_b {
        lc_from_a
    } else {
        match TensorOrder::default() {
            TensorOrder::C => la_b.shape().c(),
            TensorOrder::F => la_b.shape().f(),
        }
    };
    // generate empty c
    let device = a.device();
    let mut storage_c = unsafe { device.empty_impl(lc.bounds_index()?.1)? };
    // add provided by device
    let storage_a = a.data().storage();
    let storage_b = b.data().storage();
    device.op_mutc_refa_refb_func(&mut storage_c, &lc, storage_a, &la_b, storage_b, &lb_b, f)?;
    // return tensor
    Tensor::new_f(DataOwned::from(storage_c), lc)
}

pub fn op_muta_refb_func<RA, RB, DA, DB, TA, TB, B, F>(
    a: &mut TensorBase<RA, DA>,
    b: &TensorBase<RB, DB>,
    f: &mut F,
) -> Result<()>
where
    // lifetime and data constraints
    RA: DataMutAPI<Data = Storage<TA, B>>,
    RB: DataAPI<Data = Storage<TB, B>>,
    DA: DimAPI,
    DB: DimAPI,
    B: DeviceAPI<TA> + DeviceAPI<TB>,
    // broadcast constraints
    DA: DimMaxAPI<DB, Max = DA>,
    // operation constraints
    B: DeviceOp_MutA_RefB_API<TA, TB, DA, F>,
    F: FnMut(&mut TA, &TB),
{
    rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
    let la = a.layout();
    let lb = b.layout();
    // all layouts should be broadcastable to lc
    // we can first generate broadcasted shape, then check this
    let (la_b, lb_b) = broadcast_layout_to_first(la, lb)?;
    rstsr_assert_eq!(la_b, *la, InvalidLayout)?;
    // op provided by device
    let device = a.device().clone();
    let storage_a = a.data_mut().storage_mut();
    let storage_b = b.data().storage();
    device.op_muta_refb_func(storage_a, &la_b, storage_b, &lb_b, f)
}

pub fn op_muta_func<R, T, D, B, F>(a: &mut TensorBase<R, D>, f: &mut F) -> Result<()>
where
    R: DataMutAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T>,
    B: DeviceOp_MutA_API<T, D, F>,
    F: FnMut(&mut T),
{
    let la = a.layout().clone();
    let device = a.device().clone();
    let storage_a = a.data_mut().storage_mut();
    device.op_muta_func(storage_a, &la, f)
}

/* #endregion */
