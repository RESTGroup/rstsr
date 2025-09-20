use crate::prelude_dev::*;
use core::mem::transmute;

/* #region op_func */

pub fn op_mutc_refa_refb_func<RA, RB, RC, DA, DB, DC, TA, TB, TC, B, F>(
    c: &mut TensorAny<RC, TC, B, DC>,
    a: &TensorAny<RA, TA, B, DA>,
    b: &TensorAny<RB, TB, B, DB>,
    f: &mut F,
) -> Result<()>
where
    // lifetime and data constraints
    RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
    RC: DataMutAPI<Data = <B as DeviceRawAPI<TC>>::Raw>,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
    // broadcast constraints
    DC: DimMaxAPI<DA, Max = DC> + DimMaxAPI<DB, Max = DC>,
    // operation constraints
    B: DeviceOp_MutC_RefA_RefB_API<TA, TB, TC, DC, F>,
    F: FnMut(&mut MaybeUninit<TC>, &TA, &TB),
{
    rstsr_assert!(c.device().same_device(a.device()), DeviceMismatch)?;
    rstsr_assert!(c.device().same_device(b.device()), DeviceMismatch)?;
    let lc = c.layout();
    let la = a.layout();
    let lb = b.layout();
    let default_order = c.device().default_order();
    // all layouts should be broadcastable to lc
    // we can first generate broadcasted shape, then check this
    let (lc_b, la_b) = broadcast_layout_to_first(lc, la, default_order)?;
    rstsr_assert_eq!(lc_b, *lc, InvalidLayout)?;
    let (lc_b, lb_b) = broadcast_layout_to_first(lc, lb, default_order)?;
    rstsr_assert_eq!(lc_b, *lc, InvalidLayout)?;
    // op provided by device
    let device = c.device().clone();
    let c_raw_mut = unsafe {
        transmute::<&mut <B as DeviceRawAPI<TC>>::Raw, &mut <B as DeviceRawAPI<MaybeUninit<TC>>>::Raw>(c.raw_mut())
    };
    device.op_mutc_refa_refb_func(c_raw_mut, &lc_b, a.raw(), &la_b, b.raw(), &lb_b, f)
}

pub fn op_refa_refb_func<RA, RB, DA, DB, DC, TA, TB, TC, B, F>(
    a: &TensorAny<RA, TA, B, DA>,
    b: &TensorAny<RB, TB, B, DB>,
    f: &mut F,
) -> Result<Tensor<TC, B, DC>>
where
    // lifetime and data constraints
    RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
    // broadcast constraints
    DA: DimMaxAPI<DB, Max = DC>,
    // operation constraints
    B: DeviceOp_MutC_RefA_RefB_API<TA, TB, TC, DC, F>,
    B: DeviceCreationAnyAPI<TC>,
    F: FnMut(&mut MaybeUninit<TC>, &TA, &TB),
{
    rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
    let la = a.layout();
    let lb = b.layout();
    let default_order = a.device().default_order();
    let (la_b, lb_b) = broadcast_layout(la, lb, default_order)?;
    // generate output layout
    let lc_from_a = layout_for_array_copy(&la_b, TensorIterOrder::K)?;
    let lc_from_b = layout_for_array_copy(&lb_b, TensorIterOrder::K)?;
    let lc = if lc_from_a == lc_from_b {
        lc_from_a
    } else {
        match default_order {
            RowMajor => la_b.shape().c(),
            ColMajor => la_b.shape().f(),
        }
    };
    // generate empty c
    let device = a.device();
    let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
    // add provided by device
    device.op_mutc_refa_refb_func(storage_c.raw_mut(), &lc, a.raw(), &la_b, b.raw(), &lb_b, f)?;
    // return tensor
    let storage_c = unsafe { B::assume_init_impl(storage_c) }?;
    Tensor::new_f(storage_c, lc)
}

pub fn op_muta_refb_func<RA, RB, DA, DB, TA, TB, B, F>(
    a: &mut TensorAny<RA, TA, B, DA>,
    b: &TensorAny<RB, TB, B, DB>,
    f: &mut F,
) -> Result<()>
where
    // lifetime and data constraints
    RA: DataMutAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
    DA: DimAPI,
    DB: DimAPI,
    B: DeviceAPI<TA> + DeviceAPI<TB>,
    // broadcast constraints
    DA: DimMaxAPI<DB, Max = DA>,
    // operation constraints
    B: DeviceOp_MutA_RefB_API<TA, TB, DA, F>,
    F: FnMut(&mut MaybeUninit<TA>, &TB),
{
    rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
    let la = a.layout();
    let lb = b.layout();
    let default_order = a.device().default_order();
    // all layouts should be broadcastable to lc
    // we can first generate broadcasted shape, then check this
    let (la_b, lb_b) = broadcast_layout_to_first(la, lb, default_order)?;
    rstsr_assert_eq!(la_b, *la, InvalidLayout)?;
    // op provided by device
    let device = a.device().clone();
    let a_raw_mut = unsafe {
        transmute::<&mut <B as DeviceRawAPI<TA>>::Raw, &mut <B as DeviceRawAPI<MaybeUninit<TA>>>::Raw>(a.raw_mut())
    };
    device.op_muta_refb_func(a_raw_mut, &la_b, b.raw(), &lb_b, f)
}

pub fn op_muta_func<R, T, D, B, F>(a: &mut TensorAny<R, T, B, D>, f: &mut F) -> Result<()>
where
    R: DataMutAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T>,
    B: DeviceOp_MutA_API<T, D, F>,
    F: FnMut(&mut MaybeUninit<T>),
{
    let la = a.layout().clone();
    let device = a.device().clone();
    let a_raw_mut = unsafe {
        transmute::<&mut <B as DeviceRawAPI<T>>::Raw, &mut <B as DeviceRawAPI<MaybeUninit<T>>>::Raw>(a.raw_mut())
    };
    device.op_muta_func(a_raw_mut, &la, f)
}

/* #endregion */
