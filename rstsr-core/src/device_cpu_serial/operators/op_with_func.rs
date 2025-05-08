//! Basic math operations.
//!
//! This file assumes that layouts are pre-processed and valid.

use crate::prelude_dev::*;

/* #region impl op_func for DeviceCpuSerial */

impl<TA, TB, TC, D, F> DeviceOp_MutC_RefA_RefB_API<TA, TB, TC, D, F> for DeviceCpuSerial
where
    TA: Clone,
    TB: Clone,
    TC: Clone,
    D: DimAPI,
    F: FnMut(&mut TC, &TA, &TB) + ?Sized,
{
    fn op_mutc_refa_refb_func(
        &self,
        c: &mut Vec<TC>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: &Vec<TB>,
        lb: &Layout<D>,
        f: &mut F,
    ) -> Result<()> {
        op_mutc_refa_refb_func_cpu_serial(c, lc, a, la, b, lb, f)
    }
}

impl<TA, TB, TC, D, F> DeviceOp_MutC_RefA_NumB_API<TA, TB, TC, D, F> for DeviceCpuSerial
where
    TA: Clone,
    TC: Clone,
    D: DimAPI,
    F: FnMut(&mut TC, &TA, &TB) + ?Sized,
{
    fn op_mutc_refa_numb_func(
        &self,
        c: &mut Vec<TC>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: TB,
        f: &mut F,
    ) -> Result<()> {
        op_mutc_refa_numb_func_cpu_serial(c, lc, a, la, b, f)
    }
}

impl<TA, TB, TC, D, F> DeviceOp_MutC_NumA_RefB_API<TA, TB, TC, D, F> for DeviceCpuSerial
where
    TB: Clone,
    TC: Clone,
    D: DimAPI,
    F: FnMut(&mut TC, &TA, &TB) + ?Sized,
{
    fn op_mutc_numa_refb_func(
        &self,
        c: &mut Vec<TC>,
        lc: &Layout<D>,
        a: TA,
        b: &Vec<TB>,
        lb: &Layout<D>,
        f: &mut F,
    ) -> Result<()> {
        op_mutc_numa_refb_func_cpu_serial(c, lc, a, b, lb, f)
    }
}

impl<TA, TB, D, F> DeviceOp_MutA_RefB_API<TA, TB, D, F> for DeviceCpuSerial
where
    TA: Clone,
    TB: Clone,
    D: DimAPI,
    F: FnMut(&mut TA, &TB) + ?Sized,
{
    fn op_muta_refb_func(
        &self,
        a: &mut Vec<TA>,
        la: &Layout<D>,
        b: &Vec<TB>,
        lb: &Layout<D>,
        f: &mut F,
    ) -> Result<()> {
        op_muta_refb_func_cpu_serial(a, la, b, lb, f)
    }
}

impl<TA, TB, D, F> DeviceOp_MutA_NumB_API<TA, TB, D, F> for DeviceCpuSerial
where
    TA: Clone,
    D: DimAPI,
    F: FnMut(&mut TA, &TB) + ?Sized,
{
    fn op_muta_numb_func(&self, a: &mut Vec<TA>, la: &Layout<D>, b: TB, f: &mut F) -> Result<()> {
        op_muta_numb_func_cpu_serial(a, la, b, f)
    }
}

impl<T, D, F> DeviceOp_MutA_API<T, D, F> for DeviceCpuSerial
where
    T: Clone,
    D: DimAPI,
    F: FnMut(&mut T) + ?Sized,
{
    fn op_muta_func(&self, a: &mut Vec<T>, la: &Layout<D>, f: &mut F) -> Result<()> {
        op_muta_func_cpu_serial(a, la, f)
    }
}

/* #endregion */
