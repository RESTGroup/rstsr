//! Basic math operations.
//!
//! This file assumes that layouts are pre-processed and valid.

use crate::prelude_dev::*;

// this value is used to determine whether to use contiguous inner iteration
const CONTIG_SWITCH: usize = 16;

/* #region op_func definition */

pub fn op_mutc_refa_refb_func_cpu_serial<TA, TB, TC, D>(
    c: &mut [TC],
    lc: &Layout<D>,
    a: &[TA],
    la: &Layout<D>,
    b: &[TB],
    lb: &Layout<D>,
    mut f: impl FnMut(&mut TC, &TA, &TB),
) -> Result<()>
where
    D: DimAPI,
{
    // re-align layouts
    let layouts_full = translate_to_col_major(&[lc, la, lb], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // contiguous iteration if possible, otherwise use iterator of layout
    if size_contig >= CONTIG_SWITCH {
        let iter_c = IterLayoutColMajor::new(&layouts_contig[0])?;
        let iter_a = IterLayoutColMajor::new(&layouts_contig[1])?;
        let iter_b = IterLayoutColMajor::new(&layouts_contig[2])?;
        for (idx_c, idx_a, idx_b) in izip!(iter_c, iter_a, iter_b) {
            for i in 0..size_contig {
                f(&mut c[idx_c + i], &a[idx_a + i], &b[idx_b + i]);
            }
        }
    } else {
        let iter_c = IterLayoutColMajor::new(&layouts_full[0])?;
        let iter_a = IterLayoutColMajor::new(&layouts_full[1])?;
        let iter_b = IterLayoutColMajor::new(&layouts_full[2])?;
        for (idx_c, idx_a, idx_b) in izip!(iter_c, iter_a, iter_b) {
            f(&mut c[idx_c], &a[idx_a], &b[idx_b]);
        }
    }
    return Ok(());
}

pub fn op_mutc_refa_numb_func_cpu_serial<TA, TB, TC, D>(
    c: &mut [TC],
    lc: &Layout<D>,
    a: &[TA],
    la: &Layout<D>,
    b: TB,
    mut f: impl FnMut(&mut TC, &TA, &TB),
) -> Result<()>
where
    D: DimAPI,
{
    // re-align layouts
    let layouts_full = translate_to_col_major(&[lc, la], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // contiguous iteration if possible, otherwise use iterator of layout
    if size_contig >= CONTIG_SWITCH {
        let iter_c = IterLayoutColMajor::new(&layouts_contig[0])?;
        let iter_a = IterLayoutColMajor::new(&layouts_contig[1])?;
        for (idx_c, idx_a) in izip!(iter_c, iter_a) {
            for i in 0..size_contig {
                f(&mut c[idx_c + i], &a[idx_a + i], &b);
            }
        }
    } else {
        let iter_c = IterLayoutColMajor::new(&layouts_full[0])?;
        let iter_a = IterLayoutColMajor::new(&layouts_full[1])?;
        for (idx_c, idx_a) in izip!(iter_c, iter_a) {
            f(&mut c[idx_c], &a[idx_a], &b);
        }
    }
    return Ok(());
}

pub fn op_mutc_numa_refb_func_cpu_serial<TA, TB, TC, D>(
    c: &mut [TC],
    lc: &Layout<D>,
    a: TA,
    b: &[TB],
    lb: &Layout<D>,
    mut f: impl FnMut(&mut TC, &TA, &TB),
) -> Result<()>
where
    D: DimAPI,
{
    // re-align layouts
    let layouts_full = translate_to_col_major(&[lc, lb], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // contiguous iteration if possible, otherwise use iterator of layout
    if size_contig >= CONTIG_SWITCH {
        let iter_c = IterLayoutColMajor::new(&layouts_contig[0])?;
        let iter_b = IterLayoutColMajor::new(&layouts_contig[1])?;
        for (idx_c, idx_b) in izip!(iter_c, iter_b) {
            for i in 0..size_contig {
                f(&mut c[idx_c + i], &a, &b[idx_b + i]);
            }
        }
    } else {
        let iter_c = IterLayoutColMajor::new(&layouts_full[0])?;
        let iter_b = IterLayoutColMajor::new(&layouts_full[1])?;
        for (idx_c, idx_b) in izip!(iter_c, iter_b) {
            f(&mut c[idx_c], &a, &b[idx_b]);
        }
    }
    return Ok(());
}

pub fn op_muta_refb_func_cpu_serial<TA, TB, D>(
    a: &mut [TA],
    la: &Layout<D>,
    b: &[TB],
    lb: &Layout<D>,
    mut f: impl FnMut(&mut TA, &TB),
) -> Result<()>
where
    D: DimAPI,
{
    // re-align layouts
    let layouts_full = translate_to_col_major(&[la, lb], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // contiguous iteration if possible, otherwise use iterator of layout
    if size_contig >= CONTIG_SWITCH {
        let iter_a = IterLayoutColMajor::new(&layouts_contig[0])?;
        let iter_b = IterLayoutColMajor::new(&layouts_contig[1])?;
        for (idx_a, idx_b) in izip!(iter_a, iter_b) {
            for i in 0..size_contig {
                f(&mut a[idx_a + i], &b[idx_b + i]);
            }
        }
    } else {
        let iter_a = IterLayoutColMajor::new(&layouts_full[0])?;
        let iter_b = IterLayoutColMajor::new(&layouts_full[1])?;
        for (idx_a, idx_b) in izip!(iter_a, iter_b) {
            f(&mut a[idx_a], &b[idx_b]);
        }
    }
    return Ok(());
}

pub fn op_muta_numb_func_cpu_serial<TA, TB, D>(
    a: &mut [TA],
    la: &Layout<D>,
    b: TB,
    mut f: impl FnMut(&mut TA, &TB),
) -> Result<()>
where
    D: DimAPI,
{
    let layout = translate_to_col_major_unary(la, TensorIterOrder::G)?;
    let (layout_contig, size_contig) = translate_to_col_major_with_contig(&[&layout]);

    if size_contig >= CONTIG_SWITCH {
        let iter_a = IterLayoutColMajor::new(&layout_contig[0])?;
        for idx_a in iter_a {
            for i in 0..size_contig {
                f(&mut a[idx_a + i], &b);
            }
        }
    } else {
        let iter_a = IterLayoutColMajor::new(&layout)?;
        for idx_a in iter_a {
            f(&mut a[idx_a], &b);
        }
    }
    return Ok(());
}

pub fn op_muta_func_cpu_serial<T, D>(
    a: &mut [T],
    la: &Layout<D>,
    mut f: impl FnMut(&mut T),
) -> Result<()>
where
    D: DimAPI,
{
    let layout = translate_to_col_major_unary(la, TensorIterOrder::G)?;
    let (layout_contig, size_contig) = translate_to_col_major_with_contig(&[&layout]);

    if size_contig >= CONTIG_SWITCH {
        let iter_a = IterLayoutColMajor::new(&layout_contig[0])?;
        for idx_a in iter_a {
            for i in 0..size_contig {
                f(&mut a[idx_a + i]);
            }
        }
    } else {
        let iter_a = IterLayoutColMajor::new(&layout)?;
        for idx_a in iter_a {
            f(&mut a[idx_a]);
        }
    }
    return Ok(());
}

/* #endregion */

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
