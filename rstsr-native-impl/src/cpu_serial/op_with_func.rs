//! Basic math operations.
//!
//! This file assumes that layouts are pre-processed and valid.

use crate::prelude_dev::*;

// this value is used to determine whether to use contiguous inner iteration
const CONTIG_SWITCH: usize = 16;

pub fn op_mutc_refa_refb_func_cpu_serial<TA, TB, TC, D>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<D>,
    a: &[TA],
    la: &Layout<D>,
    b: &[TB],
    lb: &Layout<D>,
    mut f: impl FnMut(&mut MaybeUninit<TC>, &TA, &TB),
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
        let lc = &layouts_contig[0];
        let la = &layouts_contig[1];
        let lb = &layouts_contig[2];
        layout_col_major_dim_dispatch_3(lc, la, lb, |(idx_c, idx_a, idx_b)| {
            for i in 0..size_contig {
                f(&mut c[idx_c + i], &a[idx_a + i], &b[idx_b + i]);
            }
        })
    } else {
        let lc = &layouts_full[0];
        let la = &layouts_full[1];
        let lb = &layouts_full[2];
        layout_col_major_dim_dispatch_3(lc, la, lb, |(idx_c, idx_a, idx_b)| {
            f(&mut c[idx_c], &a[idx_a], &b[idx_b]);
        })
    }
}

pub fn op_mutc_refa_numb_func_cpu_serial<TA, TB, TC, D>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<D>,
    a: &[TA],
    la: &Layout<D>,
    b: TB,
    mut f: impl FnMut(&mut MaybeUninit<TC>, &TA, &TB),
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
        let lc = &layouts_contig[0];
        let la = &layouts_contig[1];
        layout_col_major_dim_dispatch_2(lc, la, |(idx_c, idx_a)| {
            for i in 0..size_contig {
                f(&mut c[idx_c + i], &a[idx_a + i], &b);
            }
        })
    } else {
        let lc = &layouts_full[0];
        let la = &layouts_full[1];
        layout_col_major_dim_dispatch_2(lc, la, |(idx_c, idx_a)| {
            f(&mut c[idx_c], &a[idx_a], &b);
        })
    }
}

pub fn op_mutc_numa_refb_func_cpu_serial<TA, TB, TC, D>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<D>,
    a: TA,
    b: &[TB],
    lb: &Layout<D>,
    mut f: impl FnMut(&mut MaybeUninit<TC>, &TA, &TB),
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
        let lc = &layouts_contig[0];
        let lb = &layouts_contig[1];
        layout_col_major_dim_dispatch_2(lc, lb, |(idx_c, idx_b)| {
            for i in 0..size_contig {
                f(&mut c[idx_c + i], &a, &b[idx_b + i]);
            }
        })
    } else {
        let lc = &layouts_full[0];
        let lb = &layouts_full[1];
        layout_col_major_dim_dispatch_2(lc, lb, |(idx_c, idx_b)| {
            f(&mut c[idx_c], &a, &b[idx_b]);
        })
    }
}

pub fn op_muta_refb_func_cpu_serial<TA, TB, D>(
    a: &mut [MaybeUninit<TA>],
    la: &Layout<D>,
    b: &[TB],
    lb: &Layout<D>,
    mut f: impl FnMut(&mut MaybeUninit<TA>, &TB),
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
        let la = &layouts_contig[0];
        let lb = &layouts_contig[1];
        layout_col_major_dim_dispatch_2(la, lb, |(idx_a, idx_b)| {
            for i in 0..size_contig {
                f(&mut a[idx_a + i], &b[idx_b + i]);
            }
        })
    } else {
        let la = &layouts_full[0];
        let lb = &layouts_full[1];
        layout_col_major_dim_dispatch_2(la, lb, |(idx_a, idx_b)| {
            f(&mut a[idx_a], &b[idx_b]);
        })
    }
}

pub fn op_muta_numb_func_cpu_serial<TA, TB, D>(
    a: &mut [MaybeUninit<TA>],
    la: &Layout<D>,
    b: TB,
    mut f: impl FnMut(&mut MaybeUninit<TA>, &TB),
) -> Result<()>
where
    D: DimAPI,
{
    let layout = translate_to_col_major_unary(la, TensorIterOrder::G)?;
    let (layout_contig, size_contig) = translate_to_col_major_with_contig(&[&layout]);

    if size_contig >= CONTIG_SWITCH {
        let la = &layout_contig[0];
        layout_col_major_dim_dispatch_1(la, |idx_a| {
            for i in 0..size_contig {
                f(&mut a[idx_a + i], &b);
            }
        })
    } else {
        let la = &layout;
        layout_col_major_dim_dispatch_1(la, |idx_a| {
            f(&mut a[idx_a], &b);
        })
    }
}

pub fn op_muta_func_cpu_serial<T, D>(
    a: &mut [MaybeUninit<T>],
    la: &Layout<D>,
    mut f: impl FnMut(&mut MaybeUninit<T>),
) -> Result<()>
where
    D: DimAPI,
{
    let layout = translate_to_col_major_unary(la, TensorIterOrder::G)?;
    let (layout_contig, size_contig) = translate_to_col_major_with_contig(&[&layout]);

    if size_contig >= CONTIG_SWITCH {
        let la = &layout_contig[0];
        layout_col_major_dim_dispatch_1(la, |idx_a| {
            for i in 0..size_contig {
                f(&mut a[idx_a + i]);
            }
        })
    } else {
        let la = &layout;
        layout_col_major_dim_dispatch_1(la, |idx_a| {
            f(&mut a[idx_a]);
        })
    }
}
