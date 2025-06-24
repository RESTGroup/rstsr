use crate::prelude_dev::*;
use rayon::prelude::*;

// this value is used to determine whether to use contiguous inner iteration
const CONTIG_SWITCH: usize = 16;
// This value is used to determine when to use parallel iteration.
// Since current task is not intensive to each element, this value is large.
const PARALLEL_SWITCH: usize = 4096;

/* #region op_func definition */

#[allow(clippy::too_many_arguments)]
pub fn op_mutc_refa_refb_func_cpu_rayon<TA, TB, TC, D, F>(
    c: &mut [TC],
    lc: &Layout<D>,
    a: &[TA],
    la: &Layout<D>,
    b: &[TB],
    lb: &Layout<D>,
    f: &mut F,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TA: Send + Sync,
    TB: Send + Sync,
    TC: Send + Sync,
    D: DimAPI,
    F: Fn(&mut TC, &TA, &TB) + ?Sized + Sync + Send,
{
    // determine whether to use parallel iteration
    let size = lc.size();
    if size < PARALLEL_SWITCH || pool.is_none() {
        return op_mutc_refa_refb_func_cpu_serial(c, lc, a, la, b, lb, f);
    }

    // re-align layouts
    let layouts_full = translate_to_col_major(&[lc, la, lb], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_outer, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // actual parallel iteration
    if size_contig >= CONTIG_SWITCH {
        // parallel for outer iteration
        let lc = &layouts_outer[0];
        let la = &layouts_outer[1];
        let lb = &layouts_outer[2];
        if size_contig < PARALLEL_SWITCH {
            // not parallel inner iteration
            let func = |(idx_c, idx_a, idx_b)| unsafe {
                let c_ptr = c.as_ptr().add(idx_c) as *mut TC;
                (0..size_contig).for_each(|idx| {
                    f(&mut *c_ptr.add(idx), &a[idx_a + idx], &b[idx_b + idx]);
                });
            };
            let task = || layout_col_major_dim_dispatch_par_3(lc, la, lb, func);
            pool.map_or_else(task, |pool| pool.install(task))
        } else {
            // parallel inner iteration
            let func = |(idx_c, idx_a, idx_b)| unsafe {
                (0..size_contig).into_par_iter().for_each(|idx| {
                    let c_ptr = c.as_ptr().add(idx_c + idx) as *mut TC;
                    f(&mut *c_ptr, &a[idx_a + idx], &b[idx_b + idx]);
                });
            };
            let task = || layout_col_major_dim_dispatch_par_3(lc, la, lb, func);
            pool.map_or_else(task, |pool| pool.install(task))
        }
    } else {
        // not possible for contiguous assign
        let lc = &layouts_full[0];
        let la = &layouts_full[1];
        let lb = &layouts_full[2];
        let func = |(idx_c, idx_a, idx_b)| unsafe {
            let c_ptr = c.as_ptr() as *mut TC;
            f(&mut *c_ptr.add(idx_c), &a[idx_a], &b[idx_b]);
        };
        let task = || layout_col_major_dim_dispatch_par_3(lc, la, lb, func);
        pool.map_or_else(task, |pool| pool.install(task))
    }
}

pub fn op_mutc_refa_numb_func_cpu_rayon<TA, TB, TC, D, F>(
    c: &mut [TC],
    lc: &Layout<D>,
    a: &[TA],
    la: &Layout<D>,
    b: TB,
    f: &mut F,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TA: Send + Sync,
    TB: Send + Sync,
    TC: Send + Sync,
    D: DimAPI,
    F: Fn(&mut TC, &TA, &TB) + ?Sized + Send + Sync,
{
    // determine whether to use parallel iteration
    let size = lc.size();
    if size < PARALLEL_SWITCH || pool.is_none() {
        return op_mutc_refa_numb_func_cpu_serial(c, lc, a, la, b, f);
    }

    // re-align layouts
    let layouts_full = translate_to_col_major(&[lc, la], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_outer, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // actual parallel iteration
    if size_contig >= CONTIG_SWITCH {
        // parallel for outer iteration
        let lc = &layouts_outer[0];
        let la = &layouts_outer[1];
        if size_contig < PARALLEL_SWITCH {
            // not parallel inner iteration
            let func = |(idx_c, idx_a)| unsafe {
                let c_ptr = c.as_ptr().add(idx_c) as *mut TC;
                (0..size_contig).for_each(|idx| {
                    f(&mut *c_ptr.add(idx), &a[idx_a + idx], &b);
                });
            };
            let task = || layout_col_major_dim_dispatch_par_2(lc, la, func);
            pool.map_or_else(task, |pool| pool.install(task))
        } else {
            // parallel inner iteration
            let func = |(idx_c, idx_a)| unsafe {
                (0..size_contig).into_par_iter().for_each(|idx| {
                    let c_ptr = c.as_ptr().add(idx_c + idx) as *mut TC;
                    f(&mut *c_ptr, &a[idx_a + idx], &b);
                });
            };
            let task = || layout_col_major_dim_dispatch_par_2(lc, la, func);
            pool.map_or_else(task, |pool| pool.install(task))
        }
    } else {
        // not possible for contiguous assign
        let lc = &layouts_full[0];
        let la = &layouts_full[1];
        let func = |(idx_c, idx_a)| unsafe {
            let c_ptr = c.as_ptr() as *mut TC;
            f(&mut *c_ptr.add(idx_c), &a[idx_a], &b);
        };
        let task = || layout_col_major_dim_dispatch_par_2(lc, la, func);
        pool.map_or_else(task, |pool| pool.install(task))
    }
}

pub fn op_mutc_numa_refb_func_cpu_rayon<TA, TB, TC, D, F>(
    c: &mut [TC],
    lc: &Layout<D>,
    a: TA,
    b: &[TB],
    lb: &Layout<D>,
    f: &mut F,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TA: Send + Sync,
    TB: Send + Sync,
    TC: Send + Sync,
    D: DimAPI,
    F: Fn(&mut TC, &TA, &TB) + ?Sized + Send + Sync,
{
    // determine whether to use parallel iteration
    let size = lc.size();
    if size < PARALLEL_SWITCH || pool.is_none() {
        return op_mutc_numa_refb_func_cpu_serial(c, lc, a, b, lb, f);
    }

    // re-align layouts
    let layouts_full = translate_to_col_major(&[lc, lb], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_outer, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // actual parallel iteration
    if size_contig >= CONTIG_SWITCH {
        // parallel for outer iteration
        let lc = &layouts_outer[0];
        let lb = &layouts_outer[1];
        if size_contig < PARALLEL_SWITCH {
            // not parallel inner iteration
            let func = |(idx_c, idx_b)| unsafe {
                let c_ptr = c.as_ptr().add(idx_c) as *mut TC;
                (0..size_contig).for_each(|idx| {
                    f(&mut *c_ptr.add(idx), &a, &b[idx_b + idx]);
                });
            };
            let task = || layout_col_major_dim_dispatch_par_2(lc, lb, func);
            pool.map_or_else(task, |pool| pool.install(task))
        } else {
            // parallel inner iteration
            let func = |(idx_c, idx_b)| unsafe {
                (0..size_contig).into_par_iter().for_each(|idx| {
                    let c_ptr = c.as_ptr().add(idx_c + idx) as *mut TC;
                    f(&mut *c_ptr, &a, &b[idx_b + idx]);
                });
            };
            let task = || layout_col_major_dim_dispatch_par_2(lc, lb, func);
            pool.map_or_else(task, |pool| pool.install(task))
        }
    } else {
        // not possible for contiguous assign
        let lc = &layouts_full[0];
        let lb = &layouts_full[1];
        let func = |(idx_c, idx_b)| unsafe {
            let c_ptr = c.as_ptr() as *mut TC;
            f(&mut *c_ptr.add(idx_c), &a, &b[idx_b]);
        };
        let task = || layout_col_major_dim_dispatch_par_2(lc, lb, func);
        pool.map_or_else(task, |pool| pool.install(task))
    }
}

pub fn op_muta_refb_func_cpu_rayon<TA, TB, D, F>(
    a: &mut [TA],
    la: &Layout<D>,
    b: &[TB],
    lb: &Layout<D>,
    f: &mut F,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TA: Send + Sync,
    TB: Send + Sync,
    D: DimAPI,
    F: Fn(&mut TA, &TB) + ?Sized + Send + Sync,
{
    // determine whether to use parallel iteration
    let size = la.size();
    if size < PARALLEL_SWITCH || pool.is_none() {
        return op_muta_refb_func_cpu_serial(a, la, b, lb, f);
    }

    // re-align layouts
    let layouts_full = translate_to_col_major(&[la, lb], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_outer, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // actual parallel iteration
    if size_contig >= CONTIG_SWITCH {
        // parallel for outer iteration
        let la = &layouts_outer[0];
        let lb = &layouts_outer[1];
        if size_contig < PARALLEL_SWITCH {
            // not parallel inner iteration
            let func = |(idx_a, idx_b)| unsafe {
                let a_ptr = a.as_ptr().add(idx_a) as *mut TA;
                (0..size_contig).for_each(|idx| {
                    f(&mut *a_ptr.add(idx), &b[idx_b + idx]);
                });
            };
            let task = || layout_col_major_dim_dispatch_par_2(la, lb, func);
            pool.map_or_else(task, |pool| pool.install(task))
        } else {
            // parallel inner iteration
            let func = |(idx_a, idx_b)| unsafe {
                (0..size_contig).into_par_iter().for_each(|idx| {
                    let a_ptr = a.as_ptr().add(idx_a + idx) as *mut TA;
                    f(&mut *a_ptr, &b[idx_b + idx]);
                });
            };
            let task = || layout_col_major_dim_dispatch_par_2(la, lb, func);
            pool.map_or_else(task, |pool| pool.install(task))
        }
    } else {
        // not possible for contiguous assign
        let la = &layouts_full[0];
        let lb = &layouts_full[1];
        let func = |(idx_a, idx_b): (usize, usize)| unsafe {
            let a_ptr = a.as_ptr() as *mut TA;
            f(&mut *a_ptr.add(idx_a), &b[idx_b]);
        };
        let task = || layout_col_major_dim_dispatch_par_2(la, lb, func);
        pool.map_or_else(task, |pool| pool.install(task))
    }
}

pub fn op_muta_numb_func_cpu_rayon<TA, TB, D, F>(
    a: &mut [TA],
    la: &Layout<D>,
    b: TB,
    f: &mut F,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TA: Send + Sync,
    TB: Send + Sync,
    D: DimAPI,
    F: Fn(&mut TA, &TB) + ?Sized + Send + Sync,
{
    // determine whether to use parallel iteration
    let size = la.size();
    if size < PARALLEL_SWITCH || pool.is_none() {
        return op_muta_numb_func_cpu_serial(a, la, b, f);
    }

    // re-align layouts
    let layout = translate_to_col_major_unary(la, TensorIterOrder::G)?;
    let (layout_contig, size_contig) = translate_to_col_major_with_contig(&[&layout]);

    // actual parallel iteration
    if size_contig >= CONTIG_SWITCH {
        // parallel for outer iteration
        let la = &layout_contig[0];
        if size_contig < PARALLEL_SWITCH {
            // not parallel inner iteration
            let func = |idx_a| unsafe {
                let a_ptr = a.as_ptr().add(idx_a) as *mut TA;
                (0..size_contig).for_each(|idx| {
                    f(&mut *a_ptr.add(idx), &b);
                });
            };
            let task = || layout_col_major_dim_dispatch_par_1(la, func);
            pool.map_or_else(task, |pool| pool.install(task))
        } else {
            // parallel inner iteration
            let func = |idx_a| unsafe {
                (0..size_contig).into_par_iter().for_each(|idx| {
                    let a_ptr = a.as_ptr().add(idx_a + idx) as *mut TA;
                    f(&mut *a_ptr, &b);
                });
            };
            let task = || layout_col_major_dim_dispatch_par_1(la, func);
            pool.map_or_else(task, |pool| pool.install(task))
        }
    } else {
        // not possible for contiguous assign
        let func = |idx_a| unsafe {
            let a_ptr = a.as_ptr() as *mut TA;
            f(&mut *a_ptr.add(idx_a), &b);
        };
        let task = || layout_col_major_dim_dispatch_par_1(&layout, func);
        pool.map_or_else(task, |pool| pool.install(task))
    }
}

pub fn op_muta_func_cpu_rayon<T, D, F>(a: &mut [T], la: &Layout<D>, f: &mut F, pool: Option<&ThreadPool>) -> Result<()>
where
    T: Send + Sync,
    D: DimAPI,
    F: Fn(&mut T) + ?Sized + Send + Sync,
{
    // determine whether to use parallel iteration
    let size = la.size();
    if size < PARALLEL_SWITCH || pool.is_none() {
        return op_muta_func_cpu_serial(a, la, f);
    }

    // re-align layouts
    let layout = translate_to_col_major_unary(la, TensorIterOrder::G)?;
    let (layout_contig, size_contig) = translate_to_col_major_with_contig(&[&layout]);

    // actual parallel iteration
    if size_contig >= CONTIG_SWITCH {
        let la = &layout_contig[0];
        if size_contig < PARALLEL_SWITCH {
            // not parallel inner iteration
            let func = |idx_a| unsafe {
                let a_ptr = a.as_ptr().add(idx_a) as *mut T;
                (0..size_contig).for_each(|idx| {
                    f(&mut *a_ptr.add(idx));
                });
            };
            let task = || layout_col_major_dim_dispatch_par_1(la, func);
            pool.map_or_else(task, |pool| pool.install(task))
        } else {
            // parallel inner iteration
            let func = |idx_a| unsafe {
                (0..size_contig).into_par_iter().for_each(|idx| {
                    let a_ptr = a.as_ptr().add(idx_a + idx) as *mut T;
                    f(&mut *a_ptr);
                });
            };
            let task = || layout_col_major_dim_dispatch_par_1(la, func);
            pool.map_or_else(task, |pool| pool.install(task))
        }
    } else {
        let func = |idx_a| unsafe {
            let a_ptr = a.as_ptr() as *mut T;
            f(&mut *a_ptr.add(idx_a));
        };
        let task = || layout_col_major_dim_dispatch_par_1(&layout, func);
        pool.map_or_else(task, |pool| pool.install(task))
    }
}
