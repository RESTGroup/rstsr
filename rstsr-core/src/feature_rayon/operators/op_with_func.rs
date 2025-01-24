use crate::prelude_dev::*;
use rayon::prelude::*;

// this value is used to determine whether to use contiguous inner iteration
const CONTIG_SWITCH: usize = 16;
// This value is used to determine when to use parallel iteration.
// Actual switch value is PARALLEL_SWITCH * RAYON_NUM_THREADS.
// Since current task is not intensive to each element, this value is large.
// 64 kB for f64
const PARALLEL_SWITCH: usize = 1024;
// Currently, we do not make contiguous parts to be parallel. Only outer
// iteration is parallelized.

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
    pool: &rayon::ThreadPool,
) -> Result<()>
where
    TA: Send + Sync,
    TB: Send + Sync,
    TC: Send + Sync,
    D: DimAPI,
    F: Fn(&mut TC, &TA, &TB) + ?Sized + Sync + Send,
{
    // determine whether to use parallel iteration
    let nthreads = pool.current_num_threads();
    let size = lc.size();
    if size < PARALLEL_SWITCH * nthreads || nthreads == 1 {
        return op_mutc_refa_refb_func_cpu_serial(c, lc, a, la, b, lb, f);
    }

    // re-align layouts
    let layouts_full = translate_to_col_major(&[lc, la, lb], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_outer, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // actual parallel iteration
    if size_contig >= CONTIG_SWITCH {
        // parallel for outer iteration
        let iter_c = IterLayoutColMajor::new(&layouts_outer[0])?;
        let iter_a = IterLayoutColMajor::new(&layouts_outer[1])?;
        let iter_b = IterLayoutColMajor::new(&layouts_outer[2])?;
        if size_contig < PARALLEL_SWITCH * nthreads {
            // not parallel inner iteration
            pool.install(|| {
                (iter_c, iter_a, iter_b).into_par_iter().for_each(|(idx_c, idx_a, idx_b)| unsafe {
                    let c_ptr = c.as_ptr().add(idx_c) as *mut TC;
                    (0..size_contig).for_each(|idx| {
                        f(&mut *c_ptr.add(idx), &a[idx_a + idx], &b[idx_b + idx]);
                    });
                });
            });
        } else {
            // parallel inner iteration
            pool.install(|| {
                (iter_c, iter_a, iter_b).into_par_iter().for_each(|(idx_c, idx_a, idx_b)| unsafe {
                    (0..size_contig).into_par_iter().for_each(|idx| {
                        let c_ptr = c.as_ptr().add(idx_c + idx) as *mut TC;
                        f(&mut *c_ptr, &a[idx_a + idx], &b[idx_b + idx]);
                    });
                });
            });
        }
    } else {
        // not possible for contiguous assign
        let iter_c = IterLayoutColMajor::new(&layouts_full[0])?;
        let iter_a = IterLayoutColMajor::new(&layouts_full[1])?;
        let iter_b = IterLayoutColMajor::new(&layouts_full[2])?;
        pool.install(|| {
            (iter_c, iter_a, iter_b).into_par_iter().for_each(|(idx_c, idx_a, idx_b)| unsafe {
                let c_ptr = c.as_ptr() as *mut TC;
                f(&mut *c_ptr.add(idx_c), &a[idx_a], &b[idx_b]);
            });
        });
    }
    return Ok(());
}

pub fn op_mutc_refa_numb_func_cpu_rayon<TA, TB, TC, D, F>(
    c: &mut [TC],
    lc: &Layout<D>,
    a: &[TA],
    la: &Layout<D>,
    b: TB,
    f: &mut F,
    pool: &rayon::ThreadPool,
) -> Result<()>
where
    TA: Send + Sync,
    TB: Send + Sync,
    TC: Send + Sync,
    D: DimAPI,
    F: Fn(&mut TC, &TA, &TB) + ?Sized + Send + Sync,
{
    // determine whether to use parallel iteration
    let nthreads = pool.current_num_threads();
    let size = lc.size();
    if size < PARALLEL_SWITCH * nthreads || nthreads == 1 {
        return op_mutc_refa_numb_func_cpu_serial(c, lc, a, la, b, f);
    }

    // re-align layouts
    let layouts_full = translate_to_col_major(&[lc, la], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_outer, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // actual parallel iteration
    if size_contig >= CONTIG_SWITCH {
        // parallel for outer iteration
        let iter_c = IterLayoutColMajor::new(&layouts_outer[0])?;
        let iter_a = IterLayoutColMajor::new(&layouts_outer[1])?;
        if size_contig < PARALLEL_SWITCH * nthreads {
            // not parallel inner iteration
            pool.install(|| {
                (iter_c, iter_a).into_par_iter().for_each(|(idx_c, idx_a)| unsafe {
                    let c_ptr = c.as_ptr().add(idx_c) as *mut TC;
                    (0..size_contig).for_each(|idx| {
                        f(&mut *c_ptr.add(idx), &a[idx_a + idx], &b);
                    });
                });
            });
        } else {
            // parallel inner iteration
            pool.install(|| {
                (iter_c, iter_a).into_par_iter().for_each(|(idx_c, idx_a)| unsafe {
                    (0..size_contig).into_par_iter().for_each(|idx| {
                        let c_ptr = c.as_ptr().add(idx_c + idx) as *mut TC;
                        f(&mut *c_ptr, &a[idx_a + idx], &b);
                    });
                });
            });
        }
    } else {
        // not possible for contiguous assign
        let iter_c = IterLayoutColMajor::new(&layouts_full[0])?;
        let iter_a = IterLayoutColMajor::new(&layouts_full[1])?;
        pool.install(|| {
            (iter_c, iter_a).into_par_iter().for_each(|(idx_c, idx_a)| unsafe {
                let c_ptr = c.as_ptr() as *mut TC;
                f(&mut *c_ptr.add(idx_c), &a[idx_a], &b);
            });
        });
    }
    return Ok(());
}

pub fn op_mutc_numa_refb_func_cpu_rayon<TA, TB, TC, D, F>(
    c: &mut [TC],
    lc: &Layout<D>,
    a: TA,
    b: &[TB],
    lb: &Layout<D>,
    f: &mut F,
    pool: &rayon::ThreadPool,
) -> Result<()>
where
    TA: Send + Sync,
    TB: Send + Sync,
    TC: Send + Sync,
    D: DimAPI,
    F: Fn(&mut TC, &TA, &TB) + ?Sized + Send + Sync,
{
    // determine whether to use parallel iteration
    let nthreads = pool.current_num_threads();
    let size = lc.size();
    if size < PARALLEL_SWITCH * nthreads || nthreads == 1 {
        return op_mutc_numa_refb_func_cpu_serial(c, lc, a, b, lb, f);
    }

    // re-align layouts
    let layouts_full = translate_to_col_major(&[lc, lb], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_outer, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // actual parallel iteration
    if size_contig >= CONTIG_SWITCH {
        // parallel for outer iteration
        let iter_c = IterLayoutColMajor::new(&layouts_outer[0])?;
        let iter_b = IterLayoutColMajor::new(&layouts_outer[1])?;
        if size_contig < PARALLEL_SWITCH * nthreads {
            // not parallel inner iteration
            pool.install(|| {
                (iter_c, iter_b).into_par_iter().for_each(|(idx_c, idx_b)| unsafe {
                    let c_ptr = c.as_ptr().add(idx_c) as *mut TC;
                    (0..size_contig).for_each(|idx| {
                        f(&mut *c_ptr.add(idx), &a, &b[idx_b + idx]);
                    });
                });
            });
        } else {
            // parallel inner iteration
            pool.install(|| {
                (iter_c, iter_b).into_par_iter().for_each(|(idx_c, idx_b)| unsafe {
                    (0..size_contig).into_par_iter().for_each(|idx| {
                        let c_ptr = c.as_ptr().add(idx_c + idx) as *mut TC;
                        f(&mut *c_ptr, &a, &b[idx_b + idx]);
                    });
                });
            });
        }
    } else {
        // not possible for contiguous assign
        let iter_c = IterLayoutColMajor::new(&layouts_full[0])?;
        let iter_b = IterLayoutColMajor::new(&layouts_full[1])?;
        pool.install(|| {
            (iter_c, iter_b).into_par_iter().for_each(|(idx_c, idx_b)| unsafe {
                let c_ptr = c.as_ptr() as *mut TC;
                f(&mut *c_ptr.add(idx_c), &a, &b[idx_b]);
            });
        });
    }
    return Ok(());
}

pub fn op_muta_refb_func_cpu_rayon<TA, TB, D, F>(
    a: &mut [TA],
    la: &Layout<D>,
    b: &[TB],
    lb: &Layout<D>,
    f: &mut F,
    pool: &rayon::ThreadPool,
) -> Result<()>
where
    TA: Send + Sync,
    TB: Send + Sync,
    D: DimAPI,
    F: Fn(&mut TA, &TB) + ?Sized + Send + Sync,
{
    // determine whether to use parallel iteration
    let nthreads = pool.current_num_threads();
    let size = la.size();
    if size < PARALLEL_SWITCH * nthreads || nthreads == 1 {
        return op_muta_refb_func_cpu_serial(a, la, b, lb, f);
    }

    // re-align layouts
    let layouts_full = translate_to_col_major(&[la, lb], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_outer, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // actual parallel iteration
    if size_contig >= CONTIG_SWITCH {
        // parallel for outer iteration
        let iter_a = IterLayoutColMajor::new(&layouts_outer[0])?;
        let iter_b = IterLayoutColMajor::new(&layouts_outer[1])?;
        if size_contig < PARALLEL_SWITCH * nthreads {
            // not parallel inner iteration
            pool.install(|| {
                (iter_a, iter_b).into_par_iter().for_each(|(idx_a, idx_b)| unsafe {
                    let a_ptr = a.as_ptr().add(idx_a) as *mut TA;
                    (0..size_contig).for_each(|idx| {
                        f(&mut *a_ptr.add(idx), &b[idx_b + idx]);
                    });
                });
            });
        } else {
            // parallel inner iteration
            pool.install(|| {
                (iter_a, iter_b).into_par_iter().for_each(|(idx_a, idx_b)| unsafe {
                    (0..size_contig).for_each(|idx| {
                        let a_ptr = a.as_ptr().add(idx_a + idx) as *mut TA;
                        f(&mut *a_ptr, &b[idx_b + idx]);
                    });
                });
            });
        }
    } else {
        // not possible for contiguous assign
        let iter_a = IterLayoutColMajor::new(&layouts_full[0])?;
        let iter_b = IterLayoutColMajor::new(&layouts_full[1])?;
        pool.install(|| {
            (iter_a, iter_b).into_par_iter().for_each(|(idx_a, idx_b)| unsafe {
                let a_ptr = a.as_ptr() as *mut TA;
                f(&mut *a_ptr.add(idx_a), &b[idx_b]);
            });
        });
    }
    return Ok(());
}

pub fn op_muta_numb_func_cpu_rayon<TA, TB, D, F>(
    a: &mut [TA],
    la: &Layout<D>,
    b: TB,
    f: &mut F,
    pool: &rayon::ThreadPool,
) -> Result<()>
where
    TA: Send + Sync,
    TB: Send + Sync,
    D: DimAPI,
    F: Fn(&mut TA, &TB) + ?Sized + Send + Sync,
{
    // determine whether to use parallel iteration
    let nthreads = pool.current_num_threads();
    let size = la.size();
    if size < PARALLEL_SWITCH * nthreads || nthreads == 1 {
        return op_muta_numb_func_cpu_serial(a, la, b, f);
    }

    // re-align layouts
    let layout = translate_to_col_major_unary(la, TensorIterOrder::G)?;
    let (layout_contig, size_contig) = translate_to_col_major_with_contig(&[&layout]);

    // actual parallel iteration
    if size_contig >= CONTIG_SWITCH {
        // parallel for outer iteration
        let iter_a = IterLayoutColMajor::new(&layout_contig[0])?;
        if size_contig < PARALLEL_SWITCH * nthreads {
            // not parallel inner iteration
            pool.install(|| {
                iter_a.into_par_iter().for_each(|idx_a| unsafe {
                    let a_ptr = a.as_ptr().add(idx_a) as *mut TA;
                    (0..size_contig).for_each(|idx| {
                        f(&mut *a_ptr.add(idx), &b);
                    });
                });
            });
        } else {
            // parallel inner iteration
            pool.install(|| {
                iter_a.into_par_iter().for_each(|idx_a| unsafe {
                    (0..size_contig).into_par_iter().for_each(|idx| {
                        let a_ptr = a.as_ptr().add(idx_a + idx) as *mut TA;
                        f(&mut *a_ptr, &b);
                    });
                });
            });
        }
    } else {
        // not possible for contiguous assign
        let iter_a = IterLayoutColMajor::new(&layout)?;
        pool.install(|| {
            iter_a.into_par_iter().for_each(|idx_a| unsafe {
                let a_ptr = a.as_ptr() as *mut TA;
                f(&mut *a_ptr.add(idx_a), &b);
            });
        });
    }
    return Ok(());
}

pub fn op_muta_func_cpu_rayon<T, D, F>(
    a: &mut [T],
    la: &Layout<D>,
    f: &mut F,
    pool: &rayon::ThreadPool,
) -> Result<()>
where
    T: Send + Sync,
    D: DimAPI,
    F: Fn(&mut T) + ?Sized + Send + Sync,
{
    // determine whether to use parallel iteration
    let nthreads = pool.current_num_threads();
    let size = la.size();
    if size < PARALLEL_SWITCH * nthreads || nthreads == 1 {
        return op_muta_func_cpu_serial(a, la, f);
    }

    // re-align layouts
    let layout = translate_to_col_major_unary(la, TensorIterOrder::G)?;
    let (layout_contig, size_contig) = translate_to_col_major_with_contig(&[&layout]);

    // actual parallel iteration
    if size_contig >= CONTIG_SWITCH {
        let iter_a = IterLayoutColMajor::new(&layout_contig[0])?;
        if size_contig < PARALLEL_SWITCH * nthreads {
            // not parallel inner iteration
            pool.install(|| {
                iter_a.into_par_iter().for_each(|idx_a| unsafe {
                    let a_ptr = a.as_ptr().add(idx_a) as *mut T;
                    (0..size_contig).for_each(|idx| {
                        f(&mut *a_ptr.add(idx));
                    });
                });
            });
        } else {
            // parallel inner iteration
            pool.install(|| {
                iter_a.into_par_iter().for_each(|idx_a| unsafe {
                    (0..size_contig).into_par_iter().for_each(|idx| {
                        let a_ptr = a.as_ptr().add(idx_a + idx) as *mut T;
                        f(&mut *a_ptr);
                    });
                });
            });
        }
    } else {
        let iter_a = IterLayoutColMajor::new(&layout)?;
        pool.install(|| {
            iter_a.into_par_iter().for_each(|idx_a| unsafe {
                let a_ptr = a.as_ptr() as *mut T;
                f(&mut *a_ptr.add(idx_a));
            });
        });
    }
    return Ok(());
}
