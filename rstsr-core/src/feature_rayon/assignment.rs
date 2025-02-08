use crate::prelude_dev::*;

// this value is used to determine whether to use contiguous inner iteration
const CONTIG_SWITCH: usize = 16;
// This value is used to determine when to use parallel iteration.
// Actual switch value is PARALLEL_SWITCH * RAYON_NUM_THREADS.
// Since current task is not intensive to each element, this value is large.
// 64 kB for f64
const PARALLEL_SWITCH: usize = 1024;
// For assignment, it is fully memory bounded; contiguous assignment is better
// handled by serial code. So we only do parallel in outer iteration
// (non-contiguous part).

pub fn assign_arbitary_cpu_rayon<T, DC, DA>(
    c: &mut [T],
    lc: &Layout<DC>,
    a: &[T],
    la: &Layout<DA>,
    pool: &rayon::ThreadPool,
) -> Result<()>
where
    T: Clone + Send + Sync,
    DC: DimAPI,
    DA: DimAPI,
{
    // determine whether to use parallel iteration
    let nthreads = pool.current_num_threads();
    let size = lc.size();
    if size < PARALLEL_SWITCH * nthreads || nthreads == 1 {
        return assign_arbitary_cpu_serial(c, lc, a, la);
    }

    // actual parallel iteration
    let contig = match TensorOrder::default() {
        TensorOrder::C => lc.c_contig() && la.c_contig(),
        TensorOrder::F => lc.f_contig() && la.f_contig(),
    };
    if contig {
        // contiguous case
        // we do not perform parallel for this case
        let offset_c = lc.offset();
        let offset_a = la.offset();
        let size = lc.size();
        c[offset_c..(offset_c + size)].clone_from_slice(&a[offset_a..(offset_a + size)]);
    } else {
        // determine order by layout preference
        let order = match TensorOrder::default() {
            TensorOrder::C => TensorIterOrder::C,
            TensorOrder::F => TensorIterOrder::F,
        };
        // generate col-major iterator
        let lc = translate_to_col_major_unary(lc, order)?;
        let la = translate_to_col_major_unary(la, order)?;
        // iterate and assign
        let func = |(idx_c, idx_a): (usize, usize)| unsafe {
            let c_ptr = c.as_ptr() as *mut T;
            *c_ptr.add(idx_c) = a[idx_a].clone();
        };
        pool.install(|| layout_col_major_dim_dispatch_par_2diff(&lc, &la, func))?;
    }
    return Ok(());
}

pub fn assign_cpu_rayon<T, D>(
    c: &mut [T],
    lc: &Layout<D>,
    a: &[T],
    la: &Layout<D>,
    pool: &rayon::ThreadPool,
) -> Result<()>
where
    T: Clone + Send + Sync,
    D: DimAPI,
{
    // determine whether to use parallel iteration
    let nthreads = pool.current_num_threads();
    let size = lc.size();
    if size < PARALLEL_SWITCH * nthreads || nthreads == 1 {
        return assign_cpu_serial(c, lc, a, la);
    }

    // re-align layouts
    let layouts_full = translate_to_col_major(&[lc, la], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // actual parallel iteration
    if size_contig < CONTIG_SWITCH {
        // not possible for contiguous assign
        let lc = &layouts_full[0];
        let la = &layouts_full[1];
        let func = |(idx_c, idx_a): (usize, usize)| unsafe {
            let c_ptr = c.as_ptr() as *mut T;
            *c_ptr.add(idx_c) = a[idx_a].clone();
        };
        pool.install(|| layout_col_major_dim_dispatch_par_2(lc, la, func))?;
    } else {
        // parallel for outer iteration
        let lc = &layouts_contig[0];
        let la = &layouts_contig[1];
        let func = |(idx_c, idx_a): (usize, usize)| unsafe {
            let c_ptr = c.as_ptr().add(idx_c) as *mut T;
            (0..size_contig).for_each(|idx| {
                *c_ptr.add(idx) = a[idx_a + idx].clone();
            })
        };
        pool.install(|| layout_col_major_dim_dispatch_par_2(lc, la, func))?;
    }
    return Ok(());
}

pub fn fill_cpu_rayon<T, D>(
    c: &mut [T],
    lc: &Layout<D>,
    fill: T,
    pool: &rayon::ThreadPool,
) -> Result<()>
where
    T: Clone + Send + Sync,
    D: DimAPI,
{
    // determine whether to use parallel iteration
    let nthreads = pool.current_num_threads();
    let size = lc.size();
    if size < PARALLEL_SWITCH * nthreads {
        return fill_cpu_serial(c, lc, fill);
    }

    // re-align layouts
    let layouts_full = [translate_to_col_major_unary(lc, TensorIterOrder::G)?];
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // actual parallel iteration
    if size_contig < CONTIG_SWITCH {
        // not possible for contiguous fill
        let lc = &layouts_full[0];
        let func = |idx_c| unsafe {
            let c_ptr = c.as_ptr() as *mut T;
            *c_ptr.add(idx_c) = fill.clone();
        };
        pool.install(|| layout_col_major_dim_dispatch_par_1(lc, func))?;
    } else {
        // parallel for outer iteration
        let lc = &layouts_contig[0];
        let func = |idx_c| unsafe {
            let c_ptr = c.as_ptr().add(idx_c) as *mut T;
            (0..size_contig).for_each(|idx| {
                *c_ptr.add(idx) = fill.clone();
            })
        };
        pool.install(|| layout_col_major_dim_dispatch_par_1(lc, func))?;
    }
    return Ok(());
}
