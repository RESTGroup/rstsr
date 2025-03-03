use crate::device_cpu_serial::reduction::*;
use crate::feature_rayon::*;
use crate::prelude_dev::*;
use core::sync::atomic::{AtomicPtr, Ordering};
use rayon::prelude::*;

// this value is used to determine whether to use contiguous inner iteration
const CONTIG_SWITCH: usize = 32;
// This value is used to determine when to use parallel iteration.
// Actual switch value is PARALLEL_SWITCH * RAYON_NUM_THREADS.
// Since current task is not intensive to each element, this value is large.
// 64 kB for f64
const PARALLEL_SWITCH: usize = 1024;
// This value is the maximum chunk in parallel iteration.
const PARALLEL_CHUNK_MAX: usize = 1024;
// Currently, we do not make contiguous parts to be parallel. Only outer
// iteration is parallelized.

pub fn reduce_all_cpu_rayon<TI, TS, TO, D, I, F, FSum, FOut>(
    a: &[TI],
    la: &Layout<D>,
    init: I,
    f: F,
    f_sum: FSum,
    f_out: FOut,
    pool: &rayon::ThreadPool,
) -> Result<TO>
where
    TI: Clone + Send + Sync,
    TS: Clone + Send + Sync,
    TO: Clone + Send + Sync,
    D: DimAPI,
    I: Fn() -> TS + Send + Sync,
    F: Fn(TS, TI) -> TS + Send + Sync,
    FSum: Fn(TS, TS) -> TS + Send + Sync,
    FOut: Fn(TS) -> TO + Send + Sync,
{
    // determine whether to use parallel iteration
    let nthreads = pool.current_num_threads();
    let size = la.size();
    if size < PARALLEL_SWITCH * nthreads {
        return reduce_all_cpu_serial(a, la, init, f, f_sum, f_out);
    }

    // re-align layout
    let layout = translate_to_col_major_unary(la, TensorIterOrder::K)?;
    let (layout_contig, size_contig) = translate_to_col_major_with_contig(&[&layout]);

    // actual parallel iteration
    if size_contig >= CONTIG_SWITCH {
        // parallel for outer iteration
        let mut acc = init();
        let iter_a = IterLayoutColMajor::new(&layout_contig[0])?;
        if size_contig < PARALLEL_SWITCH * nthreads {
            // not parallel inner iteration
            pool.install(|| {
                acc = iter_a
                    .into_par_iter()
                    .fold(&init, |acc_inner, idx_a| {
                        let slc = &a[idx_a..idx_a + size_contig];
                        f_sum(acc_inner, unrolled_reduce(slc, &init, &f, &f_sum))
                    })
                    .reduce(&init, &f_sum);
            });
        } else {
            // parallel inner iteration
            let chunk = PARALLEL_CHUNK_MAX.min(size_contig / nthreads + 1);
            acc = iter_a
                .into_par_iter()
                .fold(&init, |acc_inner, idx_a| {
                    let res = (0..size_contig)
                        .into_par_iter()
                        .step_by(chunk)
                        .fold(&init, |acc_chunk, idx| {
                            let chunk = chunk.min(size_contig - idx);
                            let start = idx_a + idx;
                            let slc = &a[start..start + chunk];
                            f_sum(acc_chunk, unrolled_reduce(slc, &init, &f, &f_sum))
                        })
                        .reduce(&init, &f_sum);
                    f_sum(acc_inner, res)
                })
                .reduce(&init, &f_sum);
        }
        return Ok(f_out(acc));
    } else {
        // manual fold when not contiguous
        let iter_a = IterLayoutColMajor::new(&layout)?;
        let mut acc = init();
        pool.install(|| {
            acc = iter_a
                .into_par_iter()
                .fold(&init, |acc, idx| f(acc, a[idx].clone()))
                .reduce(&init, &f_sum);
        });
        return Ok(f_out(acc));
    }
}

#[allow(clippy::too_many_arguments)]
pub fn reduce_axes_cpu_rayon<TI, TS, I, F, FSum, FOut>(
    a: &[TI],
    la: &Layout<IxD>,
    axes: &[isize],
    init: I,
    f: F,
    f_sum: FSum,
    f_out: FOut,
    pool: &rayon::ThreadPool,
) -> Result<(Vec<TS>, Layout<IxD>)>
where
    TI: Clone + Send + Sync,
    TS: Clone + Send + Sync,
    I: Fn() -> TS + Send + Sync,
    F: Fn(TS, TI) -> TS + Send + Sync,
    FSum: Fn(TS, TS) -> TS + Send + Sync,
    FOut: Fn(TS) -> TS + Send + Sync,
{
    // determine whether to use parallel iteration
    let nthreads = pool.current_num_threads();
    let size = la.size();
    if size < PARALLEL_SWITCH * nthreads {
        return reduce_axes_cpu_serial(a, la, axes, init, f, f_sum, f_out);
    }

    // split the layout into axes (to be summed) and the rest
    let (layout_axes, layout_rest) = la.dim_split_axes(axes)?;
    let layout_axes = translate_to_col_major_unary(&layout_axes, TensorIterOrder::default())?;

    // generate layout for result (from layout_rest)
    let layout_out = layout_for_array_copy(&layout_rest, TensorIterOrder::default())?;

    // use contiguous reduce_all only when size of contiguous part is large enough
    let (_, size_contig) = translate_to_col_major_with_contig(&[&layout_axes]);
    if size_contig >= CONTIG_SWITCH {
        // generate layouts for actual evaluation
        let layouts_swapped =
            translate_to_col_major(&[&layout_out, &layout_rest], TensorIterOrder::default())?;
        let layout_out_swapped = &layouts_swapped[0];
        let layout_rest_swapped = &layouts_swapped[1];

        // iterate both layout_rest and layout_out
        let iter_out_swapped = IterLayoutRowMajor::new(layout_out_swapped)?;
        let iter_rest_swapped = IterLayoutRowMajor::new(layout_rest_swapped)?;

        // prepare output
        let len_out = layout_out.size();
        let mut out: Vec<TS> = unsafe { uninitialized_vec(len_out) };
        let out_ptr = AtomicPtr::new(out.as_mut_ptr());

        // actual evaluation
        (iter_out_swapped, iter_rest_swapped).into_par_iter().try_for_each(
            |(idx_out, idx_rest)| -> Result<()> {
                let out_ptr = out_ptr.load(Ordering::Relaxed);
                // let out_ptr = out_ptr.get();
                let mut layout_inner = layout_axes.clone();
                unsafe { layout_inner.set_offset(idx_rest) };
                let acc = reduce_all_cpu_rayon(a, &layout_inner, &init, &f, &f_sum, &f_out, pool)?;
                unsafe { *out_ptr.add(idx_out) = acc };
                Ok(())
            },
        )?;
        return Ok((out, layout_out));
    } else {
        // iterate layout_axes
        let iter_layout_axes = IterLayoutRowMajor::new(&layout_axes)?;

        // inner layout is axes not to be summed
        let mut layout_inner = layout_rest.clone();

        // prepare output
        let len_out = layout_out.size();
        let init_val = init();
        let mut out = vec![init_val; len_out];

        // closure for adding to mutable reference
        let mut f_add = |a: &mut TS, b: &TI| *a = f(a.clone(), b.clone());

        for idx_axes in iter_layout_axes {
            unsafe { layout_inner.set_offset(idx_axes) };
            op_muta_refb_func_cpu_rayon(&mut out, &layout_out, a, &layout_inner, &mut f_add, pool)?;
        }
        let mut fin_inplace = |a: &mut TS| *a = f_out(a.clone());
        op_muta_func_cpu_rayon(&mut out, &layout_out, &mut fin_inplace, pool)?;
        return Ok((out, layout_out));
    }
}

#[allow(clippy::too_many_arguments)]
pub fn reduce_axes_difftype_cpu_rayon<TI, TS, TO, I, F, FSum, FOut>(
    a: &[TI],
    la: &Layout<IxD>,
    axes: &[isize],
    init: I,
    f: F,
    f_sum: FSum,
    f_out: FOut,
    pool: &rayon::ThreadPool,
) -> Result<(Vec<TO>, Layout<IxD>)>
where
    TI: Clone + Send + Sync,
    TS: Clone + Send + Sync,
    TO: Clone + Send + Sync,
    I: Fn() -> TS + Send + Sync,
    F: Fn(TS, TI) -> TS + Send + Sync,
    FSum: Fn(TS, TS) -> TS + Send + Sync,
    FOut: Fn(TS) -> TO + Send + Sync,
{
    // determine whether to use parallel iteration
    let nthreads = pool.current_num_threads();
    let size = la.size();
    if size < PARALLEL_SWITCH * nthreads {
        return reduce_axes_difftype_cpu_serial(a, la, axes, init, f, f_sum, f_out);
    }

    // split the layout into axes (to be summed) and the rest
    let (layout_axes, layout_rest) = la.dim_split_axes(axes)?;
    let layout_axes = translate_to_col_major_unary(&layout_axes, TensorIterOrder::default())?;

    // generate layout for result (from layout_rest)
    let layout_out = layout_for_array_copy(&layout_rest, TensorIterOrder::default())?;

    // use contiguous reduce_all only when size of contiguous part is large enough
    let (_, size_contig) = translate_to_col_major_with_contig(&[&layout_axes]);
    if size_contig >= CONTIG_SWITCH {
        // generate layouts for actual evaluation
        let layouts_swapped =
            translate_to_col_major(&[&layout_out, &layout_rest], TensorIterOrder::default())?;
        let layout_out_swapped = &layouts_swapped[0];
        let layout_rest_swapped = &layouts_swapped[1];

        // iterate both layout_rest and layout_out
        let iter_out_swapped = IterLayoutRowMajor::new(layout_out_swapped)?;
        let iter_rest_swapped = IterLayoutRowMajor::new(layout_rest_swapped)?;

        // prepare output
        let len_out = layout_out.size();
        let mut out: Vec<TO> = unsafe { uninitialized_vec(len_out) };
        let out_ptr = AtomicPtr::new(out.as_mut_ptr());

        // actual evaluation
        (iter_out_swapped, iter_rest_swapped).into_par_iter().try_for_each(
            |(idx_out, idx_rest)| -> Result<()> {
                let out_ptr = out_ptr.load(Ordering::Relaxed);
                // let out_ptr = out_ptr.get();
                let mut layout_inner = layout_axes.clone();
                unsafe { layout_inner.set_offset(idx_rest) };
                let acc = reduce_all_cpu_rayon(a, &layout_inner, &init, &f, &f_sum, &f_out, pool)?;
                unsafe { *out_ptr.add(idx_out) = acc };
                Ok(())
            },
        )?;
        return Ok((out, layout_out));
    } else {
        // iterate layout_axes
        let iter_layout_axes = IterLayoutRowMajor::new(&layout_axes)?;

        // inner layout is axes not to be summed
        let mut layout_inner = layout_rest.clone();

        // prepare output
        let len_out = layout_out.size();
        let init_val = init();
        let mut out = vec![init_val; len_out];

        // closure for adding to mutable reference
        let mut f_add = |a: &mut TS, b: &TI| *a = f(a.clone(), b.clone());

        for idx_axes in iter_layout_axes {
            unsafe { layout_inner.set_offset(idx_axes) };
            op_muta_refb_func_cpu_rayon(&mut out, &layout_out, a, &layout_inner, &mut f_add, pool)?;
        }

        let mut out_converted = unsafe { uninitialized_vec(len_out) };
        let mut f_out = |a: &mut TO, b: &TS| *a = f_out(b.clone());
        op_muta_refb_func_cpu_rayon(
            &mut out_converted,
            &layout_out,
            &out,
            &layout_out,
            &mut f_out,
            pool,
        )?;
        return Ok((out_converted, layout_out));
    }
}
