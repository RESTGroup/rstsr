use crate::device_cpu_serial::reduction::*;
use crate::prelude_dev::*;
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

pub fn reduce_all_cpu_rayon<T, D, I, F>(
    a: &[T],
    la: &Layout<D>,
    init: I,
    f: F,
    nthreads: usize,
) -> Result<T>
where
    T: Clone + Send + Sync,
    D: DimAPI,
    I: Fn() -> T + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync,
{
    // determine whether to use parallel iteration
    let size = la.size();
    if size < PARALLEL_SWITCH * nthreads {
        return reduce_all_cpu_serial(a, la, init, f);
    }

    // re-align layout
    let layout = translate_to_col_major_unary(la, TensorIterOrder::K)?;
    let (layout_contig, size_contig) = translate_to_col_major_with_contig(&[&layout]);

    // actual parallel iteration
    let pool = DeviceCpuRayon::new(nthreads).get_pool(nthreads)?;

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
                        f(acc_inner, unrolled_reduce(slc, &init, &f))
                    })
                    .reduce(&init, &f);
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
                            f(acc_chunk, unrolled_reduce(slc, &init, &f))
                        })
                        .reduce(&init, &f);
                    f(acc_inner, res)
                })
                .reduce(&init, &f);
        }
        return Ok(acc);
    } else {
        // manual fold when not contiguous
        let iter_a = IterLayoutColMajor::new(&layout)?;
        let mut acc = init();
        pool.install(|| {
            acc = iter_a
                .into_par_iter()
                .fold(&init, |acc, idx| f(acc, a[idx].clone()))
                .reduce(&init, &f);
        });
        return Ok(acc);
    }
}

#[allow(clippy::uninit_vec)]
pub fn reduce_axes_cpu_rayon<T, I, F>(
    a: &[T],
    la: &Layout<IxD>,
    axes: &[isize],
    init: I,
    f: F,
    nthreads: usize,
) -> Result<(Vec<T>, Layout<IxD>)>
where
    T: Clone + Send + Sync,
    I: Fn() -> T + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync,
{
    // determine whether to use parallel iteration
    let size = la.size();
    if size < PARALLEL_SWITCH * nthreads {
        return reduce_axes_cpu_serial(a, la, axes, init, f);
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
        let mut out: Vec<T> = Vec::with_capacity(len_out);
        unsafe { out.set_len(len_out) };
        let out_ptr = ThreadedRawPtr(out.as_mut_ptr());

        // actual evaluation
        (iter_out_swapped, iter_rest_swapped).into_par_iter().try_for_each(
            |(idx_out, idx_rest)| -> Result<()> {
                let out_ptr = out_ptr.get();
                let mut layout_inner = layout_axes.clone();
                unsafe { layout_inner.set_offset(idx_rest) };
                let acc = reduce_all_cpu_rayon(a, &layout_inner, &init, &f, nthreads)?;
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
        let mut f_add = |a: &mut T, b: &T| *a = f(a.clone(), b.clone());

        for idx_axes in iter_layout_axes {
            unsafe { layout_inner.set_offset(idx_axes) };
            op_muta_refb_func_cpu_rayon(
                &mut out,
                &layout_out,
                a,
                &layout_inner,
                &mut f_add,
                nthreads,
            )?;
        }
        return Ok((out, layout_out));
    }
}
