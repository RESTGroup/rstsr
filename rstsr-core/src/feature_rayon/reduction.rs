use crate::device_cpu_serial::reduction::*;
use crate::prelude_dev::*;
use rayon::prelude::*;

// this value is used to determine whether to use contiguous inner iteration
const CONTIG_SWITCH: usize = 32;
// This value is used to determine when to use parallel iteration.
// Actual switch value is PARALLEL_SWITCH * RAYON_NUM_THREADS.
// Since current task is not intensive to each element, this value is large.
const PARALLEL_SWITCH: usize = 128;
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
