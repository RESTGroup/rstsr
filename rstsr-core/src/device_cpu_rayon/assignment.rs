use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::device_cpu_serial::assignment::*;
use crate::prelude_dev::*;

use super::device::DeviceCpuRayon;

// this value is used to determine whether to use contiguous inner iteration
const CONTIG_SWITCH: usize = 16;
// This value is used to determine when to use parallel iteration.
// Actual switch value is PARALLEL_SWITCH * RAYON_NUM_THREADS.
// Since current task is not intensive to each element, this value is large.
const PARALLEL_SWITCH: usize = 256;
// For assignment, it is fully memory bounded; contiguous assignment is better
// handled by serial code. So we only do parallel in outer iteration
// (non-contiguous part).

pub fn assign_arbitary_cpu_rayon<T, DC, DA>(
    c: &mut [T],
    lc: &Layout<DC>,
    a: &[T],
    la: &Layout<DA>,
    nthreads: usize,
) -> Result<()>
where
    T: Clone + Send + Sync,
    DC: DimAPI,
    DA: DimAPI,
{
    // determine whether to use parallel iteration
    let size = lc.size();
    if size < PARALLEL_SWITCH * nthreads {
        return assign_arbitary_cpu_serial(c, lc, a, la);
    }

    // actual parallel iteration
    let pool = DeviceCpuRayon::new(nthreads).get_pool(nthreads)?;
    if lc.c_contig() && la.c_contig() || lc.f_contig() && la.f_contig() {
        // contiguous case
        // we do not perform parallel for this case
        let offset_c = lc.offset();
        let offset_a = la.offset();
        let size = lc.size();
        c[offset_c..(offset_c + size)].clone_from_slice(&a[offset_a..(offset_a + size)]);
    } else {
        // determine order by layout preference
        let order = {
            if lc.c_prefer() && la.c_prefer() {
                TensorIterOrder::C
            } else if lc.f_prefer() && la.f_prefer() {
                TensorIterOrder::F
            } else {
                match TensorOrder::default() {
                    TensorOrder::C => TensorIterOrder::C,
                    TensorOrder::F => TensorIterOrder::F,
                }
            }
        };
        // generate col-major iterator
        let lc = translate_to_col_major_unary(lc, order)?;
        let la = translate_to_col_major_unary(la, order)?;
        let iter_c = IterLayoutColMajor::new(&lc)?;
        let iter_a = IterLayoutColMajor::new(&la)?;
        // iterate and assign
        pool.install(|| {
            (iter_c, iter_a).into_par_iter().for_each(|(idx_c, idx_a)| unsafe {
                let c_ptr = c.as_ptr() as *mut T;
                *c_ptr.add(idx_c) = a[idx_a].clone();
            });
        });
    }
    return Ok(());
}

pub fn assign_cpu_rayon<T, D>(
    c: &mut [T],
    lc: &Layout<D>,
    a: &[T],
    la: &Layout<D>,
    nthreads: usize,
) -> Result<()>
where
    T: Clone + Send + Sync,
    D: DimAPI,
{
    // determine whether to use parallel iteration
    let size = lc.size();
    if size < PARALLEL_SWITCH * nthreads {
        return assign_cpu_serial(c, lc, a, la);
    }

    // re-align layouts
    let layouts_full = translate_to_col_major(&[lc, la], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // actual parallel iteration
    let pool = DeviceCpuRayon::new(nthreads).get_pool(nthreads)?;
    if size_contig < CONTIG_SWITCH {
        // not possible for contiguous assign
        let iter_c = IterLayoutColMajor::new(&layouts_full[0])?;
        let iter_a = IterLayoutColMajor::new(&layouts_full[1])?;
        pool.install(|| {
            (iter_c, iter_a).into_par_iter().for_each(|(idx_c, idx_a)| unsafe {
                let c_ptr = c.as_ptr() as *mut T;
                *c_ptr.add(idx_c) = a[idx_a].clone();
            });
        });
    } else {
        // parallel for outer iteration
        let iter_c = IterLayoutColMajor::new(&layouts_contig[0])?;
        let iter_a = IterLayoutColMajor::new(&layouts_contig[1])?;
        pool.install(|| {
            (iter_c, iter_a).into_par_iter().for_each(|(idx_c, idx_a)| unsafe {
                let c_ptr = c.as_ptr().add(idx_c) as *mut T;
                (0..size_contig).for_each(|idx| {
                    *c_ptr.add(idx) = a[idx_a + idx].clone();
                })
            });
        });
    }
    return Ok(());
}

pub fn fill_cpu_rayon<T, D>(c: &mut [T], lc: &Layout<D>, fill: T, nthreads: usize) -> Result<()>
where
    T: Clone + Send + Sync,
    D: DimAPI,
{
    // determine whether to use parallel iteration
    let size = lc.size();
    if size < PARALLEL_SWITCH * nthreads {
        return fill_cpu_serial(c, lc, fill);
    }

    // re-align layouts
    let layouts_full = [translate_to_col_major_unary(lc, TensorIterOrder::G)?];
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // actual parallel iteration
    let pool = DeviceCpuRayon::new(nthreads).get_pool(nthreads)?;
    if size_contig < CONTIG_SWITCH {
        // not possible for contiguous fill
        let iter_c = IterLayoutColMajor::new(&layouts_full[0])?;
        pool.install(|| {
            (iter_c).into_par_iter().for_each(|idx_c| unsafe {
                let c_ptr = c.as_ptr() as *mut T;
                *c_ptr.add(idx_c) = fill.clone();
            });
        });
    } else {
        // parallel for outer iteration
        let iter_c = IterLayoutColMajor::new(&layouts_contig[0])?;
        pool.install(|| {
            (iter_c).into_par_iter().for_each(|idx_c| unsafe {
                let c_ptr = c.as_ptr().add(idx_c) as *mut T;
                (0..size_contig).for_each(|idx| {
                    *c_ptr.add(idx) = fill.clone();
                })
            });
        });
    }
    return Ok(());
}

impl<T, DC, DA> OpAssignArbitaryAPI<T, DC, DA> for DeviceCpuRayon
where
    T: Clone + Send + Sync,
    DC: DimAPI,
    DA: DimAPI,
{
    fn assign_arbitary(
        &self,
        c: &mut Storage<T, Self>,
        lc: &Layout<DC>,
        a: &Storage<T, Self>,
        la: &Layout<DA>,
    ) -> Result<()> {
        let c = c.rawvec_mut();
        let a = a.rawvec();
        let nthreads = self.get_num_threads();
        assign_arbitary_cpu_rayon(c, lc, a, la, nthreads)
    }
}

impl<T, D> OpAssignAPI<T, D> for DeviceCpuRayon
where
    T: Clone + Send + Sync,
    D: DimAPI,
{
    fn assign(
        &self,
        c: &mut Storage<T, Self>,
        lc: &Layout<D>,
        a: &Storage<T, Self>,
        la: &Layout<D>,
    ) -> Result<()> {
        let c = c.rawvec_mut();
        let a = a.rawvec();
        let nthreads = self.get_num_threads();
        assign_cpu_rayon(c, lc, a, la, nthreads)
    }

    fn fill(&self, c: &mut Storage<T, Self>, lc: &Layout<D>, fill: T) -> Result<()> {
        let c = c.rawvec_mut();
        let nthreads = self.get_num_threads();
        fill_cpu_rayon(c, lc, fill, nthreads)
    }
}
