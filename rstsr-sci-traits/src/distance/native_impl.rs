use super::metric::{MetricDistAPI, MetricDistWeightedAPI};
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicPtr, Ordering};
use num::{Float, Zero};
use rayon::prelude::*;
use rstsr_core::prelude_dev::*;

const CACHE_SIZE: usize = 256 * 1024; // 256 KiB

/// Re-view a fully initialized `Vec<MaybeUninit<T>>` as `Vec<T>`.
///
/// # Safety
///
/// Every element of `v` must have been initialized before calling this function.
unsafe fn assume_init_vec<T>(v: Vec<MaybeUninit<T>>) -> Vec<T> {
    let (ptr, len, cap) = (v.as_ptr() as *mut T, v.len(), v.capacity());
    core::mem::forget(v);
    // SAFETY: the caller guarantees every element is initialized; the new `Vec`
    // reuses the same allocation, length and capacity.
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

pub fn cdist_serial<T, M>(
    xa: &Vec<T>,
    xb: &Vec<T>,
    la: &Layout<Ix2>,
    lb: &Layout<Ix2>,
    mut kernel: M,
    order: FlagOrder,
) -> Result<Vec<M::Out>>
where
    M: MetricDistAPI<Vec<T>>,
{
    let shape_a = la.shape();
    let shape_b = lb.shape();
    let stride_a = la.stride();
    let stride_b = lb.stride();
    let offset_a = la.offset();
    let offset_b = lb.offset();

    rstsr_assert_eq!(shape_a[1], shape_b[1], InvalidLayout, "The number of columns in xa and xb must match.")?;
    let k = shape_a[1];

    let m = shape_a[0];
    let n = shape_b[0];
    // The `MaybeUninit` element type makes the uninitialized allocation itself
    // hazard-free (see `alloc_vec_contract.md` of rstsr-common); every slot is
    // initialized below, and the buffer is re-viewed as `Vec<M::Out>` at the end.
    let mut dists: Vec<MaybeUninit<M::Out>> = unsafe { uninitialized_vec(m * n)? };

    kernel.initialize(xa, la, xb, lb)?;

    // calculate batch size based on cache size
    let size_t = std::mem::size_of::<T>();
    let batch_size = (CACHE_SIZE / (size_t * k) / 2).clamp(8, 64);

    let strided = stride_a[1] != 1 || stride_b[1] != 1;

    macro_rules! perform_batch_calc {
        ($STRIDED: ident, $ORDER: ident) => {
            for i_batch in (0..m).step_by(batch_size) {
                let batch_end = (i_batch + batch_size).min(m);
                for j_batch in (0..n).step_by(batch_size) {
                    let j_end = (j_batch + batch_size).min(n);
                    for i in i_batch..batch_end {
                        for j in j_batch..j_end {
                            let uv = (xa, xb);
                            let offsets = (
                                (offset_a as isize + i as isize * stride_a[0]) as usize,
                                (offset_b as isize + j as isize * stride_b[0]) as usize,
                            );
                            let indices = (i, j);
                            let strides = (stride_a[1], stride_b[1]);
                            let size = k;
                            let dist = kernel.distance::<{ $STRIDED }>(uv, offsets, indices, strides, size);
                            match $ORDER {
                                RowMajor => dists[i * n + j].write(dist),
                                ColMajor => dists[i + j * m].write(dist),
                            };
                        }
                    }
                }
            }
        };
    }
    match (strided, order) {
        (false, RowMajor) => perform_batch_calc!(false, RowMajor),
        (true, RowMajor) => perform_batch_calc!(true, RowMajor),
        (false, ColMajor) => perform_batch_calc!(false, ColMajor),
        (true, ColMajor) => perform_batch_calc!(true, ColMajor),
    }

    // SAFETY: every one of the `m * n` slots was written exactly once by the
    // batch loops above (each `dist` is a freshly computed value; the buffer is
    // never read before initialization).
    Ok(unsafe { assume_init_vec(dists) })
}

pub fn cdist_weighted_serial<T, M>(
    xa: &Vec<T>,
    xb: &Vec<T>,
    la: &Layout<Ix2>,
    lb: &Layout<Ix2>,
    weights: &M::Weight,
    mut kernel: M,
    order: FlagOrder,
) -> Result<Vec<M::Out>>
where
    M: MetricDistWeightedAPI<Vec<T>, Weight: AsRef<[M::Out]>, Out: Float>,
{
    let shape_a = la.shape();
    let shape_b = lb.shape();
    let stride_a = la.stride();
    let stride_b = lb.stride();
    let offset_a = la.offset();
    let offset_b = lb.offset();

    rstsr_assert_eq!(shape_a[1], shape_b[1], InvalidLayout, "The number of columns in xa and xb must match.")?;
    let k = shape_a[1];

    let m = shape_a[0];
    let n = shape_b[0];
    // The `MaybeUninit` element type makes the uninitialized allocation itself
    // hazard-free (see `alloc_vec_contract.md` of rstsr-common); every slot is
    // initialized below, and the buffer is re-viewed as `Vec<M::Out>` at the end.
    let mut dists: Vec<MaybeUninit<M::Out>> = unsafe { uninitialized_vec(m * n)? };

    kernel.weighted_initialize(xa, la, xb, lb, weights)?;

    // calculate batch size based on cache size
    let size_t = std::mem::size_of::<T>();
    let batch_size = (CACHE_SIZE / (size_t * k) / 2).clamp(8, 64);

    let strided = stride_a[1] != 1 || stride_b[1] != 1;
    let weights_sum = weights.as_ref().iter().fold(M::Out::zero(), |acc, w| acc + *w);

    macro_rules! perform_batch_calc {
        ($STRIDED: ident, $ORDER: ident) => {
            for i_batch in (0..m).step_by(batch_size) {
                let batch_end = (i_batch + batch_size).min(m);
                for j_batch in (0..n).step_by(batch_size) {
                    let j_end = (j_batch + batch_size).min(n);
                    for i in i_batch..batch_end {
                        for j in j_batch..j_end {
                            let uv = (xa, xb);
                            let offsets = (
                                (offset_a as isize + i as isize * stride_a[0]) as usize,
                                (offset_b as isize + j as isize * stride_b[0]) as usize,
                            );
                            let indices = (i, j);
                            let strides = (stride_a[1], stride_b[1]);
                            let size = k;
                            let dist = kernel.weighted_distance::<{ $STRIDED }>(
                                uv,
                                offsets,
                                indices,
                                strides,
                                size,
                                weights,
                                weights_sum,
                            );
                            match $ORDER {
                                RowMajor => dists[i * n + j].write(dist),
                                ColMajor => dists[i + j * m].write(dist),
                            };
                        }
                    }
                }
            }
        };
    }
    match (strided, order) {
        (false, RowMajor) => perform_batch_calc!(false, RowMajor),
        (true, RowMajor) => perform_batch_calc!(true, RowMajor),
        (false, ColMajor) => perform_batch_calc!(false, ColMajor),
        (true, ColMajor) => perform_batch_calc!(true, ColMajor),
    }

    // SAFETY: every one of the `m * n` slots was written exactly once by the
    // batch loops above (each `dist` is a freshly computed value; the buffer is
    // never read before initialization).
    Ok(unsafe { assume_init_vec(dists) })
}

pub fn cdist_rayon<T, M>(
    xa: &Vec<T>,
    xb: &Vec<T>,
    la: &Layout<Ix2>,
    lb: &Layout<Ix2>,
    mut kernel: M,
    order: FlagOrder,
    pool: Option<&rayon::ThreadPool>,
) -> Result<Vec<M::Out>>
where
    T: Send + Sync,
    M: MetricDistAPI<Vec<T>> + Send + Sync,
    M::Out: Send + Sync,
{
    if pool.is_none() {
        return cdist_serial(xa, xb, la, lb, kernel, order);
    }
    let pool = pool.unwrap();

    let shape_a = la.shape();
    let shape_b = lb.shape();
    let stride_a = la.stride();
    let stride_b = lb.stride();
    let offset_a = la.offset();
    let offset_b = lb.offset();

    rstsr_assert_eq!(shape_a[1], shape_b[1], InvalidLayout, "The number of columns in xa and xb must match.")?;
    let k = shape_a[1];

    let m = shape_a[0];
    let n = shape_b[0];
    // The `MaybeUninit` element type makes the uninitialized allocation itself
    // hazard-free (see `alloc_vec_contract.md` of rstsr-common); every slot is
    // initialized below, and the buffer is re-viewed as `Vec<M::Out>` at the end.
    let mut dists: Vec<MaybeUninit<M::Out>> = unsafe { uninitialized_vec(m * n)? };
    // pass mutable reference in parallel region
    let thr_dists = AtomicPtr::new(dists.as_mut_ptr());

    kernel.initialize(xa, la, xb, lb)?;

    // calculate batch size based on cache size
    let size_t = std::mem::size_of::<T>();
    let batch_size = (CACHE_SIZE / (size_t * k) / 2).clamp(8, 64);

    let strided = stride_a[1] != 1 || stride_b[1] != 1;

    macro_rules! perform_batch_calc {
        ($STRIDED: ident, $ORDER: ident) => {
            (0..m).into_par_iter().step_by(batch_size).for_each(|i_batch| {
                let batch_end = (i_batch + batch_size).min(m);
                (0..n).into_par_iter().step_by(batch_size).for_each(|j_batch| {
                    let j_end = (j_batch + batch_size).min(n);
                    // SAFETY-adjacent note: `dists_ptr` is `dists`'s base pointer hoisted
                    // through `AtomicPtr` (relaxed load; `dists` is never reassigned
                    // through it); each task writes the disjoint `i/j`-batch block.
                    let dists_ptr = thr_dists.load(Ordering::Relaxed);
                    for i in i_batch..batch_end {
                        for j in j_batch..j_end {
                            let uv = (xa, xb);
                            let offsets = (
                                (offset_a as isize + i as isize * stride_a[0]) as usize,
                                (offset_b as isize + j as isize * stride_b[0]) as usize,
                            );
                            let indices = (i, j);
                            let strides = (stride_a[1], stride_b[1]);
                            let size = k;
                            let dist = kernel.distance::<{ $STRIDED }>(uv, offsets, indices, strides, size);
                            unsafe {
                                let dist_ij = match $ORDER {
                                    RowMajor => dists_ptr.add(i * n + j),
                                    ColMajor => dists_ptr.add(i + j * m),
                                };
                                (*dist_ij).write(dist);
                            }
                        }
                    }
                })
            })
        };
    }

    pool.install(|| match (strided, order) {
        (false, RowMajor) => perform_batch_calc!(false, RowMajor),
        (true, RowMajor) => perform_batch_calc!(true, RowMajor),
        (false, ColMajor) => perform_batch_calc!(false, ColMajor),
        (true, ColMajor) => perform_batch_calc!(true, ColMajor),
    });

    // SAFETY: every one of the `m * n` slots was written exactly once by the
    // parallel batch tasks above (each `dist` is a freshly computed value; the
    // buffer is never read before initialization).
    Ok(unsafe { assume_init_vec(dists) })
}

pub fn cdist_weighted_rayon<T, M>(
    xa: &Vec<T>,
    xb: &Vec<T>,
    la: &Layout<Ix2>,
    lb: &Layout<Ix2>,
    weights: &M::Weight,
    mut kernel: M,
    order: FlagOrder,
    pool: Option<&rayon::ThreadPool>,
) -> Result<Vec<M::Out>>
where
    T: Send + Sync,
    M: MetricDistWeightedAPI<Vec<T>> + Send + Sync,
    M::Weight: AsRef<[M::Out]> + Send + Sync,
    M::Out: Float + Send + Sync,
{
    if pool.is_none() {
        return cdist_weighted_serial(xa, xb, la, lb, weights, kernel, order);
    }
    let pool = pool.unwrap();

    let shape_a = la.shape();
    let shape_b = lb.shape();
    let stride_a = la.stride();
    let stride_b = lb.stride();
    let offset_a = la.offset();
    let offset_b = lb.offset();

    rstsr_assert_eq!(shape_a[1], shape_b[1], InvalidLayout, "The number of columns in xa and xb must match.")?;
    let k = shape_a[1];

    let m = shape_a[0];
    let n = shape_b[0];
    // The `MaybeUninit` element with `write`-only access makes the
    // uninitialized allocation hazard-free (see `alloc_vec_contract.md` of
    // rstsr-common); every slot is initialized below, and the buffer is
    // re-viewed as `Vec<M::Out>` at the end.
    let mut dists: Vec<MaybeUninit<M::Out>> = unsafe { uninitialized_vec(m * n)? };
    // pass mutable reference in parallel region
    let thr_dists = AtomicPtr::new(dists.as_mut_ptr());

    kernel.weighted_initialize(xa, la, xb, lb, weights)?;

    // calculate batch size based on cache size
    let size_t = std::mem::size_of::<T>();
    let batch_size = (CACHE_SIZE / (size_t * k) / 2).clamp(8, 64);

    let strided = stride_a[1] != 1 || stride_b[1] != 1;
    let weights_sum = weights.as_ref().iter().fold(M::Out::zero(), |acc, w| acc + *w);

    macro_rules! perform_batch_calc {
        ($STRIDED: ident, $ORDER: ident) => {
            (0..m).into_par_iter().step_by(batch_size).for_each(|i_batch| {
                let batch_end = (i_batch + batch_size).min(m);
                (0..n).into_par_iter().step_by(batch_size).for_each(|j_batch| {
                    let j_end = (j_batch + batch_size).min(n);
                    // SAFETY-adjacent note: `dists_ptr` is `dists`'s base pointer hoisted
                    // through `AtomicPtr` (relaxed load; `dists` is never reassigned
                    // through it); each task writes the disjoint `i/j`-batch block.
                    let dists_ptr = thr_dists.load(Ordering::Relaxed);
                    for i in i_batch..batch_end {
                        for j in j_batch..j_end {
                            let uv = (xa, xb);
                            let offsets = (
                                (offset_a as isize + i as isize * stride_a[0]) as usize,
                                (offset_b as isize + j as isize * stride_b[0]) as usize,
                            );
                            let indices = (i, j);
                            let strides = (stride_a[1], stride_b[1]);
                            let size = k;
                            let dist = kernel.weighted_distance::<{ $STRIDED }>(
                                uv,
                                offsets,
                                indices,
                                strides,
                                size,
                                weights,
                                weights_sum,
                            );
                            unsafe {
                                let dist_ij = match $ORDER {
                                    RowMajor => dists_ptr.add(i * n + j),
                                    ColMajor => dists_ptr.add(i + j * m),
                                };
                                (*dist_ij).write(dist);
                            }
                        }
                    }
                })
            })
        };
    }

    pool.install(|| match (strided, order) {
        (false, RowMajor) => perform_batch_calc!(false, RowMajor),
        (true, RowMajor) => perform_batch_calc!(true, RowMajor),
        (false, ColMajor) => perform_batch_calc!(false, ColMajor),
        (true, ColMajor) => perform_batch_calc!(true, ColMajor),
    });

    // SAFETY: every one of the `m * n` slots was written exactly once by the
    // parallel batch tasks above (each `dist` is a freshly computed value; the
    // buffer is never read before initialization).
    Ok(unsafe { assume_init_vec(dists) })
}

/* #region tests: cdist buffer initialization discipline */

#[cfg(test)]
mod test_cdist_uninit_discipline {
    use super::*;
    use crate::distance::metric::MetricEuclidean;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Default)]
    struct Counters {
        created: AtomicUsize,
        dropped: AtomicUsize,
    }

    /// Non-POD output type with drop counting; dropping uninitialized memory,
    /// double-dropping, or leaking breaks the `created == dropped` invariant.
    struct GuardedDist {
        _payload: Vec<u8>,
        counters: Arc<Counters>,
    }

    impl GuardedDist {
        fn new(counters: Arc<Counters>) -> Self {
            counters.created.fetch_add(1, Ordering::SeqCst);
            GuardedDist { _payload: vec![0xA5; 64], counters }
        }
    }

    impl Drop for GuardedDist {
        fn drop(&mut self) {
            self.counters.dropped.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Metric whose output is the non-POD `GuardedDist` (valid instantiation:
    /// `Out` is unconstrained on `MetricDistAPI`).
    struct CountingMetric {
        counters: Arc<Counters>,
    }

    impl MetricDistAPI<Vec<f64>> for CountingMetric {
        type Out = GuardedDist;
        fn distance<const STRIDED: bool>(
            &self,
            _uv: (&Vec<f64>, &Vec<f64>),
            _offsets: (usize, usize),
            _indices: (usize, usize),
            _strides: (isize, isize),
            _size: usize,
        ) -> GuardedDist {
            GuardedDist::new(self.counters.clone())
        }
    }

    fn euclidean_cdist(order: FlagOrder) -> Vec<f64> {
        let xa = vec![0.0, 0.0, 3.0, 4.0]; // [m=2, k=2]
        let xb = vec![6.0, 8.0, 0.0, 0.0]; // [n=2, k=2]
        let la: Layout<Ix2> = Layout::new([2, 2], [2, 1], 0).unwrap();
        let lb: Layout<Ix2> = Layout::new([2, 2], [2, 1], 0).unwrap();
        cdist_serial(&xa, &xb, &la, &lb, MetricEuclidean, order).unwrap()
    }

    #[test]
    fn test_cdist_euclidean_values() {
        assert_eq!(euclidean_cdist(RowMajor), vec![10.0, 0.0, 5.0, 5.0]);
        assert_eq!(euclidean_cdist(ColMajor), vec![10.0, 5.0, 0.0, 5.0]);
    }

    #[test]
    fn test_cdist_guarded_out_serial() {
        for order in [RowMajor, ColMajor] {
            let counters = Arc::new(Counters::default());
            let xa = vec![0.0; 4]; // [m=2, k=2]
            let xb = vec![0.0; 6]; // [n=3, k=2]
            let la: Layout<Ix2> = Layout::new([2, 2], [2, 1], 0).unwrap();
            let lb: Layout<Ix2> = Layout::new([3, 2], [2, 1], 0).unwrap();
            let dists = cdist_serial(&xa, &xb, &la, &lb, CountingMetric { counters: counters.clone() }, order).unwrap();
            assert_eq!(dists.len(), 6);
            drop(dists);
            let created = counters.created.load(Ordering::SeqCst);
            let dropped = counters.dropped.load(Ordering::SeqCst);
            assert_eq!(created, 6, "one output per (i, j) pair (order = {order:?})");
            assert_eq!(dropped, created, "every output dropped exactly once (order = {order:?})");
        }
    }

    #[test]
    fn test_cdist_guarded_out_rayon() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        let counters = Arc::new(Counters::default());
        let xa = vec![0.0; 4]; // [m=2, k=2]
        let xb = vec![0.0; 6]; // [n=3, k=2]
        let la: Layout<Ix2> = Layout::new([2, 2], [2, 1], 0).unwrap();
        let lb: Layout<Ix2> = Layout::new([3, 2], [2, 1], 0).unwrap();
        let dists =
            cdist_rayon(&xa, &xb, &la, &lb, CountingMetric { counters: counters.clone() }, RowMajor, Some(&pool))
                .unwrap();
        assert_eq!(dists.len(), 6);
        drop(dists);
        let created = counters.created.load(Ordering::SeqCst);
        let dropped = counters.dropped.load(Ordering::SeqCst);
        assert_eq!(created, 6, "one output per (i, j) pair");
        assert_eq!(dropped, created, "every output dropped exactly once");
    }
}

/* #endregion */
