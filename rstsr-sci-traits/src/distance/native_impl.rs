use super::metric::{MetricDistAPI, MetricDistWeightedAPI};
use num::{Float, Zero};
use rayon::prelude::*;
use rstsr_core::prelude_dev::*;

const CACHE_SIZE: usize = 256 * 1024; // 256 KiB

pub fn cdist_serial<T, M>(
    xa: &Vec<T>,
    xb: &Vec<T>,
    la: &Layout<Ix2>,
    lb: &Layout<Ix2>,
    kernel: M,
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

    rstsr_assert_eq!(
        shape_a[1],
        shape_b[1],
        InvalidLayout,
        "The number of columns in xa and xb must match."
    )?;
    let k = shape_a[1];

    let m = shape_a[0];
    let n = shape_b[0];
    let mut dists = unsafe { uninitialized_vec::<M::Out>(m * n)? };

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
                            let dist = kernel
                                .distance::<{ $STRIDED }>(uv, offsets, indices, strides, size);
                            match $ORDER {
                                RowMajor => dists[i * n + j] = dist,
                                ColMajor => dists[i + j * m] = dist,
                            }
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

    Ok(dists)
}

pub fn cdist_weighted_serial<T, M>(
    xa: &Vec<T>,
    xb: &Vec<T>,
    la: &Layout<Ix2>,
    lb: &Layout<Ix2>,
    weights: &M::Weight,
    kernel: M,
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

    rstsr_assert_eq!(
        shape_a[1],
        shape_b[1],
        InvalidLayout,
        "The number of columns in xa and xb must match."
    )?;
    let k = shape_a[1];

    let m = shape_a[0];
    let n = shape_b[0];
    let mut dists = unsafe { uninitialized_vec::<M::Out>(m * n)? };

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
                                RowMajor => dists[i * n + j] = dist,
                                ColMajor => dists[i + j * m] = dist,
                            }
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

    Ok(dists)
}

pub fn cdist_rayon<T, M>(
    xa: &Vec<T>,
    xb: &Vec<T>,
    la: &Layout<Ix2>,
    lb: &Layout<Ix2>,
    kernel: M,
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

    rstsr_assert_eq!(
        shape_a[1],
        shape_b[1],
        InvalidLayout,
        "The number of columns in xa and xb must match."
    )?;
    let k = shape_a[1];

    let m = shape_a[0];
    let n = shape_b[0];
    let dists = unsafe { uninitialized_vec::<M::Out>(m * n)? };

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
                            let dist = kernel
                                .distance::<{ $STRIDED }>(uv, offsets, indices, strides, size);
                            unsafe {
                                let dist_ij = match $ORDER {
                                    RowMajor => dists.as_ptr().add(i * n + j) as *mut _,
                                    ColMajor => dists.as_ptr().add(i + j * m) as *mut _,
                                };
                                *dist_ij = dist;
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

    Ok(dists)
}

pub fn cdist_weighted_rayon<T, M>(
    xa: &Vec<T>,
    xb: &Vec<T>,
    la: &Layout<Ix2>,
    lb: &Layout<Ix2>,
    weights: &M::Weight,
    kernel: M,
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

    rstsr_assert_eq!(
        shape_a[1],
        shape_b[1],
        InvalidLayout,
        "The number of columns in xa and xb must match."
    )?;
    let k = shape_a[1];

    let m = shape_a[0];
    let n = shape_b[0];
    let dists = unsafe { uninitialized_vec::<M::Out>(m * n)? };

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
                                    RowMajor => dists.as_ptr().add(i * n + j) as *mut _,
                                    ColMajor => dists.as_ptr().add(i + j * m) as *mut _,
                                };
                                *dist_ij = dist;
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

    Ok(dists)
}
