//! implementation of faer matmul by basic types

#![allow(clippy::unnecessary_cast)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::too_many_arguments)]

use crate::prelude_dev::*;
use core::mem::transmute;
use core::num::NonZeroUsize;
use faer::prelude::*;
use faer::traits::ComplexField;
use num::Num;
use rayon::prelude::*;

const PARALLEL_SWITCH: usize = 64;

/* #region gemm */

pub fn gemm_faer<T>(
    c: &mut [T],
    lc: &Layout<Ix2>,
    a: &[T],
    la: &Layout<Ix2>,
    b: &[T],
    lb: &Layout<Ix2>,
    alpha: T,
    beta: T,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    T: ComplexField + Num + MulAssign<T>,
{
    let nthreads = pool.map_or_else(|| 1, |pool| pool.current_num_threads());

    // shape check
    let sc = lc.shape();
    let sa = la.shape();
    let sb = lb.shape();
    rstsr_assert_eq!(sc[0], sa[0], InvalidLayout)?;
    rstsr_assert_eq!(sa[1], sb[0], InvalidLayout)?;
    rstsr_assert_eq!(sc[1], sb[1], InvalidLayout)?;

    let faer_a = unsafe {
        MatRef::from_raw_parts(
            a.as_ptr().add(la.offset()) as *const T,
            la.shape()[0],
            la.shape()[1],
            la.stride()[0],
            la.stride()[1],
        )
    };
    let faer_b = unsafe {
        MatRef::from_raw_parts(
            b.as_ptr().add(lb.offset()) as *const T,
            lb.shape()[0],
            lb.shape()[1],
            lb.stride()[0],
            lb.stride()[1],
        )
    };
    let faer_c = unsafe {
        MatMut::from_raw_parts_mut(
            c.as_mut_ptr().add(lc.offset()) as *mut T,
            lc.shape()[0],
            lc.shape()[1],
            lc.stride()[0],
            lc.stride()[1],
        )
    };

    if beta == T::zero() {
        faer::linalg::matmul::matmul(
            faer_c,
            faer::Accum::Replace,
            faer_a,
            faer_b,
            alpha,
            faer::Par::Rayon(NonZeroUsize::new(nthreads).unwrap()),
        );
    } else {
        if beta != T::one() {
            // perform inplace multiplication
            let c = unsafe { transmute::<&mut [T], &mut [MaybeUninit<T>]>(c) };
            op_muta_numb_func_cpu_rayon(
                c,
                lc,
                beta,
                &mut |vc, vb| unsafe { *vc.assume_init_mut() *= vb.clone() },
                pool,
            )?;
        }
        faer::linalg::matmul::matmul(
            faer_c,
            faer::Accum::Add,
            faer_a,
            faer_b,
            alpha,
            faer::Par::Rayon(NonZeroUsize::new(nthreads).unwrap()),
        );
    }
    return Ok(());
}

/* #endregion */

/* #region syrk */

pub fn syrk_faer<T>(
    c: &mut [T],
    lc: &Layout<Ix2>,
    a: &[T],
    la: &Layout<Ix2>,
    uplo: FlagUpLo,
    alpha: T,
    beta: T,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    T: ComplexField + Num + MulAssign<T>,
{
    let nthreads = pool.map_or_else(|| 1, |pool| pool.current_num_threads());

    // shape check
    let sc = lc.shape();
    let sa = la.shape();
    rstsr_assert_eq!(sc[0], sc[1], InvalidLayout)?;
    rstsr_assert_eq!(sc[0], sa[0], InvalidLayout)?;

    let faer_a = unsafe {
        MatRef::from_raw_parts(
            a.as_ptr().add(la.offset()) as *const T,
            la.shape()[0],
            la.shape()[1],
            la.stride()[0],
            la.stride()[1],
        )
    };
    let faer_at = unsafe {
        MatRef::from_raw_parts(
            a.as_ptr().add(la.offset()) as *const T,
            la.shape()[1],
            la.shape()[0],
            la.stride()[1],
            la.stride()[0],
        )
    };
    let faer_c = unsafe {
        MatMut::from_raw_parts_mut(
            c.as_mut_ptr().add(lc.offset()) as *mut T,
            lc.shape()[0],
            lc.shape()[1],
            lc.stride()[0],
            lc.stride()[1],
        )
    };

    use faer::linalg::matmul::triangular::BlockStructure;
    let block_structure = match uplo {
        FlagUpLo::U => BlockStructure::TriangularUpper,
        FlagUpLo::L => BlockStructure::TriangularLower,
    };
    if beta == T::zero() {
        faer::linalg::matmul::triangular::matmul(
            faer_c,
            block_structure,
            faer::Accum::Replace,
            faer_a,
            BlockStructure::Rectangular,
            faer_at,
            BlockStructure::Rectangular,
            alpha,
            faer::Par::Rayon(NonZeroUsize::new(nthreads).unwrap()),
        );
    } else {
        if beta != T::one() {
            // perform inplace multiplication
            let c = unsafe { transmute::<&mut [T], &mut [MaybeUninit<T>]>(c) };
            op_muta_numb_func_cpu_rayon(
                c,
                lc,
                beta,
                &mut |vc, vb| unsafe { *vc.assume_init_mut() *= vb.clone() },
                pool,
            )?;
        }
        faer::linalg::matmul::triangular::matmul(
            faer_c,
            block_structure,
            faer::Accum::Add,
            faer_a,
            BlockStructure::Rectangular,
            faer_at,
            BlockStructure::Rectangular,
            alpha,
            faer::Par::Rayon(NonZeroUsize::new(nthreads).unwrap()),
        );
    }

    return Ok(());
}

pub fn gemm_with_syrk_faer<T>(
    c: &mut [T],
    lc: &Layout<Ix2>,
    a: &[T],
    la: &Layout<Ix2>,
    alpha: T,
    beta: T,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    T: ComplexField + Num + MulAssign<T>,
{
    let nthreads = pool.map_or_else(|| 1, |pool| pool.current_num_threads());

    // This function performs c = beta * c + alpha * a * a^T
    // Note that we do not assume c is symmetric, so if beta != 0, we fall back to
    // full gemm (in order not to allocate a temporary buffer)
    // beta is usually zero, in that normal use case of tensor multiplication
    // usually do not involve output matrix c
    if beta != T::zero() {
        gemm_faer(c, lc, a, la, a, &la.reverse_axes(), alpha, beta, pool)?;
    } else {
        syrk_faer(c, lc, a, la, FlagUpLo::L, alpha, beta, pool)?;
        // symmetrize
        let n = lc.shape()[0];
        if n < PARALLEL_SWITCH || nthreads == 1 {
            for i in 0..n {
                for j in 0..i {
                    let idx_ij = unsafe { lc.index_uncheck(&[i, j]) as usize };
                    let idx_ji = unsafe { lc.index_uncheck(&[j, i]) as usize };
                    c[idx_ji] = c[idx_ij].clone();
                }
            }
        } else {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(nthreads).build().unwrap();
            pool.install(|| {
                (0..n).into_par_iter().for_each(|i| {
                    (0..i).for_each(|j| unsafe {
                        let idx_ij = lc.index_uncheck(&[i, j]) as usize;
                        let idx_ji = lc.index_uncheck(&[j, i]) as usize;
                        let c_ptr_ji = c.as_ptr().add(idx_ji) as *mut T;
                        *c_ptr_ji = c[idx_ij].clone();
                    });
                });
            });
        }
    }
    return Ok(());
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore]
    fn playground_1() {
        let m = 2048;
        let n = 2049;
        let k = 2050;
        let a = (0..m * k).map(|x| x as f64).collect::<Vec<_>>();
        let b = (0..k * n).map(|x| x as f64).collect::<Vec<_>>();
        let mut c = vec![0.0; m * n];
        let la = [m, k].c();
        let lb = [k, n].c();
        let lc = [m, n].c();

        let pool = rayon::ThreadPoolBuilder::new().num_threads(16).build().unwrap();
        let pool = Some(&pool);

        let start = Instant::now();
        gemm_faer(&mut c, &lc, &a, &la, &b, &lb, 1.0, 0.0, pool).unwrap();
        println!("time: {:?}", start.elapsed());
        let start = Instant::now();
        gemm_faer(&mut c, &lc, &a, &la, &b, &lb, 1.0, 0.0, pool).unwrap();
        println!("time: {:?}", start.elapsed());
        let start = Instant::now();
        gemm_faer(&mut c, &lc, &a, &la, &b, &lb, 1.0, 0.0, pool).unwrap();
        println!("time: {:?}", start.elapsed());
    }

    #[test]
    fn playground_2() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![1.0; 4];
        let la = [2, 2].c();
        let lc = [2, 2].c();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(16).build().unwrap();
        let pool = Some(&pool);
        syrk_faer(&mut c, &lc, &a, &la, FlagUpLo::L, 2.0, 1.0, pool).unwrap();
        println!("{c:?}");
    }

    #[test]
    #[cfg(not(feature = "col_major"))]
    fn test_minimal_correctness() {
        #[allow(non_camel_case_types)]
        type c32 = num::Complex<f32>;
        let vec_a = vec![
            c32::new(0., 1.),
            c32::new(1., 2.),
            c32::new(2., 3.),
            c32::new(3., 4.),
            c32::new(4., 5.),
            c32::new(5., 6.),
        ];
        let vec_b = vec![c32::new(0., 1.), c32::new(2., 3.), c32::new(4., 5.), c32::new(6., 7.)];
        let device = DeviceFaer::default();

        let a = asarray((vec_a, &device)).into_shape([3, 2]);
        let b = asarray((vec_b, &device)).into_shape([2, 2]);
        let c = a % b;
        let sum_c = c.raw().iter().sum::<c32>();

        assert!(sum_c.re - -78.0 < 1e-5);
        assert!(sum_c.im - 270.0 < 1e-5);
    }
}
