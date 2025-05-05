//! implementation of faer matmul by basic types

#![allow(clippy::unnecessary_cast)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::too_many_arguments)]

use crate::prelude_dev::*;
use num::complex::Complex;
use rayon::prelude::*;

const PARALLEL_SWITCH: usize = 64;

/* #region gemm */

#[duplicate_item(
     ty             ty_faer                     fn_name       ;
    [f32]          [f32]                       [gemm_faer_f32];
    [f64]          [f64]                       [gemm_faer_f64];
    [Complex<f32>] [faer::complex_native::c32] [gemm_faer_c32];
    [Complex<f64>] [faer::complex_native::c64] [gemm_faer_c64];
)]
pub fn fn_name(
    c: &mut [ty],
    lc: &Layout<Ix2>,
    a: &[ty],
    la: &Layout<Ix2>,
    b: &[ty],
    lb: &Layout<Ix2>,
    alpha: ty,
    beta: ty,
    pool: Option<&ThreadPool>,
) -> Result<()>
where {
    let nthreads = pool.map_or_else(|| 1, |pool| pool.current_num_threads());

    // shape check
    let sc = lc.shape();
    let sa = la.shape();
    let sb = lb.shape();
    rstsr_assert_eq!(sc[0], sa[0], InvalidLayout)?;
    rstsr_assert_eq!(sa[1], sb[0], InvalidLayout)?;
    rstsr_assert_eq!(sc[1], sb[1], InvalidLayout)?;

    let faer_a = unsafe {
        faer::mat::from_raw_parts::<ty_faer>(
            a.as_ptr().add(la.offset()) as *const ty_faer,
            la.shape()[0],
            la.shape()[1],
            la.stride()[0],
            la.stride()[1],
        )
    };
    let faer_b = unsafe {
        faer::mat::from_raw_parts::<ty_faer>(
            b.as_ptr().add(lb.offset()) as *const ty_faer,
            lb.shape()[0],
            lb.shape()[1],
            lb.stride()[0],
            lb.stride()[1],
        )
    };
    let faer_c = unsafe {
        faer::mat::from_raw_parts_mut::<ty_faer>(
            c.as_mut_ptr().add(lc.offset()) as *mut ty_faer,
            lc.shape()[0],
            lc.shape()[1],
            lc.stride()[0],
            lc.stride()[1],
        )
    };
    faer::linalg::matmul::matmul(
        faer_c,
        faer_a,
        faer_b,
        Some(beta.into()),
        alpha.into(),
        faer::Parallelism::Rayon(nthreads),
    );
    return Ok(());
}

/* #endregion */

/* #region syrk */

#[duplicate_item(
     ty             ty_faer                     fn_name       ;
    [f32]          [f32]                       [syrk_faer_f32];
    [f64]          [f64]                       [syrk_faer_f64];
    [Complex<f32>] [faer::complex_native::c32] [syrk_faer_c32];
    [Complex<f64>] [faer::complex_native::c64] [syrk_faer_c64];
)]
pub fn fn_name(
    c: &mut [ty],
    lc: &Layout<Ix2>,
    a: &[ty],
    la: &Layout<Ix2>,
    uplo: FlagUpLo,
    alpha: ty,
    beta: ty,
    pool: Option<&ThreadPool>,
) -> Result<()> {
    let nthreads = pool.map_or_else(|| 1, |pool| pool.current_num_threads());

    // shape check
    let sc = lc.shape();
    let sa = la.shape();
    rstsr_assert_eq!(sc[0], sc[1], InvalidLayout)?;
    rstsr_assert_eq!(sc[0], sa[0], InvalidLayout)?;

    let faer_a = unsafe {
        faer::mat::from_raw_parts::<ty_faer>(
            a.as_ptr().add(la.offset()) as *const ty_faer,
            la.shape()[0],
            la.shape()[1],
            la.stride()[0],
            la.stride()[1],
        )
    };
    let faer_at = unsafe {
        faer::mat::from_raw_parts::<ty_faer>(
            a.as_ptr().add(la.offset()) as *const ty_faer,
            la.shape()[1],
            la.shape()[0],
            la.stride()[1],
            la.stride()[0],
        )
    };
    let faer_c = unsafe {
        faer::mat::from_raw_parts_mut::<ty_faer>(
            c.as_mut_ptr().add(lc.offset()) as *mut ty_faer,
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
    faer::linalg::matmul::triangular::matmul(
        faer_c,
        block_structure,
        faer_a,
        BlockStructure::Rectangular,
        faer_at,
        BlockStructure::Rectangular,
        Some(beta.into()),
        alpha.into(),
        faer::Parallelism::Rayon(nthreads),
    );
    return Ok(());
}

#[duplicate_item(
     ty             fn_name                   gemm_name       syrk_name     ;
    [f32]          [gemm_with_syrk_faer_f32] [gemm_faer_f32] [syrk_faer_f32];
    [f64]          [gemm_with_syrk_faer_f64] [gemm_faer_f64] [syrk_faer_f64];
    [Complex<f32>] [gemm_with_syrk_faer_c32] [gemm_faer_c32] [syrk_faer_c32];
    [Complex<f64>] [gemm_with_syrk_faer_c64] [gemm_faer_c64] [syrk_faer_c64];
)]
pub fn fn_name(
    c: &mut [ty],
    lc: &Layout<Ix2>,
    a: &[ty],
    la: &Layout<Ix2>,
    alpha: ty,
    beta: ty,
    pool: Option<&ThreadPool>,
) -> Result<()> {
    let nthreads = pool.map_or_else(|| 1, |pool| pool.current_num_threads());

    // This function performs c = beta * c + alpha * a * a^T
    // Note that we do not assume c is symmetric, so if beta != 0, we fall back to
    // full gemm (in order not to allocate a temporary buffer)
    // beta is usually zero, in that normal use case of tensor multiplication
    // usually do not involve output matrix c
    if beta != <ty>::from(0.0) {
        gemm_name(c, lc, a, la, a, &la.reverse_axes(), alpha, beta, pool)?;
    } else {
        syrk_name(c, lc, a, la, FlagUpLo::L, alpha, beta, pool)?;
        // symmetrize
        let n = lc.shape()[0];
        if n < PARALLEL_SWITCH || nthreads == 1 {
            for i in 0..n {
                for j in 0..i {
                    let idx_ij = unsafe { lc.index_uncheck(&[i, j]) as usize };
                    let idx_ji = unsafe { lc.index_uncheck(&[j, i]) as usize };
                    c[idx_ji] = c[idx_ij];
                }
            }
        } else {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(nthreads).build().unwrap();
            pool.install(|| {
                (0..n).into_par_iter().for_each(|i| {
                    (0..i).for_each(|j| unsafe {
                        let idx_ij = lc.index_uncheck(&[i, j]) as usize;
                        let idx_ji = lc.index_uncheck(&[j, i]) as usize;
                        let c_ptr_ji = c.as_ptr().add(idx_ji) as *mut ty;
                        *c_ptr_ji = c[idx_ij];
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
        gemm_faer_f64(&mut c, &lc, &a, &la, &b, &lb, 1.0, 0.0, pool).unwrap();
        println!("time: {:?}", start.elapsed());
        let start = Instant::now();
        gemm_faer_f64(&mut c, &lc, &a, &la, &b, &lb, 1.0, 0.0, pool).unwrap();
        println!("time: {:?}", start.elapsed());
        let start = Instant::now();
        gemm_faer_f64(&mut c, &lc, &a, &la, &b, &lb, 1.0, 0.0, pool).unwrap();
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
        syrk_faer_f64(&mut c, &lc, &a, &la, FlagUpLo::L, 2.0, 1.0, pool).unwrap();
        println!("{:?}", c);
    }
}
