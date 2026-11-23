use crate::matmul_impl::*;
use crate::prelude_dev::*;
use crate::threading::with_num_threads;
use core::any::TypeId;
use core::ops::{Add, Mul};
use core::slice::{from_raw_parts, from_raw_parts_mut};
use core::sync::atomic::{AtomicPtr, Ordering};
use num::{Complex, Zero};
use rayon::prelude::*;

// code from ndarray
fn same_type<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

#[allow(clippy::too_many_arguments)]
pub fn gemm_blas_ix2_no_conj_dispatch<TA, TB, TC>(
    c: &mut [TC],
    lc: &Layout<Ix2>,
    a: &[TA],
    la: &Layout<Ix2>,
    b: &[TB],
    lb: &Layout<Ix2>,
    alpha: TC,
    beta: TC,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TA: Clone + Send + Sync + 'static,
    TB: Clone + Send + Sync + 'static,
    TC: Clone + Send + Sync + 'static,
    TA: Mul<TB, Output = TC>,
    TB: Mul<TA, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + PartialEq,
{
    // check if syrk could be applicable
    let able_syrk = beta == TC::zero()
        && same_type::<TB, TC>()
        && same_type::<TA, TC>()
        && unsafe {
            let a_ptr = a.as_ptr().add(la.offset()) as *const TC;
            let b_ptr = b.as_ptr().add(lb.offset()) as *const TC;
            let equal_ptr = core::ptr::eq(a_ptr, b_ptr);
            let equal_shape = la.shape() == lb.reverse_axes().shape();
            let equal_stride = la.stride() == lb.reverse_axes().stride();
            equal_ptr && equal_shape && equal_stride
        };

    // type check and dispatch
    macro_rules! impl_gemm_dispatch {
        ($ty: ty, $fn_gemm_name: ident, $fn_syrk_name: ident) => {
            if (same_type::<TA, $ty>() && same_type::<TB, $ty>() && same_type::<TC, $ty>()) {
                let a_slice = unsafe { from_raw_parts(a.as_ptr() as *const $ty, a.len()) };
                let b_slice = unsafe { from_raw_parts(b.as_ptr() as *const $ty, b.len()) };
                let c_slice = unsafe { from_raw_parts_mut(c.as_mut_ptr() as *mut $ty, c.len()) };
                let alpha = unsafe { *(&alpha as *const TC as *const $ty) };
                let beta = unsafe { *(&beta as *const TC as *const $ty) };
                if able_syrk {
                    $fn_syrk_name(c_slice, lc, a_slice, la, alpha, pool)?;
                } else {
                    $fn_gemm_name(c_slice, lc, a_slice, la, b_slice, lb, alpha, beta, pool)?;
                }
                return Ok(());
            }
        };
    }

    impl_gemm_dispatch!(f32, gemm_blas_no_conj_f32, syrk_blas_no_conj_f32);
    impl_gemm_dispatch!(f64, gemm_blas_no_conj_f64, syrk_blas_no_conj_f64);
    impl_gemm_dispatch!(Complex<f32>, gemm_blas_no_conj_c32, syrk_blas_no_conj_c32);
    impl_gemm_dispatch!(Complex<f64>, gemm_blas_no_conj_c64, syrk_blas_no_conj_c64);

    // not able to be accelarated by blas_no_conj
    // fallback to naive implementation
    let c_slice = c;
    let a_slice = a;
    let b_slice = b;
    return gemm_ix2_naive_cpu_rayon(c_slice, lc, a_slice, la, b_slice, lb, alpha, beta, pool);
}

#[allow(clippy::too_many_arguments)]
fn gemv_blas_ix1_no_conj_dispatch<TA, TB, TC>(
    c: &mut [TC],
    lc: &Layout<Ix1>,
    a: &[TA],
    la: &Layout<Ix2>,
    b: &[TB],
    lb: &Layout<Ix1>,
    transa: FlagTrans,
    alpha: TC,
    beta: TC,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TA: Clone + Send + Sync + 'static,
    TB: Clone + Send + Sync + 'static,
    TC: Clone + Send + Sync + 'static,
    TA: Mul<TB, Output = TC>,
    TB: Mul<TA, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + PartialEq,
{
    // type check and dispatch
    macro_rules! impl_gemv_dispatch {
        ($ty: ty, $fn_gemv_name: ident) => {
            if (same_type::<TA, $ty>() && same_type::<TB, $ty>() && same_type::<TC, $ty>()) {
                let a_slice = unsafe { from_raw_parts(a.as_ptr() as *const $ty, a.len()) };
                let b_slice = unsafe { from_raw_parts(b.as_ptr() as *const $ty, b.len()) };
                let c_slice = unsafe { from_raw_parts_mut(c.as_mut_ptr() as *mut $ty, c.len()) };
                let alpha = unsafe { *(&alpha as *const TC as *const $ty) };
                let beta = unsafe { *(&beta as *const TC as *const $ty) };
                return $fn_gemv_name(c_slice, lc, a_slice, la, b_slice, lb, transa, alpha, beta, pool);
            }
        };
    }

    impl_gemv_dispatch!(f32, gemv_blas_no_conj_f32);
    impl_gemv_dispatch!(f64, gemv_blas_no_conj_f64);
    impl_gemv_dispatch!(Complex<f32>, gemv_blas_no_conj_c32);
    impl_gemv_dispatch!(Complex<f64>, gemv_blas_no_conj_c64);

    // not able to be accelarated by blas_no_conj (mixed dtypes)
    // fallback to naive implementation
    match transa {
        FlagTrans::N => gemv_naive_cpu_serial(c, lc, a, la, b, lb, alpha, beta),
        FlagTrans::T => gevm_naive_cpu_serial(c, lc, b, lb, a, la, alpha, beta),
        _ => rstsr_invalid!(transa),
    }
}

/// 2-D BLAS matmul: route to `gemv` when the product collapses to a vector
/// (`m == 1` or `n == 1`), otherwise to `gemm`.
#[allow(clippy::too_many_arguments)]
fn matmul_blas_ix2_no_conj_dispatch<TA, TB, TC>(
    c: &mut [TC],
    lc: &Layout<Ix2>,
    a: &[TA],
    la: &Layout<Ix2>,
    b: &[TB],
    lb: &Layout<Ix2>,
    alpha: TC,
    beta: TC,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TA: Clone + Send + Sync + 'static,
    TB: Clone + Send + Sync + 'static,
    TC: Clone + Send + Sync + 'static,
    TA: Mul<TB, Output = TC>,
    TB: Mul<TA, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + PartialEq,
{
    if lc.shape()[0] == 1 {
        // c row: c(0, j) = sum_k a(0, k) b(k, j) = b^T a(0, :)
        let lc = &lc.dim_eliminate(0)?;
        let la = &la.dim_eliminate(0)?;
        return gemv_blas_ix1_no_conj_dispatch(c, lc, b, lb, a, la, FlagTrans::T, alpha, beta, pool);
    } else if lc.shape()[1] == 1 {
        // c column: c(i, 0) = sum_k a(i, k) b(k, 0)
        let lc = &lc.dim_eliminate(1)?;
        let lb = &lb.dim_eliminate(1)?;
        return gemv_blas_ix1_no_conj_dispatch(c, lc, a, la, b, lb, FlagTrans::N, alpha, beta, pool);
    }
    gemm_blas_ix2_no_conj_dispatch(c, lc, a, la, b, lb, alpha, beta, pool)
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_row_major_blas<TA, TB, TC, DA, DB, DC>(
    c: &mut [TC],
    lc: &Layout<DC>,
    a: &[TA],
    la: &Layout<DA>,
    b: &[TB],
    lb: &Layout<DB>,
    alpha: TC,
    beta: TC,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TA: Clone + Send + Sync + 'static,
    TB: Clone + Send + Sync + 'static,
    TC: Clone + Send + Sync + 'static,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: Mul<TB, Output = TC>,
    TB: Mul<TA, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + PartialEq,
{
    // NOTE: this only works for row-major layout
    // for column-major layout, we need to transpose the input:
    // C = A * B  =>  C^T = B^T * A^T

    // quick return for empty matrix
    // in this case, we do not check the shape of a, b, c
    if lc.size() == 0 {
        return Ok(());
    }

    let nthreads = match pool {
        Some(pool) => pool.current_num_threads(),
        None => 1,
    };

    // handle special cases
    match (la.ndim(), lb.ndim(), lc.ndim()) {
        (1, 1, 0) => {
            // rule 1: vector inner dot
            let la = &la.clone().into_dim::<Ix1>().unwrap();
            let lb = &lb.clone().into_dim::<Ix1>().unwrap();
            let lc = &lc.clone().into_dim::<Ix0>().unwrap();
            let c_num = &mut c[lc.offset()];
            return with_num_threads(nthreads, || inner_dot_naive_cpu_rayon(c_num, a, la, b, lb, alpha, beta, pool));
        },
        (2, 1, 1) => {
            // rule 2a: matrix-vector
            let la = &la.clone().into_dim::<Ix2>().unwrap();
            let lb = &lb.clone().into_dim::<Ix1>().unwrap();
            let lc = &lc.clone().into_dim::<Ix1>().unwrap();
            return with_num_threads(nthreads, || {
                gemv_blas_ix1_no_conj_dispatch(c, lc, a, la, b, lb, FlagTrans::N, alpha, beta, pool)
            });
        },
        (1, 2, 1) => {
            // rule 2b: vector-matrix
            let la = &la.clone().into_dim::<Ix1>().unwrap();
            let lb = &lb.clone().into_dim::<Ix2>().unwrap();
            let lc = &lc.clone().into_dim::<Ix1>().unwrap();
            return with_num_threads(nthreads, || {
                gemv_blas_ix1_no_conj_dispatch(c, lc, b, lb, a, la, FlagTrans::T, alpha, beta, pool)
            });
        },
        (2, 2, 2) => {
            // rule 2: matrix multiplication (vector-shaped products go to gemv)
            let la = &la.clone().into_dim::<Ix2>().unwrap();
            let lb = &lb.clone().into_dim::<Ix2>().unwrap();
            let lc = &lc.clone().into_dim::<Ix2>().unwrap();
            return with_num_threads(nthreads, || {
                matmul_blas_ix2_no_conj_dispatch(c, lc, a, la, b, lb, alpha, beta, pool)
            });
        },
        _ => (),
    };

    // handle broadcasted cases
    let cfg = layout_matmul_dyn_row_major_with_lc(&la.to_dim()?, &lb.to_dim()?, &lc.to_dim()?)?;
    // rules 1 and 2 are handled above as fast paths; only the broadcasted
    // rules (3..7) reach here.
    let la_matmul = cfg.la_matmul.into_dim::<Ix2>()?;
    let lb_matmul = cfg.lb_matmul.into_dim::<Ix2>()?;
    let lc_matmul = cfg.lc_matmul.into_dim::<Ix2>()?;
    let la_rest = cfg.la_rest.unwrap();
    let lb_rest = cfg.lb_rest.unwrap();
    let lc_rest = cfg.lc_rest.unwrap();
    // now, lx_rest should have the same shape, while lx_matmul
    // should be matmulable
    // only parallel matmul when lx_rest is small (larger than
    // 2*nthreads), otherwise parallel matmul anyway
    let n_task = la_rest.size();
    let ita_rest = IterLayoutColMajor::new(&la_rest)?;
    let itb_rest = IterLayoutColMajor::new(&lb_rest)?;
    let itc_rest = IterLayoutColMajor::new(&lc_rest)?;
    // a vector-shaped inner product is memory-bound: iterate batches in the
    // main thread and let BLAS use all threads instead of spreading the
    // batches over rayon tasks with single-threaded BLAS
    let vec_inner = lc_matmul.shape()[0] == 1 || lc_matmul.shape()[1] == 1;
    if !vec_inner && n_task >= 4 * nthreads {
        // parallel outer, sequential matmul
        let c_ptr = AtomicPtr::new(c.as_mut_ptr());
        let c_len = c.len();
        let task = || {
            ita_rest.into_par_iter().zip(itb_rest).zip(itc_rest).try_for_each(
                |((ia_rest, ib_rest), ic_rest)| -> Result<()> {
                    // prepare layout
                    let mut la_m = la_matmul.clone();
                    let mut lb_m = lb_matmul.clone();
                    let mut lc_m = lc_matmul.clone();
                    unsafe {
                        la_m.set_offset(ia_rest);
                        lb_m.set_offset(ib_rest);
                        lc_m.set_offset(ic_rest);
                    }
                    // task-local slice handle for this batch, derived from the hoisted
                    // base pointer
                    let c = unsafe { from_raw_parts_mut(c_ptr.load(Ordering::Relaxed), c_len) };
                    // clone alpha and beta
                    let alpha = alpha.clone();
                    let beta = beta.clone();
                    with_num_threads(1, || {
                        matmul_blas_ix2_no_conj_dispatch(c, &lc_m, a, &la_m, b, &lb_m, alpha, beta, None)
                    })
                },
            )
        };
        match pool {
            Some(pool) => pool.install(task),
            None => task(),
        }
    } else {
        // sequential outer, parallel matmul
        with_num_threads(nthreads, || -> Result<()> {
            izip!(ita_rest, itb_rest, itc_rest).try_for_each(|(ia_rest, ib_rest, ic_rest)| {
                // prepare layout
                let mut la_m = la_matmul.clone();
                let mut lb_m = lb_matmul.clone();
                let mut lc_m = lc_matmul.clone();
                unsafe {
                    la_m.set_offset(ia_rest);
                    lb_m.set_offset(ib_rest);
                    lc_m.set_offset(ic_rest);
                }
                // clone alpha and beta
                let alpha = alpha.clone();
                let beta = beta.clone();
                matmul_blas_ix2_no_conj_dispatch(c, &lc_m, a, &la_m, b, &lb_m, alpha, beta, pool)
            })
        })
    }
}

/// Row-major driver of the fresh-output matmul: `c = alpha * (a @ b)` into
/// an uninitialized `c`.
///
/// The BLAS POD types run the existing machinery over the buffer reinterpreted
/// as initialized (`beta = 0` keeps every path write-only — the BLAS non-read
/// convention). Other dtypes use the generic write-only naive fallback.
#[allow(clippy::too_many_arguments)]
pub fn matmul_uninit_row_major_blas<TA, TB, TC, DA, DB, DC>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<DC>,
    a: &[TA],
    la: &Layout<DA>,
    b: &[TB],
    lb: &Layout<DB>,
    alpha: TC,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TA: Clone + Send + Sync + 'static,
    TB: Clone + Send + Sync + 'static,
    TC: Clone + Send + Sync + 'static,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: Mul<TB, Output = TC>,
    TB: Mul<TA, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + PartialEq,
{
    // quick return for empty matrix
    // in this case, we do not check the shape of a, b, c
    if lc.size() == 0 {
        return Ok(());
    }

    // vector inner dot: the shared naive kernel scales `c` by `beta`, so use
    // the write-only kernel instead
    if let (1, 1, 0) = (la.ndim(), lb.ndim(), lc.ndim()) {
        let la = &la.clone().into_dim::<Ix1>().unwrap();
        let lb = &lb.clone().into_dim::<Ix1>().unwrap();
        let lc = &lc.clone().into_dim::<Ix0>().unwrap();
        return inner_dot_naive_uninit_cpu_rayon(c, lc, a, la, b, lb, alpha, pool);
    }

    // type check and dispatch
    macro_rules! impl_uninit_dispatch {
        ($ty: ty) => {
            if (same_type::<TA, $ty>() && same_type::<TB, $ty>() && same_type::<TC, $ty>()) {
                // SAFETY: `TypeId` equality above proves TA = TB = TC = $ty; the
                // reinterpreted slices have exactly the original lengths, and with
                // `beta = 0` the dispatched routines only write `c` (initializing
                // every slot) — only pointer passing happens before the call.
                let a_slice = unsafe { from_raw_parts(a.as_ptr() as *const $ty, a.len()) };
                let b_slice = unsafe { from_raw_parts(b.as_ptr() as *const $ty, b.len()) };
                let c_slice = unsafe { from_raw_parts_mut(c.as_mut_ptr() as *mut $ty, c.len()) };
                let alpha = unsafe { *(&alpha as *const TC as *const $ty) };
                let beta = <$ty as Zero>::zero();
                return matmul_row_major_blas::<$ty, $ty, $ty, DA, DB, DC>(
                    c_slice, lc, a_slice, la, b_slice, lb, alpha, beta, pool,
                );
            }
        };
    }

    impl_uninit_dispatch!(f32);
    impl_uninit_dispatch!(f64);
    impl_uninit_dispatch!(Complex<f32>);
    impl_uninit_dispatch!(Complex<f64>);

    // not able to be accelarated by blas
    // fallback to naive write-only implementation
    return matmul_naive_uninit_cpu_rayon(c, lc, a, la, b, lb, alpha, pool);
}

#[allow(clippy::too_many_arguments)]
impl<TA, TB, TC, DA, DB, DC> DeviceMatMulAPI<TA, TB, TC, DA, DB, DC> for DeviceBLAS
where
    TA: Clone + Send + Sync + 'static,
    TB: Clone + Send + Sync + 'static,
    TC: Clone + Send + Sync + 'static,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: Mul<TB, Output = TC>,
    TB: Mul<TA, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + PartialEq,
    Self: DeviceRawAPI<MaybeUninit<TC>, Raw = Vec<MaybeUninit<TC>>>,
{
    fn matmul(
        &self,
        c: &mut Vec<TC>,
        lc: &Layout<DC>,
        a: &Vec<TA>,
        la: &Layout<DA>,
        b: &Vec<TB>,
        lb: &Layout<DB>,
        alpha: TC,
        beta: TC,
    ) -> Result<()> {
        let default_order = self.default_order();
        let pool = self.get_current_pool();
        match default_order {
            RowMajor => matmul_row_major_blas(c, lc, a, la, b, lb, alpha, beta, pool),
            ColMajor => {
                let la = la.reverse_axes();
                let lb = lb.reverse_axes();
                let lc = lc.reverse_axes();
                matmul_row_major_blas(c, &lc, b, &lb, a, &la, alpha, beta, pool)
            },
        }
    }

    fn matmul_uninit(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<DC>,
        a: &Vec<TA>,
        la: &Layout<DA>,
        b: &Vec<TB>,
        lb: &Layout<DB>,
        alpha: TC,
    ) -> Result<()> {
        let default_order = self.default_order();
        let pool = self.get_current_pool();
        match default_order {
            RowMajor => matmul_uninit_row_major_blas(c, lc, a, la, b, lb, alpha, pool),
            ColMajor => {
                let la = la.reverse_axes();
                let lb = lb.reverse_axes();
                let lc = lc.reverse_axes();
                matmul_uninit_row_major_blas(c, &lc, b, &lb, a, &la, alpha, pool)
            },
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rstsr_core::prelude::rstsr_funcs::matmul_from;

    #[test]
    fn test_matmul() {
        let device = DeviceBLAS::default();
        let a = linspace((0.0, 14.0, 15, &device)).into_shape([3, 5]);
        let b = linspace((0.0, 14.0, 15, &device)).into_shape([5, 3]);
        println!("{:}", &a % &b);

        let a = linspace((0.0, 14.0, 15, &device));
        let b = linspace((0.0, 14.0, 15, &device));
        println!("{:}", &a % &b);

        let a = linspace((0.0, 2.0, 3, &device));
        let b = linspace((0.0, 29.0, 30, &device)).into_shape([2, 3, 5]);
        println!("{:}", &a % &b);

        let a = linspace((0.0, 29.0, 30, &device)).into_shape([2, 3, 5]);
        let b = linspace((0.0, 4.0, 5, &device));
        println!("{:}", &a % &b);

        let a = linspace((0.0, 14.0, 15, &device)).into_shape([5, 3]);
        let b = linspace((0.0, 29.0, 30, &device)).into_shape([2, 3, 5]);
        println!("{:}", &a % &b);

        let a = linspace((0.0, 29.0, 30, &device)).into_shape([2, 3, 5]);
        let b = linspace((0.0, 14.0, 15, &device)).into_shape([5, 3]);
        println!("{:}", &a % &b);
    }

    #[test]
    #[ignore]
    fn parallel_test_full() {
        let device = DeviceBLAS::default();
        let a = linspace((0.0, 1.0, 8192 * 8192, &device)).into_shape([8192, 8192]);
        let b = linspace((0.0, 1.0, 8192 * 8192, &device)).into_shape([8192, 8192]);
        for _ in 0..10 {
            let start = std::time::Instant::now();
            let _ = &a % &b;
            println!("time: {:?}", start.elapsed());
        }
    }

    #[test]
    #[ignore]
    fn parallel_test_full_512() {
        let device = DeviceBLAS::new(1);
        let a = linspace((0.0, 1.0, 512 * 512, &device)).into_shape([512, 512]);
        let b = linspace((0.0, 1.0, 512 * 512, &device)).into_shape([512, 512]);
        for _ in 0..1000 {
            let start = std::time::Instant::now();
            let c = &a % &b;
            println!("{:?}", c.device());
            println!("time: {:?}", start.elapsed());
        }
    }

    #[test]
    #[ignore]
    fn parallel_test_par_rule7() {
        let device = DeviceBLAS::default();
        let a = linspace((0.0, 1.0, 8192 * 8192, &device)).into_shape([256, 512, 512]);
        let b = linspace((0.0, 1.0, 8192 * 8192, &device)).into_shape([256, 512, 512]);
        for i in 0..10 {
            let start = std::time::Instant::now();
            let c = &a % &b;
            println!("{:?}", c.layout());
            println!("time: {:?}", start.elapsed());
            if i == 0 {
                println!("{c:?}");
            }
        }
    }

    #[test]
    #[ignore]
    fn parallel_test_par_rule6() {
        let device = DeviceBLAS::default();
        let a = linspace((0.0, 1.0, 8192 * 8192, &device)).into_shape([256, 512, 512]);
        let b = linspace((0.0, 1.0, 512 * 512, &device)).into_shape([512, 512]);
        for i in 0..10 {
            let start = std::time::Instant::now();
            let c = &a % &b;
            println!("{:?}", c.layout());
            println!("time: {:?}", start.elapsed());
            if i == 0 {
                println!("{c:?}");
            }
        }
    }

    #[test]
    #[ignore]
    fn parallel_test_par_rule6_fprefer() {
        let device = DeviceBLAS::default();
        let a = linspace((0.0, 1.0, 8192 * 8192, &device)).into_shape([512, 512, 256]).into_reverse_axes();
        let b = linspace((0.0, 1.0, 512 * 512, &device)).into_shape([512, 512]);
        for i in 0..10 {
            let start = std::time::Instant::now();
            let c = &a % &b;
            println!("{:?}", c.layout());
            println!("time: {:?}", start.elapsed());
            if i == 0 {
                println!("{c:?}");
            }
        }
    }

    #[test]
    fn syrk_correctness() {
        let device = DeviceBLAS::default();
        let a = linspace((0.0, 1.0, 512 * 512, &device)).into_shape([512, 512]);
        let b = linspace((0.0, 1.0, 512 * 512, &device)).into_shape([512, 512]);
        let c = &a % &a.t();
        let d = &a % &b.t();
        assert!(allclose_f64(&c, &d));

        let device = DeviceBLAS::default();
        let a = linspace((0.0, 1.0, 1024 * 1024, &device)).into_shape([4, 512, 512]);
        let b = linspace((0.0, 1.0, 1024 * 1024, &device)).into_shape([4, 512, 512]);
        let c = &a % &a.swapaxes(-1, -2);
        let d = &a % &b.swapaxes(-1, -2);
        assert!(allclose_f64(&c, &d));
    }

    #[test]
    fn gemv_rule2a_matrix_vector() {
        let device = DeviceBLAS::default();
        let a = linspace((0.0, 11.0, 12, &device)).into_shape([3, 4]);
        let x = asarray((&[1.0f64, 2.0, 3.0, 4.0][..], [4].c(), &device)).into_dim::<Ix1>();

        let y = &a % &x;
        assert_eq!(y.shape(), &[3]);
        let want = asarray((&[20.0f64, 60.0, 100.0][..], [3].c(), &device)).into_dim::<Ix1>();
        assert!(allclose_f64(&y, &want));
    }

    /// `(1,2,1)`: `x @ a` goes through the transposed `gemv` arm.
    #[test]
    fn gemv_rule2b_vector_matrix() {
        let device = DeviceBLAS::default();
        let a = linspace((0.0, 11.0, 12, &device)).into_shape([3, 4]);
        let x = asarray((&[1.0f64, 2.0, 3.0][..], [3].c(), &device)).into_dim::<Ix1>();

        let y = &x % &a;
        assert_eq!(y.shape(), &[4]);
        let want = asarray((&[32.0f64, 38.0, 44.0, 50.0][..], [4].c(), &device)).into_dim::<Ix1>();
        assert!(allclose_f64(&y, &want));
    }

    /// f-prefer matrix, strided `x` and strided `y`.
    #[test]
    fn gemv_layout_variants() {
        let device = DeviceBLAS::default();
        let want = asarray((&[20.0f64, 60.0, 100.0][..], [3].c(), &device)).into_dim::<Ix1>();
        let x = asarray((&[1.0f64, 2.0, 3.0, 4.0][..], [4].c(), &device)).into_dim::<Ix1>();
        let a = linspace((0.0, 11.0, 12, &device)).into_shape([3, 4]);

        // the same 3x4 matrix stored column-major
        let a_f =
            asarray((&[0.0f64, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0][..], [3, 4].f(), &device))
                .into_dim::<Ix2>();
        assert!(a_f.f_prefer());
        assert!(allclose_f64(&(&a_f % &x), &want));

        // strided `x`: column 0 of a c-prefer [4, 2] buffer
        let base_x = asarray((&[1.0f64, 9.0, 2.0, 9.0, 3.0, 9.0, 4.0, 9.0][..], [4, 2].c(), &device)).into_dim::<Ix2>();
        let x_strided = base_x.i((.., 0));
        assert_eq!(x_strided.stride()[0], 2);
        assert!(allclose_f64(&(&a % &x_strided.into_dim::<Ix1>()), &want));

        // strided `y`: column 0 of a c-prefer [3, 2] buffer
        let mut base_y: Tensor<f64, DeviceBLAS, Ix2> = zeros_f(([3, 2].c(), &device)).unwrap().into_dim();
        matmul_from(base_y.i_mut((.., 0)).into_dim::<Ix1>(), a.view(), x.view(), 1.0, 0.0);
        assert!(allclose_f64(&base_y.i((.., 0)), &want));
        let zero = zeros_f(([3].c(), &device)).unwrap().into_dim::<Ix1>();
        assert!(allclose_f64(&base_y.i((.., 1)), &zero));
    }

    /// `beta != 0` with a strided output: the staging buffer must carry `y` in.
    #[test]
    fn gemv_beta_with_strided_output() {
        let device = DeviceBLAS::default();
        let a = linspace((0.0, 11.0, 12, &device)).into_shape([3, 4]);
        let x = asarray((&[1.0f64, 2.0, 3.0, 4.0][..], [4].c(), &device)).into_dim::<Ix1>();

        // column 0 of a c-prefer [3, 2] linspace holds [1, 3, 5] at stride 2
        let mut base = linspace((1.0, 6.0, 6, &device)).into_shape([3, 2]);
        matmul_from(base.i_mut((.., 0)).into_dim::<Ix1>(), a.view(), x.view(), 1.0, 2.0);

        let want = asarray((&[22.0f64, 66.0, 110.0][..], [3].c(), &device)).into_dim::<Ix1>();
        assert!(allclose_f64(&base.i((.., 0)), &want));
    }

    /// Empty contraction: zeros for `%`, `beta * c` for `matmul_from`.
    #[test]
    fn gemv_empty_contraction() {
        let device = DeviceBLAS::default();
        let a: Tensor<f64, DeviceBLAS, Ix2> = zeros_f(([3, 0].c(), &device)).unwrap().into_dim();
        let x: Tensor<f64, DeviceBLAS, Ix1> = zeros_f(([0].c(), &device)).unwrap().into_dim();

        let y = &a % &x;
        assert_eq!(y.shape(), &[3]);
        let zero = zeros_f(([3].c(), &device)).unwrap().into_dim::<Ix1>();
        assert!(allclose_f64(&y, &zero));

        let mut c = linspace((1.0, 3.0, 3, &device)).into_shape([3]);
        matmul_from(c.view_mut(), a.view(), x.view(), 1.0, 2.0);
        let want = asarray((&[2.0f64, 4.0, 6.0][..], [3].c(), &device)).into_dim::<Ix1>();
        assert!(allclose_f64(&c, &want));
    }

    #[test]
    fn gemv_complex_matrix_vector() {
        let device = DeviceBLAS::default();
        let a_vals = [Complex::new(1.0, 1.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0), Complex::new(4.0, -1.0)];
        let x_vals = [Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)];
        let a = asarray((&a_vals[..], [2, 2].c(), &device)).into_dim::<Ix2>();
        let x = asarray((&x_vals[..], [2].c(), &device)).into_dim::<Ix1>();

        // (1+i)*1 + 2*i = 1+3i ; 3*1 + (4-i)*i = 4+4i
        let y = (&a % &x).to_vec();
        let want = [Complex::new(1.0, 3.0), Complex::new(4.0, 4.0)];
        assert!((0..2).all(|i| (y[i] - want[i]).norm() < 1e-12), "got {:?}", y);
    }

    /// `(2,2,2)` products with a size-1 output axis collapse to `gemv`; `k == 1`
    /// stays an outer product.
    #[test]
    fn gemv_gemm_shapes_collapse() {
        let device = DeviceBLAS::default();
        let a = linspace((0.0, 11.0, 12, &device)).into_shape([3, 4]);

        let x_col = linspace((1.0, 4.0, 4, &device)).into_shape([4, 1]);
        let y_col = &a % &x_col;
        assert_eq!(y_col.shape(), &[3, 1]);
        let want_col = asarray((&[20.0f64, 60.0, 100.0][..], [3].c(), &device)).into_dim::<Ix1>();
        assert!(allclose_f64(&y_col.i((.., 0)), &want_col));

        let x_row = linspace((1.0, 3.0, 3, &device)).into_shape([1, 3]);
        let y_row = &x_row % &a;
        assert_eq!(y_row.shape(), &[1, 4]);
        let want_row = asarray((&[32.0f64, 38.0, 44.0, 50.0][..], [4].c(), &device)).into_dim::<Ix1>();
        assert!(allclose_f64(&y_row.i((0, ..)), &want_row));

        let b_col = linspace((1.0, 3.0, 3, &device)).into_shape([3, 1]);
        let dot = &x_row % &b_col;
        assert_eq!(dot.shape(), &[1, 1]);
        assert!(allclose_f64(&dot, &asarray((&[14.0f64][..], [1, 1].c(), &device)).into_dim::<Ix2>()));

        let u = linspace((1.0, 3.0, 3, &device)).into_shape([3, 1]);
        let v = linspace((1.0, 4.0, 4, &device)).into_shape([1, 4]);
        let outer = &u % &v;
        assert_eq!(outer.shape(), &[3, 4]);
        let want =
            asarray((&[1.0f64, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 3.0, 6.0, 9.0, 12.0][..], [3, 4].c(), &device))
                .into_dim::<Ix2>();
        assert!(allclose_f64(&outer, &want));
    }

    /// A batch far larger than `4 * nthreads` still runs in the main thread
    /// for vector-shaped products.
    #[test]
    fn gemv_batched_vector_shapes_large_batch() {
        let device = DeviceBLAS::default();
        let a = linspace((0.0, 3071.0, 3072, &device)).into_shape([256, 3, 4]);
        let b = linspace((1.0, 1024.0, 1024, &device)).into_shape([256, 4, 1]);

        let c = &a % &b;
        assert_eq!(c.shape(), &[256, 3, 1]);
        for i in [0, 127, 255] {
            let c_ref = &a.i(i) % &b.i(i);
            assert!(allclose_f64(&c.i(i), &c_ref));
        }
    }

    /// Neither f- nor c-prefer matrix (broadcast rows) is copied before `gemv`.
    #[test]
    fn gemv_matrix_copy_branch() {
        let device = DeviceBLAS::default();
        let row = asarray((&[1.0f64, 2.0, 3.0, 4.0][..], [1, 4].c(), &device)).into_dim::<Ix2>();
        let a = row.to_broadcast(vec![3, 4]);
        let x = asarray((&[1.0f64, 2.0, 3.0, 4.0][..], [4].c(), &device)).into_dim::<Ix1>();

        let y = &a % &x;
        assert_eq!(y.shape(), &[3]);
        let want = asarray((&[30.0f64, 30.0, 30.0][..], [3].c(), &device)).into_dim::<Ix1>();
        assert!(allclose_f64(&y, &want));
    }

    /// Batched vector-shaped products go through the serial-over-batches path.
    #[test]
    fn gemv_batched_vector_shapes() {
        let device = DeviceBLAS::default();
        let a = linspace((0.0, 23.0, 24, &device)).into_shape([2, 3, 4]);
        let b = linspace((1.0, 8.0, 8, &device)).into_shape([2, 4, 1]);

        let c = &a % &b;
        assert_eq!(c.shape(), &[2, 3, 1]);
        for i in 0..2 {
            let c_ref = &a.i(i) % &b.i(i);
            assert!(allclose_f64(&c.i(i), &c_ref));
        }

        // a 1-D operand broadcast over the batch still takes the gemv path
        let b1 = linspace((1.0, 4.0, 4, &device));
        let c1 = &a % &b1;
        assert_eq!(c1.shape(), &[2, 3]);
        for i in 0..2 {
            let c1_ref = &a.i(i) % &b1;
            assert!(allclose_f64(&c1.i(i), &c1_ref));
        }
    }

    #[test]
    #[ignore]
    fn syrk_efficiency() {
        use std::hint::black_box;
        let device = DeviceBLAS::default();
        let a = linspace((0.0, 1.0, 8192 * 8192, &device)).into_shape([256, 512, 512]);
        let b = linspace((0.0, 1.0, 8192 * 8192, &device)).into_shape([256, 512, 512]);
        for _ in 0..10 {
            let start = std::time::Instant::now();
            black_box(&a % &a.swapaxes(-1, -2));
            println!("syrk time: {:?}", start.elapsed());
            let start = std::time::Instant::now();
            black_box(&a % &b.swapaxes(-1, -2));
            println!("gemm time: {:?}", start.elapsed());
        }

        println!("---------------------");
        let a = linspace((0.0, 1.0, 8192 * 8192, &device)).into_shape([8192, 8192]);
        let b = linspace((0.0, 1.0, 8192 * 8192, &device)).into_shape([8192, 8192]);
        for _ in 0..10 {
            let start = std::time::Instant::now();
            black_box(&a % &a.swapaxes(-1, -2));
            println!("syrk time: {:?}", start.elapsed());
            let start = std::time::Instant::now();
            black_box(&a % &b.swapaxes(-1, -2));
            println!("gemm time: {:?}", start.elapsed());
        }
    }
}
