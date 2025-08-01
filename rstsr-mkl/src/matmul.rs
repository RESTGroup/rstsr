use crate::matmul_impl::*;
use crate::prelude_dev::*;
use crate::threading::with_num_threads;
use core::any::TypeId;
use core::ops::{Add, Mul};
use core::slice::{from_raw_parts, from_raw_parts_mut};
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
        (2, 2, 2) => {
            // rule 2: matrix multiplication
            let la = &la.clone().into_dim::<Ix2>().unwrap();
            let lb = &lb.clone().into_dim::<Ix2>().unwrap();
            let lc = &lc.clone().into_dim::<Ix2>().unwrap();
            return with_num_threads(nthreads, || {
                gemm_blas_ix2_no_conj_dispatch(c, lc, a, la, b, lb, alpha, beta, pool)
            });
        },
        _ => (),
    };

    // handle broadcasted cases
    // temporary variables
    let la_matmul;
    let lb_matmul;
    let lc_matmul;
    let la_rest;
    let lb_rest;
    let lc_rest;

    match (la.ndim(), lb.ndim(), lc.ndim()) {
        // we have already handled these cases
        (1, 1, 0) | (2, 2, 2) => unreachable!(),
        (1, 2.., _) => {
            // rule 3: | `        K` | `..., K, N` | `   ..., N` |
            rstsr_assert_eq!(lb.ndim(), lc.ndim() + 1, InvalidLayout)?;
            let (la_r, la_m) = la.dim_split_at(-1)?;
            let (lb_r, lb_m) = lb.dim_split_at(-2)?;
            let (lc_r, lc_m) = lc.dim_split_at(-1)?;
            la_rest = broadcast_layout_to_first(&lc_r, &la_r, RowMajor)?.1;
            lb_rest = lb_r;
            lc_rest = lc_r;
            la_matmul = la_m.dim_insert(0)?.into_dim::<Ix2>()?;
            lb_matmul = lb_m.into_dim::<Ix2>()?;
            lc_matmul = lc_m.dim_insert(0)?.into_dim::<Ix2>()?;
        },
        (2.., 1, _) => {
            // rule 4: | `..., M, K` | `        K` | `   ..., M` |
            rstsr_assert_eq!(la.ndim(), lc.ndim() + 1, InvalidLayout)?;
            let (la_r, la_m) = la.dim_split_at(-2)?;
            let (lb_r, lb_m) = lb.dim_split_at(-1)?;
            let (lc_r, lc_m) = lc.dim_split_at(-1)?;
            la_rest = la_r;
            lb_rest = broadcast_layout_to_first(&lc_r, &lb_r, RowMajor)?.1;
            lc_rest = lc_r;
            la_matmul = la_m.into_dim::<Ix2>()?;
            lb_matmul = lb_m.dim_insert(1)?.into_dim::<Ix2>()?;
            lc_matmul = lc_m.dim_insert(1)?.into_dim::<Ix2>()?;
        },
        (2, 3.., _) => {
            // rule 5: | `     M, K` | `..., K, N` | `..., M, N` |
            rstsr_assert_eq!(lb.ndim(), lc.ndim(), InvalidLayout)?;
            let (la_r, la_m) = la.dim_split_at(-2)?;
            let (lb_r, lb_m) = lb.dim_split_at(-2)?;
            let (lc_r, lc_m) = lc.dim_split_at(-2)?;
            la_rest = broadcast_layout_to_first(&lc_r, &la_r, RowMajor)?.1;
            lb_rest = lb_r;
            lc_rest = lc_r;
            la_matmul = la_m.into_dim::<Ix2>()?;
            lb_matmul = lb_m.into_dim::<Ix2>()?;
            lc_matmul = lc_m.into_dim::<Ix2>()?;
        },
        (3.., 2, _) => {
            // rule 6: | `..., M, K` | `     K, N` | `..., M, N` |
            rstsr_assert_eq!(la.ndim(), lc.ndim(), InvalidLayout)?;
            let (la_r, la_m) = la.dim_split_at(-2)?;
            let (lb_r, lb_m) = lb.dim_split_at(-2)?;
            let (lc_r, lc_m) = lc.dim_split_at(-2)?;
            la_rest = la_r;
            lb_rest = broadcast_layout_to_first(&lc_r, &lb_r, RowMajor)?.1;
            lc_rest = lc_r;
            la_matmul = la_m.into_dim::<Ix2>()?;
            lb_matmul = lb_m.into_dim::<Ix2>()?;
            lc_matmul = lc_m.into_dim::<Ix2>()?;
        },
        (3.., 3.., _) => {
            // rule 7: | `..., M, K` | `..., K, N` | `..., M, N` |
            rstsr_assert_eq!(la.ndim(), lc.ndim(), InvalidLayout)?;
            rstsr_assert_eq!(lb.ndim(), lc.ndim(), InvalidLayout)?;
            let (la_r, la_m) = la.dim_split_at(-2)?;
            let (lb_r, lb_m) = lb.dim_split_at(-2)?;
            let (lc_r, lc_m) = lc.dim_split_at(-2)?;
            la_rest = la_r;
            lb_rest = lb_r;
            lc_rest = lc_r;
            la_matmul = la_m.into_dim::<Ix2>()?;
            lb_matmul = lb_m.into_dim::<Ix2>()?;
            lc_matmul = lc_m.into_dim::<Ix2>()?;
        },
        _ => {
            rstsr_raise!(InvalidLayout, "This is not valid layout for matmul broadcasting.")?;
            unreachable!()
        },
    }
    // now, lx_rest should have the same shape, while lx_matmul
    // should be matmulable
    // only parallel matmul when lx_rest is small (larger than
    // 2*nthreads), otherwise parallel matmul anyway
    rstsr_assert_eq!(la_rest.shape(), lb_rest.shape(), InvalidLayout)?;
    rstsr_assert_eq!(lb_rest.shape(), lc_rest.shape(), InvalidLayout)?;
    let n_task = la_rest.size();
    let ita_rest = IterLayoutColMajor::new(&la_rest)?;
    let itb_rest = IterLayoutColMajor::new(&lb_rest)?;
    let itc_rest = IterLayoutColMajor::new(&lc_rest)?;
    if n_task >= 4 * nthreads {
        // parallel outer, sequential matmul
        with_num_threads(1, || {
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
                        // move mutable reference into parallel closure
                        let c = unsafe {
                            let c_ptr = c.as_ptr() as *mut TC;
                            let c_len = c.len();
                            from_raw_parts_mut(c_ptr, c_len)
                        };
                        // clone alpha and beta
                        let alpha = alpha.clone();
                        let beta = beta.clone();
                        gemm_blas_ix2_no_conj_dispatch(c, &lc_m, a, &la_m, b, &lb_m, alpha, beta, None)
                    },
                )
            };
            match pool {
                Some(pool) => pool.install(task),
                None => task(),
            }
        })
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
                gemm_blas_ix2_no_conj_dispatch(c, &lc_m, a, &la_m, b, &lb_m, alpha, beta, pool)
            })
        })
    }
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
}

#[cfg(test)]
mod test {
    use super::*;

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
