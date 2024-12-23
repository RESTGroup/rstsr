//! Implementation of faer matmul
//!
//! This implementation does not specialize gemv. We always use gemm for matmul.

use num::{Complex, Zero};

use super::matmul_impl::*;
use crate::feature_rayon::matmul_naive::{gemm_naive_rayon, inner_dot_naive_rayon};
use crate::prelude_dev::*;
use core::any::TypeId;
use core::mem::ManuallyDrop;
use core::ops::{Add, Mul};
use core::slice::{from_raw_parts, from_raw_parts_mut};
use rayon::prelude::*;

// code from ndarray
fn same_type<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

#[allow(clippy::too_many_arguments)]
pub fn gemm_faer_dispatch<TA, TB, TC>(
    c: &mut Storage<TC, DeviceFaer>,
    lc: &Layout<Ix2>,
    a: &Storage<TA, DeviceFaer>,
    la: &Layout<Ix2>,
    b: &Storage<TB, DeviceFaer>,
    lb: &Layout<Ix2>,
    alpha: TC,
    beta: TC,
    nthreads: usize,
) -> Result<()>
where
    TA: Clone + Send + Sync + 'static,
    TB: Clone + Send + Sync + 'static,
    TC: Clone + Send + Sync + 'static,
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero,
{
    // check if syrk could be applicable
    let able_syrk = if same_type::<TA, TC>() && same_type::<TB, TC>() {
        unsafe {
            let a_ptr = a.rawvec().as_ptr().add(la.offset()) as *const TC;
            let b_ptr = b.rawvec().as_ptr().add(lb.offset()) as *const TC;
            let equal_ptr = a_ptr == b_ptr;
            let equal_shape = la.shape() == lb.reverse_axes().shape();
            let equal_stride = la.stride() == lb.reverse_axes().stride();
            equal_ptr && equal_shape && equal_stride
        }
    } else {
        false
    };

    // type check and dispatch
    macro_rules! impl_gemm_dispatch {
        ($ty: ty, $fn_gemm_name: ident, $fn_syrk_name: ident) => {
            if (same_type::<TA, $ty>() && same_type::<TB, $ty>() && same_type::<TC, $ty>()) {
                let a_slice = unsafe { from_raw_parts(a.rawvec().as_ptr() as *const $ty, a.len()) };
                let b_slice = unsafe { from_raw_parts(b.rawvec().as_ptr() as *const $ty, b.len()) };
                let c_slice =
                    unsafe { from_raw_parts_mut(c.rawvec_mut().as_mut_ptr() as *mut $ty, c.len()) };
                let alpha = unsafe { *(&alpha as *const TC as *const $ty) };
                let beta = unsafe { *(&beta as *const TC as *const $ty) };
                if able_syrk {
                    $fn_syrk_name(c_slice, lc, a_slice, la, alpha, beta, nthreads)?;
                } else {
                    $fn_gemm_name(c_slice, lc, a_slice, la, b_slice, lb, alpha, beta, nthreads)?;
                }
                return Ok(());
            }
        };
    }

    impl_gemm_dispatch!(f32, gemm_faer_f32, gemm_with_syrk_faer_f32);
    impl_gemm_dispatch!(f64, gemm_faer_f64, gemm_with_syrk_faer_f64);
    impl_gemm_dispatch!(Complex<f32>, gemm_faer_c32, gemm_with_syrk_faer_c32);
    impl_gemm_dispatch!(Complex<f64>, gemm_faer_c64, gemm_with_syrk_faer_c64);

    // not able to be accelarated by faer
    // fallback to naive implementation
    let c_slice = c.rawvec_mut();
    let a_slice = a.rawvec();
    let b_slice = b.rawvec();
    return gemm_naive_rayon(c_slice, lc, a_slice, la, b_slice, lb, alpha, beta, nthreads);
}

#[allow(clippy::too_many_arguments)]
impl<TA, TB, TC, DA, DB, DC> DeviceMatMulAPI<TA, TB, TC, DA, DB, DC> for DeviceFaer
where
    TA: Clone + Send + Sync + 'static,
    TB: Clone + Send + Sync + 'static,
    TC: Clone + Send + Sync + 'static,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero,
    Self: DeviceAPI<TA, RawVec = Vec<TA>>,
    Self: DeviceAPI<TB, RawVec = Vec<TB>>,
    Self: DeviceAPI<TC, RawVec = Vec<TC>>,
{
    fn matmul(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<DC>,
        a: &Storage<TA, Self>,
        la: &Layout<DA>,
        b: &Storage<TB, Self>,
        lb: &Layout<DB>,
        alpha: TC,
        beta: TC,
    ) -> Result<()> {
        let nthreads = self.get_num_threads();
        rstsr_assert!(c.device().same_device(a.device()), DeviceMismatch)?;
        rstsr_assert!(c.device().same_device(b.device()), DeviceMismatch)?;
        let c_slice = c.rawvec_mut().as_mut_slice();
        let a_slice = a.rawvec().as_slice();
        let b_slice = b.rawvec().as_slice();
        match (la.ndim(), lb.ndim(), lc.ndim()) {
            (1, 1, 0) => {
                // rule 1: vector inner dot
                let la = &la.clone().into_dim::<Ix1>().unwrap();
                let lb = &lb.clone().into_dim::<Ix1>().unwrap();
                let lc = &lc.clone().into_dim::<Ix0>().unwrap();
                let c_num = &mut c_slice[lc.offset()];
                inner_dot_naive_rayon(c_num, a_slice, la, b_slice, lb, alpha, beta, nthreads)?;
            },
            (2, 2, 2) => {
                // rule 2: matrix multiplication
                let la = &la.clone().into_dim::<Ix2>().unwrap();
                let lb = &lb.clone().into_dim::<Ix2>().unwrap();
                let lc = &lc.clone().into_dim::<Ix2>().unwrap();
                gemm_faer_dispatch(c, lc, a, la, b, lb, alpha, beta, nthreads)?;
            },
            _ => {
                // temporary layout
                let la_matmul;
                let lb_matmul;
                let lc_matmul;
                let la_rest;
                let lb_rest;
                let lc_rest;
                match (la.ndim(), lb.ndim(), lc.ndim()) {
                    (1, 1, 0) | (2, 2, 2) => unreachable!(),
                    (1, 2.., _) => {
                        // rule 3: | `        K` | `..., K, N` | `   ..., N` |
                        rstsr_assert_eq!(lb.ndim(), lc.ndim() + 1, InvalidLayout)?;
                        let (la_r, la_m) = la.dim_split_at(-1)?;
                        let (lb_r, lb_m) = lb.dim_split_at(-2)?;
                        let (lc_r, lc_m) = lc.dim_split_at(-1)?;
                        la_rest = broadcast_layout_to_first(&lc_r, &la_r)?.1;
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
                        lb_rest = broadcast_layout_to_first(&lc_r, &lb_r)?.1;
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
                        la_rest = broadcast_layout_to_first(&lc_r, &la_r)?.1;
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
                        lb_rest = broadcast_layout_to_first(&lc_r, &lb_r)?.1;
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
                    _ => todo!(),
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
                if n_task > 2 * nthreads {
                    // parallel outer, sequential matmul
                    let pool = self.get_pool(nthreads)?;
                    pool.install(|| {
                        ita_rest.into_par_iter().zip(itb_rest).zip(itc_rest).for_each(
                            |((ia_rest, ib_rest), ic_rest)| {
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
                                let c_rawvec = unsafe {
                                    let c_ptr = c_slice.as_ptr() as *mut TC;
                                    let c_len = c_slice.len();
                                    Vec::from_raw_parts(c_ptr, c_len, c_len)
                                };
                                let c = Storage::new(c_rawvec, a.device().clone());
                                let mut c = ManuallyDrop::new(c);
                                // clone alpha and beta
                                let alpha = alpha.clone();
                                let beta = beta.clone();
                                gemm_faer_dispatch(
                                    &mut c, &lc_m, a, &la_m, b, &lb_m, alpha, beta, 1,
                                )
                                .unwrap()
                            },
                        );
                    });
                } else {
                    // sequential outer, parallel matmul
                    for (ia_rest, ib_rest, ic_rest) in izip!(ita_rest, itb_rest, itc_rest) {
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
                        gemm_faer_dispatch(c, &lc_m, a, &la_m, b, &lb_m, alpha, beta, nthreads)?;
                    }
                }
            },
        }
        return Ok(());
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_matmul() {
        let device = DeviceFaer::default();
        let a = Tensor::linspace((0.0, 14.0, 15, &device)).into_shape_assume_contig([3, 5]);
        let b = Tensor::linspace((0.0, 14.0, 15, &device)).into_shape_assume_contig([5, 3]);

        let d = &a % &b;
        println!("{d}");

        let a = Tensor::linspace((0.0, 14.0, 15, &device));
        let b = Tensor::linspace((0.0, 14.0, 15, &device));
        println!("{:}", &a % &b);

        let a = Tensor::linspace((0.0, 2.0, 3, &device));
        let b = Tensor::linspace((0.0, 29.0, 30, &device)).into_shape_assume_contig([2, 3, 5]);
        println!("{:}", &a % &b);

        let a = Tensor::linspace((0.0, 29.0, 30, &device)).into_shape_assume_contig([2, 3, 5]);
        let b = Tensor::linspace((0.0, 4.0, 5, &device));
        println!("{:}", &a % &b);

        let a = Tensor::linspace((0.0, 14.0, 15, &device)).into_shape_assume_contig([5, 3]);
        let b = Tensor::linspace((0.0, 29.0, 30, &device)).into_shape_assume_contig([2, 3, 5]);
        println!("{:}", &a % &b);

        let a = Tensor::linspace((0.0, 29.0, 30, &device)).into_shape_assume_contig([2, 3, 5]);
        let b = Tensor::linspace((0.0, 14.0, 15, &device)).into_shape_assume_contig([5, 3]);
        println!("{:}", &a % &b);
    }
}
