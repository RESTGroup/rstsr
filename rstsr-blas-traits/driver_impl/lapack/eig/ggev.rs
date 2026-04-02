#![allow(clippy::needless_range_loop)]

use crate::lapack_ffi;
use crate::DeviceBLAS;
use num::complex::ComplexFloat;
use num::{Complex, ToPrimitive, Zero};
use rstsr_blas_traits::lapack_eig::GGEVDriverAPI;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude_dev::*;
use rstsr_native_impl::prelude_dev::*;
use std::slice::from_raw_parts_mut;

#[duplicate_item(
    T     func_   ;
   [f32] [sggev_];
   [f64] [dggev_];
)]
impl GGEVDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_ggev(
        order: FlagOrder,
        jobvl: char,
        jobvr: char,
        n: usize,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        alpha: *mut Complex<T>,
        beta: *mut Complex<T>,
        vl: *mut T,
        ldvl: usize,
        vr: *mut T,
        ldvr: usize,
    ) -> blas_int {
        use lapack_ffi::lapack::func_;

        let compute_vl = jobvl == 'V' || jobvl == 'v';
        let compute_vr = jobvr == 'V' || jobvr == 'v';

        // Allocate temporary arrays for LAPACK's real format
        let mut alphar: Vec<T> = match uninitialized_vec(n) {
            Ok(alphar) => alphar,
            Err(_) => return -1010,
        };
        let mut alphai: Vec<T> = match uninitialized_vec(n) {
            Ok(alphai) => alphai,
            Err(_) => return -1010,
        };
        let mut beta_real: Vec<T> = match uninitialized_vec(n) {
            Ok(beta) => beta,
            Err(_) => return -1010,
        };

        if order == ColMajor {
            // Query optimal work array size
            let mut info = 0;
            let lwork = -1;
            let mut work_query: T = T::zero();
            func_(
                &(jobvl as _),
                &(jobvr as _),
                &(n as blas_int),
                a,
                &(lda as blas_int),
                b,
                &(ldb as blas_int),
                alphar.as_mut_ptr(),
                alphai.as_mut_ptr(),
                beta_real.as_mut_ptr(),
                vl,
                &(ldvl as blas_int),
                vr,
                &(ldvr as blas_int),
                &mut work_query,
                &lwork,
                &mut info,
            );
            if info != 0 {
                return info;
            }
            let lwork = work_query.to_usize().unwrap();

            // Allocate work array
            let mut work: Vec<T> = match uninitialized_vec(lwork) {
                Ok(work) => work,
                Err(_) => return -1010,
            };

            // Call LAPACK directly
            func_(
                &(jobvl as _),
                &(jobvr as _),
                &(n as blas_int),
                a,
                &(lda as blas_int),
                b,
                &(ldb as blas_int),
                alphar.as_mut_ptr(),
                alphai.as_mut_ptr(),
                beta_real.as_mut_ptr(),
                vl,
                &(ldvl as blas_int),
                vr,
                &(ldvr as blas_int),
                work.as_mut_ptr(),
                &(lwork as blas_int),
                &mut info,
            );

            // Convert to complex: alpha = Complex(alphar, alphai), beta = Complex(beta, 0)
            for i in 0..n {
                *alpha.add(i) = Complex::new(alphar[i], alphai[i]);
                *beta.add(i) = Complex::new(beta_real[i], T::zero());
            }

            info
        } else {
            // Row-major handling: LAPACK is column-major, so we need to transpose
            let lda_t = n.max(1);
            let ldb_t = n.max(1);
            let ldvl_t = n.max(1);
            let ldvr_t = n.max(1);

            // Allocate temporary column-major arrays
            // Note: We always allocate VL and VR arrays even if not computing them,
            // to avoid passing dangling pointers to LAPACK.
            let mut a_t: Vec<T> = match uninitialized_vec(n * lda_t) {
                Ok(a_t) => a_t,
                Err(_) => return -1010,
            };
            let mut b_t: Vec<T> = match uninitialized_vec(n * ldb_t) {
                Ok(b_t) => b_t,
                Err(_) => return -1010,
            };
            let mut vl_t: Vec<T> = match uninitialized_vec(n * ldvl_t) {
                Ok(vl_t) => vl_t,
                Err(_) => return -1011,
            };
            let mut vr_t: Vec<T> = match uninitialized_vec(n * ldvr_t) {
                Ok(vr_t) => vr_t,
                Err(_) => return -1011,
            };

            // Transpose inputs from row-major to column-major
            let a_slice = from_raw_parts_mut(a, n * lda);
            let b_slice = from_raw_parts_mut(b, n * ldb);
            let la = Layout::new_unchecked([n, n], [lda as isize, 1], 0);
            let lb = Layout::new_unchecked([n, n], [ldb as isize, 1], 0);
            let la_t = Layout::new_unchecked([n, n], [1, lda_t as isize], 0);
            let lb_t = Layout::new_unchecked([n, n], [1, ldb_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).unwrap();
            orderchange_out_r2c_ix2_cpu_serial(&mut b_t, &lb_t, b_slice, &lb).unwrap();

            // Query optimal work array size with column-major dimensions
            let mut info = 0;
            let lwork = -1;
            let mut work_query: T = T::zero();
            func_(
                &(jobvl as _),
                &(jobvr as _),
                &(n as blas_int),
                a_t.as_mut_ptr(),
                &(lda_t as blas_int),
                b_t.as_mut_ptr(),
                &(ldb_t as blas_int),
                alphar.as_mut_ptr(),
                alphai.as_mut_ptr(),
                beta_real.as_mut_ptr(),
                vl_t.as_mut_ptr(),
                &(ldvl_t as blas_int),
                vr_t.as_mut_ptr(),
                &(ldvr_t as blas_int),
                &mut work_query,
                &lwork,
                &mut info,
            );
            if info != 0 {
                return info;
            }
            let lwork = work_query.to_usize().unwrap();

            // Allocate work array
            let mut work: Vec<T> = match uninitialized_vec(lwork) {
                Ok(work) => work,
                Err(_) => return -1010,
            };

            // Call LAPACK with column-major arrays
            func_(
                &(jobvl as _),
                &(jobvr as _),
                &(n as blas_int),
                a_t.as_mut_ptr(),
                &(lda_t as blas_int),
                b_t.as_mut_ptr(),
                &(ldb_t as blas_int),
                alphar.as_mut_ptr(),
                alphai.as_mut_ptr(),
                beta_real.as_mut_ptr(),
                vl_t.as_mut_ptr(),
                &(ldvl_t as blas_int),
                vr_t.as_mut_ptr(),
                &(ldvr_t as blas_int),
                work.as_mut_ptr(),
                &(lwork as blas_int),
                &mut info,
            );
            if info != 0 {
                return info;
            }

            // Transpose outputs back to row-major
            orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).unwrap();
            orderchange_out_c2r_ix2_cpu_serial(b_slice, &lb, &b_t, &lb_t).unwrap();
            if compute_vl {
                let vl_slice = from_raw_parts_mut(vl, n * ldvl);
                let lvl = Layout::new_unchecked([n, n], [ldvl as isize, 1], 0);
                let lvl_t = Layout::new_unchecked([n, n], [1, ldvl_t as isize], 0);
                orderchange_out_c2r_ix2_cpu_serial(vl_slice, &lvl, &vl_t, &lvl_t).unwrap();
            }
            if compute_vr {
                let vr_slice = from_raw_parts_mut(vr, n * ldvr);
                let lvr = Layout::new_unchecked([n, n], [ldvr as isize, 1], 0);
                let lvr_t = Layout::new_unchecked([n, n], [1, ldvr_t as isize], 0);
                orderchange_out_c2r_ix2_cpu_serial(vr_slice, &lvr, &vr_t, &lvr_t).unwrap();
            }

            // Convert to complex: alpha = Complex(alphar, alphai), beta = Complex(beta, 0)
            for i in 0..n {
                *alpha.add(i) = Complex::new(alphar[i], alphai[i]);
                *beta.add(i) = Complex::new(beta_real[i], T::zero());
            }

            info
        }
    }
}

#[duplicate_item(
    T              func_   ;
   [Complex::<f32>] [cggev_];
   [Complex::<f64>] [zggev_];
)]
impl GGEVDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_ggev(
        order: FlagOrder,
        jobvl: char,
        jobvr: char,
        n: usize,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        alpha: *mut T, // For complex types, T = Complex<T::Real>
        beta: *mut T,  // For complex types, T = Complex<T::Real>
        vl: *mut T,
        ldvl: usize,
        vr: *mut T,
        ldvr: usize,
    ) -> blas_int {
        use lapack_ffi::lapack::func_;

        let compute_vl = jobvl == 'V' || jobvl == 'v';
        let compute_vr = jobvr == 'V' || jobvr == 'v';

        // Allocate rwork array (required for complex GGEV)
        let rwork_len = 2 * n;
        let mut rwork: Vec<<T as ComplexFloat>::Real> = match uninitialized_vec(rwork_len) {
            Ok(rwork) => rwork,
            Err(_) => return -1010,
        };

        // For complex types, alpha and beta are already complex.
        // T = Complex<T::Real>, so T* = Complex<T::Real>*
        // Cast to opaque pointer type for LAPACK FFI

        if order == ColMajor {
            // Query optimal work array size
            let mut info = 0;
            let lwork = -1;
            let mut work_query: T = T::zero();
            func_(
                &(jobvl as _),
                &(jobvr as _),
                &(n as blas_int),
                a as *mut _,
                &(lda as blas_int),
                b as *mut _,
                &(ldb as blas_int),
                alpha as *mut _,
                beta as *mut _,
                vl as *mut _,
                &(ldvl as blas_int),
                vr as *mut _,
                &(ldvr as blas_int),
                &mut work_query as *mut _ as *mut _,
                &lwork,
                rwork.as_mut_ptr() as *mut _,
                &mut info,
            );
            if info != 0 {
                return info;
            }
            let lwork = work_query.to_usize().unwrap();

            // Allocate work array
            let mut work: Vec<T> = match uninitialized_vec(lwork) {
                Ok(work) => work,
                Err(_) => return -1010,
            };

            // Call LAPACK directly
            func_(
                &(jobvl as _),
                &(jobvr as _),
                &(n as blas_int),
                a as *mut _,
                &(lda as blas_int),
                b as *mut _,
                &(ldb as blas_int),
                alpha as *mut _,
                beta as *mut _,
                vl as *mut _,
                &(ldvl as blas_int),
                vr as *mut _,
                &(ldvr as blas_int),
                work.as_mut_ptr() as *mut _,
                &(lwork as blas_int),
                rwork.as_mut_ptr() as *mut _,
                &mut info,
            );
            info
        } else {
            // Row-major handling
            let lda_t = n.max(1);
            let ldb_t = n.max(1);
            let ldvl_t = n.max(1);
            let ldvr_t = n.max(1);

            // Allocate temporary column-major arrays
            // Note: We always allocate VL and VR arrays even if not computing them,
            // to avoid passing dangling pointers to LAPACK.
            let mut a_t: Vec<T> = match uninitialized_vec(n * lda_t) {
                Ok(a_t) => a_t,
                Err(_) => return -1010,
            };
            let mut b_t: Vec<T> = match uninitialized_vec(n * ldb_t) {
                Ok(b_t) => b_t,
                Err(_) => return -1010,
            };
            let mut vl_t: Vec<T> = match uninitialized_vec(n * ldvl_t) {
                Ok(vl_t) => vl_t,
                Err(_) => return -1011,
            };
            let mut vr_t: Vec<T> = match uninitialized_vec(n * ldvr_t) {
                Ok(vr_t) => vr_t,
                Err(_) => return -1011,
            };

            // Transpose inputs from row-major to column-major
            let a_slice = from_raw_parts_mut(a, n * lda);
            let b_slice = from_raw_parts_mut(b, n * ldb);
            let la = Layout::new_unchecked([n, n], [lda as isize, 1], 0);
            let lb = Layout::new_unchecked([n, n], [ldb as isize, 1], 0);
            let la_t = Layout::new_unchecked([n, n], [1, lda_t as isize], 0);
            let lb_t = Layout::new_unchecked([n, n], [1, ldb_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).unwrap();
            orderchange_out_r2c_ix2_cpu_serial(&mut b_t, &lb_t, b_slice, &lb).unwrap();

            // Query optimal work array size with column-major dimensions
            let mut info = 0;
            let lwork = -1;
            let mut work_query: T = T::zero();
            func_(
                &(jobvl as _),
                &(jobvr as _),
                &(n as blas_int),
                a_t.as_mut_ptr() as *mut _,
                &(lda_t as blas_int),
                b_t.as_mut_ptr() as *mut _,
                &(ldb_t as blas_int),
                alpha as *mut _,
                beta as *mut _,
                vl_t.as_mut_ptr() as *mut _,
                &(ldvl_t as blas_int),
                vr_t.as_mut_ptr() as *mut _,
                &(ldvr_t as blas_int),
                &mut work_query as *mut _ as *mut _,
                &lwork,
                rwork.as_mut_ptr() as *mut _,
                &mut info,
            );
            if info != 0 {
                return info;
            }
            let lwork = work_query.to_usize().unwrap();

            // Allocate work array
            let mut work: Vec<T> = match uninitialized_vec(lwork) {
                Ok(work) => work,
                Err(_) => return -1010,
            };

            // Call LAPACK with column-major arrays
            func_(
                &(jobvl as _),
                &(jobvr as _),
                &(n as blas_int),
                a_t.as_mut_ptr() as *mut _,
                &(lda_t as blas_int),
                b_t.as_mut_ptr() as *mut _,
                &(ldb_t as blas_int),
                alpha as *mut _,
                beta as *mut _,
                vl_t.as_mut_ptr() as *mut _,
                &(ldvl_t as blas_int),
                vr_t.as_mut_ptr() as *mut _,
                &(ldvr_t as blas_int),
                work.as_mut_ptr() as *mut _,
                &(lwork as blas_int),
                rwork.as_mut_ptr() as *mut _,
                &mut info,
            );
            if info != 0 {
                return info;
            }

            // Transpose outputs back to row-major
            orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).unwrap();
            orderchange_out_c2r_ix2_cpu_serial(b_slice, &lb, &b_t, &lb_t).unwrap();
            if compute_vl {
                let vl_slice = from_raw_parts_mut(vl, n * ldvl);
                let lvl = Layout::new_unchecked([n, n], [ldvl as isize, 1], 0);
                let lvl_t = Layout::new_unchecked([n, n], [1, ldvl_t as isize], 0);
                orderchange_out_c2r_ix2_cpu_serial(vl_slice, &lvl, &vl_t, &lvl_t).unwrap();
            }
            if compute_vr {
                let vr_slice = from_raw_parts_mut(vr, n * ldvr);
                let lvr = Layout::new_unchecked([n, n], [ldvr as isize, 1], 0);
                let lvr_t = Layout::new_unchecked([n, n], [1, ldvr_t as isize], 0);
                orderchange_out_c2r_ix2_cpu_serial(vr_slice, &lvr, &vr_t, &lvr_t).unwrap();
            }

            info
        }
    }
}
