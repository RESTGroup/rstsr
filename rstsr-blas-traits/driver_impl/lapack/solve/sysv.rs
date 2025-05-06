use crate::DeviceBLAS;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude_dev::*;
use rstsr_lapack_ffi::blas::xerbla_;
use rstsr_lapack_ffi::lapacke::{LAPACK_TRANSPOSE_MEMORY_ERROR, LAPACK_WORK_MEMORY_ERROR};
use rstsr_native_impl::prelude_dev::*;
use std::slice::from_raw_parts_mut;

#[duplicate_item(
    T     func_   ;
   [f32] [ssysv_];
   [f64] [dsysv_];
)]
impl<const HERMI: bool> SYSVDriverAPI<T, HERMI> for DeviceBLAS {
    unsafe fn driver_sysv(
        order: FlagOrder,
        uplo: FlagUpLo,
        n: usize,
        nrhs: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
        b: *mut T,
        ldb: usize,
    ) -> blas_int {
        use rstsr_lapack_ffi::lapack::func_;

        unsafe fn raise_info(mut info: blas_int) -> blas_int {
            xerbla_(c"sysv".as_ptr() as _, &mut info as *mut _ as *mut _);
            return if info < 0 { info - 1 } else { info };
        }

        // Query optimal working array(s) size
        let mut info = 0;
        let lwork = -1;
        let mut work_query = 0.0;
        func_(
            &uplo.into(),
            &(n as _),
            &(nrhs as _),
            a,
            &(lda as _),
            ipiv,
            b,
            &(ldb as _),
            &mut work_query,
            &lwork,
            &mut info,
        );
        if info != 0 {
            return raise_info(info);
        }
        let lwork = work_query as usize;

        // Allocate memory for work arrays
        let mut work: Vec<T> = match uninitialized_vec(lwork) {
            Ok(work) => work,
            Err(_) => return LAPACK_WORK_MEMORY_ERROR,
        };

        if order == ColMajor {
            // Call LAPACK function and adjust info
            func_(
                &uplo.into(),
                &(n as _),
                &(nrhs as _),
                a,
                &(lda as _),
                ipiv,
                b,
                &(ldb as _),
                work.as_mut_ptr(),
                &(lwork as _),
                &mut info,
            );
            if info != 0 {
                return raise_info(info);
            }
        } else {
            let lda_t = n.max(1);
            let ldb_t = n.max(1);
            // Transpose input matrices
            let mut a_t: Vec<T> = match uninitialized_vec(n * n) {
                Ok(a_t) => a_t,
                Err(_) => return LAPACK_TRANSPOSE_MEMORY_ERROR,
            };
            let mut b_t: Vec<T> = match uninitialized_vec(n * nrhs) {
                Ok(b_t) => b_t,
                Err(_) => return LAPACK_TRANSPOSE_MEMORY_ERROR,
            };
            let a_slice = from_raw_parts_mut(a, n * lda);
            let b_slice = from_raw_parts_mut(b, n * ldb);
            let la = Layout::new_unchecked([n, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([n, n], [1, lda_t as isize], 0);
            let lb = Layout::new_unchecked([n, nrhs], [ldb as isize, 1], 0);
            let lb_t = Layout::new_unchecked([n, nrhs], [1, ldb_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).unwrap();
            orderchange_out_r2c_ix2_cpu_serial(&mut b_t, &lb_t, b_slice, &lb).unwrap();
            // Call LAPACK function and adjust info
            func_(
                &uplo.into(),
                &(n as _),
                &(nrhs as _),
                a_t.as_mut_ptr(),
                &(lda_t as _),
                ipiv,
                b_t.as_mut_ptr(),
                &(ldb_t as _),
                work.as_mut_ptr(),
                &(lwork as _),
                &mut info,
            );
            if info != 0 {
                return raise_info(info);
            }
            // Transpose output matrices
            orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).unwrap();
            orderchange_out_c2r_ix2_cpu_serial(b_slice, &lb, &b_t, &lb_t).unwrap();
        }
        return info;
    }
}

#[duplicate_item(
    T              func_   HERMI  ;
   [Complex<f32>] [csysv_] [false];
   [Complex<f32>] [chesv_] [true ];
   [Complex<f64>] [zsysv_] [false];
   [Complex<f64>] [zhesv_] [true ];
)]
impl SYSVDriverAPI<T, HERMI> for DeviceBLAS {
    unsafe fn driver_sysv(
        order: FlagOrder,
        uplo: FlagUpLo,
        n: usize,
        nrhs: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
        b: *mut T,
        ldb: usize,
    ) -> blas_int {
        use rstsr_lapack_ffi::lapack::func_;

        unsafe fn raise_info(mut info: blas_int) -> blas_int {
            xerbla_(c"sysv".as_ptr() as _, &mut info as *mut _ as *mut _);
            return if info < 0 { info - 1 } else { info };
        }

        // Query optimal working array(s) size
        let mut info = 0;
        let lwork = -1;
        let mut work_query = 0.0;
        func_(
            &uplo.into(),
            &(n as _),
            &(nrhs as _),
            a as *mut _,
            &(lda as _),
            ipiv,
            b as *mut _,
            &(ldb as _),
            &mut work_query as *mut _ as *mut _,
            &lwork,
            &mut info,
        );
        if info != 0 {
            return raise_info(info);
        }
        let lwork = work_query as usize;

        // Allocate memory for work arrays
        let mut work: Vec<T> = match uninitialized_vec(lwork) {
            Ok(work) => work,
            Err(_) => return LAPACK_WORK_MEMORY_ERROR,
        };

        if order == ColMajor {
            // Call LAPACK function and adjust info
            func_(
                &uplo.into(),
                &(n as _),
                &(nrhs as _),
                a as *mut _,
                &(lda as _),
                ipiv,
                b as *mut _,
                &(ldb as _),
                work.as_mut_ptr() as *mut _,
                &(lwork as _),
                &mut info,
            );
            if info != 0 {
                return raise_info(info);
            }
        } else {
            let lda_t = n.max(1);
            let ldb_t = n.max(1);
            // Transpose input matrices
            let mut a_t: Vec<T> = match uninitialized_vec(n * n) {
                Ok(a_t) => a_t,
                Err(_) => return LAPACK_TRANSPOSE_MEMORY_ERROR,
            };
            let mut b_t: Vec<T> = match uninitialized_vec(n * nrhs) {
                Ok(b_t) => b_t,
                Err(_) => return LAPACK_TRANSPOSE_MEMORY_ERROR,
            };
            let a_slice = from_raw_parts_mut(a, n * lda);
            let b_slice = from_raw_parts_mut(b, n * ldb);
            let la = Layout::new_unchecked([n, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([n, n], [1, lda_t as isize], 0);
            let lb = Layout::new_unchecked([n, nrhs], [ldb as isize, 1], 0);
            let lb_t = Layout::new_unchecked([n, nrhs], [1, ldb_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).unwrap();
            orderchange_out_r2c_ix2_cpu_serial(&mut b_t, &lb_t, b_slice, &lb).unwrap();
            // Call LAPACK function and adjust info
            func_(
                &uplo.into(),
                &(n as _),
                &(nrhs as _),
                a_t.as_mut_ptr() as *mut _,
                &(lda_t as _),
                ipiv,
                b_t.as_mut_ptr() as *mut _,
                &(ldb_t as _),
                work.as_mut_ptr() as *mut _,
                &(lwork as _),
                &mut info,
            );
            if info != 0 {
                return raise_info(info);
            }
            // Transpose output matrices
            orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).unwrap();
            orderchange_out_c2r_ix2_cpu_serial(b_slice, &lb, &b_t, &lb_t).unwrap();
        }
        return info;
    }
}
