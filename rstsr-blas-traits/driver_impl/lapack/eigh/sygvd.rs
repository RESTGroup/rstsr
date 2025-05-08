/* lapack_sygvd.rs */
use crate::DeviceBLAS;
use num::complex::ComplexFloat;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude_dev::*;
use rstsr_lapack_ffi::blas::xerbla_;
use rstsr_lapack_ffi::lapacke::{LAPACK_TRANSPOSE_MEMORY_ERROR, LAPACK_WORK_MEMORY_ERROR};
use rstsr_native_impl::prelude_dev::*;
use std::slice::from_raw_parts_mut;

#[duplicate_item(
    T     func_   ;
   [f32] [ssygvd_];
   [f64] [dsygvd_];
)]
impl SYGVDDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_sygvd(
        order: FlagOrder,
        itype: blas_int,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        w: *mut T,
    ) -> blas_int {
        use rstsr_lapack_ffi::lapack::func_;

        unsafe fn raise_info(mut info: blas_int) -> blas_int {
            xerbla_(c"sygvd".as_ptr() as _, &mut info as *mut _ as *mut _);
            return if info < 0 { info - 1 } else { info };
        }

        // Query optimal working array(s) size
        let mut info = 0;
        let lwork = -1;
        let liwork = -1;
        let mut work_query = 0.0;
        let mut iwork_query = 0;
        func_(
            &itype,
            &(jobz as _),
            &uplo.into(),
            &(n as _),
            a,
            &(lda as _),
            b,
            &(ldb as _),
            w,
            &mut work_query,
            &lwork,
            &mut iwork_query,
            &liwork,
            &mut info,
        );
        if info != 0 {
            return raise_info(info);
        }
        let lwork = work_query as usize;
        let liwork = iwork_query as usize;

        // Allocate memory for temporary array(s)
        let mut work: Vec<T> = match uninitialized_vec(lwork) {
            Ok(work) => work,
            Err(_) => return LAPACK_WORK_MEMORY_ERROR,
        };
        let mut iwork: Vec<blas_int> = match uninitialized_vec(liwork) {
            Ok(iwork) => iwork,
            Err(_) => return LAPACK_WORK_MEMORY_ERROR,
        };

        if order == ColMajor {
            // Call LAPACK function and adjust info
            func_(
                &itype,
                &(jobz as _),
                &uplo.into(),
                &(n as _),
                a,
                &(lda as _),
                b,
                &(ldb as _),
                w,
                work.as_mut_ptr(),
                &(lwork as _),
                iwork.as_mut_ptr(),
                &(liwork as _),
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
            let mut b_t: Vec<T> = match uninitialized_vec(n * n) {
                Ok(b_t) => b_t,
                Err(_) => return LAPACK_TRANSPOSE_MEMORY_ERROR,
            };
            let a_slice = from_raw_parts_mut(a, n * lda);
            let b_slice = from_raw_parts_mut(b, n * ldb);
            let la = Layout::new_unchecked([n, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([n, n], [1, lda_t as isize], 0);
            let lb = Layout::new_unchecked([n, n], [ldb as isize, 1], 0);
            let lb_t = Layout::new_unchecked([n, n], [1, ldb_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).unwrap();
            orderchange_out_r2c_ix2_cpu_serial(&mut b_t, &lb_t, b_slice, &lb).unwrap();
            // Call LAPACK function and adjust info
            func_(
                &itype,
                &(jobz as _),
                &uplo.into(),
                &(n as _),
                a_t.as_mut_ptr(),
                &(lda_t as _),
                b_t.as_mut_ptr(),
                &(ldb_t as _),
                w,
                work.as_mut_ptr(),
                &(lwork as _),
                iwork.as_mut_ptr(),
                &(liwork as _),
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
    T              func_   ;
   [Complex<f32>] [chegvd_];
   [Complex<f64>] [zhegvd_];
)]
impl SYGVDDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_sygvd(
        order: FlagOrder,
        itype: blas_int,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        w: *mut <T as ComplexFloat>::Real,
    ) -> blas_int {
        use rstsr_lapack_ffi::lapack::func_;

        unsafe fn raise_info(mut info: blas_int) -> blas_int {
            xerbla_(c"sygvd".as_ptr() as _, &mut info as *mut _ as *mut _);
            return if info < 0 { info - 1 } else { info };
        }

        // Query optimal working array(s) size
        let mut info = 0;
        let lwork = -1;
        let lrwork = -1;
        let liwork = -1;
        let mut work_query = 0.0;
        let mut rwork_query = 0.0;
        let mut iwork_query = 0;
        func_(
            &itype,
            &(jobz as _),
            &uplo.into(),
            &(n as _),
            a as *mut _,
            &(lda as _),
            b as *mut _,
            &(ldb as _),
            w as *mut _,
            &mut work_query as *mut _ as *mut _,
            &lwork,
            &mut rwork_query as *mut _ as *mut _,
            &lrwork,
            &mut iwork_query,
            &liwork,
            &mut info,
        );
        if info != 0 {
            return raise_info(info);
        }
        let lwork = work_query as usize;
        let lrwork = rwork_query as usize;
        let liwork = iwork_query as usize;

        // Allocate memory for temporary array(s)
        let mut work: Vec<T> = match uninitialized_vec(lwork) {
            Ok(work) => work,
            Err(_) => return LAPACK_WORK_MEMORY_ERROR,
        };
        let mut rwork: Vec<<T as ComplexFloat>::Real> = match uninitialized_vec(lrwork) {
            Ok(rwork) => rwork,
            Err(_) => return LAPACK_WORK_MEMORY_ERROR,
        };
        let mut iwork: Vec<blas_int> = match uninitialized_vec(liwork) {
            Ok(iwork) => iwork,
            Err(_) => return LAPACK_WORK_MEMORY_ERROR,
        };

        if order == ColMajor {
            // Call LAPACK function and adjust info
            func_(
                &itype,
                &(jobz as _),
                &uplo.into(),
                &(n as _),
                a as *mut _,
                &(lda as _),
                b as *mut _,
                &(ldb as _),
                w as *mut _,
                work.as_mut_ptr() as *mut _,
                &(lwork as _),
                rwork.as_mut_ptr() as *mut _,
                &(lrwork as _),
                iwork.as_mut_ptr() as *mut _,
                &(liwork as _),
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
            let mut b_t: Vec<T> = match uninitialized_vec(n * n) {
                Ok(b_t) => b_t,
                Err(_) => return LAPACK_TRANSPOSE_MEMORY_ERROR,
            };
            let a_slice = from_raw_parts_mut(a, n * lda);
            let b_slice = from_raw_parts_mut(b, n * ldb);
            let la = Layout::new_unchecked([n, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([n, n], [1, lda_t as isize], 0);
            let lb = Layout::new_unchecked([n, n], [ldb as isize, 1], 0);
            let lb_t = Layout::new_unchecked([n, n], [1, ldb_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).unwrap();
            orderchange_out_r2c_ix2_cpu_serial(&mut b_t, &lb_t, b_slice, &lb).unwrap();
            // Call LAPACK function and adjust info
            func_(
                &itype,
                &(jobz as _),
                &uplo.into(),
                &(n as _),
                a_t.as_mut_ptr() as *mut _,
                &(lda_t as _),
                b_t.as_mut_ptr() as *mut _,
                &(ldb_t as _),
                w as *mut _,
                work.as_mut_ptr() as *mut _,
                &(lwork as _),
                rwork.as_mut_ptr() as *mut _,
                &(lrwork as _),
                iwork.as_mut_ptr(),
                &(liwork as _),
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
