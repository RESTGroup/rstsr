use crate::lapack_ffi;
use crate::DeviceBLAS;
use num::{Complex, Zero};
use rstsr_blas_traits::prelude::*;
use rstsr_blas_traits::util::util_error::lwork_from_query;
use rstsr_common::prelude_dev::*;

use rstsr_native_impl::prelude_dev::*;

use std::slice::from_raw_parts_mut;

#[duplicate_item(
    T     fn_work     func_    ;
   [f32] [ssysv_work] [ssysv_];
   [f64] [dsysv_work] [dsysv_];
)]
pub unsafe fn fn_work(
    order: FlagOrder,
    uplo: FlagUpLo,
    n: usize,
    nrhs: usize,
    a: *mut T,
    lda: usize,
    ipiv: *mut blas_int,
    b: *mut T,
    ldb: usize,
    work: *mut T,
    lwork: blas_int,
    info: *mut blas_int,
) -> blas_int {
    use lapack_ffi::lapack::func_;

    match order {
        ColMajor => {
            func_(&uplo.into(), &(n as _), &(nrhs as _), a, &(lda as _), ipiv, b, &(ldb as _), work, &lwork, info);
            *info
        },
        RowMajor => {
            let lda_t = n.max(1);
            let ldb_t = n.max(1);
            // Check leading dimension(s); Fortran (reference-LAPACK) argument indices
            if lda < n {
                *info = -5;
                return *info;
            }
            if ldb < nrhs {
                *info = -8;
                return *info;
            }
            // Query optimal working array(s) size if requested
            if lwork == -1 {
                func_(
                    &uplo.into(),
                    &(n as _),
                    &(nrhs as _),
                    a,
                    &(lda_t as _),
                    ipiv,
                    b,
                    &(ldb_t as _),
                    work,
                    &lwork,
                    info,
                );
                return *info;
            }
            // Allocate memory for temporary array(s)
            let mut a_t: Vec<T> = match uninitialized_vec(lda_t * n.max(1)) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };
            let mut b_t: Vec<T> = match uninitialized_vec(ldb_t * nrhs.max(1)) {
                Ok(b_t) => b_t,
                Err(_) => return -1011,
            };
            let a_slice = from_raw_parts_mut(a, n * lda);
            let b_slice = from_raw_parts_mut(b, n * ldb);
            let la = Layout::new_unchecked([n, n], [lda as isize, 1], 0);
            let lb = Layout::new_unchecked([n, nrhs], [ldb as isize, 1], 0);
            let la_t = Layout::new_unchecked([n, n], [1, lda_t as isize], 0);
            let lb_t = Layout::new_unchecked([n, nrhs], [1, ldb_t as isize], 0);
            // Transpose input matrices
            if orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).is_err() {
                return -1011;
            }
            if orderchange_out_r2c_ix2_cpu_serial(&mut b_t, &lb_t, b_slice, &lb).is_err() {
                return -1011;
            }
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
                work,
                &lwork,
                info,
            );
            if *info != 0 {
                return *info;
            }
            // Transpose output matrices
            if orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).is_err() {
                return -1011;
            }
            if orderchange_out_c2r_ix2_cpu_serial(b_slice, &lb, &b_t, &lb_t).is_err() {
                return -1011;
            }
            *info
        },
    }
}

#[duplicate_item(
    T     fn_work     func_    ;
   [f32] [ssysv_work] [ssysv_];
   [f64] [dsysv_work] [dsysv_];
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
        // query optimal workspace size
        let mut info: blas_int = 0;
        let mut work_query: T = <T as Zero>::zero();
        let info_query = fn_work(order, uplo, n, nrhs, a, lda, ipiv, b, ldb, &mut work_query, -1, &mut info);
        if info_query != 0 {
            return info_query;
        }
        let Some(lwork) = lwork_from_query(work_query) else {
            return -1010;
        };

        // allocate memory for work array
        let mut work: Vec<T> = match uninitialized_vec(lwork) {
            Ok(work) => work,
            Err(_) => return -1010,
        };

        // call LAPACK routine
        fn_work(order, uplo, n, nrhs, a, lda, ipiv, b, ldb, work.as_mut_ptr(), lwork as blas_int, &mut info)
    }
}

#[duplicate_item(
    T              fn_work      func_    HERMI  ;
   [Complex<f32>] [csysv_work] [csysv_] [false];
   [Complex<f32>] [chesv_work] [chesv_] [true ];
   [Complex<f64>] [zsysv_work] [zsysv_] [false];
   [Complex<f64>] [zhesv_work] [zhesv_] [true ];
)]
pub unsafe fn fn_work(
    order: FlagOrder,
    uplo: FlagUpLo,
    n: usize,
    nrhs: usize,
    a: *mut T,
    lda: usize,
    ipiv: *mut blas_int,
    b: *mut T,
    ldb: usize,
    work: *mut T,
    lwork: blas_int,
    info: *mut blas_int,
) -> blas_int {
    use lapack_ffi::lapack::func_;

    match order {
        ColMajor => {
            func_(
                &uplo.into(),
                &(n as _),
                &(nrhs as _),
                a as *mut _,
                &(lda as _),
                ipiv,
                b as *mut _,
                &(ldb as _),
                work as *mut _,
                &lwork,
                info,
            );
            *info
        },
        RowMajor => {
            let lda_t = n.max(1);
            let ldb_t = n.max(1);
            // Check leading dimension(s); Fortran (reference-LAPACK) argument indices
            if lda < n {
                *info = -5;
                return *info;
            }
            if ldb < nrhs {
                *info = -8;
                return *info;
            }
            // Query optimal working array(s) size if requested
            if lwork == -1 {
                func_(
                    &uplo.into(),
                    &(n as _),
                    &(nrhs as _),
                    a as *mut _,
                    &(lda_t as _),
                    ipiv,
                    b as *mut _,
                    &(ldb_t as _),
                    work as *mut _,
                    &lwork,
                    info,
                );
                return *info;
            }
            // Allocate memory for temporary array(s)
            let mut a_t: Vec<T> = match uninitialized_vec(lda_t * n.max(1)) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };
            let mut b_t: Vec<T> = match uninitialized_vec(ldb_t * nrhs.max(1)) {
                Ok(b_t) => b_t,
                Err(_) => return -1011,
            };
            let a_slice = from_raw_parts_mut(a, n * lda);
            let b_slice = from_raw_parts_mut(b, n * ldb);
            let la = Layout::new_unchecked([n, n], [lda as isize, 1], 0);
            let lb = Layout::new_unchecked([n, nrhs], [ldb as isize, 1], 0);
            let la_t = Layout::new_unchecked([n, n], [1, lda_t as isize], 0);
            let lb_t = Layout::new_unchecked([n, nrhs], [1, ldb_t as isize], 0);
            // Transpose input matrices
            if orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).is_err() {
                return -1011;
            }
            if orderchange_out_r2c_ix2_cpu_serial(&mut b_t, &lb_t, b_slice, &lb).is_err() {
                return -1011;
            }
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
                work as *mut _,
                &lwork,
                info,
            );
            if *info != 0 {
                return *info;
            }
            // Transpose output matrices
            if orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).is_err() {
                return -1011;
            }
            if orderchange_out_c2r_ix2_cpu_serial(b_slice, &lb, &b_t, &lb_t).is_err() {
                return -1011;
            }
            *info
        },
    }
}

#[duplicate_item(
    T              fn_work      func_    HERMI  ;
   [Complex<f32>] [csysv_work] [csysv_] [false];
   [Complex<f32>] [chesv_work] [chesv_] [true ];
   [Complex<f64>] [zsysv_work] [zsysv_] [false];
   [Complex<f64>] [zhesv_work] [zhesv_] [true ];
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
        // query optimal workspace size
        let mut info: blas_int = 0;
        let mut work_query: T = <T as Zero>::zero();
        let info_query = fn_work(order, uplo, n, nrhs, a, lda, ipiv, b, ldb, &mut work_query, -1, &mut info);
        if info_query != 0 {
            return info_query;
        }
        let Some(lwork) = lwork_from_query(work_query) else {
            return -1010;
        };

        // allocate memory for work array
        let mut work: Vec<T> = match uninitialized_vec(lwork) {
            Ok(work) => work,
            Err(_) => return -1010,
        };

        // call LAPACK routine
        fn_work(order, uplo, n, nrhs, a, lda, ipiv, b, ldb, work.as_mut_ptr(), lwork as blas_int, &mut info)
    }
}
