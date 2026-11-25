use crate::lapack_ffi;
use crate::DeviceBLAS;
use num::complex::ComplexFloat;
use num::{Complex, Zero};
use rstsr_blas_traits::prelude::*;
use rstsr_blas_traits::util::util_error::lwork_from_query;
use rstsr_common::prelude_dev::*;

use rstsr_native_impl::prelude_dev::*;

use std::slice::from_raw_parts_mut;

#[duplicate_item(
    T     fn_work     func_    ;
   [f32] [ssygv_work] [ssygv_];
   [f64] [dsygv_work] [dsygv_];
)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fn_work(
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
    work: *mut T,
    lwork: blas_int,
    info: *mut blas_int,
) -> blas_int {
    use lapack_ffi::lapack::func_;

    match order {
        ColMajor => {
            func_(&itype, &(jobz as _), &uplo.into(), &(n as _), a, &(lda as _), b, &(ldb as _), w, work, &lwork, info);
            *info
        },
        RowMajor => {
            let lda_t = n.max(1);
            let ldb_t = n.max(1);
            // Check leading dimension(s); Fortran (reference-LAPACK) argument indices
            if lda < n {
                *info = -6;
                return *info;
            }
            if ldb < n {
                *info = -8;
                return *info;
            }
            // Query optimal working array(s) size if requested
            if lwork == -1 {
                func_(
                    &itype,
                    &(jobz as _),
                    &uplo.into(),
                    &(n as _),
                    a,
                    &(lda_t as _),
                    b,
                    &(ldb_t as _),
                    w,
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
            let mut b_t: Vec<T> = match uninitialized_vec(ldb_t * n.max(1)) {
                Ok(b_t) => b_t,
                Err(_) => return -1011,
            };
            let a_slice = from_raw_parts_mut(a, n * lda);
            let b_slice = from_raw_parts_mut(b, n * ldb);
            let la = Layout::new_unchecked([n, n], [lda as isize, 1], 0);
            let lb = Layout::new_unchecked([n, n], [ldb as isize, 1], 0);
            let la_t = Layout::new_unchecked([n, n], [1, lda_t as isize], 0);
            let lb_t = Layout::new_unchecked([n, n], [1, ldb_t as isize], 0);
            // Transpose input matrices
            if orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).is_err() {
                return -1011;
            }
            if orderchange_out_r2c_ix2_cpu_serial(&mut b_t, &lb_t, b_slice, &lb).is_err() {
                return -1011;
            }
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
   [f32] [ssygv_work] [ssygv_];
   [f64] [dsygv_work] [dsygv_];
)]
impl SYGVDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_sygv(
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
        // query optimal workspace size
        let mut info: blas_int = 0;
        let mut work_query: T = <T as Zero>::zero();
        let info_query = fn_work(order, itype, jobz, uplo, n, a, lda, b, ldb, w, &mut work_query, -1, &mut info);
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
        fn_work(order, itype, jobz, uplo, n, a, lda, b, ldb, w, work.as_mut_ptr(), lwork as blas_int, &mut info)
    }
}

#[duplicate_item(
    T              fn_work     func_    ;
   [Complex<f32>] [chegv_work] [chegv_];
   [Complex<f64>] [zhegv_work] [zhegv_];
)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fn_work(
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
    work: *mut T,
    lwork: blas_int,
    rwork: *mut <T as ComplexFloat>::Real,
    info: *mut blas_int,
) -> blas_int {
    use lapack_ffi::lapack::func_;

    match order {
        ColMajor => {
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
                work as *mut _,
                &lwork,
                rwork as *mut _,
                info,
            );
            *info
        },
        RowMajor => {
            let lda_t = n.max(1);
            let ldb_t = n.max(1);
            // Check leading dimension(s); Fortran (reference-LAPACK) argument indices
            if lda < n {
                *info = -6;
                return *info;
            }
            if ldb < n {
                *info = -8;
                return *info;
            }
            // Query optimal working array(s) size if requested
            if lwork == -1 {
                func_(
                    &itype,
                    &(jobz as _),
                    &uplo.into(),
                    &(n as _),
                    a as *mut _,
                    &(lda_t as _),
                    b as *mut _,
                    &(ldb_t as _),
                    w as *mut _,
                    work as *mut _,
                    &lwork,
                    rwork as *mut _,
                    info,
                );
                return *info;
            }
            // Allocate memory for temporary array(s)
            let mut a_t: Vec<T> = match uninitialized_vec(lda_t * n.max(1)) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };
            let mut b_t: Vec<T> = match uninitialized_vec(ldb_t * n.max(1)) {
                Ok(b_t) => b_t,
                Err(_) => return -1011,
            };
            let a_slice = from_raw_parts_mut(a, n * lda);
            let b_slice = from_raw_parts_mut(b, n * ldb);
            let la = Layout::new_unchecked([n, n], [lda as isize, 1], 0);
            let lb = Layout::new_unchecked([n, n], [ldb as isize, 1], 0);
            let la_t = Layout::new_unchecked([n, n], [1, lda_t as isize], 0);
            let lb_t = Layout::new_unchecked([n, n], [1, ldb_t as isize], 0);
            // Transpose input matrices
            if orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).is_err() {
                return -1011;
            }
            if orderchange_out_r2c_ix2_cpu_serial(&mut b_t, &lb_t, b_slice, &lb).is_err() {
                return -1011;
            }
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
                work as *mut _,
                &lwork,
                rwork as *mut _,
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
    T              fn_work     func_    ;
   [Complex<f32>] [chegv_work] [chegv_];
   [Complex<f64>] [zhegv_work] [zhegv_];
)]
impl SYGVDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_sygv(
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
        // complex rwork: documented fixed minimum, allocated before the query
        let mut rwork: Vec<<T as ComplexFloat>::Real> = match uninitialized_vec((3 * n).saturating_sub(2).max(1)) {
            Ok(rwork) => rwork,
            Err(_) => return -1010,
        };

        // query optimal workspace size
        let mut info: blas_int = 0;
        let mut work_query: T = <T as Zero>::zero();
        let info_query =
            fn_work(order, itype, jobz, uplo, n, a, lda, b, ldb, w, &mut work_query, -1, rwork.as_mut_ptr(), &mut info);
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
        fn_work(
            order,
            itype,
            jobz,
            uplo,
            n,
            a,
            lda,
            b,
            ldb,
            w,
            work.as_mut_ptr(),
            lwork as blas_int,
            rwork.as_mut_ptr(),
            &mut info,
        )
    }
}
