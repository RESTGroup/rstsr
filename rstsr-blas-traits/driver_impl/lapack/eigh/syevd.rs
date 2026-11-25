use crate::lapack_ffi;
use crate::DeviceBLAS;
use num::complex::ComplexFloat;
use num::{Complex, Zero};
use rstsr_blas_traits::prelude::*;
use rstsr_blas_traits::util::util_error::{liwork_from_query, lwork_from_query};
use rstsr_common::prelude_dev::*;

use rstsr_native_impl::prelude_dev::*;

use std::slice::from_raw_parts_mut;

#[duplicate_item(
    T      fn_work      func_    ;
   [f32] [ssyevd_work] [ssyevd_];
   [f64] [dsyevd_work] [dsyevd_];
)]
pub unsafe fn fn_work(
    order: FlagOrder,
    jobz: char,
    uplo: FlagUpLo,
    n: usize,
    a: *mut T,
    lda: usize,
    w: *mut T,
    work: *mut T,
    lwork: blas_int,
    iwork: *mut blas_int,
    liwork: blas_int,
    info: *mut blas_int,
) -> blas_int {
    use lapack_ffi::lapack::func_;

    match order {
        ColMajor => {
            func_(&(jobz as _), &uplo.into(), &(n as _), a, &(lda as _), w, work, &lwork, iwork, &liwork, info);
            *info
        },
        RowMajor => {
            let lda_t = n.max(1);
            // Check leading dimension(s); Fortran (reference-LAPACK) argument indices
            if lda < n {
                *info = -5;
                return *info;
            }
            // Query optimal working array(s) size if requested
            if lwork == -1 || liwork == -1 {
                func_(&(jobz as _), &uplo.into(), &(n as _), a, &(lda_t as _), w, work, &lwork, iwork, &liwork, info);
                return *info;
            }
            // Allocate memory for temporary array(s)
            let mut a_t: Vec<T> = match uninitialized_vec(lda_t * n.max(1)) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };
            let a_slice = from_raw_parts_mut(a, n * lda);
            let la = Layout::new_unchecked([n, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([n, n], [1, lda_t as isize], 0);
            // Transpose input matrices
            if orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).is_err() {
                return -1011;
            }
            // Call LAPACK function and adjust info
            func_(
                &(jobz as _),
                &uplo.into(),
                &(n as _),
                a_t.as_mut_ptr(),
                &(lda_t as _),
                w,
                work,
                &lwork,
                iwork,
                &liwork,
                info,
            );
            if *info != 0 {
                return *info;
            }
            // Transpose output matrices
            if orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).is_err() {
                return -1011;
            }
            *info
        },
    }
}

#[duplicate_item(
    T      fn_work      func_    ;
   [f32] [ssyevd_work] [ssyevd_];
   [f64] [dsyevd_work] [dsyevd_];
)]
impl SYEVDDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_syevd(
        order: FlagOrder,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        w: *mut T,
    ) -> blas_int {
        // double query: lwork and liwork in one call
        let mut info: blas_int = 0;
        let mut work_query: T = <T as Zero>::zero();
        let mut iwork_query: blas_int = 0;
        let info_query = fn_work(order, jobz, uplo, n, a, lda, w, &mut work_query, -1, &mut iwork_query, -1, &mut info);
        if info_query != 0 {
            return info_query;
        }
        let Some(lwork) = lwork_from_query(work_query) else {
            return -1010;
        };
        let Some(liwork) = liwork_from_query(iwork_query) else {
            return -1010;
        };

        // allocate memory for work arrays
        let mut work: Vec<T> = match uninitialized_vec(lwork) {
            Ok(work) => work,
            Err(_) => return -1010,
        };
        let mut iwork: Vec<blas_int> = match uninitialized_vec(liwork) {
            Ok(iwork) => iwork,
            Err(_) => return -1010,
        };

        // call LAPACK routine
        fn_work(
            order,
            jobz,
            uplo,
            n,
            a,
            lda,
            w,
            work.as_mut_ptr(),
            lwork as blas_int,
            iwork.as_mut_ptr(),
            liwork as blas_int,
            &mut info,
        )
    }
}

#[duplicate_item(
    T              fn_work      func_    ;
   [Complex<f32>] [cheevd_work] [cheevd_];
   [Complex<f64>] [zheevd_work] [zheevd_];
)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fn_work(
    order: FlagOrder,
    jobz: char,
    uplo: FlagUpLo,
    n: usize,
    a: *mut T,
    lda: usize,
    w: *mut <T as ComplexFloat>::Real,
    work: *mut T,
    lwork: blas_int,
    rwork: *mut <T as ComplexFloat>::Real,
    lrwork: blas_int,
    iwork: *mut blas_int,
    liwork: blas_int,
    info: *mut blas_int,
) -> blas_int {
    use lapack_ffi::lapack::func_;

    match order {
        ColMajor => {
            func_(
                &(jobz as _),
                &uplo.into(),
                &(n as _),
                a as *mut _,
                &(lda as _),
                w as *mut _,
                work as *mut _,
                &lwork,
                rwork as *mut _,
                &lrwork,
                iwork,
                &liwork,
                info,
            );
            *info
        },
        RowMajor => {
            let lda_t = n.max(1);
            // Check leading dimension(s); Fortran (reference-LAPACK) argument indices
            if lda < n {
                *info = -5;
                return *info;
            }
            // Query optimal working array(s) size if requested
            if lwork == -1 || lrwork == -1 || liwork == -1 {
                func_(
                    &(jobz as _),
                    &uplo.into(),
                    &(n as _),
                    a as *mut _,
                    &(lda_t as _),
                    w as *mut _,
                    work as *mut _,
                    &lwork,
                    rwork as *mut _,
                    &lrwork,
                    iwork,
                    &liwork,
                    info,
                );
                return *info;
            }
            // Allocate memory for temporary array(s)
            let mut a_t: Vec<T> = match uninitialized_vec(lda_t * n.max(1)) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };
            let a_slice = from_raw_parts_mut(a, n * lda);
            let la = Layout::new_unchecked([n, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([n, n], [1, lda_t as isize], 0);
            // Transpose input matrices
            if orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).is_err() {
                return -1011;
            }
            // Call LAPACK function and adjust info
            func_(
                &(jobz as _),
                &uplo.into(),
                &(n as _),
                a_t.as_mut_ptr() as *mut _,
                &(lda_t as _),
                w as *mut _,
                work as *mut _,
                &lwork,
                rwork as *mut _,
                &lrwork,
                iwork,
                &liwork,
                info,
            );
            if *info != 0 {
                return *info;
            }
            // Transpose output matrices
            if orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).is_err() {
                return -1011;
            }
            *info
        },
    }
}

#[duplicate_item(
    T              fn_work      func_    ;
   [Complex<f32>] [cheevd_work] [cheevd_];
   [Complex<f64>] [zheevd_work] [zheevd_];
)]
impl SYEVDDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_syevd(
        order: FlagOrder,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        w: *mut <T as ComplexFloat>::Real,
    ) -> blas_int {
        // double query: lwork, lrwork and liwork in one call (rwork queried via
        // a real-typed slot; the callee writes its magnitude there)
        let mut info: blas_int = 0;
        let mut work_query: T = <T as Zero>::zero();
        let mut rwork_query: <T as ComplexFloat>::Real = <T as ComplexFloat>::Real::zero();
        let mut iwork_query: blas_int = 0;
        let info_query = fn_work(
            order,
            jobz,
            uplo,
            n,
            a,
            lda,
            w,
            &mut work_query,
            -1,
            &mut rwork_query,
            -1,
            &mut iwork_query,
            -1,
            &mut info,
        );
        if info_query != 0 {
            return info_query;
        }
        let Some(lwork) = lwork_from_query(work_query) else {
            return -1010;
        };
        let Some(lrwork) = lwork_from_query(rwork_query) else {
            return -1010;
        };
        let Some(liwork) = liwork_from_query(iwork_query) else {
            return -1010;
        };

        // allocate memory for work arrays
        let mut work: Vec<T> = match uninitialized_vec(lwork) {
            Ok(work) => work,
            Err(_) => return -1010,
        };
        let mut rwork: Vec<<T as ComplexFloat>::Real> = match uninitialized_vec(lrwork) {
            Ok(rwork) => rwork,
            Err(_) => return -1010,
        };
        let mut iwork: Vec<blas_int> = match uninitialized_vec(liwork) {
            Ok(iwork) => iwork,
            Err(_) => return -1010,
        };

        // call LAPACK routine
        fn_work(
            order,
            jobz,
            uplo,
            n,
            a,
            lda,
            w,
            work.as_mut_ptr(),
            lwork as blas_int,
            rwork.as_mut_ptr(),
            lrwork as blas_int,
            iwork.as_mut_ptr(),
            liwork as blas_int,
            &mut info,
        )
    }
}
