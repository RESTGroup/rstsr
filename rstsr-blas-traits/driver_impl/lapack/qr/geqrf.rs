use crate::lapack_ffi;
use crate::DeviceBLAS;
use num::{Complex, Zero};
use rstsr_blas_traits::prelude::*;
use rstsr_blas_traits::util::util_error::lwork_from_query;
use rstsr_common::prelude_dev::*;

use rstsr_native_impl::prelude_dev::*;

use std::slice::from_raw_parts_mut;

#[duplicate_item(
    T          fn_work      func_    ;
   [f32] [sgeqrf_work] [sgeqrf_];
   [f64] [dgeqrf_work] [dgeqrf_];
)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fn_work(
    order: FlagOrder,
    m: usize,
    n: usize,
    a: *mut T,
    lda: usize,
    tau: *mut T,
    work: *mut T,
    lwork: blas_int,
    info: *mut blas_int,
) -> blas_int {
    use lapack_ffi::lapack::func_;

    match order {
        ColMajor => {
            func_(&(m as _), &(n as _), a, &(lda as _), tau, work, &lwork, info);
            *info
        },
        RowMajor => {
            let lda_t = m.max(1);
            // Check leading dimension(s); Fortran (reference-LAPACK) argument indices
            if lda < n {
                *info = -5;
                return *info;
            }
            // Query optimal working array(s) size if requested
            if lwork == -1 {
                func_(&(m as _), &(n as _), a, &(lda_t as _), tau, work, &lwork, info);
                return *info;
            }
            // Allocate memory for temporary array(s)
            let mut a_t: Vec<T> = match uninitialized_vec(lda_t * n.max(1)) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };
            let a_slice = from_raw_parts_mut(a, m * lda);
            let la = Layout::new_unchecked([m, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([m, n], [1, lda_t as isize], 0);
            // Transpose input matrices
            if orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).is_err() {
                return -1011;
            }
            // Call LAPACK function
            func_(&(m as _), &(n as _), a_t.as_mut_ptr(), &(lda_t as _), tau, work, &lwork, info);
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
    T          fn_work      func_    ;
   [f32] [sgeqrf_work] [sgeqrf_];
   [f64] [dgeqrf_work] [dgeqrf_];
)]
impl GEQRFDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_geqrf(order: FlagOrder, m: usize, n: usize, a: *mut T, lda: usize, tau: *mut T) -> blas_int {
        // query optimal workspace size
        let mut info: blas_int = 0;
        let mut work_query: T = <T as Zero>::zero();
        let info_query = fn_work(order, m, n, a, lda, tau, &mut work_query, -1, &mut info);
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
        fn_work(order, m, n, a, lda, tau, work.as_mut_ptr(), lwork as blas_int, &mut info)
    }
}

#[duplicate_item(
    T          fn_work      func_    ;
   [Complex<f32>] [cgeqrf_work] [cgeqrf_];
   [Complex<f64>] [zgeqrf_work] [zgeqrf_];
)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fn_work(
    order: FlagOrder,
    m: usize,
    n: usize,
    a: *mut T,
    lda: usize,
    tau: *mut T,
    work: *mut T,
    lwork: blas_int,
    info: *mut blas_int,
) -> blas_int {
    use lapack_ffi::lapack::func_;

    match order {
        ColMajor => {
            func_(&(m as _), &(n as _), a as *mut _, &(lda as _), tau as *mut _, work as *mut _, &lwork, info);
            *info
        },
        RowMajor => {
            let lda_t = m.max(1);
            // Check leading dimension(s); Fortran (reference-LAPACK) argument indices
            if lda < n {
                *info = -5;
                return *info;
            }
            // Query optimal working array(s) size if requested
            if lwork == -1 {
                func_(&(m as _), &(n as _), a as *mut _, &(lda_t as _), tau as *mut _, work as *mut _, &lwork, info);
                return *info;
            }
            // Allocate memory for temporary array(s)
            let mut a_t: Vec<T> = match uninitialized_vec(lda_t * n.max(1)) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };
            let a_slice = from_raw_parts_mut(a, m * lda);
            let la = Layout::new_unchecked([m, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([m, n], [1, lda_t as isize], 0);
            // Transpose input matrices
            if orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).is_err() {
                return -1011;
            }
            // Call LAPACK function
            func_(
                &(m as _),
                &(n as _),
                a_t.as_mut_ptr() as *mut _,
                &(lda_t as _),
                tau as *mut _,
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
            *info
        },
    }
}

#[duplicate_item(
    T          fn_work      func_    ;
   [Complex<f32>] [cgeqrf_work] [cgeqrf_];
   [Complex<f64>] [zgeqrf_work] [zgeqrf_];
)]
impl GEQRFDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_geqrf(order: FlagOrder, m: usize, n: usize, a: *mut T, lda: usize, tau: *mut T) -> blas_int {
        // query optimal workspace size
        let mut info: blas_int = 0;
        let mut work_query: T = <T as Zero>::zero();
        let info_query = fn_work(order, m, n, a, lda, tau, &mut work_query, -1, &mut info);
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
        fn_work(order, m, n, a, lda, tau, work.as_mut_ptr(), lwork as blas_int, &mut info)
    }
}
