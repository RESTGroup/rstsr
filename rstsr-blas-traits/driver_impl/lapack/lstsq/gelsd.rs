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
    T          fn_work      func_      ;
   [f32] [sgelsd_work] [sgelsd_] ;
   [f64] [dgelsd_work] [dgelsd_] ;
)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fn_work(
    order: FlagOrder,
    m: usize,
    n: usize,
    nrhs: usize,
    a: *mut T,
    lda: usize,
    b: *mut T,
    ldb: usize,
    s: *mut <T as ComplexFloat>::Real,
    rcond: <T as ComplexFloat>::Real,
    rank: *mut blas_int,
    work: *mut T,
    lwork: blas_int,
    iwork: *mut blas_int,
    info: *mut blas_int,
) -> blas_int {
    use lapack_ffi::lapack::func_;

    match order {
        ColMajor => {
            func_(
                &(m as _),
                &(n as _),
                &(nrhs as _),
                a,
                &(lda as _),
                b,
                &(ldb as _),
                s,
                &rcond,
                rank,
                work,
                &lwork,
                iwork,
                info,
            );
            *info
        },
        RowMajor => {
            // row-major: ld counts the row length; LAPACK's argument indices
            if lda < n {
                *info = -5;
                return *info;
            }
            if ldb < nrhs {
                *info = -7;
                return *info;
            }
            let lda_t = m.max(1);
            let ldb_t = m.max(n).max(1);
            // query: sizes only, the user arrays are not read
            if lwork == -1 {
                func_(
                    &(m as _),
                    &(n as _),
                    &(nrhs as _),
                    a,
                    &(lda_t as _),
                    b,
                    &(ldb_t as _),
                    s,
                    &rcond,
                    rank,
                    work,
                    &lwork,
                    iwork,
                    info,
                );
                return *info;
            }

            let mut a_t: Vec<T> = match uninitialized_vec(lda_t * n.max(1)) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };
            let mut b_t: Vec<T> = match uninitialized_vec(ldb_t * nrhs.max(1)) {
                Ok(b_t) => b_t,
                Err(_) => return -1011,
            };

            // spans cover the logical elements only: (rows - 1) * ld + cols
            let a_span = if m == 0 || n == 0 { 0 } else { (m - 1) * lda + n };
            let a_slice = from_raw_parts_mut(a, a_span);
            let la = Layout::new_unchecked([m, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([m, n], [1, lda_t as isize], 0);
            if orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).is_err() {
                return -1011;
            }
            // the solution of the wide case lives in rows 0..n of b
            let rows = m.max(n);
            let b_span = if rows == 0 || nrhs == 0 { 0 } else { (rows - 1) * ldb + nrhs };
            let b_slice = from_raw_parts_mut(b, b_span);
            let lb = Layout::new_unchecked([rows, nrhs], [ldb as isize, 1], 0);
            let lb_t = Layout::new_unchecked([rows, nrhs], [1, ldb_t as isize], 0);
            if orderchange_out_r2c_ix2_cpu_serial(&mut b_t, &lb_t, b_slice, &lb).is_err() {
                return -1011;
            }

            func_(
                &(m as _),
                &(n as _),
                &(nrhs as _),
                a_t.as_mut_ptr(),
                &(lda_t as _),
                b_t.as_mut_ptr(),
                &(ldb_t as _),
                s,
                &rcond,
                rank,
                work,
                &lwork,
                iwork,
                info,
            );
            if *info != 0 {
                return *info;
            }
            // a is destroyed by GELSD; only the solution/residuals in b are live
            if orderchange_out_c2r_ix2_cpu_serial(b_slice, &lb, &b_t, &lb_t).is_err() {
                return -1011;
            }
            *info
        },
    }
}

#[duplicate_item(
    T          fn_work      func_      ;
   [f32] [sgelsd_work] [sgelsd_] ;
   [f64] [dgelsd_work] [dgelsd_] ;
)]
impl GELSDDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_gelsd(
        order: FlagOrder,
        m: usize,
        n: usize,
        nrhs: usize,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        s: *mut <T as ComplexFloat>::Real,
        rcond: <T as ComplexFloat>::Real,
        rank: *mut blas_int,
    ) -> blas_int {
        // the query writes the minimum LIWORK into iwork[0]
        let mut iwork: Vec<blas_int> = match uninitialized_vec(1) {
            Ok(iwork) => iwork,
            Err(_) => return -1010,
        };

        let mut info: blas_int = 0;
        let mut work_query: T = <T as Zero>::zero();
        let info_query = fn_work(
            order,
            m,
            n,
            nrhs,
            a,
            lda,
            b,
            ldb,
            s,
            rcond,
            rank,
            &mut work_query,
            -1,
            iwork.as_mut_ptr(),
            &mut info,
        );
        if info_query != 0 {
            return info_query;
        }
        let Some(lwork) = lwork_from_query(work_query) else {
            return -1010;
        };
        let Some(liwork) = liwork_from_query(iwork[0]) else {
            return -1010;
        };

        let mut work: Vec<T> = match uninitialized_vec(lwork) {
            Ok(work) => work,
            Err(_) => return -1010,
        };
        let mut iwork: Vec<blas_int> = match uninitialized_vec(liwork) {
            Ok(iwork) => iwork,
            Err(_) => return -1010,
        };

        fn_work(
            order,
            m,
            n,
            nrhs,
            a,
            lda,
            b,
            ldb,
            s,
            rcond,
            rank,
            work.as_mut_ptr(),
            lwork as blas_int,
            iwork.as_mut_ptr(),
            &mut info,
        )
    }
}

#[duplicate_item(
    T              fn_work      func_      ;
   [Complex<f32>] [cgelsd_work] [cgelsd_] ;
   [Complex<f64>] [zgelsd_work] [zgelsd_] ;
)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fn_work(
    order: FlagOrder,
    m: usize,
    n: usize,
    nrhs: usize,
    a: *mut T,
    lda: usize,
    b: *mut T,
    ldb: usize,
    s: *mut <T as ComplexFloat>::Real,
    rcond: <T as ComplexFloat>::Real,
    rank: *mut blas_int,
    work: *mut T,
    lwork: blas_int,
    rwork: *mut <T as ComplexFloat>::Real,
    iwork: *mut blas_int,
    info: *mut blas_int,
) -> blas_int {
    use lapack_ffi::lapack::func_;

    match order {
        ColMajor => {
            func_(
                &(m as _),
                &(n as _),
                &(nrhs as _),
                a as *mut _,
                &(lda as _),
                b as *mut _,
                &(ldb as _),
                s as *mut _,
                &rcond,
                rank,
                work as *mut _,
                &lwork,
                rwork as *mut _,
                iwork,
                info,
            );
            *info
        },
        RowMajor => {
            // row-major: ld counts the row length; LAPACK's argument indices
            if lda < n {
                *info = -5;
                return *info;
            }
            if ldb < nrhs {
                *info = -7;
                return *info;
            }
            let lda_t = m.max(1);
            let ldb_t = m.max(n).max(1);
            // query: sizes only, the user arrays are not read
            if lwork == -1 {
                func_(
                    &(m as _),
                    &(n as _),
                    &(nrhs as _),
                    a as *mut _,
                    &(lda_t as _),
                    b as *mut _,
                    &(ldb_t as _),
                    s as *mut _,
                    &rcond,
                    rank,
                    work as *mut _,
                    &lwork,
                    rwork as *mut _,
                    iwork,
                    info,
                );
                return *info;
            }

            let mut a_t: Vec<T> = match uninitialized_vec(lda_t * n.max(1)) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };
            let mut b_t: Vec<T> = match uninitialized_vec(ldb_t * nrhs.max(1)) {
                Ok(b_t) => b_t,
                Err(_) => return -1011,
            };

            // spans cover the logical elements only: (rows - 1) * ld + cols
            let a_span = if m == 0 || n == 0 { 0 } else { (m - 1) * lda + n };
            let a_slice = from_raw_parts_mut(a, a_span);
            let la = Layout::new_unchecked([m, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([m, n], [1, lda_t as isize], 0);
            if orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).is_err() {
                return -1011;
            }
            // the solution of the wide case lives in rows 0..n of b
            let rows = m.max(n);
            let b_span = if rows == 0 || nrhs == 0 { 0 } else { (rows - 1) * ldb + nrhs };
            let b_slice = from_raw_parts_mut(b, b_span);
            let lb = Layout::new_unchecked([rows, nrhs], [ldb as isize, 1], 0);
            let lb_t = Layout::new_unchecked([rows, nrhs], [1, ldb_t as isize], 0);
            if orderchange_out_r2c_ix2_cpu_serial(&mut b_t, &lb_t, b_slice, &lb).is_err() {
                return -1011;
            }

            func_(
                &(m as _),
                &(n as _),
                &(nrhs as _),
                a_t.as_mut_ptr() as *mut _,
                &(lda_t as _),
                b_t.as_mut_ptr() as *mut _,
                &(ldb_t as _),
                s as *mut _,
                &rcond,
                rank,
                work as *mut _,
                &lwork,
                rwork as *mut _,
                iwork,
                info,
            );
            if *info != 0 {
                return *info;
            }
            // a is destroyed by GELSD; only the solution/residuals in b are live
            if orderchange_out_c2r_ix2_cpu_serial(b_slice, &lb, &b_t, &lb_t).is_err() {
                return -1011;
            }
            *info
        },
    }
}

#[duplicate_item(
    T              fn_work      func_      ;
   [Complex<f32>] [cgelsd_work] [cgelsd_] ;
   [Complex<f64>] [zgelsd_work] [zgelsd_] ;
)]
impl GELSDDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_gelsd(
        order: FlagOrder,
        m: usize,
        n: usize,
        nrhs: usize,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        s: *mut <T as ComplexFloat>::Real,
        rcond: <T as ComplexFloat>::Real,
        rank: *mut blas_int,
    ) -> blas_int {
        // the query writes the minimum LIWORK and LRWORK into iwork[0]/rwork[0]
        let mut iwork: Vec<blas_int> = match uninitialized_vec(1) {
            Ok(iwork) => iwork,
            Err(_) => return -1010,
        };
        let mut rwork: Vec<<T as ComplexFloat>::Real> = match uninitialized_vec(1) {
            Ok(rwork) => rwork,
            Err(_) => return -1010,
        };

        let mut info: blas_int = 0;
        let mut work_query: T = <T as Zero>::zero();
        let info_query = fn_work(
            order,
            m,
            n,
            nrhs,
            a,
            lda,
            b,
            ldb,
            s,
            rcond,
            rank,
            &mut work_query,
            -1,
            rwork.as_mut_ptr(),
            iwork.as_mut_ptr(),
            &mut info,
        );
        if info_query != 0 {
            return info_query;
        }
        let Some(lwork) = lwork_from_query(work_query) else {
            return -1010;
        };
        let Some(liwork) = liwork_from_query(iwork[0]) else {
            return -1010;
        };
        let Some(lrwork) = lwork_from_query(rwork[0]) else {
            return -1010;
        };

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

        fn_work(
            order,
            m,
            n,
            nrhs,
            a,
            lda,
            b,
            ldb,
            s,
            rcond,
            rank,
            work.as_mut_ptr(),
            lwork as blas_int,
            rwork.as_mut_ptr(),
            iwork.as_mut_ptr(),
            &mut info,
        )
    }
}
