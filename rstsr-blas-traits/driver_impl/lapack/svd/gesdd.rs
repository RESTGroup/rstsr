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
    T          fn_work        func_    ;
   [f32] [sgesdd_work] [sgesdd_];
   [f64] [dgesdd_work] [dgesdd_];
)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fn_work(
    order: FlagOrder,
    jobz: char,
    m: usize,
    n: usize,
    a: *mut T,
    lda: usize,
    s: *mut T,
    u: *mut T,
    ldu: usize,
    vt: *mut T,
    ldvt: usize,
    work: *mut T,
    lwork: blas_int,
    iwork: *mut blas_int,
    info: *mut blas_int,
) -> blas_int {
    use lapack_ffi::lapack::func_;

    match order {
        ColMajor => {
            func_(
                &(jobz as _),
                &(m as _),
                &(n as _),
                a,
                &(lda as _),
                s,
                u,
                &(ldu as _),
                vt,
                &(ldvt as _),
                work,
                &lwork,
                iwork,
                info,
            );
            *info
        },
        RowMajor => {
            let nrows_u = match jobz {
                'A' | 'S' => m,
                'O' if m < n => m,
                _ => 0,
            };
            let ncols_u = match jobz {
                'A' => m,
                'O' if m < n => m,
                'S' => m.min(n),
                _ => 0,
            };
            let nrows_vt = match jobz {
                'A' => n,
                'S' => m.min(n),
                'O' if m >= n => n,
                _ => 0,
            };
            let ncols_vt = match jobz {
                'A' | 'S' => n,
                'O' if m >= n => n,
                _ => 0,
            };
            let lda_t = m.max(1);
            let ldu_t = nrows_u.max(1);
            let ldvt_t = nrows_vt.max(1);
            // Check leading dimension(s); Fortran (reference-LAPACK) argument indices
            if lda < n {
                *info = -5;
                return *info;
            }
            if ldu < ncols_u {
                *info = -8;
                return *info;
            }
            if ldvt < ncols_vt {
                *info = -10;
                return *info;
            }
            // Query optimal working array(s) size if requested
            if lwork == -1 {
                func_(
                    &(jobz as _),
                    &(m as _),
                    &(n as _),
                    a,
                    &(lda_t as _),
                    s,
                    std::ptr::null_mut(),
                    &(ldu_t as _),
                    std::ptr::null_mut(),
                    &(ldvt_t as _),
                    work,
                    &lwork,
                    iwork,
                    info,
                );
                return *info;
            }

            // Allocate memory for temporary array(s)
            let mut a_t: Vec<T> = match uninitialized_vec(lda_t * n.max(1)) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };
            let mut u_t = if nrows_u > 0 {
                match uninitialized_vec(ldu_t * ncols_u.max(1)) {
                    Ok(u_t) => Some(u_t),
                    Err(_) => return -1011,
                }
            } else {
                None
            };
            let mut vt_t = if nrows_vt > 0 {
                match uninitialized_vec(ldvt_t * ncols_vt.max(1)) {
                    Ok(vt_t) => Some(vt_t),
                    Err(_) => return -1011,
                }
            } else {
                None
            };

            let a_slice = from_raw_parts_mut(a, m * lda);
            let la = Layout::new_unchecked([m, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([m, n], [1, lda_t as isize], 0);
            // Transpose input matrices
            if orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).is_err() {
                return -1011;
            }

            func_(
                &(jobz as _),
                &(m as _),
                &(n as _),
                a_t.as_mut_ptr(),
                &(lda_t as _),
                s,
                u_t.as_mut().map_or(std::ptr::null_mut(), |v| v.as_mut_ptr()),
                &(ldu_t as _),
                vt_t.as_mut().map_or(std::ptr::null_mut(), |v| v.as_mut_ptr()),
                &(ldvt_t as _),
                work,
                &lwork,
                iwork,
                info,
            );
            if *info != 0 {
                return *info;
            }

            // Transpose output matrices
            if orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).is_err() {
                return -1011;
            }
            if let Some(u_t) = u_t {
                let u_slice = from_raw_parts_mut(u, nrows_u * ldu);
                let lu = Layout::new_unchecked([nrows_u, ncols_u], [ldu as isize, 1], 0);
                let lu_t = Layout::new_unchecked([nrows_u, ncols_u], [1, ldu_t as isize], 0);
                if orderchange_out_c2r_ix2_cpu_serial(u_slice, &lu, &u_t, &lu_t).is_err() {
                    return -1011;
                }
            }
            if let Some(vt_t) = vt_t {
                let vt_slice = from_raw_parts_mut(vt, nrows_vt * ldvt);
                let lvt = Layout::new_unchecked([nrows_vt, n], [ldvt as isize, 1], 0);
                let lvt_t = Layout::new_unchecked([nrows_vt, n], [1, ldvt_t as isize], 0);
                if orderchange_out_c2r_ix2_cpu_serial(vt_slice, &lvt, &vt_t, &lvt_t).is_err() {
                    return -1011;
                }
            }
            *info
        },
    }
}

#[duplicate_item(
    T          fn_work        func_    ;
   [f32] [sgesdd_work] [sgesdd_];
   [f64] [dgesdd_work] [dgesdd_];
)]
impl GESDDDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_gesdd(
        order: FlagOrder,
        jobz: char,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        s: *mut T,
        u: *mut T,
        ldu: usize,
        vt: *mut T,
        ldvt: usize,
    ) -> blas_int {
        // iwork: documented fixed minimum, allocated before the work query
        let mut iwork: Vec<blas_int> = match uninitialized_vec((8 * m.min(n)).max(1)) {
            Ok(iwork) => iwork,
            Err(_) => return -1010,
        };

        // query optimal workspace size
        let mut info: blas_int = 0;
        let mut work_query: T = <T as Zero>::zero();
        let info_query =
            fn_work(order, jobz, m, n, a, lda, s, u, ldu, vt, ldvt, &mut work_query, -1, iwork.as_mut_ptr(), &mut info);
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
            jobz,
            m,
            n,
            a,
            lda,
            s,
            u,
            ldu,
            vt,
            ldvt,
            work.as_mut_ptr(),
            lwork as blas_int,
            iwork.as_mut_ptr(),
            &mut info,
        )
    }
}

#[duplicate_item(
    T              fn_work        func_    ;
   [Complex<f32>] [cgesdd_work] [cgesdd_];
   [Complex<f64>] [zgesdd_work] [zgesdd_];
)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fn_work(
    order: FlagOrder,
    jobz: char,
    m: usize,
    n: usize,
    a: *mut T,
    lda: usize,
    s: *mut <T as ComplexFloat>::Real,
    u: *mut T,
    ldu: usize,
    vt: *mut T,
    ldvt: usize,
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
                &(jobz as _),
                &(m as _),
                &(n as _),
                a as *mut _,
                &(lda as _),
                s as *mut _,
                u as *mut _,
                &(ldu as _),
                vt as *mut _,
                &(ldvt as _),
                work as *mut _,
                &lwork,
                rwork as *mut _,
                iwork,
                info,
            );
            *info
        },
        RowMajor => {
            let nrows_u = match jobz {
                'A' | 'S' => m,
                'O' if m < n => m,
                _ => 0,
            };
            let ncols_u = match jobz {
                'A' => m,
                'O' if m < n => m,
                'S' => m.min(n),
                _ => 0,
            };
            let nrows_vt = match jobz {
                'A' => n,
                'S' => m.min(n),
                'O' if m >= n => n,
                _ => 0,
            };
            let ncols_vt = match jobz {
                'A' | 'S' => n,
                'O' if m >= n => n,
                _ => 0,
            };
            let lda_t = m.max(1);
            let ldu_t = nrows_u.max(1);
            let ldvt_t = nrows_vt.max(1);
            // Check leading dimension(s); Fortran (reference-LAPACK) argument indices
            if lda < n {
                *info = -5;
                return *info;
            }
            if ldu < ncols_u {
                *info = -8;
                return *info;
            }
            if ldvt < ncols_vt {
                *info = -10;
                return *info;
            }
            // Query optimal working array(s) size if requested
            if lwork == -1 {
                func_(
                    &(jobz as _),
                    &(m as _),
                    &(n as _),
                    a as *mut _,
                    &(lda_t as _),
                    s as *mut _,
                    std::ptr::null_mut(),
                    &(ldu_t as _),
                    std::ptr::null_mut(),
                    &(ldvt_t as _),
                    work as *mut _,
                    &lwork,
                    rwork as *mut _,
                    iwork,
                    info,
                );
                return *info;
            }

            // Allocate memory for temporary array(s)
            let mut a_t: Vec<T> = match uninitialized_vec(lda_t * n.max(1)) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };
            let mut u_t = if nrows_u > 0 {
                match uninitialized_vec(ldu_t * ncols_u.max(1)) {
                    Ok(u_t) => Some(u_t),
                    Err(_) => return -1011,
                }
            } else {
                None
            };
            let mut vt_t = if nrows_vt > 0 {
                match uninitialized_vec(ldvt_t * ncols_vt.max(1)) {
                    Ok(vt_t) => Some(vt_t),
                    Err(_) => return -1011,
                }
            } else {
                None
            };

            let a_slice = from_raw_parts_mut(a, m * lda);
            let la = Layout::new_unchecked([m, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([m, n], [1, lda_t as isize], 0);
            // Transpose input matrices
            if orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).is_err() {
                return -1011;
            }

            func_(
                &(jobz as _),
                &(m as _),
                &(n as _),
                a_t.as_mut_ptr() as *mut _,
                &(lda_t as _),
                s as *mut _,
                u_t.as_mut().map_or(std::ptr::null_mut(), |v| v.as_mut_ptr()) as *mut _,
                &(ldu_t as _),
                vt_t.as_mut().map_or(std::ptr::null_mut(), |v| v.as_mut_ptr()) as *mut _,
                &(ldvt_t as _),
                work as *mut _,
                &lwork,
                rwork as *mut _,
                iwork,
                info,
            );
            if *info != 0 {
                return *info;
            }

            // Transpose output matrices
            if orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).is_err() {
                return -1011;
            }
            if let Some(u_t) = u_t {
                let u_slice = from_raw_parts_mut(u, nrows_u * ldu);
                let lu = Layout::new_unchecked([nrows_u, ncols_u], [ldu as isize, 1], 0);
                let lu_t = Layout::new_unchecked([nrows_u, ncols_u], [1, ldu_t as isize], 0);
                if orderchange_out_c2r_ix2_cpu_serial(u_slice, &lu, &u_t, &lu_t).is_err() {
                    return -1011;
                }
            }
            if let Some(vt_t) = vt_t {
                let vt_slice = from_raw_parts_mut(vt, nrows_vt * ldvt);
                let lvt = Layout::new_unchecked([nrows_vt, n], [ldvt as isize, 1], 0);
                let lvt_t = Layout::new_unchecked([nrows_vt, n], [1, ldvt_t as isize], 0);
                if orderchange_out_c2r_ix2_cpu_serial(vt_slice, &lvt, &vt_t, &lvt_t).is_err() {
                    return -1011;
                }
            }
            *info
        },
    }
}

#[duplicate_item(
    T              fn_work        func_    ;
   [Complex<f32>] [cgesdd_work] [cgesdd_];
   [Complex<f64>] [zgesdd_work] [zgesdd_];
)]
impl GESDDDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_gesdd(
        order: FlagOrder,
        jobz: char,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        s: *mut <T as ComplexFloat>::Real,
        u: *mut T,
        ldu: usize,
        vt: *mut T,
        ldvt: usize,
    ) -> blas_int {
        let min_mn = m.min(n);
        let max_mn = m.max(n);

        // iwork: documented fixed minimum, allocated before the work query
        let mut iwork: Vec<blas_int> = match uninitialized_vec((8 * min_mn).max(1)) {
            Ok(iwork) => iwork,
            Err(_) => return -1010,
        };
        // complex rwork: LAPACK >= 3.6 documented formula (no query)
        let lrwork = if jobz == 'N' {
            (7 * min_mn).max(1)
        } else {
            (min_mn * (5 * min_mn + 7).max(2 * max_mn + 2 * min_mn + 1)).max(1)
        };
        let mut rwork: Vec<<T as ComplexFloat>::Real> = match uninitialized_vec(lrwork) {
            Ok(rwork) => rwork,
            Err(_) => return -1010,
        };

        // query optimal workspace size
        let mut info: blas_int = 0;
        let mut work_query: T = <T as Zero>::zero();
        let info_query = fn_work(
            order,
            jobz,
            m,
            n,
            a,
            lda,
            s,
            u,
            ldu,
            vt,
            ldvt,
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

        // allocate memory for work array
        let mut work: Vec<T> = match uninitialized_vec(lwork) {
            Ok(work) => work,
            Err(_) => return -1010,
        };

        // call LAPACK routine
        fn_work(
            order,
            jobz,
            m,
            n,
            a,
            lda,
            s,
            u,
            ldu,
            vt,
            ldvt,
            work.as_mut_ptr(),
            lwork as blas_int,
            rwork.as_mut_ptr(),
            iwork.as_mut_ptr(),
            &mut info,
        )
    }
}
