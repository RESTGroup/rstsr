use crate::lapack_ffi;
use crate::DeviceBLAS;
use num::complex::ComplexFloat;
use num::{Complex, Zero};
use rstsr_blas_traits::prelude::*;
use rstsr_blas_traits::util::util_eig::{pack_eigenvalues, unpack_real_eigvecs};
use rstsr_blas_traits::util::util_error::lwork_from_query;
use rstsr_common::prelude_dev::*;

use rstsr_native_impl::prelude_dev::*;

use std::slice::from_raw_parts_mut;

#[duplicate_item(
    T          fn_work      func_    ;
   [f32] [sggev_work] [sggev_];
   [f64] [dggev_work] [dggev_];
)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fn_work(
    order: FlagOrder,
    jobvl: char,
    jobvr: char,
    n: usize,
    a: *mut T,
    lda: usize,
    b: *mut T,
    ldb: usize,
    alphar: *mut T,
    alphai: *mut T,
    beta: *mut T,
    vl: *mut T,
    ldvl: usize,
    vr: *mut T,
    ldvr: usize,
    work: *mut T,
    lwork: blas_int,
    info: *mut blas_int,
) -> blas_int {
    use lapack_ffi::lapack::func_;

    match order {
        ColMajor => {
            func_(
                &(jobvl as _),
                &(jobvr as _),
                &(n as _),
                a,
                &(lda as _),
                b,
                &(ldb as _),
                alphar,
                alphai,
                beta,
                vl,
                &(ldvl as _),
                vr,
                &(ldvr as _),
                work,
                &lwork,
                info,
            );
            *info
        },
        RowMajor => {
            let ncols_vl = if jobvl == 'V' { n } else { 1 };
            let ncols_vr = if jobvr == 'V' { n } else { 1 };
            let lda_t = n.max(1);
            let ldb_t = n.max(1);
            let ldvl_t = n.max(1);
            let ldvr_t = n.max(1);
            // Check leading dimension(s); Fortran (reference-LAPACK) argument indices
            if lda < n {
                *info = -5;
                return *info;
            }
            if ldb < n {
                *info = -7;
                return *info;
            }
            if ldvl < ncols_vl {
                *info = -12;
                return *info;
            }
            if ldvr < ncols_vr {
                *info = -14;
                return *info;
            }
            // Query optimal working array(s) size if requested
            if lwork == -1 {
                func_(
                    &(jobvl as _),
                    &(jobvr as _),
                    &(n as _),
                    a,
                    &(lda_t as _),
                    b,
                    &(ldb_t as _),
                    alphar,
                    alphai,
                    beta,
                    vl,
                    &(ldvl_t as _),
                    vr,
                    &(ldvr_t as _),
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
            let mut vl_t = if jobvl == 'V' {
                match uninitialized_vec(ldvl_t * n.max(1)) {
                    Ok(vl_t) => Some(vl_t),
                    Err(_) => return -1011,
                }
            } else {
                None
            };
            let mut vr_t = if jobvr == 'V' {
                match uninitialized_vec(ldvr_t * n.max(1)) {
                    Ok(vr_t) => Some(vr_t),
                    Err(_) => return -1011,
                }
            } else {
                None
            };

            let a_slice = from_raw_parts_mut(a, n * lda);
            let la = Layout::new_unchecked([n, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([n, n], [1, lda_t as isize], 0);
            let b_slice = from_raw_parts_mut(b, n * ldb);
            let lb = Layout::new_unchecked([n, n], [ldb as isize, 1], 0);
            let lb_t = Layout::new_unchecked([n, n], [1, ldb_t as isize], 0);
            // Transpose input matrices
            if orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).is_err() {
                return -1011;
            }
            if orderchange_out_r2c_ix2_cpu_serial(&mut b_t, &lb_t, b_slice, &lb).is_err() {
                return -1011;
            }

            // Call LAPACK function
            func_(
                &(jobvl as _),
                &(jobvr as _),
                &(n as _),
                a_t.as_mut_ptr(),
                &(lda_t as _),
                b_t.as_mut_ptr(),
                &(ldb_t as _),
                alphar,
                alphai,
                beta,
                vl_t.as_mut().map_or(std::ptr::null_mut(), |v| v.as_mut_ptr()),
                &(ldvl_t as _),
                vr_t.as_mut().map_or(std::ptr::null_mut(), |v| v.as_mut_ptr()),
                &(ldvr_t as _),
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
            if let Some(vl_t) = vl_t {
                let vl_slice = from_raw_parts_mut(vl, n * ldvl);
                let lvl = Layout::new_unchecked([n, n], [ldvl as isize, 1], 0);
                let lvl_t = Layout::new_unchecked([n, n], [1, ldvl_t as isize], 0);
                if orderchange_out_c2r_ix2_cpu_serial(vl_slice, &lvl, &vl_t, &lvl_t).is_err() {
                    return -1011;
                }
            }
            if let Some(vr_t) = vr_t {
                let vr_slice = from_raw_parts_mut(vr, n * ldvr);
                let lvr = Layout::new_unchecked([n, n], [ldvr as isize, 1], 0);
                let lvr_t = Layout::new_unchecked([n, n], [1, ldvr_t as isize], 0);
                if orderchange_out_c2r_ix2_cpu_serial(vr_slice, &lvr, &vr_t, &lvr_t).is_err() {
                    return -1011;
                }
            }
            *info
        },
    }
}

#[duplicate_item(
    T          fn_work      func_    ;
   [f32] [sggev_work] [sggev_];
   [f64] [dggev_work] [dggev_];
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
        alpha: *mut Complex<<T as ComplexFloat>::Real>,
        beta: *mut T,
        vl: *mut Complex<<T as ComplexFloat>::Real>,
        ldvl: usize,
        vr: *mut Complex<<T as ComplexFloat>::Real>,
        ldvr: usize,
    ) -> blas_int {
        // real LAPACK slots (ALPHAR/ALPHAI/BETA and the pair-packed VL/VR),
        // assembled into the complex slots after the call
        let mut alphar: Vec<T> = match uninitialized_vec(n) {
            Ok(alphar) => alphar,
            Err(_) => return -1010,
        };
        let mut alphai: Vec<T> = match uninitialized_vec(n) {
            Ok(alphai) => alphai,
            Err(_) => return -1010,
        };
        let mut beta_real: Vec<T> = match uninitialized_vec(n) {
            Ok(beta_real) => beta_real,
            Err(_) => return -1010,
        };
        let mut vl_real = if jobvl == 'V' {
            match uninitialized_vec(ldvl * n) {
                Ok(vl_real) => Some(vl_real),
                Err(_) => return -1010,
            }
        } else {
            None
        };
        let mut vr_real = if jobvr == 'V' {
            match uninitialized_vec(ldvr * n) {
                Ok(vr_real) => Some(vr_real),
                Err(_) => return -1010,
            }
        } else {
            None
        };
        let ptr_vl_real = vl_real.as_mut().map_or(std::ptr::null_mut(), |v| v.as_mut_ptr());
        let ptr_vr_real = vr_real.as_mut().map_or(std::ptr::null_mut(), |v| v.as_mut_ptr());

        // query optimal workspace size
        let mut info: blas_int = 0;
        let mut work_query: T = <T as Zero>::zero();
        fn_work(
            order,
            jobvl,
            jobvr,
            n,
            a,
            lda,
            b,
            ldb,
            alphar.as_mut_ptr(),
            alphai.as_mut_ptr(),
            beta_real.as_mut_ptr(),
            ptr_vl_real,
            ldvl,
            ptr_vr_real,
            ldvr,
            &mut work_query,
            -1,
            &mut info,
        );
        if info != 0 {
            return info;
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
            jobvl,
            jobvr,
            n,
            a,
            lda,
            b,
            ldb,
            alphar.as_mut_ptr(),
            alphai.as_mut_ptr(),
            beta_real.as_mut_ptr(),
            ptr_vl_real,
            ldvl,
            ptr_vr_real,
            ldvr,
            work.as_mut_ptr(),
            lwork as blas_int,
            &mut info,
        );
        if info != 0 {
            return info;
        }

        // assemble the complex slots
        pack_eigenvalues(&alphar, &alphai, from_raw_parts_mut(alpha, n));
        from_raw_parts_mut(beta, n).copy_from_slice(&beta_real);
        if let Some(vl_real) = vl_real {
            unpack_real_eigvecs(order, n, ldvl, &vl_real, &alphai, from_raw_parts_mut(vl, n * ldvl));
        }
        if let Some(vr_real) = vr_real {
            unpack_real_eigvecs(order, n, ldvr, &vr_real, &alphai, from_raw_parts_mut(vr, n * ldvr));
        }
        info
    }
}

#[duplicate_item(
    T              fn_work      func_    ;
   [Complex<f32>] [cggev_work] [cggev_];
   [Complex<f64>] [zggev_work] [zggev_];
)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fn_work(
    order: FlagOrder,
    jobvl: char,
    jobvr: char,
    n: usize,
    a: *mut T,
    lda: usize,
    b: *mut T,
    ldb: usize,
    alpha: *mut T,
    beta: *mut <T as ComplexFloat>::Real,
    vl: *mut T,
    ldvl: usize,
    vr: *mut T,
    ldvr: usize,
    work: *mut T,
    lwork: blas_int,
    rwork: *mut <T as ComplexFloat>::Real,
    info: *mut blas_int,
) -> blas_int {
    use lapack_ffi::lapack::func_;

    match order {
        ColMajor => {
            func_(
                &(jobvl as _),
                &(jobvr as _),
                &(n as _),
                a as *mut _,
                &(lda as _),
                b as *mut _,
                &(ldb as _),
                alpha as *mut _,
                beta as *mut _,
                vl as *mut _,
                &(ldvl as _),
                vr as *mut _,
                &(ldvr as _),
                work as *mut _,
                &lwork,
                rwork as *mut _,
                info,
            );
            *info
        },
        RowMajor => {
            let ncols_vl = if jobvl == 'V' { n } else { 1 };
            let ncols_vr = if jobvr == 'V' { n } else { 1 };
            let lda_t = n.max(1);
            let ldb_t = n.max(1);
            let ldvl_t = n.max(1);
            let ldvr_t = n.max(1);
            // Check leading dimension(s); Fortran (reference-LAPACK) argument indices
            if lda < n {
                *info = -5;
                return *info;
            }
            if ldb < n {
                *info = -7;
                return *info;
            }
            if ldvl < ncols_vl {
                *info = -11;
                return *info;
            }
            if ldvr < ncols_vr {
                *info = -13;
                return *info;
            }
            // Query optimal working array(s) size if requested
            if lwork == -1 {
                func_(
                    &(jobvl as _),
                    &(jobvr as _),
                    &(n as _),
                    a as *mut _,
                    &(lda_t as _),
                    b as *mut _,
                    &(ldb_t as _),
                    alpha as *mut _,
                    beta as *mut _,
                    vl as *mut _,
                    &(ldvl_t as _),
                    vr as *mut _,
                    &(ldvr_t as _),
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
            let mut vl_t = if jobvl == 'V' {
                match uninitialized_vec(ldvl_t * n.max(1)) {
                    Ok(vl_t) => Some(vl_t),
                    Err(_) => return -1011,
                }
            } else {
                None
            };
            let mut vr_t = if jobvr == 'V' {
                match uninitialized_vec(ldvr_t * n.max(1)) {
                    Ok(vr_t) => Some(vr_t),
                    Err(_) => return -1011,
                }
            } else {
                None
            };

            let a_slice = from_raw_parts_mut(a, n * lda);
            let la = Layout::new_unchecked([n, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([n, n], [1, lda_t as isize], 0);
            let b_slice = from_raw_parts_mut(b, n * ldb);
            let lb = Layout::new_unchecked([n, n], [ldb as isize, 1], 0);
            let lb_t = Layout::new_unchecked([n, n], [1, ldb_t as isize], 0);
            // Transpose input matrices
            if orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).is_err() {
                return -1011;
            }
            if orderchange_out_r2c_ix2_cpu_serial(&mut b_t, &lb_t, b_slice, &lb).is_err() {
                return -1011;
            }

            // Call LAPACK function
            func_(
                &(jobvl as _),
                &(jobvr as _),
                &(n as _),
                a_t.as_mut_ptr() as *mut _,
                &(lda_t as _),
                b_t.as_mut_ptr() as *mut _,
                &(ldb_t as _),
                alpha as *mut _,
                beta as *mut _,
                vl_t.as_mut().map_or(std::ptr::null_mut(), |v| v.as_mut_ptr()) as *mut _,
                &(ldvl_t as _),
                vr_t.as_mut().map_or(std::ptr::null_mut(), |v| v.as_mut_ptr()) as *mut _,
                &(ldvr_t as _),
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
            if let Some(vl_t) = vl_t {
                let vl_slice = from_raw_parts_mut(vl, n * ldvl);
                let lvl = Layout::new_unchecked([n, n], [ldvl as isize, 1], 0);
                let lvl_t = Layout::new_unchecked([n, n], [1, ldvl_t as isize], 0);
                if orderchange_out_c2r_ix2_cpu_serial(vl_slice, &lvl, &vl_t, &lvl_t).is_err() {
                    return -1011;
                }
            }
            if let Some(vr_t) = vr_t {
                let vr_slice = from_raw_parts_mut(vr, n * ldvr);
                let lvr = Layout::new_unchecked([n, n], [ldvr as isize, 1], 0);
                let lvr_t = Layout::new_unchecked([n, n], [1, ldvr_t as isize], 0);
                if orderchange_out_c2r_ix2_cpu_serial(vr_slice, &lvr, &vr_t, &lvr_t).is_err() {
                    return -1011;
                }
            }
            *info
        },
    }
}

#[duplicate_item(
    T              fn_work      func_    ;
   [Complex<f32>] [cggev_work] [cggev_];
   [Complex<f64>] [zggev_work] [zggev_];
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
        alpha: *mut Complex<<T as ComplexFloat>::Real>,
        beta: *mut T,
        vl: *mut Complex<<T as ComplexFloat>::Real>,
        ldvl: usize,
        vr: *mut Complex<<T as ComplexFloat>::Real>,
        ldvr: usize,
    ) -> blas_int {
        // complex rwork: documented fixed size (no query), allocated before the work query
        let lrwork = (8 * n).max(1);
        let mut rwork: Vec<<T as ComplexFloat>::Real> = match uninitialized_vec(lrwork) {
            Ok(rwork) => rwork,
            Err(_) => return -1010,
        };

        // query optimal workspace size
        let mut info: blas_int = 0;
        let mut work_query: T = <T as Zero>::zero();
        fn_work(
            order,
            jobvl,
            jobvr,
            n,
            a,
            lda,
            b,
            ldb,
            alpha as *mut _,
            beta as *mut _,
            vl as *mut _,
            ldvl,
            vr as *mut _,
            ldvr,
            &mut work_query,
            -1,
            rwork.as_mut_ptr(),
            &mut info,
        );
        if info != 0 {
            return info;
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
            jobvl,
            jobvr,
            n,
            a,
            lda,
            b,
            ldb,
            alpha as *mut _,
            beta as *mut _,
            vl as *mut _,
            ldvl,
            vr as *mut _,
            ldvr,
            work.as_mut_ptr(),
            lwork as blas_int,
            rwork.as_mut_ptr(),
            &mut info,
        )
    }
}
