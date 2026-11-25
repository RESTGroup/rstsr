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
   [f32] [sgesvd_work] [sgesvd_];
   [f64] [dgesvd_work] [dgesvd_];
)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fn_work(
    order: FlagOrder,
    jobu: char,
    jobvt: char,
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
    info: *mut blas_int,
) -> blas_int {
    use lapack_ffi::lapack::func_;

    match order {
        ColMajor => {
            func_(
                &(jobu as _),
                &(jobvt as _),
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
                info,
            );
            *info
        },
        RowMajor => {
            let nrows_u = match jobu {
                'A' | 'S' => m,
                _ => 0,
            };
            let ncols_u = match jobu {
                'A' => m,
                'S' => m.min(n),
                _ => 0,
            };
            let nrows_vt = match jobvt {
                'A' => n,
                'S' => m.min(n),
                _ => 0,
            };
            let ncols_vt = match jobvt {
                'A' | 'S' => n,
                _ => 0,
            };
            let lda_t = m.max(1);
            let ldu_t = nrows_u.max(1);
            let ldvt_t = nrows_vt.max(1);
            // Check leading dimension(s); Fortran (reference-LAPACK) argument indices
            if lda < n {
                *info = -6;
                return *info;
            }
            if (jobu == 'A' || jobu == 'S') && ldu < ncols_u {
                *info = -9;
                return *info;
            }
            if (jobvt == 'A' || jobvt == 'S') && ldvt < ncols_vt {
                *info = -11;
                return *info;
            }
            // Query optimal working array(s) size if requested
            if lwork == -1 {
                func_(
                    &(jobu as _),
                    &(jobvt as _),
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
                    info,
                );
                return *info;
            }

            // Allocate memory for temporary array(s)
            let mut a_t: Vec<T> = match uninitialized_vec(lda_t * n.max(1)) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };
            let mut u_t = if jobu == 'A' || jobu == 'S' {
                match uninitialized_vec(ldu_t * ncols_u.max(1)) {
                    Ok(u_t) => Some(u_t),
                    Err(_) => return -1011,
                }
            } else {
                None
            };
            let mut vt_t = if jobvt == 'A' || jobvt == 'S' {
                match uninitialized_vec(ldvt_t * n.max(1)) {
                    Ok(vt_t) => Some(vt_t),
                    Err(_) => return -1011,
                }
            } else {
                None
            };

            // transpose in
            let a_slice = from_raw_parts_mut(a, m * lda);
            let la = Layout::new_unchecked([m, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([m, n], [1, lda_t as isize], 0);
            // Transpose input matrices
            if orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).is_err() {
                return -1011;
            }

            // call with transposed buffers; null where the job produces nothing
            func_(
                &(jobu as _),
                &(jobvt as _),
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
                info,
            );
            if *info != 0 {
                return *info;
            }

            // transpose out
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
   [f32] [sgesvd_work] [sgesvd_];
   [f64] [dgesvd_work] [dgesvd_];
)]
impl GESVDDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_gesvd(
        order: FlagOrder,
        jobu: char,
        jobvt: char,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        s: *mut T,
        u: *mut T,
        ldu: usize,
        vt: *mut T,
        ldvt: usize,
        superb: *mut T,
    ) -> blas_int {
        // query optimal workspace size
        let mut info: blas_int = 0;
        let mut work_query: T = <T as Zero>::zero();
        let info_query = fn_work(order, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, &mut work_query, -1, &mut info);
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
        let info = fn_work(
            order,
            jobu,
            jobvt,
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
            &mut info,
        );

        // Backup superb from workspace (unconditional, per LAPACKE)
        for i in 0..m.min(n).saturating_sub(1) {
            superb.add(i).write(work[i + 1]);
        }
        info
    }
}

#[duplicate_item(
    T              fn_work        func_    ;
   [Complex<f32>] [cgesvd_work] [cgesvd_];
   [Complex<f64>] [zgesvd_work] [zgesvd_];
)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fn_work(
    order: FlagOrder,
    jobu: char,
    jobvt: char,
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
    info: *mut blas_int,
) -> blas_int {
    use lapack_ffi::lapack::func_;

    match order {
        ColMajor => {
            func_(
                &(jobu as _),
                &(jobvt as _),
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
                info,
            );
            *info
        },
        RowMajor => {
            let nrows_u = match jobu {
                'A' | 'S' => m,
                _ => 0,
            };
            let ncols_u = match jobu {
                'A' => m,
                'S' => m.min(n),
                _ => 0,
            };
            let nrows_vt = match jobvt {
                'A' => n,
                'S' => m.min(n),
                _ => 0,
            };
            let ncols_vt = match jobvt {
                'A' | 'S' => n,
                _ => 0,
            };
            let lda_t = m.max(1);
            let ldu_t = nrows_u.max(1);
            let ldvt_t = nrows_vt.max(1);
            // Check leading dimension(s); Fortran (reference-LAPACK) argument indices
            if lda < n {
                *info = -6;
                return *info;
            }
            if (jobu == 'A' || jobu == 'S') && ldu < ncols_u {
                *info = -9;
                return *info;
            }
            if (jobvt == 'A' || jobvt == 'S') && ldvt < ncols_vt {
                *info = -11;
                return *info;
            }
            // Query optimal working array(s) size if requested
            if lwork == -1 {
                func_(
                    &(jobu as _),
                    &(jobvt as _),
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
                    info,
                );
                return *info;
            }

            // Allocate memory for temporary array(s)
            let mut a_t: Vec<T> = match uninitialized_vec(lda_t * n.max(1)) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };
            let mut u_t = if jobu == 'A' || jobu == 'S' {
                match uninitialized_vec(ldu_t * ncols_u.max(1)) {
                    Ok(u_t) => Some(u_t),
                    Err(_) => return -1011,
                }
            } else {
                None
            };
            let mut vt_t = if jobvt == 'A' || jobvt == 'S' {
                match uninitialized_vec(ldvt_t * n.max(1)) {
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
                &(jobu as _),
                &(jobvt as _),
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
   [Complex<f32>] [cgesvd_work] [cgesvd_];
   [Complex<f64>] [zgesvd_work] [zgesvd_];
)]
impl GESVDDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_gesvd(
        order: FlagOrder,
        jobu: char,
        jobvt: char,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        s: *mut <T as ComplexFloat>::Real,
        u: *mut T,
        ldu: usize,
        vt: *mut T,
        ldvt: usize,
        superb: *mut <T as ComplexFloat>::Real,
    ) -> blas_int {
        // complex rwork: documented fixed minimum, allocated before query
        let mut rwork: Vec<<T as ComplexFloat>::Real> = match uninitialized_vec((5 * m.min(n)).max(1)) {
            Ok(rwork) => rwork,
            Err(_) => return -1010,
        };

        // query optimal workspace size
        let mut info: blas_int = 0;
        let mut work_query: T = <T as Zero>::zero();
        let info_query = fn_work(
            order,
            jobu,
            jobvt,
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
        let info = fn_work(
            order,
            jobu,
            jobvt,
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
            &mut info,
        );

        // Backup superb from rwork (unconditional, per LAPACKE)
        #[allow(clippy::needless_range_loop)]
        for i in 0..m.min(n).saturating_sub(1) {
            superb.add(i).write(rwork[i]);
        }
        info
    }
}
