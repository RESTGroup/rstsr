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
   [f32] [sgesdd_];
   [f64] [dgesdd_];
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
        use rstsr_lapack_ffi::lapack::func_;

        unsafe fn raise_info(mut info: blas_int) -> blas_int {
            xerbla_(c"gesdd".as_ptr() as _, &mut info as *mut _ as *mut _);
            return if info < 0 { info - 1 } else { info };
        }

        // Query optimal working array(s) size
        let mut info = 0;
        let lwork = -1;
        let mut work_query = 0.0;
        func_(
            &(jobz as _),
            &(m as _),
            &(n as _),
            a,
            &(m.max(n) as _),
            s,
            u,
            &(m.max(n) as _),
            vt,
            &(m.max(n) as _),
            &mut work_query,
            &lwork,
            std::ptr::null_mut(),
            &mut info,
        );
        if info != 0 {
            return raise_info(info);
        }
        let lwork = work_query as usize;
        let liwork = 8 * m.min(n);

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
                work.as_mut_ptr(),
                &(lwork as _),
                iwork.as_mut_ptr(),
                &mut info,
            );
            if info != 0 {
                return raise_info(info);
            }
        } else {
            let lda_t = m.max(1);
            let nrows_u = if jobz == 'A' || jobz == 'S' || (jobz == 'O' && m < n) { m } else { 1 };
            let ncols_u = if jobz == 'A' || (jobz == 'O' && m < n) {
                m
            } else if jobz == 'S' {
                m.min(n)
            } else {
                1
            };
            let nrows_vt = if jobz == 'A' || (jobz == 'O' && m >= n) {
                n
            } else if jobz == 'S' {
                m.min(n)
            } else {
                1
            };
            let ldu_t = nrows_u.max(1);
            let ldvt_t = nrows_vt.max(1);

            // Transpose input matrices
            let mut a_t: Vec<T> = match uninitialized_vec(m * n) {
                Ok(a_t) => a_t,
                Err(_) => return LAPACK_TRANSPOSE_MEMORY_ERROR,
            };
            let a_slice = from_raw_parts_mut(a, m * lda);
            let la = Layout::new_unchecked([m, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([m, n], [1, lda_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).unwrap();

            let mut u_t = if jobz == 'A' || jobz == 'S' || (jobz == 'O' && m < n) {
                match uninitialized_vec(nrows_u * ncols_u) {
                    Ok(u_t) => Some(u_t),
                    Err(_) => return LAPACK_TRANSPOSE_MEMORY_ERROR,
                }
            } else {
                None
            };

            let mut vt_t = if jobz == 'A' || jobz == 'S' || (jobz == 'O' && m >= n) {
                match uninitialized_vec(nrows_vt * n) {
                    Ok(vt_t) => Some(vt_t),
                    Err(_) => return LAPACK_TRANSPOSE_MEMORY_ERROR,
                }
            } else {
                None
            };

            // Call LAPACK function and adjust info
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
                work.as_mut_ptr(),
                &(lwork as _),
                iwork.as_mut_ptr(),
                &mut info,
            );
            if info != 0 {
                return raise_info(info);
            }

            // Transpose output matrices
            orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).unwrap();

            if let Some(u_t) = u_t {
                let u_slice = from_raw_parts_mut(u, nrows_u * ldu);
                let lu = Layout::new_unchecked([nrows_u, ncols_u], [ldu as isize, 1], 0);
                let lu_t = Layout::new_unchecked([nrows_u, ncols_u], [1, ldu_t as isize], 0);
                orderchange_out_c2r_ix2_cpu_serial(u_slice, &lu, &u_t, &lu_t).unwrap();
            }

            if let Some(vt_t) = vt_t {
                let vt_slice = from_raw_parts_mut(vt, nrows_vt * ldvt);
                let lvt = Layout::new_unchecked([nrows_vt, n], [ldvt as isize, 1], 0);
                let lvt_t = Layout::new_unchecked([nrows_vt, n], [1, ldvt_t as isize], 0);
                orderchange_out_c2r_ix2_cpu_serial(vt_slice, &lvt, &vt_t, &lvt_t).unwrap();
            }
        }
        return info;
    }
}

#[duplicate_item(
    T              func_   ;
   [Complex<f32>] [cgesdd_];
   [Complex<f64>] [zgesdd_];
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
        use rstsr_lapack_ffi::lapack::func_;

        unsafe fn raise_info(mut info: blas_int) -> blas_int {
            xerbla_(c"gesdd".as_ptr() as _, &mut info as *mut _ as *mut _);
            return if info < 0 { info - 1 } else { info };
        }

        // Query optimal working array(s) size
        let mut info = 0;
        let lwork = -1;
        let lrwork =
            if jobz == 'N' { 7 * m.min(n) } else { m.min(n) * (5 * m.min(n) + 7).max(2 * m.max(n) + 2 * m.min(n) + 1) };
        let mut work_query = Complex::new(0.0, 0.0);
        let mut rwork_query = 0.0;
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
            &mut work_query as *mut _ as *mut _,
            &lwork,
            &mut rwork_query as *mut _ as *mut _,
            std::ptr::null_mut(),
            &mut info,
        );
        if info != 0 {
            return raise_info(info);
        }
        let lwork = work_query.re as usize;
        let liwork = 8 * m.min(n);

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
                work.as_mut_ptr() as *mut _,
                &(lwork as _),
                rwork.as_mut_ptr() as *mut _,
                iwork.as_mut_ptr(),
                &mut info,
            );
            if info != 0 {
                return raise_info(info);
            }
        } else {
            let lda_t = m.max(1);
            let nrows_u = if jobz == 'A' || jobz == 'S' || (jobz == 'O' && m < n) { m } else { 1 };
            let ncols_u = if jobz == 'A' || (jobz == 'O' && m < n) {
                m
            } else if jobz == 'S' {
                m.min(n)
            } else {
                1
            };
            let nrows_vt = if jobz == 'A' || (jobz == 'O' && m >= n) {
                n
            } else if jobz == 'S' {
                m.min(n)
            } else {
                1
            };
            let ldu_t = nrows_u.max(1);
            let ldvt_t = nrows_vt.max(1);

            // Transpose input matrices
            let mut a_t: Vec<T> = match uninitialized_vec(m * n) {
                Ok(a_t) => a_t,
                Err(_) => return LAPACK_TRANSPOSE_MEMORY_ERROR,
            };
            let a_slice = from_raw_parts_mut(a, m * lda);
            let la = Layout::new_unchecked([m, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([m, n], [1, lda_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).unwrap();

            let mut u_t = if jobz == 'A' || jobz == 'S' || (jobz == 'O' && m < n) {
                match uninitialized_vec(nrows_u * ncols_u) {
                    Ok(u_t) => Some(u_t),
                    Err(_) => return LAPACK_TRANSPOSE_MEMORY_ERROR,
                }
            } else {
                None
            };

            let mut vt_t = if jobz == 'A' || jobz == 'S' || (jobz == 'O' && m >= n) {
                match uninitialized_vec(nrows_vt * n) {
                    Ok(vt_t) => Some(vt_t),
                    Err(_) => return LAPACK_TRANSPOSE_MEMORY_ERROR,
                }
            } else {
                None
            };

            // Call LAPACK function and adjust info
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
                work.as_mut_ptr() as *mut _,
                &(lwork as _),
                rwork.as_mut_ptr() as *mut _,
                iwork.as_mut_ptr(),
                &mut info,
            );
            if info != 0 {
                return raise_info(info);
            }

            // Transpose output matrices
            orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).unwrap();

            if let Some(u_t) = u_t {
                let u_slice = from_raw_parts_mut(u, nrows_u * ldu);
                let lu = Layout::new_unchecked([nrows_u, ncols_u], [ldu as isize, 1], 0);
                let lu_t = Layout::new_unchecked([nrows_u, ncols_u], [1, ldu_t as isize], 0);
                orderchange_out_c2r_ix2_cpu_serial(u_slice, &lu, &u_t, &lu_t).unwrap();
            }

            if let Some(vt_t) = vt_t {
                let vt_slice = from_raw_parts_mut(vt, nrows_vt * ldvt);
                let lvt = Layout::new_unchecked([nrows_vt, n], [ldvt as isize, 1], 0);
                let lvt_t = Layout::new_unchecked([nrows_vt, n], [1, ldvt_t as isize], 0);
                orderchange_out_c2r_ix2_cpu_serial(vt_slice, &lvt, &vt_t, &lvt_t).unwrap();
            }
        }
        return info;
    }
}
