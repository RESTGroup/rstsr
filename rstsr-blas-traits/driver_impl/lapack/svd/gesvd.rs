use crate::DeviceBLAS;
use num::complex::ComplexFloat;
use num::{Complex, Zero};
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude_dev::*;
use rstsr_lapack_ffi::blas::xerbla_;
use rstsr_lapack_ffi::lapacke::{LAPACK_TRANSPOSE_MEMORY_ERROR, LAPACK_WORK_MEMORY_ERROR};
use rstsr_native_impl::prelude_dev::*;
use std::slice::from_raw_parts_mut;

#[duplicate_item(
    T     func_   ;
   [f32] [sgesvd_];
   [f64] [dgesvd_];
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
        use rstsr_lapack_ffi::lapack::func_;

        unsafe fn raise_info(mut info: blas_int) -> blas_int {
            xerbla_(c"gesvd".as_ptr() as _, &mut info as *mut _ as *mut _);
            return if info < 0 { info - 1 } else { info };
        }

        // Query optimal working array size
        let mut info = 0;
        let lwork = -1;
        let mut work_query = 0.0;
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
            &mut work_query,
            &lwork,
            &mut info,
        );
        if info != 0 {
            return raise_info(info);
        }
        let lwork = work_query as usize;

        // Allocate memory for work array
        let mut work: Vec<T> = match uninitialized_vec(lwork) {
            Ok(work) => work,
            Err(_) => return LAPACK_WORK_MEMORY_ERROR,
        };

        if order == ColMajor {
            // Call LAPACK function
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
                work.as_mut_ptr(),
                &(lwork as _),
                &mut info,
            );
            if info != 0 {
                return raise_info(info);
            }
        } else {
            let lda_t = m.max(1);
            let nrows_u = if jobu == 'A' || jobu == 'S' { m } else { 1 };
            let ncols_u = if jobu == 'A' {
                m
            } else if jobu == 'S' {
                m.min(n)
            } else {
                1
            };
            let nrows_vt = if jobvt == 'A' {
                n
            } else if jobvt == 'S' {
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

            let mut u_t = if jobu == 'A' || jobu == 'S' {
                match uninitialized_vec(nrows_u * ncols_u) {
                    Ok(u_t) => u_t,
                    Err(_) => return LAPACK_TRANSPOSE_MEMORY_ERROR,
                }
            } else {
                Vec::new()
            };

            let mut vt_t = if jobvt == 'A' || jobvt == 'S' {
                match uninitialized_vec(nrows_vt * n) {
                    Ok(vt_t) => vt_t,
                    Err(_) => return LAPACK_TRANSPOSE_MEMORY_ERROR,
                }
            } else {
                Vec::new()
            };

            // Call LAPACK function
            func_(
                &(jobu as _),
                &(jobvt as _),
                &(m as _),
                &(n as _),
                a_t.as_mut_ptr(),
                &(lda_t as _),
                s,
                if jobu == 'A' || jobu == 'S' { u_t.as_mut_ptr() } else { u },
                &(ldu_t as _),
                if jobvt == 'A' || jobvt == 'S' { vt_t.as_mut_ptr() } else { vt },
                &(ldvt_t as _),
                work.as_mut_ptr(),
                &(lwork as _),
                &mut info,
            );
            if info != 0 {
                return raise_info(info);
            }

            // Transpose output matrices
            orderchange_out_r2c_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).unwrap();

            if jobu == 'A' || jobu == 'S' {
                let u_slice = from_raw_parts_mut(u, nrows_u * ldu);
                let lu = Layout::new_unchecked([nrows_u, ncols_u], [ldu as isize, 1], 0);
                let lu_t = Layout::new_unchecked([nrows_u, ncols_u], [1, ldu_t as isize], 0);
                orderchange_out_r2c_ix2_cpu_serial(u_slice, &lu, &u_t, &lu_t).unwrap();
            }

            if jobvt == 'A' || jobvt == 'S' {
                let vt_slice = from_raw_parts_mut(vt, nrows_vt * ldvt);
                let lvt = Layout::new_unchecked([nrows_vt, n], [ldvt as isize, 1], 0);
                let lvt_t = Layout::new_unchecked([nrows_vt, n], [1, ldvt_t as isize], 0);
                orderchange_out_r2c_ix2_cpu_serial(vt_slice, &lvt, &vt_t, &lvt_t).unwrap();
            }
        }

        // Backup superb data
        let min_mn = m.min(n);
        for i in 0..min_mn - 1 {
            superb.add(i).write(work[i + 1]);
        }

        return info;
    }
}

#[duplicate_item(
    T              func_   ;
   [Complex<f32>] [cgesvd_];
   [Complex<f64>] [zgesvd_];
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
        use rstsr_lapack_ffi::lapack::func_;

        unsafe fn raise_info(mut info: blas_int) -> blas_int {
            xerbla_(c"gesvd".as_ptr() as _, &mut info as *mut _ as *mut _);
            return if info < 0 { info - 1 } else { info };
        }

        // Allocate rwork
        let min_mn = m.min(n);
        let mut rwork: Vec<<T as ComplexFloat>::Real> = match uninitialized_vec(5 * min_mn) {
            Ok(rwork) => rwork,
            Err(_) => return LAPACK_WORK_MEMORY_ERROR,
        };

        // Query optimal working array size
        let mut info = 0;
        let lwork = -1;
        let mut work_query = <T as Zero>::zero();
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
            &mut work_query as *mut _ as *mut _,
            &lwork,
            rwork.as_mut_ptr() as *mut _,
            &mut info,
        );
        if info != 0 {
            return raise_info(info);
        }
        let lwork = work_query.re() as usize;

        // Allocate memory for work array
        let mut work: Vec<T> = match uninitialized_vec(lwork) {
            Ok(work) => work,
            Err(_) => return LAPACK_WORK_MEMORY_ERROR,
        };

        if order == ColMajor {
            // Call LAPACK function
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
                work.as_mut_ptr() as *mut _,
                &(lwork as _),
                rwork.as_mut_ptr() as *mut _,
                &mut info,
            );
            if info != 0 {
                return raise_info(info);
            }
        } else {
            let lda_t = m.max(1);
            let nrows_u = if jobu == 'A' || jobu == 'S' { m } else { 1 };
            let ncols_u = if jobu == 'A' {
                m
            } else if jobu == 'S' {
                m.min(n)
            } else {
                1
            };
            let nrows_vt = if jobvt == 'A' {
                n
            } else if jobvt == 'S' {
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

            let mut u_t = if jobu == 'A' || jobu == 'S' {
                match uninitialized_vec(nrows_u * ncols_u) {
                    Ok(u_t) => u_t,
                    Err(_) => return LAPACK_TRANSPOSE_MEMORY_ERROR,
                }
            } else {
                Vec::new()
            };

            let mut vt_t = if jobvt == 'A' || jobvt == 'S' {
                match uninitialized_vec(nrows_vt * n) {
                    Ok(vt_t) => vt_t,
                    Err(_) => return LAPACK_TRANSPOSE_MEMORY_ERROR,
                }
            } else {
                Vec::new()
            };

            // Call LAPACK function
            func_(
                &(jobu as _),
                &(jobvt as _),
                &(m as _),
                &(n as _),
                a_t.as_mut_ptr() as *mut _,
                &(lda_t as _),
                s as *mut _,
                if jobu == 'A' || jobu == 'S' { u_t.as_mut_ptr() as *mut _ } else { u as *mut _ },
                &(ldu_t as _),
                if jobvt == 'A' || jobvt == 'S' {
                    vt_t.as_mut_ptr() as *mut _
                } else {
                    vt as *mut _
                },
                &(ldvt_t as _),
                work.as_mut_ptr() as *mut _,
                &(lwork as _),
                rwork.as_mut_ptr() as *mut _,
                &mut info,
            );
            if info != 0 {
                return raise_info(info);
            }

            // Transpose output matrices
            orderchange_out_r2c_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).unwrap();

            if jobu == 'A' || jobu == 'S' {
                let u_slice = from_raw_parts_mut(u, nrows_u * ldu);
                let lu = Layout::new_unchecked([nrows_u, ncols_u], [ldu as isize, 1], 0);
                let lu_t = Layout::new_unchecked([nrows_u, ncols_u], [1, ldu_t as isize], 0);
                orderchange_out_r2c_ix2_cpu_serial(u_slice, &lu, &u_t, &lu_t).unwrap();
            }

            if jobvt == 'A' || jobvt == 'S' {
                let vt_slice = from_raw_parts_mut(vt, nrows_vt * ldvt);
                let lvt = Layout::new_unchecked([nrows_vt, n], [ldvt as isize, 1], 0);
                let lvt_t = Layout::new_unchecked([nrows_vt, n], [1, ldvt_t as isize], 0);
                orderchange_out_r2c_ix2_cpu_serial(vt_slice, &lvt, &vt_t, &lvt_t).unwrap();
            }
        }

        // Backup superb data
        #[allow(clippy::needless_range_loop)]
        for i in 0..min_mn - 1 {
            superb.add(i).write(rwork[i]);
        }

        return info;
    }
}
