use crate::lapack_ffi;
use crate::DeviceBLAS;
use num::complex::ComplexFloat;
use num::{Complex, ToPrimitive, Zero};
use rstsr_blas_traits::lapack_eigh::SYGVXDriverAPI;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude_dev::*;
use rstsr_native_impl::prelude_dev::*;
use std::slice::from_raw_parts_mut;

#[duplicate_item(
    T     func_   ;
   [f32] [ssygvx_];
   [f64] [dsygvx_];
)]
impl SYGVXDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_sygvx(
        order: FlagOrder,
        itype: blas_int,
        jobz: char,
        range: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        vl: T,
        vu: T,
        il: blas_int,
        iu: blas_int,
        abstol: T,
        m: *mut blas_int,
        w: *mut T,
        z: *mut T,
        ldz: usize,
        ifail: *mut blas_int,
    ) -> blas_int {
        use lapack_ffi::lapack::func_;

        let compute_z = jobz == 'V' || jobz == 'v';

        // Determine max number of eigenvalues based on range
        let max_m = match range {
            'A' | 'a' => n,
            'I' | 'i' => (iu - il + 1) as usize,
            'V' | 'v' => n,
            _ => n,
        };

        if order == ColMajor {
            // Query optimal working array sizes
            let mut info = 0;
            let lwork = -1;
            let mut work_query: T = T::zero();
            func_(
                &itype,
                &(jobz as _),
                &(range as _),
                &uplo.into(),
                &(n as _),
                a,
                &(lda as _),
                b,
                &(ldb as _),
                &vl,
                &vu,
                &il,
                &iu,
                &abstol,
                m,
                w,
                z,
                &(ldz as _),
                &mut work_query,
                &lwork,
                std::ptr::null_mut(), // iwork not needed for query
                ifail,
                &mut info,
            );
            if info != 0 {
                return info;
            }
            let lwork = work_query.to_usize().unwrap();

            // Allocate work arrays (iwork is always 5*n for SYGVX)
            let mut work: Vec<T> = match uninitialized_vec(lwork) {
                Ok(work) => work,
                Err(_) => return -1010,
            };
            let mut iwork: Vec<blas_int> = match uninitialized_vec(5 * n) {
                Ok(iwork) => iwork,
                Err(_) => return -1010,
            };

            // Call LAPACK
            func_(
                &itype,
                &(jobz as _),
                &(range as _),
                &uplo.into(),
                &(n as _),
                a,
                &(lda as _),
                b,
                &(ldb as _),
                &vl,
                &vu,
                &il,
                &iu,
                &abstol,
                m,
                w,
                z,
                &(ldz as _),
                work.as_mut_ptr(),
                &(lwork as _),
                iwork.as_mut_ptr(),
                ifail,
                &mut info,
            );
            info
        } else {
            // Row-major handling
            let lda_t = n.max(1);
            let ldb_t = n.max(1);
            let ldz_t = n.max(1);

            // Allocate temporary column-major arrays
            let mut a_t: Vec<T> = match uninitialized_vec(n * lda_t) {
                Ok(a_t) => a_t,
                Err(_) => return -1010,
            };
            let mut b_t: Vec<T> = match uninitialized_vec(n * ldb_t) {
                Ok(b_t) => b_t,
                Err(_) => return -1010,
            };
            let mut z_t: Vec<T> = if compute_z {
                match uninitialized_vec(n * max_m) {
                    Ok(z_t) => z_t,
                    Err(_) => return -1011,
                }
            } else {
                Vec::new()
            };
            let mut ifail_t: Vec<blas_int> = if compute_z {
                match uninitialized_vec(n) {
                    Ok(ifail_t) => ifail_t,
                    Err(_) => return -1012,
                }
            } else {
                Vec::new()
            };

            // Transpose input A from row-major to column-major
            let a_slice = from_raw_parts_mut(a, n * lda);
            let la = Layout::new_unchecked([n, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([n, n], [1, lda_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).unwrap();

            // Transpose input B from row-major to column-major
            let b_slice = from_raw_parts_mut(b, n * ldb);
            let lb = Layout::new_unchecked([n, n], [ldb as isize, 1], 0);
            let lb_t = Layout::new_unchecked([n, n], [1, ldb_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut b_t, &lb_t, b_slice, &lb).unwrap();

            // Query optimal work array sizes
            let mut info = 0;
            let lwork = -1;
            let mut work_query: T = T::zero();
            func_(
                &itype,
                &(jobz as _),
                &(range as _),
                &uplo.into(),
                &(n as _),
                a_t.as_mut_ptr(),
                &(lda_t as _),
                b_t.as_mut_ptr(),
                &(ldb_t as _),
                &vl,
                &vu,
                &il,
                &iu,
                &abstol,
                m,
                w,
                z_t.as_mut_ptr(),
                &(ldz_t as _),
                &mut work_query,
                &lwork,
                std::ptr::null_mut(), // iwork not needed for query
                ifail_t.as_mut_ptr(),
                &mut info,
            );
            if info != 0 {
                return info;
            }
            let lwork = work_query.to_usize().unwrap();

            // Allocate work arrays (iwork is always 5*n for SYGVX)
            let mut work: Vec<T> = match uninitialized_vec(lwork) {
                Ok(work) => work,
                Err(_) => return -1010,
            };
            let mut iwork: Vec<blas_int> = match uninitialized_vec(5 * n) {
                Ok(iwork) => iwork,
                Err(_) => return -1010,
            };

            // Call LAPACK with column-major arrays
            func_(
                &itype,
                &(jobz as _),
                &(range as _),
                &uplo.into(),
                &(n as _),
                a_t.as_mut_ptr(),
                &(lda_t as _),
                b_t.as_mut_ptr(),
                &(ldb_t as _),
                &vl,
                &vu,
                &il,
                &iu,
                &abstol,
                m,
                w,
                z_t.as_mut_ptr(),
                &(ldz_t as _),
                work.as_mut_ptr(),
                &(lwork as _),
                iwork.as_mut_ptr(),
                ifail_t.as_mut_ptr(),
                &mut info,
            );
            if info != 0 {
                return info;
            }

            // Transpose output A back to row-major
            orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).unwrap();

            // Transpose output B back to row-major
            orderchange_out_c2r_ix2_cpu_serial(b_slice, &lb, &b_t, &lb_t).unwrap();

            // Transpose output Z from column-major to row-major
            if compute_z && *m > 0 {
                let m_found = *m as usize;
                let z_slice = from_raw_parts_mut(z, n * ldz);
                // Source: column-major (n, m_found) with strides (1, n)
                let lz_t = Layout::new_unchecked([n, m_found], [1, n as isize], 0);
                // Dest: row-major (n, m_found) with strides (ldz, 1)
                let lz = Layout::new_unchecked([n, m_found], [ldz as isize, 1], 0);
                orderchange_out_c2r_ix2_cpu_serial(z_slice, &lz, &z_t, &lz_t).unwrap();

                // Copy ifail
                let ifail_slice = from_raw_parts_mut(ifail, n);
                ifail_slice.copy_from_slice(&ifail_t[..n]);
            }

            info
        }
    }
}

#[duplicate_item(
    T              func_   ;
   [Complex::<f32>] [chegvx_];
   [Complex::<f64>] [zhegvx_];
)]
impl SYGVXDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_sygvx(
        order: FlagOrder,
        itype: blas_int,
        jobz: char,
        range: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        vl: <T as ComplexFloat>::Real,
        vu: <T as ComplexFloat>::Real,
        il: blas_int,
        iu: blas_int,
        abstol: <T as ComplexFloat>::Real,
        m: *mut blas_int,
        w: *mut <T as ComplexFloat>::Real,
        z: *mut T,
        ldz: usize,
        ifail: *mut blas_int,
    ) -> blas_int {
        use lapack_ffi::lapack::func_;

        let compute_z = jobz == 'V' || jobz == 'v';

        // Determine max number of eigenvalues based on range
        let max_m = match range {
            'A' | 'a' => n,
            'I' | 'i' => (iu - il + 1) as usize,
            'V' | 'v' => n,
            _ => n,
        };

        // Allocate memory for rwork array
        let rwork_len = (7 * n).max(1);
        let mut rwork: Vec<<T as ComplexFloat>::Real> = match uninitialized_vec(rwork_len) {
            Ok(rwork) => rwork,
            Err(_) => return -1010,
        };

        if order == ColMajor {
            // Query optimal working array sizes
            let mut info = 0;
            let lwork = -1;
            let mut work_query = T::zero();
            let mut iwork_query: blas_int = 0;
            func_(
                &itype,
                &(jobz as _),
                &(range as _),
                &uplo.into(),
                &(n as _),
                a as *mut _,
                &(lda as _),
                b as *mut _,
                &(ldb as _),
                &vl,
                &vu,
                &il,
                &iu,
                &abstol,
                m,
                w,
                z as *mut _,
                &(ldz as _),
                &mut work_query as *mut _ as *mut _,
                &lwork,
                rwork.as_mut_ptr(),
                &mut iwork_query,
                ifail,
                &mut info,
            );
            if info != 0 {
                return info;
            }
            let lwork = work_query.to_usize().unwrap();
            let liwork = iwork_query as usize;

            // Allocate work arrays
            let mut work: Vec<T> = match uninitialized_vec(lwork) {
                Ok(work) => work,
                Err(_) => return -1010,
            };
            let mut iwork: Vec<blas_int> = match uninitialized_vec(liwork) {
                Ok(iwork) => iwork,
                Err(_) => return -1010,
            };

            // Call LAPACK
            func_(
                &itype,
                &(jobz as _),
                &(range as _),
                &uplo.into(),
                &(n as _),
                a as *mut _,
                &(lda as _),
                b as *mut _,
                &(ldb as _),
                &vl,
                &vu,
                &il,
                &iu,
                &abstol,
                m,
                w,
                z as *mut _,
                &(ldz as _),
                work.as_mut_ptr() as *mut _,
                &(lwork as _),
                rwork.as_mut_ptr(),
                iwork.as_mut_ptr(),
                ifail,
                &mut info,
            );
            info
        } else {
            // Row-major handling
            let lda_t = n.max(1);
            let ldb_t = n.max(1);
            let ldz_t = n.max(1);

            // Allocate temporary column-major arrays
            let mut a_t: Vec<T> = match uninitialized_vec(n * lda_t) {
                Ok(a_t) => a_t,
                Err(_) => return -1010,
            };
            let mut b_t: Vec<T> = match uninitialized_vec(n * ldb_t) {
                Ok(b_t) => b_t,
                Err(_) => return -1010,
            };
            let mut z_t: Vec<T> = if compute_z {
                match uninitialized_vec(n * max_m) {
                    Ok(z_t) => z_t,
                    Err(_) => return -1011,
                }
            } else {
                Vec::new()
            };
            let mut ifail_t: Vec<blas_int> = if compute_z {
                match uninitialized_vec(n) {
                    Ok(ifail_t) => ifail_t,
                    Err(_) => return -1012,
                }
            } else {
                Vec::new()
            };

            // Transpose input A from row-major to column-major
            let a_slice = from_raw_parts_mut(a, n * lda);
            let la = Layout::new_unchecked([n, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([n, n], [1, lda_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).unwrap();

            // Transpose input B from row-major to column-major
            let b_slice = from_raw_parts_mut(b, n * ldb);
            let lb = Layout::new_unchecked([n, n], [ldb as isize, 1], 0);
            let lb_t = Layout::new_unchecked([n, n], [1, ldb_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut b_t, &lb_t, b_slice, &lb).unwrap();

            // Query optimal work array sizes
            let mut info = 0;
            let lwork = -1;
            let mut work_query = T::zero();
            let mut iwork_query: blas_int = 0;
            func_(
                &itype,
                &(jobz as _),
                &(range as _),
                &uplo.into(),
                &(n as _),
                a_t.as_mut_ptr() as *mut _,
                &(lda_t as _),
                b_t.as_mut_ptr() as *mut _,
                &(ldb_t as _),
                &vl,
                &vu,
                &il,
                &iu,
                &abstol,
                m,
                w,
                z_t.as_mut_ptr() as *mut _,
                &(ldz_t as _),
                &mut work_query as *mut _ as *mut _,
                &lwork,
                rwork.as_mut_ptr(),
                &mut iwork_query,
                ifail_t.as_mut_ptr(),
                &mut info,
            );
            if info != 0 {
                return info;
            }
            let lwork = work_query.to_usize().unwrap();
            let liwork = iwork_query as usize;

            // Allocate work arrays
            let mut work: Vec<T> = match uninitialized_vec(lwork) {
                Ok(work) => work,
                Err(_) => return -1010,
            };
            let mut iwork: Vec<blas_int> = match uninitialized_vec(liwork) {
                Ok(iwork) => iwork,
                Err(_) => return -1010,
            };

            // Call LAPACK with column-major arrays
            func_(
                &itype,
                &(jobz as _),
                &(range as _),
                &uplo.into(),
                &(n as _),
                a_t.as_mut_ptr() as *mut _,
                &(lda_t as _),
                b_t.as_mut_ptr() as *mut _,
                &(ldb_t as _),
                &vl,
                &vu,
                &il,
                &iu,
                &abstol,
                m,
                w,
                z_t.as_mut_ptr() as *mut _,
                &(ldz_t as _),
                work.as_mut_ptr() as *mut _,
                &(lwork as _),
                rwork.as_mut_ptr(),
                iwork.as_mut_ptr(),
                ifail_t.as_mut_ptr(),
                &mut info,
            );
            if info != 0 {
                return info;
            }

            // Transpose output A back to row-major
            orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).unwrap();

            // Transpose output B back to row-major
            orderchange_out_c2r_ix2_cpu_serial(b_slice, &lb, &b_t, &lb_t).unwrap();

            // Transpose output Z from column-major to row-major
            if compute_z && *m > 0 {
                let m_found = *m as usize;
                let z_slice = from_raw_parts_mut(z, n * ldz);
                // Source: column-major (n, m_found) with strides (1, n)
                let lz_t = Layout::new_unchecked([n, m_found], [1, n as isize], 0);
                // Dest: row-major (n, m_found) with strides (ldz, 1)
                let lz = Layout::new_unchecked([n, m_found], [ldz as isize, 1], 0);
                orderchange_out_c2r_ix2_cpu_serial(z_slice, &lz, &z_t, &lz_t).unwrap();

                // Copy ifail
                let ifail_slice = from_raw_parts_mut(ifail, n);
                ifail_slice.copy_from_slice(&ifail_t[..n]);
            }

            info
        }
    }
}
