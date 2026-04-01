use crate::lapack_ffi;
use crate::DeviceBLAS;
use num::complex::ComplexFloat;
use num::{Complex, ToPrimitive, Zero};
use rstsr_blas_traits::lapack_eigh::SYEVRDriverAPI;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude_dev::*;
use rstsr_native_impl::prelude_dev::*;
use std::slice::from_raw_parts_mut;

#[duplicate_item(
    T     func_   ;
   [f32] [ssyevr_];
   [f64] [dsyevr_];
)]
impl SYEVRDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_syevr(
        order: FlagOrder,
        jobz: char,
        range: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        vl: T,
        vu: T,
        il: blas_int,
        iu: blas_int,
        abstol: T,
        m: *mut blas_int,
        w: *mut T,
        z: *mut T,
        ldz: usize,
        isuppz: *mut blas_int,
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
            let liwork = -1;
            let mut work_query: T = T::zero();
            let mut iwork_query: blas_int = 0;
            func_(
                &(jobz as _),
                &(range as _),
                &uplo.into(),
                &(n as _),
                a,
                &(lda as _),
                &vl,
                &vu,
                &il,
                &iu,
                &abstol,
                m,
                w,
                z,
                &(ldz as _),
                isuppz,
                &mut work_query,
                &lwork,
                &mut iwork_query,
                &liwork,
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
                &(jobz as _),
                &(range as _),
                &uplo.into(),
                &(n as _),
                a,
                &(lda as _),
                &vl,
                &vu,
                &il,
                &iu,
                &abstol,
                m,
                w,
                z,
                &(ldz as _),
                isuppz,
                work.as_mut_ptr(),
                &(lwork as _),
                iwork.as_mut_ptr(),
                &(liwork as _),
                &mut info,
            );
            info
        } else {
            // Row-major handling
            // For row-major, Z is (max_m, n) with ldz = max_m
            // LAPACK expects column-major Z as (n, max_m) with ldz = n
            let lda_t = n.max(1);
            let ldz_t = n.max(1);

            // Allocate temporary column-major arrays
            let mut a_t: Vec<T> = match uninitialized_vec(n * lda_t) {
                Ok(a_t) => a_t,
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
            let mut isuppz_t: Vec<blas_int> = if compute_z {
                match uninitialized_vec(2 * max_m) {
                    Ok(isuppz_t) => isuppz_t,
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

            // Query optimal work array sizes
            let mut info = 0;
            let lwork = -1;
            let liwork = -1;
            let mut work_query: T = T::zero();
            let mut iwork_query: blas_int = 0;
            func_(
                &(jobz as _),
                &(range as _),
                &uplo.into(),
                &(n as _),
                a_t.as_mut_ptr(),
                &(lda_t as _),
                &vl,
                &vu,
                &il,
                &iu,
                &abstol,
                m,
                w,
                z_t.as_mut_ptr(),
                &(ldz_t as _),
                isuppz_t.as_mut_ptr(),
                &mut work_query,
                &lwork,
                &mut iwork_query,
                &liwork,
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
                &(jobz as _),
                &(range as _),
                &uplo.into(),
                &(n as _),
                a_t.as_mut_ptr(),
                &(lda_t as _),
                &vl,
                &vu,
                &il,
                &iu,
                &abstol,
                m,
                w,
                z_t.as_mut_ptr(),
                &(ldz_t as _),
                isuppz_t.as_mut_ptr(),
                work.as_mut_ptr(),
                &(lwork as _),
                iwork.as_mut_ptr(),
                &(liwork as _),
                &mut info,
            );
            if info != 0 {
                return info;
            }

            // Transpose output A back to row-major
            orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).unwrap();

            // Transpose output Z from column-major to row-major
            // Source: z_t has shape (n, m_found) in column-major with ldz_t = n
            // Dest: z has shape (n, m_found) in row-major with ldz
            if compute_z && *m > 0 {
                let m_found = *m as usize;
                let z_slice = from_raw_parts_mut(z, n * ldz);
                // Source: column-major (n, m_found) with strides (1, n)
                let lz_t = Layout::new_unchecked([n, m_found], [1, n as isize], 0);
                // Dest: row-major (n, m_found) with strides (ldz, 1)
                let lz = Layout::new_unchecked([n, m_found], [ldz as isize, 1], 0);
                orderchange_out_c2r_ix2_cpu_serial(z_slice, &lz, &z_t, &lz_t).unwrap();

                // Copy isuppz
                let isuppz_slice = from_raw_parts_mut(isuppz, 2 * m_found);
                isuppz_slice.copy_from_slice(&isuppz_t[..2 * m_found]);
            }

            info
        }
    }
}

#[duplicate_item(
    T              func_   ;
   [Complex::<f32>] [cheevr_];
   [Complex::<f64>] [zheevr_];
)]
impl SYEVRDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_syevr(
        order: FlagOrder,
        jobz: char,
        range: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        vl: <T as ComplexFloat>::Real,
        vu: <T as ComplexFloat>::Real,
        il: blas_int,
        iu: blas_int,
        abstol: <T as ComplexFloat>::Real,
        m: *mut blas_int,
        w: *mut <T as ComplexFloat>::Real,
        z: *mut T,
        ldz: usize,
        isuppz: *mut blas_int,
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
            let lrwork = -1;
            let liwork = -1;
            let mut work_query = T::zero();
            let mut rwork_query = <T as ComplexFloat>::Real::zero();
            let mut iwork_query: blas_int = 0;
            func_(
                &(jobz as _),
                &(range as _),
                &uplo.into(),
                &(n as _),
                a as *mut _,
                &(lda as _),
                &vl,
                &vu,
                &il,
                &iu,
                &abstol,
                m,
                w,
                z as *mut _,
                &(ldz as _),
                isuppz,
                &mut work_query as *mut _ as *mut _,
                &lwork,
                &mut rwork_query as *mut _ as *mut _,
                &lrwork,
                &mut iwork_query,
                &liwork,
                &mut info,
            );
            if info != 0 {
                return info;
            }
            let lwork = work_query.to_usize().unwrap();
            let lrwork = rwork_query.to_usize().unwrap();
            let liwork = iwork_query as usize;

            // Allocate work arrays
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

            // Call LAPACK
            func_(
                &(jobz as _),
                &(range as _),
                &uplo.into(),
                &(n as _),
                a as *mut _,
                &(lda as _),
                &vl,
                &vu,
                &il,
                &iu,
                &abstol,
                m,
                w,
                z as *mut _,
                &(ldz as _),
                isuppz,
                work.as_mut_ptr() as *mut _,
                &(lwork as _),
                rwork.as_mut_ptr(),
                &(lrwork as _),
                iwork.as_mut_ptr(),
                &(liwork as _),
                &mut info,
            );
            info
        } else {
            // Row-major handling
            let lda_t = n.max(1);
            let ldz_t = n.max(1);

            // Allocate temporary column-major arrays
            let mut a_t: Vec<T> = match uninitialized_vec(n * lda_t) {
                Ok(a_t) => a_t,
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
            let mut isuppz_t: Vec<blas_int> = if compute_z {
                match uninitialized_vec(2 * max_m) {
                    Ok(isuppz_t) => isuppz_t,
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

            // Query optimal work array sizes
            let mut info = 0;
            let lwork = -1;
            let lrwork = -1;
            let liwork = -1;
            let mut work_query = T::zero();
            let mut rwork_query = <T as ComplexFloat>::Real::zero();
            let mut iwork_query: blas_int = 0;
            func_(
                &(jobz as _),
                &(range as _),
                &uplo.into(),
                &(n as _),
                a_t.as_mut_ptr() as *mut _,
                &(lda_t as _),
                &vl,
                &vu,
                &il,
                &iu,
                &abstol,
                m,
                w,
                z_t.as_mut_ptr() as *mut _,
                &(ldz_t as _),
                isuppz_t.as_mut_ptr(),
                &mut work_query as *mut _ as *mut _,
                &lwork,
                &mut rwork_query as *mut _ as *mut _,
                &lrwork,
                &mut iwork_query,
                &liwork,
                &mut info,
            );
            if info != 0 {
                return info;
            }
            let lwork = work_query.to_usize().unwrap();
            let lrwork = rwork_query.to_usize().unwrap();
            let liwork = iwork_query as usize;

            // Allocate work arrays
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

            // Call LAPACK with column-major arrays
            func_(
                &(jobz as _),
                &(range as _),
                &uplo.into(),
                &(n as _),
                a_t.as_mut_ptr() as *mut _,
                &(lda_t as _),
                &vl,
                &vu,
                &il,
                &iu,
                &abstol,
                m,
                w,
                z_t.as_mut_ptr() as *mut _,
                &(ldz_t as _),
                isuppz_t.as_mut_ptr(),
                work.as_mut_ptr() as *mut _,
                &(lwork as _),
                rwork.as_mut_ptr(),
                &(lrwork as _),
                iwork.as_mut_ptr(),
                &(liwork as _),
                &mut info,
            );
            if info != 0 {
                return info;
            }

            // Transpose output A back to row-major
            orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).unwrap();

            // Transpose output Z from column-major to row-major
            // Source: z_t has shape (n, m_found) in column-major with ldz_t = n
            // Dest: z has shape (n, m_found) in row-major with ldz
            if compute_z && *m > 0 {
                let m_found = *m as usize;
                let z_slice = from_raw_parts_mut(z, n * ldz);
                // Source: column-major (n, m_found) with strides (1, n)
                let lz_t = Layout::new_unchecked([n, m_found], [1, n as isize], 0);
                // Dest: row-major (n, m_found) with strides (ldz, 1)
                let lz = Layout::new_unchecked([n, m_found], [ldz as isize, 1], 0);
                orderchange_out_c2r_ix2_cpu_serial(z_slice, &lz, &z_t, &lz_t).unwrap();

                // Copy isuppz
                let isuppz_slice = from_raw_parts_mut(isuppz, 2 * m_found);
                isuppz_slice.copy_from_slice(&isuppz_t[..2 * m_found]);
            }

            info
        }
    }
}
