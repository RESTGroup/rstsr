use crate::lapack_ffi;
use crate::DeviceBLAS;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude_dev::*;

use rstsr_native_impl::prelude_dev::*;
use std::slice::from_raw_parts_mut;

#[duplicate_item(
    T     func_   ;
   [f32] [sgetri_];
   [f64] [dgetri_];
)]
impl GETRIDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_getri(order: FlagOrder, n: usize, a: *mut T, lda: usize, ipiv: *mut blas_int) -> blas_int {
        use lapack_ffi::lapack::func_;

        // Query optimal working array(s) size
        let mut info = 0;
        let lwork = -1;
        let mut work_query = 0.0;
        func_(&(n as _), a, &(lda as _), ipiv, &mut work_query, &lwork, &mut info);
        if info != 0 {
            return info;
        }
        let lwork = work_query as usize;

        // Allocate memory for work arrays
        let mut work: Vec<T> = match uninitialized_vec(lwork) {
            Ok(work) => work,
            Err(_) => return -1010,
        };

        if order == ColMajor {
            // Call LAPACK function and adjust info
            func_(&(n as _), a, &(lda as _), ipiv, work.as_mut_ptr(), &(lwork as _), &mut info);
            if info != 0 {
                return info;
            }
        } else {
            let lda_t = n.max(1);
            // Transpose input matrices
            let mut a_t: Vec<T> = match uninitialized_vec(n * n) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };
            let a_slice = from_raw_parts_mut(a, n * lda);
            let la = Layout::new_unchecked([n, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([n, n], [1, lda_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).unwrap();
            // Call LAPACK function and adjust info
            func_(&(n as _), a_t.as_mut_ptr(), &(lda_t as _), ipiv, work.as_mut_ptr(), &(lwork as _), &mut info);
            if info != 0 {
                return info;
            }
            // Transpose output matrices
            orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).unwrap();
        }
        return info;
    }
}

#[duplicate_item(
    T              func_   ;
   [Complex<f32>] [cgetri_];
   [Complex<f64>] [zgetri_];
)]
impl GETRIDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_getri(order: FlagOrder, n: usize, a: *mut T, lda: usize, ipiv: *mut blas_int) -> blas_int {
        use lapack_ffi::lapack::func_;

        // Query optimal working array(s) size
        let mut info = 0;
        let lwork = -1;
        let mut work_query: T = num::zero();
        func_(&(n as _), a as *mut _, &(lda as _), ipiv, &mut work_query as *mut _ as *mut _, &lwork, &mut info);
        if info != 0 {
            return info;
        }
        let lwork = work_query.re as usize;

        // Allocate memory for work arrays
        let mut work: Vec<T> = match uninitialized_vec(lwork) {
            Ok(work) => work,
            Err(_) => return -1010,
        };

        if order == ColMajor {
            // Call LAPACK function and adjust info
            func_(&(n as _), a as *mut _, &(lda as _), ipiv, work.as_mut_ptr() as *mut _, &(lwork as _), &mut info);
            if info != 0 {
                return info;
            }
        } else {
            let lda_t = n.max(1);
            // Transpose input matrices
            let mut a_t: Vec<T> = match uninitialized_vec(n * n) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };
            let a_slice = from_raw_parts_mut(a, n * lda);
            let la = Layout::new_unchecked([n, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([n, n], [1, lda_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).unwrap();
            // Call LAPACK function and adjust info
            func_(
                &(n as _),
                a_t.as_mut_ptr() as *mut _,
                &(lda_t as _),
                ipiv,
                work.as_mut_ptr() as *mut _,
                &(lwork as _),
                &mut info,
            );
            if info != 0 {
                return info;
            }
            // Transpose output matrices
            orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).unwrap();
        }
        return info;
    }
}
