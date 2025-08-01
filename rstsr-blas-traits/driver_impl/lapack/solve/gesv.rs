use crate::lapack_ffi;
use crate::DeviceBLAS;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude_dev::*;
use rstsr_native_impl::prelude_dev::*;
use std::slice::from_raw_parts_mut;

#[duplicate_item(
    T     func_   ;
   [f32] [sgesv_];
   [f64] [dgesv_];
)]
impl GESVDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_gesv(
        order: FlagOrder,
        n: usize,
        nrhs: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
        b: *mut T,
        ldb: usize,
    ) -> blas_int {
        use lapack_ffi::lapack::func_;

        let mut info = 0;

        if order == ColMajor {
            func_(&(n as _), &(nrhs as _), a, &(lda as _), ipiv, b, &(ldb as _), &mut info);
            if info != 0 {
                return info;
            }
        } else {
            let lda_t = n.max(1);
            let ldb_t = n.max(1);
            // Transpose input matrices
            let mut a_t: Vec<T> = match uninitialized_vec(n * n) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };
            let mut b_t: Vec<T> = match uninitialized_vec(n * nrhs) {
                Ok(b_t) => b_t,
                Err(_) => return -1011,
            };
            let a_slice = from_raw_parts_mut(a, n * lda);
            let b_slice = from_raw_parts_mut(b, n * ldb);
            let la = Layout::new_unchecked([n, n], [lda as isize, 1], 0);
            let lb = Layout::new_unchecked([n, nrhs], [ldb as isize, 1], 0);
            let la_t = Layout::new_unchecked([n, n], [1, lda_t as isize], 0);
            let lb_t = Layout::new_unchecked([n, nrhs], [1, ldb_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).unwrap();
            orderchange_out_r2c_ix2_cpu_serial(&mut b_t, &lb_t, b_slice, &lb).unwrap();
            // Call LAPACK function
            func_(
                &(n as _),
                &(nrhs as _),
                a_t.as_mut_ptr(),
                &(lda_t as _),
                ipiv,
                b_t.as_mut_ptr(),
                &(ldb_t as _),
                &mut info,
            );
            if info != 0 {
                return info;
            }
            // Transpose output matrices
            orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).unwrap();
            orderchange_out_c2r_ix2_cpu_serial(b_slice, &lb, &b_t, &lb_t).unwrap();
        }
        return info;
    }
}

#[duplicate_item(
    T              func_   ;
   [Complex<f32>] [cgesv_];
   [Complex<f64>] [zgesv_];
)]
impl GESVDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_gesv(
        order: FlagOrder,
        n: usize,
        nrhs: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
        b: *mut T,
        ldb: usize,
    ) -> blas_int {
        use lapack_ffi::lapack::func_;

        let mut info = 0;

        if order == ColMajor {
            func_(&(n as _), &(nrhs as _), a as *mut _, &(lda as _), ipiv, b as *mut _, &(ldb as _), &mut info);
            if info != 0 {
                return info;
            }
        } else {
            let lda_t = n.max(1);
            let ldb_t = n.max(1);
            // Transpose input matrices
            let mut a_t: Vec<T> = match uninitialized_vec(n * n) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };
            let mut b_t: Vec<T> = match uninitialized_vec(n * nrhs) {
                Ok(b_t) => b_t,
                Err(_) => return -1011,
            };
            let a_slice = from_raw_parts_mut(a, n * lda);
            let b_slice = from_raw_parts_mut(b, n * ldb);
            let la = Layout::new_unchecked([n, n], [lda as isize, 1], 0);
            let lb = Layout::new_unchecked([n, nrhs], [ldb as isize, 1], 0);
            let la_t = Layout::new_unchecked([n, n], [1, lda_t as isize], 0);
            let lb_t = Layout::new_unchecked([n, nrhs], [1, ldb_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).unwrap();
            orderchange_out_r2c_ix2_cpu_serial(&mut b_t, &lb_t, b_slice, &lb).unwrap();
            // Call LAPACK function
            func_(
                &(n as _),
                &(nrhs as _),
                a_t.as_mut_ptr() as *mut _,
                &(lda_t as _),
                ipiv,
                b_t.as_mut_ptr() as *mut _,
                &(ldb_t as _),
                &mut info,
            );
            if info != 0 {
                return info;
            }
            // Transpose output matrices
            orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).unwrap();
            orderchange_out_c2r_ix2_cpu_serial(b_slice, &lb, &b_t, &lb_t).unwrap();
        }
        return info;
    }
}
