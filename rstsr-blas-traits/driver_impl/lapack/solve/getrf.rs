use crate::DeviceBLAS;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude_dev::*;
use rstsr_lapack_ffi::blas::xerbla_;
use rstsr_lapack_ffi::lapacke::LAPACK_TRANSPOSE_MEMORY_ERROR;
use rstsr_native_impl::prelude_dev::*;
use std::slice::from_raw_parts_mut;

#[duplicate_item(
    T     func_   ;
   [f32] [sgetrf_];
   [f64] [dgetrf_];
)]
impl GETRFDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_getrf(
        order: FlagOrder,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
    ) -> blas_int {
        use rstsr_lapack_ffi::lapack::func_;

        unsafe fn raise_info(mut info: blas_int) -> blas_int {
            xerbla_(c"getrf".as_ptr() as _, &mut info as *mut _ as *mut _);
            return if info < 0 { info - 1 } else { info };
        }

        let mut info = 0;

        if order == ColMajor {
            // Call LAPACK function and adjust info
            func_(&(m as _), &(n as _), a, &(lda as _), ipiv, &mut info);
            if info != 0 {
                return raise_info(info);
            }
        } else {
            let lda_t = m.max(1);
            // Transpose input matrices
            let mut a_t: Vec<T> = match uninitialized_vec(m * n) {
                Ok(a_t) => a_t,
                Err(_) => return LAPACK_TRANSPOSE_MEMORY_ERROR,
            };
            let a_slice = from_raw_parts_mut(a, m * lda);
            let la = Layout::new_unchecked([m, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([m, n], [1, lda_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).unwrap();
            // Call LAPACK function and adjust info
            func_(&(m as _), &(n as _), a_t.as_mut_ptr(), &(lda_t as _), ipiv, &mut info);
            if info != 0 {
                return raise_info(info);
            }
            // Transpose output matrices
            orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).unwrap();
        }
        return info;
    }
}

#[duplicate_item(
    T              func_   ;
   [Complex<f32>] [cgetrf_];
   [Complex<f64>] [zgetrf_];
)]
impl GETRFDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_getrf(
        order: FlagOrder,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
    ) -> blas_int {
        use rstsr_lapack_ffi::lapack::func_;

        unsafe fn raise_info(mut info: blas_int) -> blas_int {
            xerbla_(c"getrf".as_ptr() as _, &mut info as *mut _ as *mut _);
            return if info < 0 { info - 1 } else { info };
        }

        let mut info = 0;

        if order == ColMajor {
            // Call LAPACK function and adjust info
            func_(&(m as _), &(n as _), a as *mut _, &(lda as _), ipiv, &mut info);
            if info != 0 {
                return raise_info(info);
            }
        } else {
            let lda_t = m.max(1);
            // Transpose input matrices
            let mut a_t: Vec<T> = match uninitialized_vec(m * n) {
                Ok(a_t) => a_t,
                Err(_) => return LAPACK_TRANSPOSE_MEMORY_ERROR,
            };
            let a_slice = from_raw_parts_mut(a, m * lda);
            let la = Layout::new_unchecked([m, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([m, n], [1, lda_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).unwrap();
            // Call LAPACK function and adjust info
            func_(&(m as _), &(n as _), a_t.as_mut_ptr() as *mut _, &(lda_t as _), ipiv, &mut info);
            if info != 0 {
                return raise_info(info);
            }
            // Transpose output matrices
            orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).unwrap();
        }
        return info;
    }
}
