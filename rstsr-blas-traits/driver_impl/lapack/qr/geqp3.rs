use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::complex::ComplexFloat;
use num::{Complex, ToPrimitive, Zero};
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude_dev::*;
use rstsr_native_impl::prelude_dev::*;
use std::slice::from_raw_parts_mut;

#[duplicate_item(
    T     func_   ;
   [f32] [sgeqp3_];
   [f64] [dgeqp3_];
)]
impl GEQP3DriverAPI<T> for DeviceBLAS {
    unsafe fn driver_geqp3(
        order: FlagOrder,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        jpvt: *mut blas_int,
        tau: *mut T,
    ) -> blas_int {
        use lapack_ffi::lapack::func_;

        // Query optimal working array(s) size
        let mut info = 0;
        let lwork = -1;
        let mut work_query: T = T::zero();
        func_(&(m as _), &(n as _), a, &(lda as _), jpvt, tau, &mut work_query, &lwork, &mut info);
        if info != 0 {
            return info;
        }
        let lwork = work_query.to_usize().unwrap();

        // Allocate memory for work arrays
        let mut work: Vec<T> = match uninitialized_vec(lwork) {
            Ok(work) => work,
            Err(_) => return -1010,
        };

        if order == ColMajor {
            // Call LAPACK function directly
            func_(&(m as _), &(n as _), a, &(lda as _), jpvt, tau, work.as_mut_ptr(), &(lwork as _), &mut info);
        } else {
            // Row-major: need to transpose
            let lda_t = m.max(1);
            let size_a = m * n;

            // Allocate temporary buffer for transposed A
            let mut a_t: Vec<T> = match uninitialized_vec(size_a) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };

            // Transpose A from row-major to column-major
            let a_slice = from_raw_parts_mut(a, size_a);
            let la = Layout::new_unchecked([m, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([m, n], [1, lda_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).unwrap();

            // Call LAPACK function
            func_(
                &(m as _),
                &(n as _),
                a_t.as_mut_ptr(),
                &(lda_t as _),
                jpvt,
                tau,
                work.as_mut_ptr(),
                &(lwork as _),
                &mut info,
            );
            if info != 0 {
                return info;
            }

            // Transpose A back from column-major to row-major
            orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).unwrap();
        }

        info
    }
}

#[duplicate_item(
    T              func_   ;
   [Complex<f32>] [cgeqp3_];
   [Complex<f64>] [zgeqp3_];
)]
impl GEQP3DriverAPI<T> for DeviceBLAS {
    unsafe fn driver_geqp3(
        order: FlagOrder,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        jpvt: *mut blas_int,
        tau: *mut T,
    ) -> blas_int {
        use lapack_ffi::lapack::func_;

        // Allocate memory for rwork array (complex version needs rwork)
        let rwork_len = 2 * n;
        let mut rwork: Vec<<T as ComplexFloat>::Real> = match uninitialized_vec(rwork_len) {
            Ok(rwork) => rwork,
            Err(_) => return -1010,
        };

        // Query optimal working array(s) size
        let mut info = 0;
        let lwork = -1;
        let mut work_query = <T as ComplexFloat>::Real::zero();
        func_(
            &(m as _),
            &(n as _),
            a as *mut _,
            &(lda as _),
            jpvt,
            tau as *mut _,
            &mut work_query as *mut _ as *mut _,
            &lwork,
            rwork.as_mut_ptr(),
            &mut info,
        );
        if info != 0 {
            return info;
        }
        let lwork = work_query.to_usize().unwrap();

        // Allocate memory for work arrays
        let mut work: Vec<T> = match uninitialized_vec(lwork) {
            Ok(work) => work,
            Err(_) => return -1010,
        };

        if order == ColMajor {
            // Call LAPACK function directly
            func_(
                &(m as _),
                &(n as _),
                a as *mut _,
                &(lda as _),
                jpvt,
                tau as *mut _,
                work.as_mut_ptr() as *mut _,
                &(lwork as _),
                rwork.as_mut_ptr(),
                &mut info,
            );
        } else {
            // Row-major: need to transpose
            let lda_t = m.max(1);
            let size_a = m * n;

            // Allocate temporary buffer for transposed A
            let mut a_t: Vec<T> = match uninitialized_vec(size_a) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };

            // Transpose A from row-major to column-major
            let a_slice = from_raw_parts_mut(a, size_a);
            let la = Layout::new_unchecked([m, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([m, n], [1, lda_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).unwrap();

            // Call LAPACK function
            func_(
                &(m as _),
                &(n as _),
                a_t.as_mut_ptr() as *mut _,
                &(lda_t as _),
                jpvt,
                tau as *mut _,
                work.as_mut_ptr() as *mut _,
                &(lwork as _),
                rwork.as_mut_ptr(),
                &mut info,
            );
            if info != 0 {
                return info;
            }

            // Transpose A back from column-major to row-major
            orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).unwrap();
        }

        info
    }
}
