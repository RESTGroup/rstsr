use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::{Complex, ToPrimitive};
use rstsr_blas_traits::lapack_qr::ORMQRDriverAPI;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude_dev::*;
use rstsr_native_impl::prelude_dev::*;
use std::slice::from_raw_parts_mut;

// ORMQR for real types
#[duplicate_item(
    T     func_   ;
   [f32] [sormqr_];
   [f64] [dormqr_];
)]
impl ORMQRDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_ormqr(
        order: FlagOrder,
        side: FlagSide,
        trans: FlagTrans,
        m: usize,
        n: usize,
        k: usize,
        a: *const T,
        lda: usize,
        tau: *const T,
        c: *mut T,
        ldc: usize,
    ) -> blas_int {
        use lapack_ffi::lapack::func_;
        use num::Zero;

        // Determine the number of rows in A based on side
        let r = if side == FlagSide::L { m } else { n };

        if order == ColMajor {
            // Query optimal working array(s) size
            let mut info = 0;
            let lwork = -1;
            let mut work_query: T = T::zero();
            func_(
                &side.into(),
                &trans.into(),
                &(m as _),
                &(n as _),
                &(k as _),
                a,
                &(lda as _),
                tau,
                c,
                &(ldc as _),
                &mut work_query,
                &lwork,
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

            // Call LAPACK function directly
            func_(
                &side.into(),
                &trans.into(),
                &(m as _),
                &(n as _),
                &(k as _),
                a,
                &(lda as _),
                tau,
                c,
                &(ldc as _),
                work.as_mut_ptr(),
                &(lwork as _),
                &mut info,
            );
            info
        } else {
            // Row-major: need to transpose A and C
            let lda_t = r.max(1);
            let ldc_t = m.max(1);
            let size_a = r * k;
            let size_c = m * n;

            // Allocate temporary buffers
            let mut a_t: Vec<T> = match uninitialized_vec(size_a) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };
            let mut c_t: Vec<T> = match uninitialized_vec(size_c) {
                Ok(c_t) => c_t,
                Err(_) => return -1011,
            };

            // Transpose A from row-major to column-major
            let a_slice = from_raw_parts_mut(a as *mut T, size_a);
            let la = Layout::new_unchecked([r, k], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([r, k], [1, lda_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).unwrap();

            // Transpose C from row-major to column-major
            let c_slice = from_raw_parts_mut(c, size_c);
            let lc = Layout::new_unchecked([m, n], [ldc as isize, 1], 0);
            let lc_t = Layout::new_unchecked([m, n], [1, ldc_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut c_t, &lc_t, c_slice, &lc).unwrap();

            // Query optimal working array(s) size
            let mut info = 0;
            let lwork = -1;
            let mut work_query: T = T::zero();
            func_(
                &side.into(),
                &trans.into(),
                &(m as _),
                &(n as _),
                &(k as _),
                a_t.as_ptr(),
                &(lda_t as _),
                tau,
                c_t.as_mut_ptr(),
                &(ldc_t as _),
                &mut work_query,
                &lwork,
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

            // Call LAPACK function
            func_(
                &side.into(),
                &trans.into(),
                &(m as _),
                &(n as _),
                &(k as _),
                a_t.as_ptr(),
                &(lda_t as _),
                tau,
                c_t.as_mut_ptr(),
                &(ldc_t as _),
                work.as_mut_ptr(),
                &(lwork as _),
                &mut info,
            );
            if info != 0 {
                return info;
            }

            // Transpose C back from column-major to row-major
            orderchange_out_c2r_ix2_cpu_serial(c_slice, &lc, &c_t, &lc_t).unwrap();

            info
        }
    }
}

// UNMQR for complex types
#[duplicate_item(
    T              func_   ;
   [Complex::<f32>] [cunmqr_];
   [Complex::<f64>] [zunmqr_];
)]
impl ORMQRDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_ormqr(
        order: FlagOrder,
        side: FlagSide,
        trans: FlagTrans,
        m: usize,
        n: usize,
        k: usize,
        a: *const T,
        lda: usize,
        tau: *const T,
        c: *mut T,
        ldc: usize,
    ) -> blas_int {
        use lapack_ffi::lapack::func_;
        use num::Zero;

        // Determine the number of rows in A based on side
        let r = if side == FlagSide::L { m } else { n };

        if order == ColMajor {
            // Query optimal working array(s) size
            let mut info = 0;
            let lwork = -1;
            let mut work_query: T = T::zero();
            func_(
                &side.into(),
                &trans.into(),
                &(m as _),
                &(n as _),
                &(k as _),
                a as *const _,
                &(lda as _),
                tau as *const _,
                c as *mut _,
                &(ldc as _),
                &mut work_query as *mut _ as *mut _,
                &lwork,
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

            // Call LAPACK function directly
            func_(
                &side.into(),
                &trans.into(),
                &(m as _),
                &(n as _),
                &(k as _),
                a as *const _,
                &(lda as _),
                tau as *const _,
                c as *mut _,
                &(ldc as _),
                work.as_mut_ptr() as *mut _,
                &(lwork as _),
                &mut info,
            );
            info
        } else {
            // Row-major: need to transpose A and C
            let lda_t = r.max(1);
            let ldc_t = m.max(1);
            let size_a = r * k;
            let size_c = m * n;

            // Allocate temporary buffers
            let mut a_t: Vec<T> = match uninitialized_vec(size_a) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };
            let mut c_t: Vec<T> = match uninitialized_vec(size_c) {
                Ok(c_t) => c_t,
                Err(_) => return -1011,
            };

            // Transpose A from row-major to column-major
            let a_slice = from_raw_parts_mut(a as *mut T, size_a);
            let la = Layout::new_unchecked([r, k], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([r, k], [1, lda_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).unwrap();

            // Transpose C from row-major to column-major
            let c_slice = from_raw_parts_mut(c, size_c);
            let lc = Layout::new_unchecked([m, n], [ldc as isize, 1], 0);
            let lc_t = Layout::new_unchecked([m, n], [1, ldc_t as isize], 0);
            orderchange_out_r2c_ix2_cpu_serial(&mut c_t, &lc_t, c_slice, &lc).unwrap();

            // Query optimal working array(s) size
            let mut info = 0;
            let lwork = -1;
            let mut work_query: T = T::zero();
            func_(
                &side.into(),
                &trans.into(),
                &(m as _),
                &(n as _),
                &(k as _),
                a_t.as_ptr() as *const _,
                &(lda_t as _),
                tau as *const _,
                c_t.as_mut_ptr() as *mut _,
                &(ldc_t as _),
                &mut work_query as *mut _ as *mut _,
                &lwork,
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

            // Call LAPACK function
            func_(
                &side.into(),
                &trans.into(),
                &(m as _),
                &(n as _),
                &(k as _),
                a_t.as_ptr() as *const _,
                &(lda_t as _),
                tau as *const _,
                c_t.as_mut_ptr() as *mut _,
                &(ldc_t as _),
                work.as_mut_ptr() as *mut _,
                &(lwork as _),
                &mut info,
            );
            if info != 0 {
                return info;
            }

            // Transpose C back from column-major to row-major
            orderchange_out_c2r_ix2_cpu_serial(c_slice, &lc, &c_t, &lc_t).unwrap();

            info
        }
    }
}
