use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::Complex;
use rstsr_blas_traits::lapack_eig::GEEVDriverAPI;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude::*;
use rstsr_core::prelude_dev::uninitialized_vec;

#[duplicate_item(
    T     lapacke_func  ;
   [f32] [LAPACKE_sgeev];
   [f64] [LAPACKE_dgeev];
)]
impl GEEVDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_geev(
        order: FlagOrder,
        jobvl: char,
        jobvr: char,
        n: usize,
        a: *mut T,
        lda: usize,
        w: *mut Complex<T>,
        vl: *mut T,
        ldvl: usize,
        vr: *mut T,
        ldvr: usize,
    ) -> blas_int {
        // For real types, LAPACKE_sgeev/dgeev returns wr and wi separately.
        // We need to allocate temporary arrays, call LAPACKE, then combine into complex.
        let mut wr: Vec<T> = match uninitialized_vec(n) {
            Ok(wr) => wr,
            Err(_) => return -1010,
        };
        let mut wi: Vec<T> = match uninitialized_vec(n) {
            Ok(wi) => wi,
            Err(_) => return -1010,
        };

        let result = lapack_ffi::lapacke::lapacke_func(
            order as _,
            jobvl as _,
            jobvr as _,
            n as _,
            a,
            lda as _,
            wr.as_mut_ptr(),
            wi.as_mut_ptr(),
            vl,
            ldvl as _,
            vr,
            ldvr as _,
        );

        // Combine wr and wi into complex eigenvalues
        for i in 0..n {
            *w.add(i) = Complex::new(wr[i], wi[i]);
        }

        result
    }
}

#[duplicate_item(
    T              lapacke_func  ;
   [Complex::<f32>] [LAPACKE_cgeev];
   [Complex::<f64>] [LAPACKE_zgeev];
)]
impl GEEVDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_geev(
        order: FlagOrder,
        jobvl: char,
        jobvr: char,
        n: usize,
        a: *mut T,
        lda: usize,
        w: *mut T, // For complex types, T = Complex<T::Real>
        vl: *mut T,
        ldvl: usize,
        vr: *mut T,
        ldvr: usize,
    ) -> blas_int {
        // For complex types, LAPACKE_cgeev/zgeev has a single complex eigenvalue array w.
        // The w pointer is already Complex<T::Real>*, which matches T* for complex types.
        // We can pass it directly.
        let result = lapack_ffi::lapacke::lapacke_func(
            order as _,
            jobvl as _,
            jobvr as _,
            n as _,
            a as *mut _,
            lda as _,
            w as *mut _, // T and Complex<T::Real> are the same for complex types
            vl as *mut _,
            ldvl as _,
            vr as *mut _,
            ldvr as _,
        );

        result
    }
}
