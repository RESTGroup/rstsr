use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::complex::ComplexFloat;
use num::Complex;
use rstsr_blas_traits::lapack_eig::GGEVDriverAPI;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude::*;
use rstsr_core::prelude_dev::uninitialized_vec;

#[duplicate_item(
    T     lapacke_func  ;
   [f32] [LAPACKE_sggev];
   [f64] [LAPACKE_dggev];
)]
impl GGEVDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_ggev(
        order: FlagOrder,
        jobvl: char,
        jobvr: char,
        n: usize,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        alphar: *mut T,
        alphai: *mut T,
        beta: *mut T,
        _betai: *mut T,  // unused for real types
        vl: *mut T,
        ldvl: usize,
        vr: *mut T,
        ldvr: usize,
    ) -> blas_int {
        let result = lapack_ffi::lapacke::lapacke_func(
            order as _, jobvl as _, jobvr as _, n as _, a, lda as _, b, ldb as _, alphar, alphai, beta, vl, ldvl as _,
            vr, ldvr as _,
        );
        result
    }
}

#[duplicate_item(
    T              lapacke_func  ;
   [Complex<f32>] [LAPACKE_cggev];
   [Complex<f64>] [LAPACKE_zggev];
)]
impl GGEVDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_ggev(
        order: FlagOrder,
        jobvl: char,
        jobvr: char,
        n: usize,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        alphar: *mut <T as ComplexFloat>::Real,
        alphai: *mut <T as ComplexFloat>::Real,
        beta: *mut <T as ComplexFloat>::Real,
        betai: *mut <T as ComplexFloat>::Real,
        vl: *mut T,
        ldvl: usize,
        vr: *mut T,
        ldvr: usize,
    ) -> blas_int {
        // For complex types, LAPACKE_cggev and LAPACKE_zggev have complex alpha and beta arrays.
        // We allocate temporary complex arrays, call LAPACKE, then split into real components.

        // Allocate temporary complex arrays for alpha and beta
        let mut alpha: Vec<T> = match uninitialized_vec(n) {
            Ok(alpha) => alpha,
            Err(_) => return -1010,
        };
        let mut beta_complex: Vec<T> = match uninitialized_vec(n) {
            Ok(beta) => beta,
            Err(_) => return -1010,
        };

        // Convert T pointers to the correct type for LAPACKE
        // For complex types, we need to pass Complex<T::Real>* pointers
        let ptr_alpha = alpha.as_mut_ptr();
        let ptr_beta = beta_complex.as_mut_ptr();

        let result = lapack_ffi::lapacke::lapacke_func(
            order as _,
            jobvl as _,
            jobvr as _,
            n as _,
            a as *mut _,
            lda as _,
            b as *mut _,
            ldb as _,
            ptr_alpha as *mut _,
            ptr_beta as *mut _,
            vl as *mut _,
            ldvl as _,
            vr as *mut _,
            ldvr as _,
        );

        // Copy complex eigenvalues to alphar (real part) and alphai (imaginary part)
        // For complex GGEV, both alpha and beta are complex
        // Eigenvalue = alpha/beta
        for i in 0..n {
            let val_alpha = alpha[i];
            *alphar.add(i) = val_alpha.re;
            *alphai.add(i) = val_alpha.im;

            // For beta, we store both real and imaginary parts
            let val_beta = beta_complex[i];
            *beta.add(i) = val_beta.re;
            *betai.add(i) = val_beta.im;
        }

        result
    }
}