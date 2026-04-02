use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::{Complex, Zero};
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
        alpha: *mut Complex<T>,
        beta: *mut Complex<T>,
        vl: *mut T,
        ldvl: usize,
        vr: *mut T,
        ldvr: usize,
    ) -> blas_int {
        // For real types, LAPACKE_sggev/dggev returns alphar, alphai, beta separately.
        // beta is real for real GGEV, so we set its imaginary part to 0.
        let mut alphar: Vec<T> = match uninitialized_vec(n) {
            Ok(alphar) => alphar,
            Err(_) => return -1010,
        };
        let mut alphai: Vec<T> = match uninitialized_vec(n) {
            Ok(alphai) => alphai,
            Err(_) => return -1010,
        };
        let mut beta_real: Vec<T> = match uninitialized_vec(n) {
            Ok(beta) => beta,
            Err(_) => return -1010,
        };

        let result = lapack_ffi::lapacke::lapacke_func(
            order as _,
            jobvl as _,
            jobvr as _,
            n as _,
            a,
            lda as _,
            b,
            ldb as _,
            alphar.as_mut_ptr(),
            alphai.as_mut_ptr(),
            beta_real.as_mut_ptr(),
            vl,
            ldvl as _,
            vr,
            ldvr as _,
        );

        // Combine into complex: alpha = Complex(alphar, alphai), beta = Complex(beta, 0)
        for i in 0..n {
            *alpha.add(i) = Complex::new(alphar[i], alphai[i]);
            *beta.add(i) = Complex::new(beta_real[i], T::zero());
        }

        result
    }
}

#[duplicate_item(
    T              lapacke_func  ;
   [Complex::<f32>] [LAPACKE_cggev];
   [Complex::<f64>] [LAPACKE_zggev];
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
        alpha: *mut T, // For complex types, T = Complex<T::Real>
        beta: *mut T,  // For complex types, T = Complex<T::Real>
        vl: *mut T,
        ldvl: usize,
        vr: *mut T,
        ldvr: usize,
    ) -> blas_int {
        // For complex types, LAPACKE_cggev/zggev has complex alpha and beta arrays.
        // Both alpha and beta are complex, matching our interface directly.
        // T is Complex<f32> or Complex<f64>, so T = Complex<T::Real>.
        // The pointers can be cast directly.
        let result = lapack_ffi::lapacke::lapacke_func(
            order as _,
            jobvl as _,
            jobvr as _,
            n as _,
            a as *mut _,
            lda as _,
            b as *mut _,
            ldb as _,
            alpha as *mut _, // T* and Complex<T::Real>* are the same
            beta as *mut _,  // T* and Complex<T::Real>* are the same
            vl as *mut _,
            ldvl as _,
            vr as *mut _,
            ldvr as _,
        );

        result
    }
}
