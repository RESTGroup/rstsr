use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::complex::ComplexFloat;
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
        wr: *mut T,
        wi: *mut T,
        vl: *mut T,
        ldvl: usize,
        vr: *mut T,
        ldvr: usize,
    ) -> blas_int {
        let result = lapack_ffi::lapacke::lapacke_func(
            order as _, jobvl as _, jobvr as _, n as _, a, lda as _, wr, wi, vl, ldvl as _, vr, ldvr as _,
        );
        result
    }
}

#[duplicate_item(
    T              lapacke_func  ;
   [Complex<f32>] [LAPACKE_cgeev];
   [Complex<f64>] [LAPACKE_zgeev];
)]
impl GEEVDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_geev(
        order: FlagOrder,
        jobvl: char,
        jobvr: char,
        n: usize,
        a: *mut T,
        lda: usize,
        wr: *mut <T as ComplexFloat>::Real,
        wi: *mut <T as ComplexFloat>::Real,
        vl: *mut T,
        ldvl: usize,
        vr: *mut T,
        ldvr: usize,
    ) -> blas_int {
        // For complex types, LAPACKE_cgeev and LAPACKE_zgeev have a single complex eigenvalue array w
        // We need to allocate a temporary complex array, call LAPACKE, then split into wr and wi

        // Allocate temporary complex array for eigenvalues
        let mut w: Vec<T> = match uninitialized_vec(n) {
            Ok(w) => w,
            Err(_) => return -1010,
        };

        let result = lapack_ffi::lapacke::lapacke_func(
            order as _,
            jobvl as _,
            jobvr as _,
            n as _,
            a as *mut _,
            lda as _,
            w.as_mut_ptr() as *mut _,
            vl as *mut _,
            ldvl as _,
            vr as *mut _,
            ldvr as _,
        );

        // Copy complex eigenvalues to wr (real part) and wi (imaginary part)
        for i in 0..n {
            let val = w[i];
            *wr.add(i) = val.re;
            *wi.add(i) = val.im;
        }

        result
    }
}
