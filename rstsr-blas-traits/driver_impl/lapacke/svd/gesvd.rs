use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::complex::ComplexFloat;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude::*;

#[duplicate_item(
    T     lapacke_func   ;
   [f32] [LAPACKE_sgesvd];
   [f64] [LAPACKE_dgesvd];
)]
impl GESVDDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_gesvd(
        order: FlagOrder,
        jobu: char,
        jobvt: char,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        s: *mut T,
        u: *mut T,
        ldu: usize,
        vt: *mut T,
        ldvt: usize,
        superb: *mut T,
    ) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(
            order as _, jobu as _, jobvt as _, m as _, n as _, a, lda as _, s, u, ldu as _, vt, ldvt as _, superb,
        )
    }
}

#[duplicate_item(
    T              lapacke_func   ;
   [Complex<f32>] [LAPACKE_cgesvd];
   [Complex<f64>] [LAPACKE_zgesvd];
)]
impl GESVDDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_gesvd(
        order: FlagOrder,
        jobu: char,
        jobvt: char,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        s: *mut <T as ComplexFloat>::Real,
        u: *mut T,
        ldu: usize,
        vt: *mut T,
        ldvt: usize,
        superb: *mut <T as ComplexFloat>::Real,
    ) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(
            order as _,
            jobu as _,
            jobvt as _,
            m as _,
            n as _,
            a as *mut _,
            lda as _,
            s,
            u as *mut _,
            ldu as _,
            vt as *mut _,
            ldvt as _,
            superb,
        )
    }
}
