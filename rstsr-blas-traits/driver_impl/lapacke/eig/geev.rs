use crate::lapack_ffi;
use crate::DeviceBLAS;
use duplicate::duplicate_item;
use num::complex::ComplexFloat;
use num::{Complex, Zero};
use rstsr_blas_traits::prelude::*;
use rstsr_blas_traits::util::util_eig::{pack_eigenvalues, unpack_real_eigvecs};
use rstsr_common::prelude::*;
use std::slice::from_raw_parts_mut;

#[duplicate_item(
    T     lapacke_func   ;
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
        vl: *mut Complex<T>,
        ldvl: usize,
        vr: *mut Complex<T>,
        ldvr: usize,
    ) -> blas_int {
        // real LAPACKE slots (WR/WI and the pair-packed VL/VR), assembled into
        // the complex slots below
        let mut wr: Vec<T> = vec![<T as Zero>::zero(); n];
        let mut wi: Vec<T> = vec![<T as Zero>::zero(); n];
        let mut vl_real = if jobvl == 'V' { Some(vec![<T as Zero>::zero(); ldvl * n]) } else { None };
        let mut vr_real = if jobvr == 'V' { Some(vec![<T as Zero>::zero(); ldvr * n]) } else { None };
        let ptr_vl_real = vl_real.as_mut().map_or(std::ptr::null_mut(), |v| v.as_mut_ptr());
        let ptr_vr_real = vr_real.as_mut().map_or(std::ptr::null_mut(), |v| v.as_mut_ptr());

        let info = lapack_ffi::lapacke::lapacke_func(
            order as _,
            jobvl as _,
            jobvr as _,
            n as _,
            a,
            lda as _,
            wr.as_mut_ptr(),
            wi.as_mut_ptr(),
            ptr_vl_real,
            ldvl as _,
            ptr_vr_real,
            ldvr as _,
        );
        if info != 0 {
            return info;
        }

        pack_eigenvalues(&wr, &wi, from_raw_parts_mut(w, n));
        if let Some(vl_real) = vl_real {
            unpack_real_eigvecs(order, n, ldvl, &vl_real, &wi, from_raw_parts_mut(vl, n * ldvl));
        }
        if let Some(vr_real) = vr_real {
            unpack_real_eigvecs(order, n, ldvr, &vr_real, &wi, from_raw_parts_mut(vr, n * ldvr));
        }
        info
    }
}

#[duplicate_item(
    T              lapacke_func   ;
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
        w: *mut Complex<<T as ComplexFloat>::Real>,
        vl: *mut Complex<<T as ComplexFloat>::Real>,
        ldvl: usize,
        vr: *mut Complex<<T as ComplexFloat>::Real>,
        ldvr: usize,
    ) -> blas_int {
        lapack_ffi::lapacke::lapacke_func(
            order as _,
            jobvl as _,
            jobvr as _,
            n as _,
            a as *mut _,
            lda as _,
            w as *mut _,
            vl as *mut _,
            ldvl as _,
            vr as *mut _,
            ldvr as _,
        )
    }
}
