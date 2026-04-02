use crate::prelude_dev::*;
use num::Complex;
use rstsr_core::prelude_dev::*;

/// LAPACK GEEV driver API for general eigenvalue problem.
///
/// Computes eigenvalues and optionally left/right eigenvectors of a general matrix.
///
/// Eigenvalues are returned as complex values (w) for both real and complex types.
/// For real types, LAPACK returns eigenvalues as (wr, wi) pairs; the driver converts
/// them to complex format internally.
pub trait GEEVDriverAPI<T>
where
    T: BlasFloat,
{
    /// LAPACK GEEV driver function.
    ///
    /// # Safety
    ///
    /// This function calls LAPACK routines directly and modifies memory in place.
    ///
    /// # Parameters
    /// - `w`: Pointer to complex eigenvalue array of length n. For real types, this will be
    ///   populated from LAPACK's wr/wi outputs. For complex types, this directly receives LAPACK's
    ///   complex w output.
    unsafe fn driver_geev(
        order: FlagOrder,
        jobvl: char,
        jobvr: char,
        n: usize,
        a: *mut T,
        lda: usize,
        w: *mut Complex<T::Real>,
        vl: *mut T,
        ldvl: usize,
        vr: *mut T,
        ldvr: usize,
    ) -> blas_int;
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct GEEV_<'a, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,

    #[builder(setter(into), default = "false")]
    pub left: bool,
    #[builder(setter(into), default = "true")]
    pub right: bool,
}

impl<'a, B, T> GEEV_<'a, B, T>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + GEEVDriverAPI<T>,
    B: DeviceAPI<Complex<T::Real>>,
    B: DeviceCreationAnyAPI<Complex<T::Real>>,
    B: DeviceRawAPI<Complex<T::Real>, Raw = Vec<Complex<T::Real>>>,
{
    pub fn internal_run(
        self,
    ) -> Result<(
        Tensor<Complex<T::Real>, B, Ix1>,
        Option<Tensor<T, B, Ix2>>,
        Option<Tensor<T, B, Ix2>>,
        TensorMutable<'a, T, B, Ix2>,
    )> {
        let Self { a, left, right } = self;

        let device = a.device().clone();
        let mut a = overwritable_convert(a)?;
        let order = if a.f_prefer() && !a.c_prefer() { ColMajor } else { RowMajor };

        let [n, m] = *a.view().shape();
        rstsr_assert_eq!(n, m, InvalidLayout)?;

        let lda = a.view().ld(order).unwrap();

        // Determine jobvl and jobvr
        let jobvl = if left { 'V' } else { 'N' };
        let jobvr = if right { 'V' } else { 'N' };

        // Allocate complex eigenvalue array
        let mut w = unsafe { empty_f(([n].c(), &device))?.into_dim::<Ix1>() };

        let ptr_a = a.view_mut().as_mut_ptr();
        let ptr_w = w.as_mut_ptr();

        // Allocate vl and vr if needed
        let mut vl = if left { Some(unsafe { empty_f(([n, n].c(), &device))?.into_dim::<Ix2>() }) } else { None };
        let mut vr = if right { Some(unsafe { empty_f(([n, n].c(), &device))?.into_dim::<Ix2>() }) } else { None };

        let ptr_vl = vl.as_mut().map(|v| v.view_mut().as_mut_ptr()).unwrap_or(std::ptr::null_mut());
        let ptr_vr = vr.as_mut().map(|v| v.view_mut().as_mut_ptr()).unwrap_or(std::ptr::null_mut());

        let ldvl = if left { n } else { 1 };
        let ldvr = if right { n } else { 1 };

        // run driver
        let info = unsafe { B::driver_geev(order, jobvl, jobvr, n, ptr_a, lda, ptr_w, ptr_vl, ldvl, ptr_vr, ldvr) };
        let info = info as i32;
        if info != 0 {
            rstsr_errcode!(info, "Lapack GEEV")?;
        }

        Ok((w, vl, vr, a.clone_to_mut()))
    }

    pub fn run(
        self,
    ) -> Result<(
        Tensor<Complex<T::Real>, B, Ix1>,
        Option<Tensor<T, B, Ix2>>,
        Option<Tensor<T, B, Ix2>>,
        TensorMutable<'a, T, B, Ix2>,
    )> {
        self.internal_run()
    }
}

pub type GEEV<'a, B, T> = GEEV_Builder<'a, B, T>;
pub type SGEEV<'a, B> = GEEV<'a, B, f32>;
pub type DGEEV<'a, B> = GEEV<'a, B, f64>;
pub type CGEEV<'a, B> = GEEV<'a, B, Complex<f32>>;
pub type ZGEEV<'a, B> = GEEV<'a, B, Complex<f64>>;
