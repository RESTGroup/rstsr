use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

/// LAPACK GGEV driver API for generalized eigenvalue problem.
///
/// Computes eigenvalues and optionally left/right eigenvectors for the generalized
/// eigenvalue problem: A*v = λ*B*v.
///
/// Eigenvalues are returned as (alpha, beta) pairs where λ = alpha/beta.
///
/// For real types (f32, f64), eigenvalues are returned as (alphar, alphai, beta).
/// For complex types, eigenvalues are returned as (alpha, beta) complex arrays.
pub trait GGEVDriverAPI<T>
where
    T: BlasFloat,
{
    /// LAPACK GGEV driver function for real types.
    ///
    /// # Safety
    ///
    /// This function calls LAPACK routines directly and modifies memory in place.
    unsafe fn driver_ggev(
        order: FlagOrder,
        jobvl: char,
        jobvr: char,
        n: usize,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        // For real types: alphar, alphai, beta are separate arrays
        // For complex types: alpha and beta are complex arrays
        alphar: *mut T::Real,
        alphai: *mut T::Real,
        beta: *mut T::Real,
        vl: *mut T,
        ldvl: usize,
        vr: *mut T,
        ldvr: usize,
    ) -> blas_int;
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct GGEV_<'a, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
    #[builder(setter(into))]
    pub b: TensorReference<'a, T, B, Ix2>,

    #[builder(setter(into), default = "false")]
    pub left: bool,
    #[builder(setter(into), default = "true")]
    pub right: bool,
}

impl<'a, B, T> GGEV_<'a, B, T>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + GGEVDriverAPI<T>,
{
    pub fn internal_run(
        self,
    ) -> Result<(
        Tensor<T::Real, B, Ix1>,
        Tensor<T::Real, B, Ix1>,
        Tensor<T::Real, B, Ix1>,
        Option<Tensor<T, B, Ix2>>,
        Option<Tensor<T, B, Ix2>>,
        TensorMutable<'a, T, B, Ix2>,
        TensorMutable<'a, T, B, Ix2>,
    )> {
        let Self { a, b, left, right } = self;

        rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
        let device = a.device().clone();
        let order = if a.f_prefer() && !a.c_prefer() { ColMajor } else { RowMajor };
        let mut a = overwritable_convert_with_order(a, order)?;
        let mut b = overwritable_convert_with_order(b, order)?;

        let [n, m] = *a.view().shape();
        rstsr_assert_eq!(n, m, InvalidLayout)?;
        rstsr_assert_eq!(b.view().shape(), &[n, n], InvalidLayout)?;

        let lda = a.view().ld(order).unwrap();
        let ldb = b.view().ld(order).unwrap();

        // Determine jobvl and jobvr
        let jobvl = if left { 'V' } else { 'N' };
        let jobvr = if right { 'V' } else { 'N' };

        // Allocate output arrays
        // For real and complex types, we use alphar, alphai, beta layout
        // (for complex, alpha_re, alpha_im, beta_re are stored in these arrays)
        let mut alphar = unsafe { empty_f(([n].c(), &device))?.into_dim::<Ix1>() };
        let mut alphai = unsafe { empty_f(([n].c(), &device))?.into_dim::<Ix1>() };
        let mut beta = unsafe { empty_f(([n].c(), &device))?.into_dim::<Ix1>() };

        let ptr_a = a.view_mut().as_mut_ptr();
        let ptr_b = b.view_mut().as_mut_ptr();
        let ptr_alphar = alphar.as_mut_ptr();
        let ptr_alphai = alphai.as_mut_ptr();
        let ptr_beta = beta.as_mut_ptr();

        // Allocate vl and vr if needed
        let mut vl = if left { Some(unsafe { empty_f(([n, n].c(), &device))?.into_dim::<Ix2>() }) } else { None };
        let mut vr = if right { Some(unsafe { empty_f(([n, n].c(), &device))?.into_dim::<Ix2>() }) } else { None };

        let ptr_vl = vl.as_mut().map(|v| v.view_mut().as_mut_ptr()).unwrap_or(std::ptr::null_mut());
        let ptr_vr = vr.as_mut().map(|v| v.view_mut().as_mut_ptr()).unwrap_or(std::ptr::null_mut());

        let ldvl = if left { n } else { 1 };
        let ldvr = if right { n } else { 1 };

        // run driver
        let info = unsafe {
            B::driver_ggev(
                order, jobvl, jobvr, n, ptr_a, lda, ptr_b, ldb, ptr_alphar, ptr_alphai, ptr_beta, ptr_vl, ldvl, ptr_vr,
                ldvr,
            )
        };
        let info = info as i32;
        if info != 0 {
            rstsr_errcode!(info, "Lapack GGEV")?;
        }

        Ok((alphar, alphai, beta, vl, vr, a.clone_to_mut(), b.clone_to_mut()))
    }

    pub fn run(
        self,
    ) -> Result<(
        Tensor<T::Real, B, Ix1>,
        Tensor<T::Real, B, Ix1>,
        Tensor<T::Real, B, Ix1>,
        Option<Tensor<T, B, Ix2>>,
        Option<Tensor<T, B, Ix2>>,
        TensorMutable<'a, T, B, Ix2>,
        TensorMutable<'a, T, B, Ix2>,
    )> {
        self.internal_run()
    }
}

pub type GGEV<'a, B, T> = GGEV_Builder<'a, B, T>;
pub type SGGEV<'a, B> = GGEV<'a, B, f32>;
pub type DGGEV<'a, B> = GGEV<'a, B, f64>;
pub type CGGEV<'a, B> = GGEV<'a, B, Complex<f32>>;
pub type ZGGEV<'a, B> = GGEV<'a, B, Complex<f64>>;
