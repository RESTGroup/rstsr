use crate::lapack_eig::{ComplexEigvals, ComplexEigvecs};
use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait GEEVDriverAPI<T>
where
    T: BlasFloat,
{
    /// `a` is destroyed. For a real `T` the vendor implementation assembles
    /// the complex `w`/`vl`/`vr` slots from LAPACK's real `WR`/`WI`/`VL`/`VR`;
    /// `vl`/`vr` are not referenced when the matching job is `'N'`.
    unsafe fn driver_geev(
        order: FlagOrder,
        jobvl: char,
        jobvr: char,
        n: usize,
        a: *mut T,
        lda: usize,
        w: *mut Complex<T::Real>,
        vl: *mut Complex<T::Real>,
        ldvl: usize,
        vr: *mut Complex<T::Real>,
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
    /// Compute the left eigenvectors (`uᴴ A = λ uᴴ`); defaults to `false`.
    #[builder(setter(into), default = "false")]
    pub left: bool,
    /// Compute the right eigenvectors (`A v = λ v`); defaults to `true`.
    #[builder(setter(into), default = "true")]
    pub right: bool,
}

impl<'a, B, T> GEEV_<'a, B, T>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T>
        + GEEVDriverAPI<T>
        + DeviceAPI<Complex<T::Real>, Raw = Vec<Complex<T::Real>>>
        + DeviceCreationAnyAPI<Complex<T::Real>>
        + DeviceCreationNumAPI<Complex<T::Real>>,
{
    /// Run `?GEEV`: the spectrum and the requested eigenvector blocks, all in
    /// the call order. An owned or `view_mut()` input whose layout is in the
    /// call order is consumed as scratch (destroyed); any other input is
    /// decomposed from a copy and left intact.
    pub fn internal_run(
        self,
    ) -> Result<(ComplexEigvals<B, T>, Option<ComplexEigvecs<B, T>>, Option<ComplexEigvecs<B, T>>)> {
        let Self { a, left, right } = self;

        let device = a.device().clone();
        let order = get_output_order(&[], &[(a.c_prefer(), a.f_prefer())], device.default_order());
        let mut a = overwritable_convert_with_order(a, order)?;

        let [n, m] = *a.view().shape();
        rstsr_assert_eq!(n, m, InvalidLayout, "Lapack GEEV: A must be square, got {n} x {m}")?;

        // empty A: no eigenvalues, empty eigenvector blocks (the driver is not called)
        if n == 0 {
            let w = zeros_f(([0], &device))?.into_dim::<Ix1>();
            let vl = if left { Some(zeros_f(([0, 0], &device))?.into_dim::<Ix2>()) } else { None };
            let vr = if right { Some(zeros_f(([0, 0], &device))?.into_dim::<Ix2>()) } else { None };
            return Ok((w, vl, vr));
        }

        let lda = a.view().testify_ld(order, "A")?;
        let ptr_a = a.view_mut().as_mut_ptr();
        let mut w = unsafe { empty_f(([n].c(), &device))?.into_dim::<Ix1>() };
        let ptr_w = w.as_mut_ptr();
        let mut vl = if left { Some(unsafe { empty_f(([n, n], order, &device))?.into_dim::<Ix2>() }) } else { None };
        let mut vr = if right { Some(unsafe { empty_f(([n, n], order, &device))?.into_dim::<Ix2>() }) } else { None };
        let (ptr_vl, ldvl) = match vl.as_mut() {
            Some(v) => (v.view_mut().as_mut_ptr(), v.view().testify_ld(order, "VL")?),
            None => (std::ptr::null_mut(), 1),
        };
        let (ptr_vr, ldvr) = match vr.as_mut() {
            Some(v) => (v.view_mut().as_mut_ptr(), v.view().testify_ld(order, "VR")?),
            None => (std::ptr::null_mut(), 1),
        };

        // run driver
        let info = unsafe {
            B::driver_geev(
                order,
                if left { 'V' } else { 'N' },
                if right { 'V' } else { 'N' },
                n,
                ptr_a,
                lda,
                ptr_w,
                ptr_vl,
                ldvl,
                ptr_vr,
                ldvr,
            )
        };
        map_lapack_info(info, "Lapack GEEV")?;

        Ok((w, vl, vr))
    }

    pub fn run(self) -> Result<(ComplexEigvals<B, T>, Option<ComplexEigvecs<B, T>>, Option<ComplexEigvecs<B, T>>)> {
        self.internal_run()
    }
}

pub type GEEV<'a, B, T> = GEEV_Builder<'a, B, T>;
pub type SGEEV<'a, B> = GEEV<'a, B, f32>;
pub type DGEEV<'a, B> = GEEV<'a, B, f64>;
pub type CGEEV<'a, B> = GEEV<'a, B, Complex<f32>>;
pub type ZGEEV<'a, B> = GEEV<'a, B, Complex<f64>>;
