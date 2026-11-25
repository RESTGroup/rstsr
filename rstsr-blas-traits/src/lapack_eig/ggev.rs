use crate::lapack_eig::{ComplexEigvals, ComplexEigvecs};
use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait GGEVDriverAPI<T>
where
    T: BlasFloat,
{
    /// `a` and `b` are destroyed. For a real `T` the vendor implementation
    /// assembles the complex `alpha`/`vl`/`vr` slots from LAPACK's real
    /// `ALPHAR`/`ALPHAI`/`VL`/`VR`. `beta` has the routine's own precision
    /// (real for `S/DGGEV`, complex for `C/ZGGEV`, as in LAPACK); `vl`/`vr`
    /// are not referenced when the matching job is `'N'`.
    unsafe fn driver_ggev(
        order: FlagOrder,
        jobvl: char,
        jobvr: char,
        n: usize,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        alpha: *mut Complex<T::Real>,
        beta: *mut T,
        vl: *mut Complex<T::Real>,
        ldvl: usize,
        vr: *mut Complex<T::Real>,
        ldvr: usize,
    ) -> blas_int;
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct GGEV_<'a, 'b, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
    #[builder(setter(into))]
    pub b: TensorReference<'b, T, B, Ix2>,

    /// Compute the left eigenvectors (`uᴴ A = λ uᴴ B`); defaults to `false`.
    #[builder(setter(into), default = "false")]
    pub left: bool,
    /// Compute the right eigenvectors (`A v = λ B v`); defaults to `true`.
    #[builder(setter(into), default = "true")]
    pub right: bool,
}

impl<'a, 'b, B, T> GGEV_<'a, 'b, B, T>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T>
        + GGEVDriverAPI<T>
        + DeviceAPI<Complex<T::Real>, Raw = Vec<Complex<T::Real>>>
        + DeviceCreationAnyAPI<Complex<T::Real>>
        + DeviceCreationNumAPI<Complex<T::Real>>,
{
    /// Run `?GGEV` for `A v = λ B v`: the `(alpha, beta)` pair with
    /// `lambda = alpha / beta` and the requested eigenvector blocks, all in
    /// the call order; both inputs are destroyed. LAPACK's scaling of the
    /// vectors (largest component with `abs(re) + abs(im) = 1`) is kept.
    /// `beta` is real for a real `T` and complex otherwise (LAPACK's `BETA`).
    pub fn internal_run(
        self,
    ) -> Result<(ComplexEigvals<B, T>, Tensor<T, B, Ix1>, Option<ComplexEigvecs<B, T>>, Option<ComplexEigvecs<B, T>>)>
    {
        let Self { a, b, left, right } = self;

        rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
        let device = a.device().clone();
        let order = get_output_order(
            &[],
            &[(a.c_prefer(), a.f_prefer()), (b.c_prefer(), b.f_prefer())],
            device.default_order(),
        );
        let mut a = overwritable_convert_with_order(a, order)?;
        let mut b = overwritable_convert_with_order(b, order)?;

        let [n, m] = *a.view().shape();
        rstsr_assert_eq!(n, m, InvalidLayout, "Lapack GGEV: A must be square, got {n} x {m}")?;
        rstsr_assert_eq!(b.view().shape(), &[n, n], InvalidLayout, "Lapack GGEV: B must be {n} x {n}")?;

        // empty problem: no eigenvalues, empty eigenvector blocks (the driver is not called)
        if n == 0 {
            let alpha = zeros_f(([0], &device))?.into_dim::<Ix1>();
            let beta = zeros_f(([0], &device))?.into_dim::<Ix1>();
            let vl = if left { Some(zeros_f(([0, 0], &device))?.into_dim::<Ix2>()) } else { None };
            let vr = if right { Some(zeros_f(([0, 0], &device))?.into_dim::<Ix2>()) } else { None };
            return Ok((alpha, beta, vl, vr));
        }

        let lda = a.view().testify_ld(order, "A")?;
        let ldb = b.view().testify_ld(order, "B")?;
        let ptr_a = a.view_mut().as_mut_ptr();
        let ptr_b = b.view_mut().as_mut_ptr();
        let mut alpha = unsafe { empty_f(([n].c(), &device))?.into_dim::<Ix1>() };
        let mut beta = unsafe { empty_f(([n].c(), &device))?.into_dim::<Ix1>() };
        let ptr_alpha = alpha.as_mut_ptr();
        let ptr_beta = beta.as_mut_ptr();
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
            B::driver_ggev(
                order,
                if left { 'V' } else { 'N' },
                if right { 'V' } else { 'N' },
                n,
                ptr_a,
                lda,
                ptr_b,
                ldb,
                ptr_alpha,
                ptr_beta,
                ptr_vl,
                ldvl,
                ptr_vr,
                ldvr,
            )
        };
        map_lapack_info(info, "Lapack GGEV")?;

        Ok((alpha, beta, vl, vr))
    }

    pub fn run(
        self,
    ) -> Result<(ComplexEigvals<B, T>, Tensor<T, B, Ix1>, Option<ComplexEigvecs<B, T>>, Option<ComplexEigvecs<B, T>>)>
    {
        self.internal_run()
    }
}

pub type GGEV<'a, 'b, B, T> = GGEV_Builder<'a, 'b, B, T>;
pub type SGGEV<'a, 'b, B> = GGEV<'a, 'b, B, f32>;
pub type DGGEV<'a, 'b, B> = GGEV<'a, 'b, B, f64>;
pub type CGGEV<'a, 'b, B> = GGEV<'a, 'b, B, Complex<f32>>;
pub type ZGGEV<'a, 'b, B> = GGEV<'a, 'b, B, Complex<f64>>;
