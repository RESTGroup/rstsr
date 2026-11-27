use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait GETRFDriverAPI<T> {
    unsafe fn driver_getrf(
        order: FlagOrder,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
    ) -> blas_int;
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct GETRF_<'a, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
}

impl<'a, B, T> GETRF_<'a, B, T>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + GETRFDriverAPI<T>,
{
    /// LU factorization without the singularity check: the factorized matrix
    /// (`L` below the diagonal, unit diagonal not stored, `U` on and above it),
    /// the 0-based pivots and LAPACK's `info`.
    ///
    /// `info > 0` reports `U(info, info)` exactly zero: the factorization is
    /// complete and the factors are valid, `U` is singular.
    pub fn run_with_info(self) -> Result<(TensorMutable2<'a, T, B>, Tensor<blas_int, B, Ix1>, blas_int)> {
        let Self { a } = self;

        let device = a.device().clone();
        let mut a = overwritable_convert(a)?;
        let order = get_output_order(&[], &[(a.c_prefer(), a.f_prefer())], device.default_order());

        let [m, n] = *a.view().shape();
        let k = m.min(n);

        // empty A: no pivots
        if k == 0 {
            let ipiv = unsafe { empty_f(([k].c(), &device))?.into_dim::<Ix1>() };
            return Ok((a.clone_to_mut(), ipiv, 0));
        }

        let lda = a.view().testify_ld(order, "A")?;
        let mut ipiv = unsafe { empty_f(([k].c(), &device))?.into_dim::<Ix1>() };
        let ptr_a = a.view_mut().as_mut_ptr();
        let ptr_ipiv = ipiv.as_mut_ptr();

        // run driver
        let info = unsafe { B::driver_getrf(order, m, n, ptr_a, lda, ptr_ipiv) };
        // an illegal argument or a failed row-major bridge leaves no usable factors
        if info < 0 {
            map_lapack_info(info, "Lapack GETRF")?;
        }

        // rust is 0-indexed
        ipiv -= 1;

        Ok((a.clone_to_mut(), ipiv, info))
    }

    /// LU factorization; an exactly-zero pivot (`info > 0`) is an error.
    pub fn internal_run(self) -> Result<(TensorMutable2<'a, T, B>, Tensor<blas_int, B, Ix1>)> {
        let (a, ipiv, info) = self.run_with_info()?;
        map_lapack_info(info, "Lapack GETRF")?;
        Ok((a, ipiv))
    }

    pub fn run(self) -> Result<(TensorMutable2<'a, T, B>, Tensor<blas_int, B, Ix1>)> {
        self.internal_run()
    }
}

pub type GETRF<'a, B, T> = GETRF_Builder<'a, B, T>;
pub type SGETRF<'a, B> = GETRF<'a, B, f32>;
pub type DGETRF<'a, B> = GETRF<'a, B, f64>;
pub type CGETRF<'a, B> = GETRF<'a, B, Complex<f32>>;
pub type ZGETRF<'a, B> = GETRF<'a, B, Complex<f64>>;
