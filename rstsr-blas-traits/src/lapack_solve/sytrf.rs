use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait SYTRFDriverAPI<T, const HERMI: bool> {
    unsafe fn driver_sytrf(
        order: FlagOrder,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
    ) -> blas_int;
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct SYTRF_<'a, B, T, const HERMI: bool>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,

    #[builder(setter(into), default = "None")]
    pub uplo: Option<FlagUpLo>,
}

impl<'a, B, T, const HERMI: bool> SYTRF_<'a, B, T, HERMI>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + SYTRFDriverAPI<T, HERMI>,
{
    /// Bunch–Kaufman factorization without the singularity check: the packed
    /// factors (`?SYTRF` / `?HETRF`), LAPACK's signed `ipiv` and `info`.
    ///
    /// `info > 0` reports `D(info, info)` exactly zero: the factorization is
    /// complete and the factors are valid, `D` is singular.
    pub fn run_with_info(self) -> Result<(TensorMutable2<'a, T, B>, Tensor<blas_int, B, Ix1>, blas_int)> {
        let Self { a, uplo } = self;

        let device = a.device().clone();
        let mut a = overwritable_convert(a)?;
        let order = get_output_order(&[], &[(a.c_prefer(), a.f_prefer())], device.default_order());
        let uplo = uplo.unwrap_or_else(|| match device.default_order() {
            RowMajor => Lower,
            ColMajor => Upper,
        });

        let [n, _] = *a.view().shape();
        rstsr_assert_eq!(a.view().shape(), &[n, n], InvalidLayout, "SYTRF: A shape")?;

        // empty A: no pivots
        if n == 0 {
            let ipiv = unsafe { empty_f(([n], &device))?.into_dim::<Ix1>() };
            return Ok((a.clone_to_mut(), ipiv, 0));
        }

        let lda = a.view().testify_ld(order, "A")?;
        let mut ipiv = unsafe { empty_f(([n], &device))?.into_dim::<Ix1>() };
        let ptr_a = a.view_mut().as_mut_ptr();
        let ptr_ipiv = ipiv.as_mut_ptr();

        // run driver
        let info = unsafe { B::driver_sytrf(order, uplo, n, ptr_a, lda, ptr_ipiv) };
        // an illegal argument or a failed row-major bridge leaves no usable factors
        if info < 0 {
            map_lapack_info(info, "Lapack SYTRF")?;
        }

        Ok((a.clone_to_mut(), ipiv, info))
    }

    /// Bunch–Kaufman factorization; an exactly-zero pivot (`info > 0`) is an
    /// error. Use [`SYTRF_::run_with_info`] to obtain the factors.
    pub fn internal_run(self) -> Result<(TensorMutable2<'a, T, B>, Tensor<blas_int, B, Ix1>)> {
        let (a, ipiv, info) = self.run_with_info()?;
        map_lapack_info(info, "Lapack SYTRF")?;
        Ok((a, ipiv))
    }

    pub fn run(self) -> Result<(TensorMutable2<'a, T, B>, Tensor<blas_int, B, Ix1>)> {
        self.internal_run()
    }
}

pub type SYTRF<'a, B, T, const HERMI: bool> = SYTRF_Builder<'a, B, T, HERMI>;
pub type SSYTRF<'a, B> = SYTRF<'a, B, f32, false>;
pub type DSYTRF<'a, B> = SYTRF<'a, B, f64, false>;
pub type CSYTRF<'a, B> = SYTRF<'a, B, Complex<f32>, false>;
pub type ZSYTRF<'a, B> = SYTRF<'a, B, Complex<f64>, false>;
pub type CHETRF<'a, B> = SYTRF<'a, B, Complex<f32>, true>;
pub type ZHETRF<'a, B> = SYTRF<'a, B, Complex<f64>, true>;
