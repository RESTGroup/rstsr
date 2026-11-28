use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait ORGQRDriverAPI<T> {
    unsafe fn driver_orgqr(
        order: FlagOrder,
        m: usize,
        n: usize,
        k: usize,
        a: *mut T,
        lda: usize,
        tau: *const T,
    ) -> blas_int;
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct ORGQR_<'a, 'tau, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
    #[builder(setter(into))]
    pub tau: TensorView<'tau, T, B, Ix1>,
    /// Number of reflectors applied (at most `a.ncol()`); defaults to `a.ncol()`.
    /// Columns of `a` beyond `k` receive an orthonormal completion.
    #[builder(setter(into, strip_option), default = "None")]
    pub k: Option<usize>,
}

impl<'a, 'tau, B, T> ORGQR_<'a, 'tau, B, T>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + ORGQRDriverAPI<T>,
{
    /// Generate the first `k` columns of `Q` from the output of [`GEQRF_`]; the
    /// result overwrites `a`.
    pub fn internal_run(self) -> Result<TensorMutable2<'a, T, B>> {
        let Self { a, tau, k } = self;

        let device = a.device().clone();
        let order = get_output_order(&[], &[(a.c_prefer(), a.f_prefer())], device.default_order());
        let mut a = overwritable_convert_with_order(a, order)?;
        let tau = tau.into_contig_f(ColMajor)?;

        let [m, n] = *a.view().shape();
        let k = k.unwrap_or(n);

        // perform check
        rstsr_assert!(m >= n && n >= k, InvalidLayout, "Lapack ORGQR: requires m >= n >= k, got {m} x {n}, k = {k}")?;
        rstsr_assert!(tau.size() >= k, InvalidLayout, "Lapack ORGQR: tau must have at least k elements")?;

        // nothing to generate
        if k == 0 || m == 0 {
            return Ok(a.clone_to_mut());
        }

        let lda = a.view().testify_ld(order, "A")?;
        let ptr_a = a.view_mut().as_mut_ptr();
        let ptr_tau = tau.as_ptr();

        // run driver
        let info = unsafe { B::driver_orgqr(order, m, n, k, ptr_a, lda, ptr_tau) };
        map_lapack_info(info, "Lapack ORGQR")?;

        Ok(a.clone_to_mut())
    }

    pub fn run(self) -> Result<TensorMutable2<'a, T, B>> {
        self.internal_run()
    }
}

pub type ORGQR<'a, 'tau, B, T> = ORGQR_Builder<'a, 'tau, B, T>;
pub type SORGQR<'a, 'tau, B> = ORGQR<'a, 'tau, B, f32>;
pub type DORGQR<'a, 'tau, B> = ORGQR<'a, 'tau, B, f64>;
pub type CORGQR<'a, 'tau, B> = ORGQR<'a, 'tau, B, Complex<f32>>;
pub type ZORGQR<'a, 'tau, B> = ORGQR<'a, 'tau, B, Complex<f64>>;
