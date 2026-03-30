use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait ORGQRDriverAPI<T>
where
    T: BlasFloat,
{
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
pub struct ORGQR_<'a, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
    #[builder(setter(into))]
    pub tau: TensorReference<'a, T, B, Ix1>,
    #[builder(default)]
    pub k: Option<usize>,
}

impl<'a, B, T> ORGQR_<'a, B, T>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + ORGQRDriverAPI<T>,
{
    pub fn internal_run(self) -> Result<TensorMutable<'a, T, B, Ix2>> {
        let Self { a, tau, k } = self;

        let device = a.device().clone();
        let order = match (a.c_prefer(), a.f_prefer()) {
            (true, false) => RowMajor,
            (false, true) => ColMajor,
            (false, false) | (true, true) => device.default_order(),
        };
        let mut a = overwritable_convert_with_order(a, order)?;
        let [m, n] = *a.view().shape();
        let k = k.unwrap_or_else(|| {
            let [tau_len] = *tau.view().shape();
            tau_len.min(n)
        });

        rstsr_assert!(k <= n, InvalidLayout, "ORGQR: k must be <= n")?;
        {
            let [tau_len] = *tau.view().shape();
            rstsr_assert!(k <= tau_len, InvalidLayout, "ORGQR: k must be <= tau.len()")?;
        }

        let lda = a.view().ld(order).unwrap();

        let ptr_a = a.view_mut().as_mut_ptr();
        let ptr_tau = tau.as_ptr();

        // run driver
        let info = unsafe { B::driver_orgqr(order, m, n, k, ptr_a, lda, ptr_tau) };
        let info = info as i32;
        if info != 0 {
            rstsr_errcode!(info, "Lapack ORGQR")?;
        }

        Ok(a.clone_to_mut())
    }

    pub fn run(self) -> Result<TensorMutable<'a, T, B, Ix2>> {
        self.internal_run()
    }
}

pub type ORGQR<'a, B, T> = ORGQR_Builder<'a, B, T>;
pub type SORGQR<'a, B> = ORGQR<'a, B, f32>;
pub type DORGQR<'a, B> = ORGQR<'a, B, f64>;
pub type CUNGQR<'a, B> = ORGQR<'a, B, Complex<f32>>;
pub type ZUNGQR<'a, B> = ORGQR<'a, B, Complex<f64>>;
