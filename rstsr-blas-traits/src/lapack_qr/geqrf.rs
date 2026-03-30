use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait GEQRFDriverAPI<T>
where
    T: BlasFloat,
{
    unsafe fn driver_geqrf(order: FlagOrder, m: usize, n: usize, a: *mut T, lda: usize, tau: *mut T) -> blas_int;
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct GEQRF_<'a, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
}

impl<'a, B, T> GEQRF_<'a, B, T>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + GEQRFDriverAPI<T>,
{
    pub fn internal_run(self) -> Result<(TensorMutable<'a, T, B, Ix2>, Tensor<T, B, Ix1>)> {
        let Self { a } = self;

        let device = a.device().clone();
        let order = match (a.c_prefer(), a.f_prefer()) {
            (true, false) => RowMajor,
            (false, true) => ColMajor,
            (false, false) | (true, true) => device.default_order(),
        };
        let mut a = overwritable_convert_with_order(a, order)?;
        let [m, n] = *a.view().shape();
        let minmn = m.min(n);
        let lda = a.view().ld(order).unwrap();

        // Allocate tau array
        let mut tau = unsafe { empty_f(([minmn].c(), &device))?.into_dim::<Ix1>() };

        let ptr_a = a.view_mut().as_mut_ptr();
        let ptr_tau = tau.as_mut_ptr();

        // run driver
        let info = unsafe { B::driver_geqrf(order, m, n, ptr_a, lda, ptr_tau) };
        let info = info as i32;
        if info != 0 {
            rstsr_errcode!(info, "Lapack GEQRF")?;
        }

        Ok((a.clone_to_mut(), tau))
    }

    pub fn run(self) -> Result<(TensorMutable<'a, T, B, Ix2>, Tensor<T, B, Ix1>)> {
        self.internal_run()
    }
}

pub type GEQRF<'a, B, T> = GEQRF_Builder<'a, B, T>;
pub type SGEQRF<'a, B> = GEQRF<'a, B, f32>;
pub type DGEQRF<'a, B> = GEQRF<'a, B, f64>;
pub type CGEQRF<'a, B> = GEQRF<'a, B, Complex<f32>>;
pub type ZGEQRF<'a, B> = GEQRF<'a, B, Complex<f64>>;
