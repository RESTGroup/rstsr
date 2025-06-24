use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait POTRFDriverAPI<T> {
    unsafe fn driver_potrf(order: FlagOrder, uplo: FlagUpLo, n: usize, a: *mut T, lda: usize) -> blas_int;
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct POTRF_<'a, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,

    #[builder(setter(into), default = "None")]
    pub uplo: Option<FlagUpLo>,
}

impl<'a, B, T> POTRF_<'a, B, T>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + POTRFDriverAPI<T>,
{
    pub fn internal_run(self) -> Result<TensorMutable2<'a, T, B>> {
        let Self { a, uplo } = self;

        let device = a.device().clone();
        let uplo = uplo.unwrap_or_else(|| match device.default_order() {
            RowMajor => Lower,
            ColMajor => Upper,
        });
        let mut a = overwritable_convert(a)?;
        let order = if a.f_prefer() && !a.c_prefer() { ColMajor } else { RowMajor };

        // perform check
        rstsr_assert_eq!(a.view().nrow(), a.view().ncol(), InvalidLayout, "Lapack POTRF: A must be square")?;

        let n = a.view().nrow();
        let lda = a.view().ld(order).unwrap();
        let ptr_a = a.view_mut().as_mut_ptr();

        // run driver
        let info = unsafe { B::driver_potrf(order, uplo, n, ptr_a, lda) };
        let info = info as i32;
        if info != 0 {
            rstsr_errcode!(info, "Lapack POTRF")?;
        }

        Ok(a)
    }

    pub fn run(self) -> Result<TensorMutable2<'a, T, B>> {
        self.internal_run()
    }
}

pub type POTRF<'a, B, T> = POTRF_Builder<'a, B, T>;
pub type SPOTRF<'a, B> = POTRF<'a, B, f32>;
pub type DPOTRF<'a, B> = POTRF<'a, B, f64>;
pub type CPOTRF<'a, B> = POTRF<'a, B, Complex<f32>>;
pub type ZPOTRF<'a, B> = POTRF<'a, B, Complex<f64>>;
