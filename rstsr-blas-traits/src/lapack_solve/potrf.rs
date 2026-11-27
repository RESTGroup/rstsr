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
    /// Factor the Hermitian matrix `a` in place.
    ///
    /// A c-prefer input is factored through its transposed view (with `uplo`
    /// flipped), so the driver call is column-major in both storage orders and
    /// the factor lands in the requested triangle of the buffer.
    pub fn internal_run(self) -> Result<TensorMutable2<'a, T, B>> {
        let Self { a, uplo } = self;

        let device = a.device().clone();
        let uplo = uplo.unwrap_or_else(|| match device.default_order() {
            RowMajor => Lower,
            ColMajor => Upper,
        });
        let mut a = overwritable_convert(a)?;
        let layout_order = get_output_order(&[], &[(a.c_prefer(), a.f_prefer())], device.default_order());

        // perform check
        rstsr_assert_eq!(a.view().nrow(), a.view().ncol(), InvalidLayout, "Lapack POTRF: A must be square")?;

        let n = a.view().nrow();

        // empty A: nothing to factor
        if n == 0 {
            return Ok(a);
        }

        // a c-prefer buffer is factored through its transposed view (see the docstring)
        let uplo_call = match layout_order {
            ColMajor => uplo,
            RowMajor => uplo.flip(),
        };
        let lda = a.view().testify_ld(layout_order, "A")?;
        let ptr_a = a.view_mut().as_mut_ptr();

        // run driver
        let info = unsafe { B::driver_potrf(ColMajor, uplo_call, n, ptr_a, lda) };
        map_lapack_info(info, "Lapack POTRF")?;

        Ok(a)
    }

    /// Factor the Hermitian matrix `a` in place; a c-prefer input is factored
    /// through its transposed view (see [`POTRF_::internal_run`]).
    pub fn run(self) -> Result<TensorMutable2<'a, T, B>> {
        self.internal_run()
    }
}

pub type POTRF<'a, B, T> = POTRF_Builder<'a, B, T>;
pub type SPOTRF<'a, B> = POTRF<'a, B, f32>;
pub type DPOTRF<'a, B> = POTRF<'a, B, f64>;
pub type CPOTRF<'a, B> = POTRF<'a, B, Complex<f32>>;
pub type ZPOTRF<'a, B> = POTRF<'a, B, Complex<f64>>;
