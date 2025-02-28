use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait GETRIDriverAPI<T> {
    unsafe fn driver_getri(
        order: FlagOrder,
        n: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blasint,
    ) -> blasint;
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std)]
pub struct GETRI_<'a, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
    pub ipiv: Vec<blasint>,
}

impl<'a, B, T> GETRI_<'a, B, T>
where
    T: BlasFloat,
    B: GETRIDriverAPI<T> + DeviceAPI<T, Raw = Vec<T>> + DeviceComplexFloatAPI<T, Ix2>,
{
    pub fn internal_run(self) -> Result<TensorMutable2<'a, T, B>> {
        let Self { a, mut ipiv } = self;

        let mut a = overwritable_convert(a)?;
        let order = if a.f_prefer() && !a.c_prefer() { ColMajor } else { RowMajor };

        // perform check
        rstsr_assert_eq!(
            a.view().nrow(),
            a.view().ncol(),
            InvalidLayout,
            "Lapack GETRI: A must be square"
        )?;

        let n = a.view().nrow();
        let lda = a.view().ld(order).unwrap();
        let ptr_a = a.view_mut().as_mut_ptr();

        // run driver
        let info = unsafe { B::driver_getri(order, n, ptr_a, lda, ipiv.as_mut_ptr()) };
        let info = info as i32;
        if info != 0 {
            rstsr_errcode!(info, "Lapack GETRI")?;
        }

        Ok(a.clone_to_mut())
    }

    pub fn run(self) -> Result<TensorMutable2<'a, T, B>> {
        self.internal_run()
    }
}

pub type GETRI<'a, B, T> = GETRI_Builder<'a, B, T>;
pub type SGETRI<'a, B> = GETRI<'a, B, f32>;
pub type DGETRI<'a, B> = GETRI<'a, B, f64>;
pub type CGETRI<'a, B> = GETRI<'a, B, Complex<f32>>;
pub type ZGETRI<'a, B> = GETRI<'a, B, Complex<f64>>;
