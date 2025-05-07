use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait GETRIDriverAPI<T> {
    unsafe fn driver_getri(
        order: FlagOrder,
        n: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
    ) -> blas_int;
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct GETRI_<'a, 'ipiv, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T> + DeviceAPI<blas_int>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
    pub ipiv: TensorView<'ipiv, blas_int, B, Ix1>,
}

impl<'a, B, T> GETRI_<'a, '_, B, T>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + GETRIDriverAPI<T>,
{
    pub fn internal_run(self) -> Result<TensorMutable2<'a, T, B>> {
        let Self { a, ipiv } = self;

        let mut a = overwritable_convert(a)?;
        let order = if a.f_prefer() && !a.c_prefer() { ColMajor } else { RowMajor };
        let mut ipiv = ipiv.into_contig_f(ColMajor)?;

        // rust is 1-indexed
        ipiv += 1;

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
        let ptr_ipiv = ipiv.as_mut_ptr();

        // run driver
        let info = unsafe { B::driver_getri(order, n, ptr_a, lda, ptr_ipiv) };
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

pub type GETRI<'a, 'ipiv, B, T> = GETRI_Builder<'a, 'ipiv, B, T>;
pub type SGETRI<'a, 'ipiv, B> = GETRI<'a, 'ipiv, B, f32>;
pub type DGETRI<'a, 'ipiv, B> = GETRI<'a, 'ipiv, B, f64>;
pub type CGETRI<'a, 'ipiv, B> = GETRI<'a, 'ipiv, B, Complex<f32>>;
pub type ZGETRI<'a, 'ipiv, B> = GETRI<'a, 'ipiv, B, Complex<f64>>;
