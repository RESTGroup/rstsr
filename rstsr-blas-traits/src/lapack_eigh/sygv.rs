use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait SYGVDriverAPI<T>
where
    T: BlasFloat,
{
    unsafe fn driver_sygv(
        order: FlagOrder,
        itype: blas_int,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        w: *mut T::Real,
    ) -> blas_int;
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct SYGV_<'a, 'b, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
    #[builder(setter(into))]
    pub b: TensorReference<'b, T, B, Ix2>,

    #[builder(setter(into), default = "1")]
    pub itype: blas_int,
    #[builder(setter(into), default = "'V'")]
    pub jobz: char,
    #[builder(setter(into), default = "None")]
    pub uplo: Option<FlagUpLo>,
}

impl<'a, B, T> SYGV_<'a, '_, B, T>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + SYGVDriverAPI<T>,
{
    pub fn internal_run(self) -> Result<(Tensor<T::Real, B, Ix1>, TensorMutable<'a, T, B, Ix2>)> {
        let Self { a, b, itype, jobz, uplo } = self;

        rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
        let device = a.device().clone();
        let uplo = uplo.unwrap_or_else(|| match device.default_order() {
            RowMajor => Lower,
            ColMajor => Upper,
        });
        let order = if a.f_prefer() && !a.c_prefer() { ColMajor } else { RowMajor };
        let mut a = overwritable_convert_with_order(a, order)?;
        let mut b = overwritable_convert_with_order(b, order)?;

        let n = a.view().nrow();
        rstsr_assert_eq!(a.view().shape(), &[n, n], InvalidLayout)?;
        rstsr_assert_eq!(b.view().shape(), &[n, n], InvalidLayout)?;

        let lda = a.view().ld(order).unwrap();
        let ldb = b.view().ld(order).unwrap();
        let mut w = unsafe { empty_f(([n].c(), &device))?.into_dim::<Ix1>() };
        let ptr_a = a.view_mut().as_mut_ptr();
        let ptr_b = b.view_mut().as_mut_ptr();
        let ptr_w = w.as_mut_ptr();

        // run driver
        let info = unsafe { B::driver_sygv(order, itype, jobz, uplo, n, ptr_a, lda, ptr_b, ldb, ptr_w) };
        rstsr_assert_eq!(info, 0, InvalidLayout)?;

        Ok((w, a.clone_to_mut()))
    }

    pub fn run(self) -> Result<(Tensor<T::Real, B, Ix1>, TensorMutable<'a, T, B, Ix2>)> {
        self.internal_run()
    }
}

pub type SYGV<'a, 'b, B, T> = SYGV_Builder<'a, 'b, B, T>;
pub type SSYGV<'a, 'b, B> = SYGV<'a, 'b, B, f32>;
pub type DSYGV<'a, 'b, B> = SYGV<'a, 'b, B, f64>;
pub type CHEGV<'a, 'b, B> = SYGV<'a, 'b, B, Complex<f32>>;
pub type ZHEGV<'a, 'b, B> = SYGV<'a, 'b, B, Complex<f64>>;
