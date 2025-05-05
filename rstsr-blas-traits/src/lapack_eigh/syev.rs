use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait SYEVDriverAPI<T>
where
    T: BlasFloat,
{
    unsafe fn driver_syev(
        order: FlagOrder,
        jobz: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        w: *mut T::Real,
    ) -> blas_int;
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct SYEV_<'a, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,

    #[builder(setter(into), default = "'V'")]
    pub jobz: char,
    #[builder(setter(into), default = "Lower")]
    pub uplo: FlagUpLo,
}

impl<'a, B, T> SYEV_<'a, B, T>
where
    T: BlasFloat,
    B: SYEVDriverAPI<T>
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<T::Real, Raw = Vec<T::Real>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceComplexFloatAPI<T::Real, Ix2>,
{
    pub fn internal_run(self) -> Result<(Tensor<T::Real, B, Ix1>, TensorMutable<'a, T, B, Ix2>)> {
        let Self { a, jobz, uplo } = self;

        let device = a.device().clone();
        let mut a = overwritable_convert(a)?;
        let order = if a.f_prefer() && !a.c_prefer() { ColMajor } else { RowMajor };

        let [n, m] = *a.view().shape();
        rstsr_assert_eq!(n, m, InvalidLayout)?;

        let lda = a.view().ld(order).unwrap();
        let mut w = unsafe { empty_f(([n].c(), &device))?.into_dim::<Ix1>() };
        let ptr_a = a.view_mut().as_mut_ptr();
        let ptr_w = w.as_mut_ptr();

        // run driver
        let info = unsafe { B::driver_syev(order, jobz, uplo, n, ptr_a, lda, ptr_w) };
        let info = info as i32;
        if info != 0 {
            rstsr_errcode!(info, "Lapack SYEV")?;
        }

        Ok((w, a.clone_to_mut()))
    }

    pub fn run(self) -> Result<(Tensor<T::Real, B, Ix1>, TensorMutable<'a, T, B, Ix2>)> {
        self.internal_run()
    }
}

pub type SYEV<'a, B, T> = SYEV_Builder<'a, B, T>;
pub type SSYEV<'a, B> = SYEV<'a, B, f32>;
pub type DSYEV<'a, B> = SYEV<'a, B, f64>;
pub type CHEEV<'a, B> = SYEV<'a, B, Complex<f32>>;
pub type ZHEEV<'a, B> = SYEV<'a, B, Complex<f64>>;
