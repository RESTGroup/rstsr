use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait GETRFDriverAPI<T> {
    unsafe fn driver_getrf(
        order: FlagOrder,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
    ) -> blas_int;
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct GETRF_<'a, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
}

impl<'a, B, T> GETRF_<'a, B, T>
where
    T: BlasFloat,
    B: GETRFDriverAPI<T>
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blas_int, Raw = Vec<blas_int>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blas_int, Ix1>,
{
    pub fn internal_run(self) -> Result<(TensorMutable2<'a, T, B>, Tensor<blas_int, B, Ix1>)> {
        let Self { a } = self;

        let device = a.device().clone();
        let mut a = overwritable_convert(a)?;
        let order = if a.f_prefer() && !a.c_prefer() { ColMajor } else { RowMajor };

        let [m, n] = *a.view().shape();
        let lda = a.view().ld(order).unwrap();
        let mut ipiv = unsafe { empty_f(([n].c(), &device))?.into_dim::<Ix1>() };
        let ptr_a = a.view_mut().as_mut_ptr();
        let ptr_ipiv = ipiv.as_mut_ptr();

        // run driver
        let info = unsafe { B::driver_getrf(order, m, n, ptr_a, lda, ptr_ipiv) };
        let info = info as i32;
        if info != 0 {
            rstsr_errcode!(info, "Lapack GETRF")?;
        }

        Ok((a.clone_to_mut(), ipiv))
    }

    pub fn run(self) -> Result<(TensorMutable2<'a, T, B>, Tensor<blas_int, B, Ix1>)> {
        self.internal_run()
    }
}

pub type GETRF<'a, B, T> = GETRF_Builder<'a, B, T>;
pub type SGETRF<'a, B> = GETRF<'a, B, f32>;
pub type DGETRF<'a, B> = GETRF<'a, B, f64>;
pub type CGETRF<'a, B> = GETRF<'a, B, Complex<f32>>;
pub type ZGETRF<'a, B> = GETRF<'a, B, Complex<f64>>;
