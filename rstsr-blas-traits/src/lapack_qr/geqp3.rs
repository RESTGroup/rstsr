use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait GEQP3DriverAPI<T>
where
    T: BlasFloat,
{
    unsafe fn driver_geqp3(
        order: FlagOrder,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        jpvt: *mut blas_int,
        tau: *mut T,
    ) -> blas_int;
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct GEQP3_<'a, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T> + DeviceAPI<blas_int>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
}

impl<'a, B, T> GEQP3_<'a, B, T>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + GEQP3DriverAPI<T>,
{
    pub fn internal_run(self) -> Result<(TensorMutable<'a, T, B, Ix2>, Tensor<blas_int, B, Ix1>, Tensor<T, B, Ix1>)> {
        let Self { a } = self;

        let device = a.device().clone();
        let mut a = overwritable_convert(a)?;
        let order = if a.f_prefer() && !a.c_prefer() { ColMajor } else { RowMajor };
        let [m, n] = *a.view().shape();
        let minmn = m.min(n);
        let lda = a.view().ld(order).unwrap();

        // Allocate jpvt and tau arrays
        let mut jpvt = unsafe { empty_f(([n].c(), &device))?.into_dim::<Ix1>() };
        let mut tau = unsafe { empty_f(([minmn].c(), &device))?.into_dim::<Ix1>() };

        // Initialize jpvt to zeros (all columns are free to pivot)
        jpvt.fill(0);

        let ptr_a = a.view_mut().as_mut_ptr();
        let ptr_jpvt = jpvt.as_mut_ptr();
        let ptr_tau = tau.as_mut_ptr();

        // run driver
        let info = unsafe { B::driver_geqp3(order, m, n, ptr_a, lda, ptr_jpvt, ptr_tau) };
        let info = info as i32;
        if info != 0 {
            rstsr_errcode!(info, "Lapack GEQP3")?;
        }

        // Convert jpvt from 1-based to 0-based indexing
        jpvt -= 1;

        Ok((a.clone_to_mut(), jpvt, tau))
    }

    pub fn run(self) -> Result<(TensorMutable<'a, T, B, Ix2>, Tensor<blas_int, B, Ix1>, Tensor<T, B, Ix1>)> {
        self.internal_run()
    }
}

pub type GEQP3<'a, B, T> = GEQP3_Builder<'a, B, T>;
pub type SGEQP3<'a, B> = GEQP3<'a, B, f32>;
pub type DGEQP3<'a, B> = GEQP3<'a, B, f64>;
pub type CGEQP3<'a, B> = GEQP3<'a, B, Complex<f32>>;
pub type ZGEQP3<'a, B> = GEQP3<'a, B, Complex<f64>>;
