use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait GESVDriverAPI<T> {
    unsafe fn driver_gesv(
        order: FlagOrder,
        n: usize,
        nrhs: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
        b: *mut T,
        ldb: usize,
    ) -> blas_int;
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct GESV_<'a, 'b, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
    #[builder(setter(into))]
    pub b: TensorReference<'b, T, B, Ix2>,
}

impl<'a, 'b, B, T> GESV_<'a, 'b, B, T>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + GESVDriverAPI<T>,
{
    pub fn internal_run(
        self,
    ) -> Result<(TensorMutable2<'a, T, B>, TensorMutable2<'b, T, B>, Tensor<blas_int, B, Ix1>)>
    {
        let Self { a, b } = self;

        let device = a.device().clone();
        let mut b = overwritable_convert(b)?;
        let order = if b.f_prefer() && !b.c_prefer() { ColMajor } else { RowMajor };
        let mut a = overwritable_convert_with_order(a, order)?;

        let [n, nrhs] = *b.view().shape();
        rstsr_assert_eq!(a.view().shape(), &[n, n], InvalidLayout, "GESV: A shape")?;
        let lda = a.view().ld(order).unwrap();
        let ldb = b.view().ld(order).unwrap();
        let mut ipiv = unsafe { empty_f(([n], &device))?.into_dim::<Ix1>() };
        let ptr_a = a.view_mut().as_mut_ptr();
        let ptr_b = b.view_mut().as_mut_ptr();
        let ptr_ipiv = ipiv.as_mut_ptr();

        // run driver
        let info = unsafe { B::driver_gesv(order, n, nrhs, ptr_a, lda, ptr_ipiv, ptr_b, ldb) };
        let info = info as i32;
        if info != 0 {
            rstsr_errcode!(info, "Lapack GESV")?;
        }

        Ok((a.clone_to_mut(), b.clone_to_mut(), ipiv))
    }

    pub fn run(
        self,
    ) -> Result<(TensorMutable2<'a, T, B>, TensorMutable2<'b, T, B>, Tensor<blas_int, B, Ix1>)>
    {
        self.internal_run()
    }
}

pub type GESV<'a, 'b, B, T> = GESV_Builder<'a, 'b, B, T>;
pub type SGESV<'a, 'b, B> = GESV<'a, 'b, B, f32>;
pub type DGESV<'a, 'b, B> = GESV<'a, 'b, B, f64>;
pub type CGESV<'a, 'b, B> = GESV<'a, 'b, B, Complex<f32>>;
pub type ZGESV<'a, 'b, B> = GESV<'a, 'b, B, Complex<f64>>;
