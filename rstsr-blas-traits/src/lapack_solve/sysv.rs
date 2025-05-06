use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait SYSVDriverAPI<T, const HERMI: bool> {
    unsafe fn driver_sysv(
        order: FlagOrder,
        uplo: FlagUpLo,
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
pub struct SYSV_<'a, 'b, B, T, const HERMI: bool>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
    #[builder(setter(into))]
    pub b: TensorReference<'b, T, B, Ix2>,

    #[builder(setter(into), default = "Lower")]
    pub uplo: FlagUpLo,
}

impl<'a, 'b, B, T, const HERMI: bool> SYSV_<'a, 'b, B, T, HERMI>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + SYSVDriverAPI<T, HERMI>,
{
    pub fn internal_run(
        self,
    ) -> Result<(TensorMutable2<'a, T, B>, TensorMutable2<'b, T, B>, Tensor<blas_int, B, Ix1>)>
    {
        let Self { a, b, uplo } = self;

        let device = a.device().clone();
        let mut b = overwritable_convert(b)?;
        let order = if b.f_prefer() && !b.c_prefer() { ColMajor } else { RowMajor };
        let mut a = overwritable_convert_with_order(a, order)?;

        let [n, nrhs] = *b.view().shape();
        rstsr_assert_eq!(a.view().shape(), &[n, n], InvalidLayout, "SYSV: A shape")?;
        let lda = a.view().ld(order).unwrap();
        let ldb = b.view().ld(order).unwrap();
        let mut ipiv = unsafe { empty_f(([n], &device))?.into_dim::<Ix1>() };
        let ptr_a = a.view_mut().as_mut_ptr();
        let ptr_b = b.view_mut().as_mut_ptr();
        let ptr_ipiv = ipiv.as_mut_ptr();

        // run driver
        let info =
            unsafe { B::driver_sysv(order, uplo, n, nrhs, ptr_a, lda, ptr_ipiv, ptr_b, ldb) };
        let info = info as i32;
        if info != 0 {
            rstsr_errcode!(info, "Lapack SYSV")?;
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

pub type SYSV<'a, 'b, B, T, const HERMI: bool> = SYSV_Builder<'a, 'b, B, T, HERMI>;
pub type SSYSV<'a, 'b, B> = SYSV<'a, 'b, B, f32, true>;
pub type DSYSV<'a, 'b, B> = SYSV<'a, 'b, B, f64, true>;
pub type CSYSV<'a, 'b, B> = SYSV<'a, 'b, B, Complex<f32>, false>;
pub type ZSYSV<'a, 'b, B> = SYSV<'a, 'b, B, Complex<f64>, false>;
pub type CHESV<'a, 'b, B> = SYSV<'a, 'b, B, Complex<f32>, true>;
pub type ZHESV<'a, 'b, B> = SYSV<'a, 'b, B, Complex<f64>, true>;
