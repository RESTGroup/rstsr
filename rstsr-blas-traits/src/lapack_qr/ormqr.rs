use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait ORMQRDriverAPI<T>
where
    T: BlasFloat,
{
    unsafe fn driver_ormqr(
        order: FlagOrder,
        side: FlagSide,
        trans: FlagTrans,
        m: usize,
        n: usize,
        k: usize,
        a: *const T,
        lda: usize,
        tau: *const T,
        c: *mut T,
        ldc: usize,
    ) -> blas_int;
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct ORMQR_<'a, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
    #[builder(setter(into))]
    pub tau: TensorReference<'a, T, B, Ix1>,
    #[builder(setter(into))]
    pub c: TensorReference<'a, T, B, Ix2>,
    #[builder(default)]
    pub side: FlagSide,
    #[builder(default)]
    pub trans: FlagTrans,
}

impl<'a, B, T> ORMQR_<'a, B, T>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + ORMQRDriverAPI<T>,
{
    pub fn internal_run(self) -> Result<TensorMutable<'a, T, B, Ix2>> {
        let Self { a, tau, c, side, trans } = self;

        let mut c = overwritable_convert(c)?;
        let order = if c.f_prefer() && !c.c_prefer() { ColMajor } else { RowMajor };

        let a_shape = *a.view().shape();
        let c_shape = *c.view().shape();

        // Determine k from tau length
        let [k] = *tau.view().shape();

        // Determine m, n based on side
        let (m, n) = match side {
            FlagSide::L => {
                rstsr_assert_eq!(
                    a_shape[0],
                    c_shape[0],
                    InvalidLayout,
                    "ORMQR: A rows must match C rows for Left side"
                )?;
                (c_shape[0], c_shape[1])
            },
            FlagSide::R => {
                rstsr_assert_eq!(
                    a_shape[0],
                    c_shape[1],
                    InvalidLayout,
                    "ORMQR: A rows must match C cols for Right side"
                )?;
                (c_shape[0], c_shape[1])
            },
        };

        let ldc = c.view().ld(order).unwrap();

        // Convert A to the same order as C (read-only, so make a copy if needed)
        let a_cow = a.to_contig_f(order)?;
        let a_view = a_cow.view();

        let lda = a_view.ld(order).unwrap();

        let ptr_a = a_view.as_ptr();
        let ptr_tau = tau.as_ptr();
        let ptr_c = c.view_mut().as_mut_ptr();

        // run driver
        let info = unsafe { B::driver_ormqr(order, side, trans, m, n, k, ptr_a, lda, ptr_tau, ptr_c, ldc) };
        let info = info as i32;
        if info != 0 {
            rstsr_errcode!(info, "Lapack ORMQR")?;
        }

        Ok(c.clone_to_mut())
    }

    pub fn run(self) -> Result<TensorMutable<'a, T, B, Ix2>> {
        self.internal_run()
    }
}

pub type ORMQR<'a, B, T> = ORMQR_Builder<'a, B, T>;
pub type SORMQR<'a, B> = ORMQR<'a, B, f32>;
pub type DORMQR<'a, B> = ORMQR<'a, B, f64>;
pub type CUNMQR<'a, B> = ORMQR<'a, B, Complex<f32>>;
pub type ZUNMQR<'a, B> = ORMQR<'a, B, Complex<f64>>;
