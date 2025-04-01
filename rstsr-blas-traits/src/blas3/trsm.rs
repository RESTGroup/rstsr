use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait TRSMDriverAPI<T> {
    unsafe fn driver_trsm(
        order: FlagOrder,
        side: FlagSide,
        uplo: FlagUpLo,
        transa: FlagTrans,
        diag: FlagDiag,
        m: usize,
        n: usize,
        alpha: T,
        a: *const T,
        lda: usize,
        b: *mut T,
        ldb: usize,
    );
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct TRSM_<'a, 'b, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    pub a: TensorView<'a, T, B, Ix2>,
    #[builder(setter(into))]
    pub b: TensorReference<'b, T, B, Ix2>,

    #[builder(setter(into), default = "T::one()")]
    pub alpha: T,
    #[builder(setter(into), default = "Left")]
    pub side: FlagSide,
    #[builder(setter(into), default = "Lower")]
    pub uplo: FlagUpLo,
    #[builder(setter(into), default = "NoTrans")]
    pub transa: FlagTrans,
    #[builder(setter(into), default = "NonUnit")]
    pub diag: FlagDiag,
    #[builder(setter(into, strip_option), default = "None")]
    pub order: Option<FlagOrder>,
}

impl<'b, B, T> TRSM_<'_, 'b, B, T>
where
    T: BlasFloat,
    B: TRSMDriverAPI<T> + DeviceAPI<T, Raw = Vec<T>> + DeviceComplexFloatAPI<T, Ix2>,
{
    pub fn run(self) -> Result<TensorMutable2<'b, T, B>> {
        let Self { a, b, alpha, side, uplo, transa, diag, order } = self;

        // determine preferred layout
        let order_b = (b.c_prefer(), b.f_prefer());
        let order = order.map(|order| match order {
            ColMajor => (true, false),
            RowMajor => (false, true),
        });

        let default_order = b.device().default_order();
        let order = get_output_order(&[order, Some(order_b)], &[], default_order);
        if order == ColMajor {
            let (transa_new, a_cow) = flip_trans(ColMajor, transa, a, false)?;
            let uplo = if transa_new != transa { uplo.flip()? } else { uplo };
            let obj = TRSM_ {
                a: a_cow.view(),
                b,
                alpha,
                side,
                uplo,
                transa: transa_new,
                diag,
                order: Some(ColMajor),
            };
            obj.internal_run()
        } else {
            let (transa_new, a_cow) = flip_trans(RowMajor, transa, a, false)?;
            let uplo = if transa_new != transa { uplo.flip()? } else { uplo };
            let obj = TRSM_ {
                a: a_cow.t(),
                b: b.into_reverse_axes(),
                alpha,
                side: side.flip()?,
                uplo: uplo.flip()?,
                transa: transa_new,
                diag,
                order: Some(ColMajor),
            };
            Ok(obj.internal_run()?.into_reverse_axes())
        }
    }

    pub fn internal_run(self) -> Result<TensorMutable2<'b, T, B>> {
        let Self { a, b, alpha, side, uplo, transa, diag, order } = self;

        // this function only accepts column major
        rstsr_assert_eq!(order, Some(ColMajor), RuntimeError)?;
        rstsr_assert!(a.f_prefer(), RuntimeError)?;

        // device check
        rstsr_assert!(a.device().same_device(b.device()), DeviceError)?;

        // initialize intent(hide)
        let [m, n] = *b.shape();
        let lda = a.ld(ColMajor).unwrap();

        // perform check
        match side {
            Left => rstsr_assert_eq!(a.shape(), &[m, m], InvalidLayout)?,
            Right => rstsr_assert_eq!(a.shape(), &[n, n], InvalidLayout)?,
            _ => rstsr_invalid!(side)?,
        };

        // prepare output
        let mut b = overwritable_convert_with_order(b, ColMajor)?;

        // perform blas
        let ptr_a = a.raw().as_ptr();
        let ptr_b = b.view_mut().raw_mut().as_mut_ptr();

        unsafe {
            B::driver_trsm(ColMajor, side, uplo, transa, diag, m, n, alpha, ptr_a, lda, ptr_b, m)
        };

        Ok(b.clone_to_mut())
    }
}

pub type TRSM<'a, 'b, B, T> = TRSM_Builder<'a, 'b, B, T>;
pub type STRSM<'a, 'b, B> = TRSM<'a, 'b, B, f32>;
pub type DTRSM<'a, 'b, B> = TRSM<'a, 'b, B, f64>;
pub type CTRSM<'a, 'b, B> = TRSM<'a, 'b, B, Complex<f32>>;
pub type ZTRSM<'a, 'b, B> = TRSM<'a, 'b, B, Complex<f64>>;
