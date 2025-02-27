use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

/* #region driver */

pub trait GEMMDriverAPI<T> {
    unsafe fn driver_gemm(
        order: TensorOrder,
        transa: FlagTrans,
        transb: FlagTrans,
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        a: *const T,
        lda: usize,
        b: *const T,
        ldb: usize,
        beta: T,
        c: *mut T,
        ldc: usize,
    );
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std)]
pub struct GEMM_<'a, 'b, 'c, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    pub a: TensorView<'a, T, B, Ix2>,
    pub b: TensorView<'b, T, B, Ix2>,

    #[builder(setter(into, strip_option), default = "None")]
    pub c: Option<TensorViewMut<'c, T, B, Ix2>>,
    #[builder(setter(into), default = "T::one()")]
    pub alpha: T,
    #[builder(setter(into), default = "T::zero()")]
    pub beta: T,
    #[builder(setter(into), default = "FlagTrans::N")]
    pub transa: FlagTrans,
    #[builder(setter(into), default = "FlagTrans::N")]
    pub transb: FlagTrans,
    #[builder(setter(into, strip_option), default = "None")]
    pub order: Option<TensorOrder>,
}

/* #endregion */

/* #region builder */

impl<'c, B, T> GEMM_<'_, '_, 'c, B, T>
where
    T: BlasFloat + Num,
    B: GEMMDriverAPI<T>
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceRawAPI<T, Raw = Vec<T>>
        + DeviceCreationNumAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, Ix2, Ix2>
        + OpAssignAPI<T, Ix2>
        + DeviceConjAPI<T, Ix2, TOut = T>,
{
    pub fn run(self) -> Result<TensorMutable2<'c, T, B>> {
        let Self { a, b, c, alpha, beta, transa, transb, order } = self;

        // determine preferred layout
        let order_a = (a.c_prefer(), a.f_prefer());
        let order_b = (b.c_prefer(), b.f_prefer());
        let order_c = c.as_ref().map(|c| (c.c_prefer(), c.f_prefer()));
        let order = order.map(|order| match order {
            TensorOrder::F => (true, false),
            TensorOrder::C => (false, true),
        });

        let order = get_order_row_preferred(&[order, order_c], &[order_a, order_b]);
        if order == TensorOrder::F {
            // f-prefer: C = op(A) op(B)
            let (transa, a_cow) = flip_trans(order, transa, a, false)?;
            let (transb, b_cow) = flip_trans(order, transb, b, false)?;
            let obj = GEMM_ {
                a: a_cow.view(),
                b: b_cow.view(),
                c,
                alpha,
                beta,
                transa,
                transb,
                order: Some(TensorOrder::F),
            };
            obj.internal_run()
        } else {
            // c-prefer: C' = op(B') op(A')
            let (transa, a_cow) = flip_trans(order, transa, a, false)?;
            let (transb, b_cow) = flip_trans(order, transb, b, false)?;
            let obj = GEMM_ {
                a: b_cow.t(),
                b: a_cow.t(),
                c: c.map(|c| c.into_reverse_axes()),
                alpha,
                beta,
                transa: transb,
                transb: transa,
                order: Some(TensorOrder::F),
            };
            Ok(obj.internal_run()?.into_reverse_axes())
        }
    }

    pub fn internal_run(self) -> Result<TensorMutable2<'c, T, B>> {
        // this function only accepts column major
        let Self { a, b, c, alpha, beta, transa, transb, order } = self;

        rstsr_assert_eq!(order, Some(TensorOrder::F), RuntimeError)?;
        rstsr_assert!(a.f_prefer(), RuntimeError)?;
        rstsr_assert!(b.f_prefer(), RuntimeError)?;

        // device check
        rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;

        // initialize intent(hide)
        let (m, k) = match transa {
            FlagTrans::N => (a.nrow(), a.ncol()),
            FlagTrans::T | FlagTrans::C => (a.ncol(), a.nrow()),
            _ => rstsr_invalid!(transa)?,
        };

        let n = match transb {
            FlagTrans::N => b.ncol(),
            FlagTrans::T | FlagTrans::C => b.nrow(),
            _ => rstsr_invalid!(transb)?,
        };
        let lda = a.ld_col().unwrap();
        let ldb = b.ld_col().unwrap();

        // perform check
        match transb {
            FlagTrans::N => rstsr_assert_eq!(b.nrow(), k, InvalidLayout)?,
            FlagTrans::T | FlagTrans::C => rstsr_assert_eq!(b.ncol(), k, InvalidLayout)?,
            _ => rstsr_invalid!(transb)?,
        }

        // optional intent(out)
        let mut c: TensorMutable2<T, B> = match c {
            Some(c) => {
                rstsr_assert_eq!(c.nrow(), m, InvalidLayout)?;
                rstsr_assert_eq!(c.ncol(), n, InvalidLayout)?;
                if c.f_prefer() {
                    TensorMutable::Mut(c)
                } else {
                    let c_buffer = c.to_contig_f(TensorOrder::F)?.into_owned();
                    TensorMutable::ToBeCloned(c, c_buffer)
                }
            },
            None => TensorMutable2::Owned(zeros_f(([m, n].f(), a.device()))?.into_dim()),
        };

        // perform gemm
        let ptr_a = a.raw().as_ptr();
        let ptr_b = b.raw().as_ptr();
        let ptr_c = c.view_mut().raw_mut().as_mut_ptr();

        unsafe {
            B::driver_gemm(
                ColMajor, transa, transb, m, n, k, alpha, ptr_a, lda, ptr_b, ldb, beta, ptr_c, m,
            );
        }

        Ok(c.clone_to_mut())
    }
}

pub type GEMM<'a, 'b, 'c, B, T> = GEMM_Builder<'a, 'b, 'c, B, T>;
pub type SGEMM<'a, 'b, 'c, B> = GEMM<'a, 'b, 'c, B, f32>;
pub type DGEMM<'a, 'b, 'c, B> = GEMM<'a, 'b, 'c, B, f64>;
pub type CGEMM<'a, 'b, 'c, B> = GEMM<'a, 'b, 'c, B, Complex<f32>>;
pub type ZGEMM<'a, 'b, 'c, B> = GEMM<'a, 'b, 'c, B, Complex<f64>>;

/* #endregion */
