use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait SYHEMMDriverAPI<T, const HERMI: bool> {
    unsafe fn driver_syhemm(
        order: FlagOrder,
        side: FlagSide,
        uplo: FlagUpLo,
        m: usize,
        n: usize,
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

/* #endregion */

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct SYHEMM_<'a, 'b, 'c, B, T, const HERMI: bool>
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
    #[builder(setter(into), default = "FlagSide::L")]
    pub side: FlagSide,
    #[builder(setter(into), default = "FlagUpLo::L")]
    pub uplo: FlagUpLo,
    #[builder(setter(into, strip_option), default = "None")]
    pub order: Option<FlagOrder>,
}

impl<'c, B, T, const HERMI: bool> SYHEMM_<'_, '_, 'c, B, T, HERMI>
where
    T: BlasFloat + Num,
    B: SYHEMMDriverAPI<T, HERMI> + DeviceAPI<T, Raw = Vec<T>> + DeviceComplexFloatAPI<T, Ix2>,
{
    pub fn run(self) -> Result<TensorMutable2<'c, T, B>> {
        let Self { a, b, c, alpha, beta, side, uplo, order } = self;

        // determine preferred layout
        let order_c = c.as_ref().map(|c| (c.c_prefer(), c.f_prefer()));
        let order = order.map(|order| match order {
            ColMajor => (true, false),
            RowMajor => (false, true),
        });

        let default_order = a.device().default_order();
        let order = get_output_order(&[order, order_c], &[], default_order);
        if order == ColMajor {
            let (uplo, a_cow) = match (HERMI, a.f_prefer()) {
                (false, false) => (uplo.flip()?, a.to_contig_f(ColMajor)?),
                _ => (uplo, a.to_contig_f(ColMajor)?),
            };
            let b_cow = b.to_contig_f(ColMajor)?;
            let obj = SYHEMM_ {
                a: a_cow.view(),
                b: b_cow.view(),
                c,
                alpha,
                beta,
                side,
                uplo,
                order: Some(ColMajor),
            };
            obj.internal_run()
        } else {
            let (uplo, a_cow) = match (HERMI, a.c_prefer()) {
                (false, false) => (uplo.flip()?, a.to_contig_f(RowMajor)?),
                _ => (uplo, a.to_contig_f(RowMajor)?),
            };
            let b_cow = b.to_contig_f(RowMajor)?;
            let obj = SYHEMM_ {
                a: a_cow.t(),
                b: b_cow.t(),
                c: c.map(|c| c.into_reverse_axes()),
                alpha,
                beta,
                side: side.flip()?,
                uplo: uplo.flip()?,
                order: Some(ColMajor),
            };
            Ok(obj.internal_run()?.into_reverse_axes())
        }
    }

    pub fn internal_run(self) -> Result<TensorMutable2<'c, T, B>> {
        let Self { a, b, c, alpha, beta, side, uplo, order } = self;

        // this function only accepts column major
        rstsr_assert_eq!(order, Some(ColMajor), RuntimeError)?;
        rstsr_assert!(a.f_prefer(), RuntimeError)?;
        rstsr_assert!(b.f_prefer(), RuntimeError)?;

        // device check
        rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;

        // initialize intent(hide)
        let m = b.nrow();
        let n = a.ncol();
        let lda = a.ld_col().unwrap();
        let ldb = b.ld_col().unwrap();

        // perform check
        match side {
            Left => rstsr_assert_eq!(a.shape(), &[m, m], InvalidLayout)?,
            Right => rstsr_assert_eq!(a.shape(), &[n, n], InvalidLayout)?,
            _ => rstsr_invalid!(side)?,
        }

        // optional intent(out)
        let mut c = if let Some(c) = c {
            rstsr_assert!(a.device().same_device(c.device()), DeviceMismatch)?;
            rstsr_assert_eq!(c.shape(), &[m, n], InvalidLayout)?;
            if c.f_prefer() {
                TensorMutable::Mut(c)
            } else {
                let c_buffer = c.to_contig_f(ColMajor)?.into_owned();
                TensorMutable::ToBeCloned(c, c_buffer)
            }
        } else {
            TensorMutable2::Owned(zeros_f(([m, n].f(), a.device()))?.into_dim())
        };

        // perform blas
        let ptr_a = a.raw().as_ptr();
        let ptr_b = b.raw().as_ptr();
        let ptr_c = c.view_mut().raw_mut().as_mut_ptr();

        unsafe {
            B::driver_syhemm(
                ColMajor, side, uplo, m, n, alpha, ptr_a, lda, ptr_b, ldb, beta, ptr_c, m,
            );
        }

        Ok(c.clone_to_mut())
    }
}

pub type SYHEMM<'a, 'b, 'c, B, T, const HERMI: bool> = SYHEMM_Builder<'a, 'b, 'c, B, T, HERMI>;
pub type SSYMM<'a, 'b, 'c, B> = SYHEMM<'a, 'b, 'c, B, f32, false>;
pub type DSYMM<'a, 'b, 'c, B> = SYHEMM<'a, 'b, 'c, B, f64, false>;
pub type CSYMM<'a, 'b, 'c, B> = SYHEMM<'a, 'b, 'c, B, Complex<f32>, false>;
pub type ZSYMM<'a, 'b, 'c, B> = SYHEMM<'a, 'b, 'c, B, Complex<f64>, false>;
pub type CHEMM<'a, 'b, 'c, B> = SYHEMM<'a, 'b, 'c, B, Complex<f32>, true>;
pub type ZHEMM<'a, 'b, 'c, B> = SYHEMM<'a, 'b, 'c, B, Complex<f64>, true>;
