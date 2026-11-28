use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

/// BLAS-2 `gemv` driver: `y := alpha * op(A) x + beta * y`.
///
/// # Safety
///
/// `a` addresses an `m`-by-`n` column-major matrix with leading dimension
/// `lda`; `x` and `y` address unit-stride vectors of `len_x` and `len_y`
/// elements that overlap neither `a` nor each other, with `(len_x, len_y)`
/// equal to `(n, m)` for [`FlagTrans::N`] and to `(m, n)` otherwise.
pub trait GEMVDriverAPI<T> {
    unsafe fn driver_gemv(
        order: FlagOrder,
        transa: FlagTrans,
        m: usize,
        n: usize,
        alpha: T,
        a: *const T,
        lda: usize,
        x: *const T,
        beta: T,
        y: *mut T,
    );
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct GEMV_<'a, 'b, 'c, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    pub a: TensorView<'a, T, B, Ix2>,
    pub x: TensorView<'b, T, B, Ix1>,

    #[builder(setter(into, strip_option), default = "None")]
    pub y: Option<TensorViewMut<'c, T, B, Ix1>>,
    #[builder(setter(into), default = "T::one()")]
    pub alpha: T,
    #[builder(setter(into), default = "T::zero()")]
    pub beta: T,
    #[builder(setter(into), default = "NoTrans")]
    pub transa: FlagTrans,
}

impl<'c, B, T> GEMV_<'_, '_, 'c, B, T>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + GEMVDriverAPI<T>,
{
    /// Run `y := alpha * op(A) x + beta * y`, returning the output vector.
    ///
    /// There is no `order` parameter: the result is one-dimensional, and the
    /// matrix is always normalised to column-major before the driver call.
    ///
    /// # Errors
    ///
    /// Returns an error for [`FlagTrans::CN`], which has no CBLAS counterpart:
    /// conjugate the matrix and pass [`FlagTrans::N`] instead (for real scalars
    /// `C` and `T` coincide).
    pub fn run(self) -> Result<TensorMutable1<'c, T, B>> {
        let Self { a, x, y, alpha, beta, transa } = self;

        if transa == FlagTrans::CN {
            rstsr_invalid!(transa)?;
        }

        // f-prefer passes through, c-prefer is viewed transposed, and a
        // neither-prefer layout or a `C` on c-prefer is copied
        let (transa, a_cow) = flip_trans(ColMajor, transa, a, false)?;
        let obj = GEMV_ { a: a_cow.view(), x, y, alpha, beta, transa };
        obj.internal_run()
    }

    /// Column-major entry; `a` must already be f-prefer when it is read.
    pub fn internal_run(self) -> Result<TensorMutable1<'c, T, B>> {
        let Self { a, x, y, alpha, beta, transa } = self;

        let (m, n) = (a.nrow(), a.ncol());
        let (len_x, len_y) = match transa {
            FlagTrans::N => (n, m),
            FlagTrans::T | FlagTrans::C => (m, n),
            _ => rstsr_invalid!(transa)?,
        };

        // device and shape checks
        rstsr_assert!(a.device().same_device(x.device()), DeviceMismatch)?;
        rstsr_assert_eq!(x.shape()[0], len_x, InvalidLayout)?;
        if let Some(y) = &y {
            rstsr_assert!(a.device().same_device(y.device()), DeviceMismatch)?;
            rstsr_assert_eq!(y.shape()[0], len_y, InvalidLayout)?;
        }

        // a non-unit-stride `y` is written through a contiguous buffer
        let mut y = match y {
            Some(y) if len_y <= 1 || y.stride()[0] == 1 => TensorMutable::Mut(y),
            Some(y) => {
                let y_buffer = y.to_contig_f(ColMajor)?.into_owned();
                TensorMutable::ToBeCloned(y, y_buffer)
            },
            None => TensorMutable1::Owned(zeros_f(([len_y].f(), a.device()))?.into_dim()),
        };

        // empty output: nothing is read or written
        if len_y == 0 {
            return Ok(y.clone_to_mut());
        }

        // `k == 0`: BLAS returns before scaling `y`, so scale it here
        if len_x == 0 {
            if beta != T::one() {
                match &mut y {
                    // an owned output is allocated zeroed
                    TensorMutable::Owned(_) => (),
                    TensorMutable::Mut(view) => unsafe { scale_vector(view.as_mut_ptr(), len_y, beta) },
                    TensorMutable::ToBeCloned(_, buffer) => unsafe { scale_vector(buffer.as_mut_ptr(), len_y, beta) },
                }
            }
            return Ok(y.clone_to_mut());
        }

        let lda = a.testify_ld(ColMajor, "A")?;

        // `x` is read-only: a non-unit-stride vector is copied once
        let x_cow;
        let x_view = if len_x <= 1 || x.stride()[0] == 1 {
            x.view()
        } else {
            x_cow = x.to_contig_f(ColMajor)?;
            x_cow.view()
        };

        let mut y_view = y.view_mut();
        let ptr_a = a.as_ptr();
        let ptr_x = x_view.as_ptr();
        let ptr_y = y_view.as_mut_ptr();

        // SAFETY: dimensions and the leading dimension are checked above; `x`
        // and `y` address `len_x` and `len_y` live elements of `a`'s device
        unsafe {
            B::driver_gemv(ColMajor, transa, m, n, alpha, ptr_a, lda, ptr_x, beta, ptr_y);
        }

        Ok(y.clone_to_mut())
    }
}

/// `ptr[0..len] *= beta` over consecutive elements.
///
/// # Safety
///
/// `ptr` addresses `len` initialized, writable elements of `T`.
unsafe fn scale_vector<T>(ptr: *mut T, len: usize, beta: T)
where
    T: BlasFloat,
{
    for i in 0..len {
        let p = ptr.add(i);
        *p = *p * beta;
    }
}

pub type GEMV<'a, 'b, 'c, B, T> = GEMV_Builder<'a, 'b, 'c, B, T>;
pub type SGEMV<'a, 'b, 'c, B> = GEMV<'a, 'b, 'c, B, f32>;
pub type DGEMV<'a, 'b, 'c, B> = GEMV<'a, 'b, 'c, B, f64>;
pub type CGEMV<'a, 'b, 'c, B> = GEMV<'a, 'b, 'c, B, Complex<f32>>;
pub type ZGEMV<'a, 'b, 'c, B> = GEMV<'a, 'b, 'c, B, Complex<f64>>;
