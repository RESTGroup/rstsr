use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

/// BLAS-2 `gemv` driver: `y := alpha * op(A) x + beta * y`.
///
/// The vectors are unit-stride by contract; callers that hold a strided vector
/// make it contiguous before the call (the driver layer has no view-level stride
/// mechanism, mirroring the shape-derived leading dimensions of BLAS-3).
pub trait GEMVDriverAPI<T> {
    unsafe fn driver_gemv(
        order: FlagOrder,
        trans: FlagTrans,
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
    B: BlasDriverBaseAPI<T> + GEMVDriverAPI<T> + OpAssignArbitaryAPI<T, Ix1, Ix1>,
{
    /// Run `y := alpha * op(A) x + beta * y`, returning the output vector.
    ///
    /// The container is order-free: the vectors are one-dimensional, so the
    /// device default order has no observable meaning, and `a`'s layout
    /// preference alone decides the `Op` the driver receives.
    ///
    /// # Errors
    ///
    /// Returns an error for [`FlagTrans::CN`]: the portable CBLAS surface has no
    /// conj-no-trans flag (OpenBLAS has one as an extension, the other backends
    /// do not). Conjugate the matrix and use [`FlagTrans::N`] instead. For real
    /// scalars `C` and `T` are identical, so `C` already covers conjugation.
    pub fn run(self) -> Result<TensorMutable1<'c, T, B>> {
        let Self { a, x, y, alpha, beta, transa } = self;

        // conj-no-trans has no CBLAS counterpart
        if transa == FlagTrans::CN {
            rstsr_invalid!(transa)?;
        }

        // Normalize the matrix to column-major: f-prefer is passed through,
        // c-prefer is viewed transposed with the flag flipped, and only a
        // neither-prefer layout (or a `C` on a c-prefer buffer) is copied.
        let (transa, a_cow) = flip_trans(ColMajor, transa, a, false)?;
        let obj = GEMV_ { a: a_cow.view(), x, y, alpha, beta, transa };
        obj.internal_run()
    }

    /// Column-major-only entry; `a` must already be f-prefer.
    pub fn internal_run(self) -> Result<TensorMutable1<'c, T, B>> {
        let Self { a, x, y, alpha, beta, transa } = self;

        let (m, n) = (a.nrow(), a.ncol());
        // `lx` contracts, `ly` is the output length
        let (lx, ly) = match transa {
            FlagTrans::N => (n, m),
            FlagTrans::T | FlagTrans::C => (m, n),
            _ => rstsr_invalid!(transa)?,
        };

        // device and shape checks
        rstsr_assert!(a.device().same_device(x.device()), DeviceMismatch)?;
        rstsr_assert_eq!(x.shape()[0], lx, InvalidLayout)?;
        if let Some(y) = &y {
            rstsr_assert!(a.device().same_device(y.device()), DeviceMismatch)?;
            rstsr_assert_eq!(y.shape()[0], ly, InvalidLayout)?;
        }

        // Output conversion: a non-unit-stride `y` is written through a
        // contiguous buffer and copied back (the `GEMM_` pattern).
        let mut y = match y {
            Some(y) if ly <= 1 || y.stride()[0] == 1 => TensorMutable::Mut(y),
            Some(y) => {
                let y_buffer = y.to_contig_f(ColMajor)?.into_owned();
                TensorMutable::ToBeCloned(y, y_buffer)
            },
            None => TensorMutable1::Owned(zeros_f(([ly].f(), a.device()))?.into_dim()),
        };

        // empty output: nothing is read or written
        if ly == 0 {
            return Ok(y.clone_to_mut());
        }

        // `k == 0`: BLAS returns before scaling `y`, so `y := beta * y` is
        // applied here (the freshly allocated zero vector needs nothing).
        if lx == 0 {
            if beta != T::zero() && beta != T::one() {
                let (ptr, len) = match &mut y {
                    TensorMutable::Mut(view) => (view.as_mut_ptr(), ly),
                    TensorMutable::ToBeCloned(_, buffer) => (buffer.as_mut_ptr(), ly),
                    TensorMutable::Owned(_) => (core::ptr::null_mut(), 0),
                };
                for i in 0..len {
                    // SAFETY: `ptr` addresses `len` consecutive elements of an
                    // initialized buffer (unit-stride view or contiguous copy).
                    unsafe {
                        let p = ptr.add(i);
                        *p = *p * beta;
                    }
                }
            }
            return Ok(y.clone_to_mut());
        }

        // `a` is column-major after `flip_trans`; `testify_ld` enforces the
        // preference itself and reports the operand on failure
        let lda = a.testify_ld(ColMajor, "A")?;

        // `x` is read-only: a non-unit-stride vector is copied once
        let x_cow;
        let x_view = if lx <= 1 || x.stride()[0] == 1 {
            x.view()
        } else {
            x_cow = x.to_contig_f(ColMajor)?;
            x_cow.view()
        };

        let mut y_view = y.view_mut();
        let ptr_a = a.as_ptr();
        let ptr_x = x_view.as_ptr();
        let ptr_y = y_view.as_mut_ptr();

        unsafe {
            B::driver_gemv(ColMajor, transa, m, n, alpha, ptr_a, lda, ptr_x, beta, ptr_y);
        }

        Ok(y.clone_to_mut())
    }
}

pub type GEMV<'a, 'b, 'c, B, T> = GEMV_Builder<'a, 'b, 'c, B, T>;
pub type SGEMV<'a, 'b, 'c, B> = GEMV<'a, 'b, 'c, B, f32>;
pub type DGEMV<'a, 'b, 'c, B> = GEMV<'a, 'b, 'c, B, f64>;
pub type CGEMV<'a, 'b, 'c, B> = GEMV<'a, 'b, 'c, B, Complex<f32>>;
pub type ZGEMV<'a, 'b, 'c, B> = GEMV<'a, 'b, 'c, B, Complex<f64>>;
