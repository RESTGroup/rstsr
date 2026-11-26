use crate::prelude_dev::*;
use num::{Float, Zero};
use rstsr_core::prelude_dev::*;

pub trait GELSDDriverAPI<T>
where
    T: BlasFloat,
{
    unsafe fn driver_gelsd(
        order: FlagOrder,
        m: usize,
        n: usize,
        nrhs: usize,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        s: *mut T::Real,
        rcond: T::Real,
        rank: *mut blas_int,
    ) -> blas_int;
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct GELSD_<'a, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
    #[builder(setter(into))]
    pub b: TensorReference<'a, T, B, Ix2>,
    pub rcond: T::Real,
}

impl<'a, B, T> GELSD_<'a, B, T>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T>
        + GELSDDriverAPI<T>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationAnyAPI<T::Real>
        + OpAssignArbitaryAPI<T, Ix2, Ix2>
        + OpAssignAPI<T, Ix2>,
{
    pub fn internal_run(
        self,
    ) -> Result<(Tensor<T, B, Ix2>, Tensor<T::Real, B, Ix1>, usize, Option<Tensor<T::Real, B, Ix1>>)> {
        let Self { a, b, rcond } = self;
        let device = a.device().clone();

        let order = get_output_order(
            &[],
            &[(a.c_prefer(), a.f_prefer()), (b.c_prefer(), b.f_prefer())],
            device.default_order(),
        );

        let [m, n] = *a.shape();
        let [b_rows, k] = *b.shape();
        rstsr_assert_eq!(
            b_rows,
            m,
            InvalidLayout,
            "Lapack GELSD: b must have the same number of rows as a, got {} vs {}",
            b_rows,
            m
        )?;
        if <T::Real as Float>::is_nan(rcond) {
            return Err(rstsr_error!(InvalidValue, "Lapack GELSD: rcond must not be NaN"));
        }

        // zero-size: GELSD quick-returns before any work; synthesize the outputs
        if n == 0 || m == 0 {
            let x = zeros_f(([n, k], order, &device))?.into_dim::<Ix2>();
            let s = zeros_f(([0], &device))?.into_dim::<Ix1>();
            // n == 0 < m: residuals are the column sums of |b|^2; none otherwise
            let residuals = if n == 0 && m > 0 {
                let mut b_work = zeros_f(([m, k], order, &device))?.into_dim::<Ix2>();
                b_work.assign_f(b)?;
                Some(residual_sums(&b_work, 0, m, k, order)?)
            } else {
                None
            };
            return Ok((x, s, 0, residuals));
        }

        // GELSD destroys a; clone-or-consume it in the call order
        let mut a_work = overwritable_convert_with_order(a, order)?;

        // b workspace: max(m, n) rows; pad nrhs to 1 (DLALSD requires nrhs >= 1)
        let k_call = k.max(1);
        let mut b_work = zeros_f(([m.max(n), k_call], order, &device))?.into_dim::<Ix2>();
        if k > 0 {
            b_work.i_mut((..m, ..)).assign_f(b)?;
        }
        let mut s = zeros_f(([m.min(n)], &device))?.into_dim::<Ix1>();

        let lda = a_work.view().testify_ld(order, "A")?;
        let ldb = b_work.testify_ld(order, "B")?;

        let mut rank: blas_int = 0;
        let info = unsafe {
            B::driver_gelsd(
                order,
                m,
                n,
                k_call,
                a_work.view_mut().as_mut_ptr(),
                lda,
                b_work.as_mut_ptr(),
                ldb,
                s.as_mut_ptr(),
                rcond,
                &mut rank,
            )
        };
        map_gelsd_info(info)?;
        let rank = rank as usize;

        let mut x = zeros_f(([n, k], order, &device))?.into_dim::<Ix2>();
        if k > 0 {
            x.assign_f(b_work.i((..n, ..)))?;
        }
        // residuals only for an over-determined full-rank system (m > n)
        let residuals = if m > n && rank == n {
            if k == 0 {
                Some(zeros_f(([0], &device))?.into_dim::<Ix1>())
            } else {
                Some(residual_sums(&b_work, n, m, k, order)?)
            }
        } else {
            None
        };
        Ok((x, s, rank, residuals))
    }

    pub fn run(self) -> Result<(Tensor<T, B, Ix2>, Tensor<T::Real, B, Ix1>, usize, Option<Tensor<T::Real, B, Ix1>>)> {
        self.internal_run()
    }
}

/// Sum of `|b[i, j]|^2` over the row range `start..end` for every column.
/// An empty `b` has no leading dimension; the result is empty.
fn residual_sums<T, B>(
    b_work: &Tensor<T, B, Ix2>,
    start: usize,
    end: usize,
    k: usize,
    order: FlagOrder,
) -> Result<Tensor<T::Real, B, Ix1>>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + DeviceCreationNumAPI<T::Real>,
{
    let device = b_work.device().clone();
    let mut out = zeros_f(([k], &device))?.into_dim::<Ix1>();
    if k == 0 {
        return Ok(out);
    }
    let ld = b_work.testify_ld(order, "B")?;
    let (stride_i, stride_j) = match order {
        RowMajor => (ld as isize, 1isize),
        ColMajor => (1isize, ld as isize),
    };
    let raw = b_work.raw();
    for j in 0..k {
        let mut acc = T::Real::zero();
        for i in start..end {
            let z = raw[(i as isize * stride_i + j as isize * stride_j) as usize].abs();
            acc = acc + z * z;
        }
        out[[j]] = acc;
    }
    Ok(out)
}

/// Map a raw `GELSD` info return to a rstsr error.
///
/// Netlib LAPACK master (`3d90a787f`, 2026-11; in no release up to 3.12.1)
/// reports a NaN/Inf `A` as `info = -4` (its `A` argument; `B` is never
/// checked); libraries without that check may fail to converge instead.
/// LAPACKE reports it as `-5`/`-7`/`-10` (its `matrix_layout` shifts the
/// Fortran indices). In the `lapack` configuration
/// `-5`/`-7` are illegal-LDA/LDB codes, which the builder rules out before
/// the call. A positive `info` means the underlying SVD did not converge.
fn map_gelsd_info(info: blas_int) -> Result<()> {
    let code = blas_int_as_i32(info);
    if code == -4 || code == -5 || code == -7 || code == -10 {
        return Err(rstsr_error!(InvalidValue, "Lapack GELSD: A or B contains a NaN or infinity"));
    }
    if code > 0 {
        return Err(rstsr_error!(
            InvalidValue,
            "Lapack GELSD: SVD failed to converge (info = {code} off-diagonal elements of an intermediate bidiagonal form did not converge to zero)"
        ));
    }
    map_lapack_info(info, "Lapack GELSD")
}

pub type GELSD<'a, B, T> = GELSD_Builder<'a, B, T>;
pub type SGELSD<'a, B> = GELSD<'a, B, f32>;
pub type DGELSD<'a, B> = GELSD<'a, B, f64>;
pub type CGELSD<'a, B> = GELSD<'a, B, Complex<f32>>;
pub type ZGELSD<'a, B> = GELSD<'a, B, Complex<f64>>;
