use crate::prelude_dev::*;
use num::complex::ComplexFloat;
use num::ToPrimitive;
use rstsr_core::prelude_dev::*;

/// Map a raw LAPACK `info` return to a rstsr error.
///
/// `routine` is the full routine label as reported to the user, e.g.
/// `"Lapack GELSD"`.
///
/// `info < 0`: illegal argument — an internal bug (builders validate all
/// arguments before the FFI call), reported with the Fortran argument index.
/// `info > 0`: algorithmic failure, semantics defined by the routine itself.
/// `info == 0` returns `Ok`.
pub fn map_lapack_info(info: blas_int, routine: &str) -> Result<()> {
    if info == 0 {
        return Ok(());
    }
    // tier-internal allocation sentinels (-1010/-1011)
    if info == -1010 || info == -1011 {
        return Err(rstsr_error!(MemoryError, "{routine}: workspace allocation failed"));
    }
    let info = blas_int_as_i32(info);
    let neg_info = -info;
    if info < 0 {
        Err(rstsr_error!(
            InvalidValue,
            "{routine}: illegal argument in argument {neg_info} (internal bug, please report)"
        ))
    } else {
        Err(rstsr_error!(InvalidValue, "{routine} (info = {info})"))
    }
}

/// Checked conversion of `blas_int` to `i32` (the info channel of
/// `RSTSRError::ErrorCode`); saturates instead of truncating under ilp64.
pub fn blas_int_as_i32(x: blas_int) -> i32 {
    #[cfg(feature = "ilp64")]
    {
        i32::try_from(x).unwrap_or(if x > 0 { i32::MAX } else { i32::MIN })
    }
    #[cfg(not(feature = "ilp64"))]
    {
        x
    }
}

/// Hardened read-back of a LAPACK workspace-query value (ADR-0009 §4):
/// one-ulp round-up guard for f32-mantissa query results, checked
/// conversion into `blas_int` range, and the `MAX(1, ·)` clamp.
/// `None` — unusable query value (non-finite or negative; vendor malfunction).
pub fn lwork_from_query<T>(work_query: T) -> Option<usize>
where
    T: ComplexFloat,
    T::Real: ToPrimitive,
{
    let re: f64 = work_query.re().to_f64()?;
    if !re.is_finite() || re < 0.0 {
        return None;
    }
    // f32-mantissa query values >= 2^24 truncate low on conversion: grow one ulp
    let grown = if size_of::<T::Real>() == 4 { re * (1.0 + f32::EPSILON as f64) } else { re };
    if grown >= blas_int::MAX as f64 {
        return None;
    }
    Some((grown as usize).max(1))
}

/// Integer counterpart of [`lwork_from_query`] for queries that return
/// `blas_int` directly (e.g. the liwork slot of the double-query routines):
/// negative or `MAX(1, ·)`-clamped, range-checked, no float handling.
pub fn liwork_from_query(iwork_query: blas_int) -> Option<usize> {
    if iwork_query < 0 {
        return None;
    }
    usize::try_from(iwork_query).ok().map(|v| v.max(1))
}
