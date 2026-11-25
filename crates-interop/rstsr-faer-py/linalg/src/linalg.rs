//! linalg surface (the array-API `linalg` extension) over rstsr's existing
//! entries.
//!
//! Wrapper-only: each function marshals a `NativeArray` through an rstsr entry
//! that already exists on the faer device — the tensor-tier entries
//! (`matmul`, `vecdot`, `cross`, `matrix_transpose`, `diagonal`, `trace`) or
//! the faer linalg entries (`cholesky`, `det`, `eig`, `eigh`, `eigvals`,
//! `eigvalsh`, `inv`, `matrix_norm`, `matrix_power`, `matrix_rank`, `pinv`,
//! `qr`, `solve`, `svd`, `svdvals`, `vector_norm`). Every dispatch body is
//! written once against a concrete dtype arm, so no generic trait bound is
//! stated here and no faer type is named.
//!
//! Every member of the standard's `linalg` extension is provided here — none is
//! stubbed.
//!
//! `solve` accepts a same-dtype float/complex pair only; the Python layer
//! promotes a mixed-dtype pair first — see `DTYPE-CASTS.md` in this crate.

use core::mem::MaybeUninit;
use core::ops::{Add, Mul, Sub};

use num::complex::ComplexFloat;
use num::{Complex, FromPrimitive, One, Zero};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PySequence, PySequenceMethods};
use rstsr::prelude::rt;
use rstsr::prelude::*;
use rstsr_core::operators::exports::{
    DeviceExtCrossAPI, DeviceExtMatMulAPI, DeviceExtOuterAPI, DeviceExtTensordotAPI, DeviceExtVecdotAPI,
};
use rstsr_core::operators::reduction::OpSumAPI;
use rstsr_core::storage::exports::{DeviceCreationAnyAPI, DeviceRawAPI};
use rstsr_dtype_traits::{DTypePromoteAPI, ExtNum};

use rstsr_faer_py_core::idx_lift;
use rstsr_faer_py_core::{
    any_of, dispatch_bin_promote, dispatch_bin_promote_arith, dispatch_t_numeric_same, err_py, lift, type_err,
    AnyTensor, AnyTensorRef, FTensor, IntoAnyTensor, NativeArray,
};

/// `rt::Result<FTensor<R>>` -> erased handle, by the result's own dtype, so a
/// real-valued eigenvalue/singular-value output lands in the right variant.
fn any_res<R>(r: rt::Result<FTensor<R>>) -> PyResult<AnyTensor>
where
    FTensor<R>: IntoAnyTensor,
{
    err_py(r).map(|t| t.into_any())
}

/// Unary dispatch over the four faer dtypes (float/complex); `$body` is written
/// once against the bound name `$a` and monomorphizes per arm.
macro_rules! unary_fc {
    ($x:expr, |$a:ident| $body:expr) => {
        match &$x.t {
            AnyTensor::F32($a) => $body,
            AnyTensor::F64($a) => $body,
            AnyTensor::C32($a) => $body,
            AnyTensor::C64($a) => $body,
            _ => type_err("linalg: rstsr/faer provides this only for float32/float64/complex64/complex128"),
        }
    };
}

/// Unary dispatch over all 13 dtypes (matrix_transpose, diagonal).
macro_rules! unary_all {
    ($x:expr, |$a:ident| $body:expr) => {
        match &$x.t {
            AnyTensor::Bool($a) => $body,
            AnyTensor::I8($a) => $body,
            AnyTensor::I16($a) => $body,
            AnyTensor::I32($a) => $body,
            AnyTensor::I64($a) => $body,
            AnyTensor::U8($a) => $body,
            AnyTensor::U16($a) => $body,
            AnyTensor::U32($a) => $body,
            AnyTensor::U64($a) => $body,
            AnyTensor::F32($a) => $body,
            AnyTensor::F64($a) => $body,
            AnyTensor::C32($a) => $body,
            AnyTensor::C64($a) => $body,
        }
    };
}

/// Same-dtype binary dispatch over the four faer dtypes (solve).
macro_rules! bin_fc {
    ($x1:expr, $x2:expr, |$a:ident, $b:ident| $body:expr) => {
        match (&$x1.t, &$x2.t) {
            (AnyTensor::F32($a), AnyTensor::F32($b)) => $body,
            (AnyTensor::F64($a), AnyTensor::F64($b)) => $body,
            (AnyTensor::C32($a), AnyTensor::C32($b)) => $body,
            (AnyTensor::C64($a), AnyTensor::C64($b)) => $body,
            (AnyTensor::Bool(_), _) | (_, AnyTensor::Bool(_)) => type_err("solve: bool dtype is not defined"),
            _ => type_err("solve: x1 and x2 must share one float/complex dtype (rstsr gap G-009); cast first"),
        }
    };
}

// ---------------------------------------------------------- faer factorizations

/// Lower/upper Cholesky factor (`upper` selects the triangle).
#[pyfunction]
pub fn linalg_cholesky(x: &NativeArray, upper: bool) -> PyResult<NativeArray> {
    let uplo = if upper { Upper } else { Lower };
    let t = unary_fc!(x, |a| any_res(rt::linalg::cholesky_f((a, uplo))))?;
    Ok(NativeArray { t })
}

/// Determinant: a 0-d array for a 2-D input, or the batch-shaped array for a
/// stack of matrices.
#[pyfunction]
pub fn linalg_det(x: &NativeArray) -> PyResult<NativeArray> {
    let t = unary_fc!(x, |a| any_res(rt::linalg::det_f(a)))?;
    Ok(NativeArray { t })
}

/// General (non-Hermitian) eigendecomposition; both outputs are complex — the
/// complex counterpart of a real input. Eigenvalues are unsorted.
#[pyfunction]
pub fn linalg_eig(x: &NativeArray) -> PyResult<(NativeArray, NativeArray)> {
    let (w, v) = unary_fc!(x, |a| {
        let r = err_py(rt::linalg::eig_f(a))?;
        Ok((any_of(r.eigenvalues), any_of(r.right.unwrap())))
    })?;
    Ok((NativeArray { t: w }, NativeArray { t: v }))
}

/// Symmetric/Hermitian eigendecomposition; the eigenvector triangle follows the
/// device default order (rstsr has no `uplo` argument in the array-API sense).
#[pyfunction]
pub fn linalg_eigh(x: &NativeArray) -> PyResult<(NativeArray, NativeArray)> {
    let (w, v) = unary_fc!(x, |a| {
        let r = err_py(rt::linalg::eigh_f((a, None::<FlagUpLo>)))?;
        Ok((any_of(r.eigenvalues), any_of(r.eigenvectors)))
    })?;
    Ok((NativeArray { t: w }, NativeArray { t: v }))
}

/// Eigenvalues of a symmetric/Hermitian matrix.
#[pyfunction]
pub fn linalg_eigvalsh(x: &NativeArray) -> PyResult<NativeArray> {
    let t = unary_fc!(x, |a| any_res(rt::linalg::eigvalsh_f((a, None::<FlagUpLo>))))?;
    Ok(NativeArray { t })
}

/// Eigenvalues of a general (non-Hermitian) matrix; complex, unsorted.
#[pyfunction]
pub fn linalg_eigvals(x: &NativeArray) -> PyResult<NativeArray> {
    let t = unary_fc!(x, |a| any_res(rt::linalg::eigvals_f(a)))?;
    Ok(NativeArray { t })
}

/// Matrix inverse.
#[pyfunction]
pub fn linalg_inv(x: &NativeArray) -> PyResult<NativeArray> {
    let t = unary_fc!(x, |a| any_res(rt::linalg::inv_f(a)))?;
    Ok(NativeArray { t })
}

/// Integer matrix power; a negative exponent inverts the base first.
#[pyfunction]
pub fn linalg_matrix_power(x: &NativeArray, n: isize) -> PyResult<NativeArray> {
    let t = unary_fc!(x, |a| any_res(rt::linalg::matrix_power_f((a, n))))?;
    Ok(NativeArray { t })
}

/// A Python norm order (`None`, `'fro'`/`'nuc'`, an int, or a float) to `NormOrd`.
fn py_to_ord(ord: &pyo3::Bound<'_, PyAny>) -> PyResult<rt::linalg::NormOrd> {
    if ord.is_none() {
        return Ok(rt::linalg::NormOrd::None);
    }
    if let Ok(s) = ord.extract::<String>() {
        return err_py(rt::linalg::NormOrd::try_from(s.as_str()));
    }
    if let Ok(v) = ord.extract::<isize>() {
        return err_py(rt::linalg::NormOrd::try_from(v));
    }
    if let Ok(v) = ord.extract::<f64>() {
        return err_py(rt::linalg::NormOrd::try_from(v));
    }
    type_err("norm: ord must be None, 'fro'/'nuc', an int, or a float")
}

/// Matrix norm; `ord` is `'fro'` (default), `'nuc'`, `±1`, `±2`, `±inf`, or `None`.
#[pyfunction]
pub fn linalg_matrix_norm(x: &NativeArray, keepdims: bool, ord: &pyo3::Bound<'_, PyAny>) -> PyResult<NativeArray> {
    let ord = py_to_ord(ord)?;
    let t = unary_fc!(x, |a| any_res(rt::linalg::matrix_norm_f((a, rt::linalg::MatrixNormArgs { ord, keepdims }))))?;
    Ok(NativeArray { t })
}

/// `rtol` for `matrix_rank`: a bare Python float becomes a scalar tolerance (no
/// device), a tensor one per matrix. The element type is the input's **real**
/// projection `T::Real` — float32 for both `f32` and `c32`, float64 for both
/// `f64` and `c64`; the Python layer has already cast an array to that dtype.
fn matrix_rank_rtol<T>(
    rtol: Option<&pyo3::Bound<'_, PyAny>>,
) -> PyResult<rt::linalg::MatrixRankRtol<T::Real, DeviceFaer>>
where
    T: ComplexFloat,
    T::Real: num::FromPrimitive + Clone + Send + Sync + 'static,
    AnyTensor: AnyTensorRef<T::Real>,
{
    let Some(r) = rtol else {
        return Ok(rt::linalg::MatrixRankRtol::None);
    };
    if let Ok(f) = r.extract::<f64>() {
        let f = T::Real::from_f64(f)
            .ok_or_else(|| PyTypeError::new_err("matrix_rank: rtol is not representable in the input's real dtype"))?;
        return Ok(rt::linalg::MatrixRankRtol::Float(f));
    }
    match r.cast::<NativeArray>()?.borrow().t.tensor_ref() {
        Some(t) => Ok(rt::linalg::MatrixRankRtol::Array(t.clone())),
        None => type_err("matrix_rank: rtol must have the input's real floating dtype"),
    }
}

/// Matrix rank (the count of singular values above the tolerance). `rtol` is
/// absent/`None` for the default `max(M, N) * eps`, a Python float for one
/// tolerance on every matrix, or an array in the input's real dtype for one
/// tolerance per matrix — the kernel broadcasts it against the batch (the
/// Python layer only casts the dtype; binding marshalling, not a promotion
/// pre-cast). The rank is an index output, lifted to the namespace's int64.
#[pyfunction]
pub fn linalg_matrix_rank(x: &NativeArray, rtol: Option<&pyo3::Bound<'_, PyAny>>) -> PyResult<NativeArray> {
    match &x.t {
        AnyTensor::F32(a) => idx_lift(rt::linalg::matrix_rank_f((a, matrix_rank_rtol::<f32>(rtol)?))),
        AnyTensor::C32(a) => idx_lift(rt::linalg::matrix_rank_f((a, matrix_rank_rtol::<Complex<f32>>(rtol)?))),
        AnyTensor::F64(a) => idx_lift(rt::linalg::matrix_rank_f((a, matrix_rank_rtol::<f64>(rtol)?))),
        AnyTensor::C64(a) => idx_lift(rt::linalg::matrix_rank_f((a, matrix_rank_rtol::<Complex<f64>>(rtol)?))),
        _ => type_err("linalg: matrix_rank is provided only for float/complex dtypes"),
    }
}

/// Vector norm; `ord` is a number (default `2`) or `±inf`, and `axis` (already
/// marshalled by the Python wrapper) is `None` for the whole array or a list of
/// axes.
#[pyfunction]
pub fn linalg_vector_norm(
    x: &NativeArray,
    ord: &pyo3::Bound<'_, PyAny>,
    axis: Option<Vec<isize>>,
    keepdims: bool,
) -> PyResult<NativeArray> {
    let ord = py_to_ord(ord)?;
    let axes = match axis {
        Some(v) => AxesIndex::Vec(v),
        None => AxesIndex::None,
    };
    let t =
        unary_fc!(x, |a| any_res(rt::linalg::vector_norm_f((a, rt::linalg::VectorNormArgs { ord, axes, keepdims }))))?;
    Ok(NativeArray { t })
}

/// `rtol` for `pinv`: a bare Python float becomes a scalar tolerance (no
/// device), a tensor one per matrix. Same marshalling as `matrix_rank_rtol`.
fn pinv_rtol<T>(rtol: Option<&pyo3::Bound<'_, PyAny>>) -> PyResult<rt::linalg::PinvRtol<T::Real, DeviceFaer>>
where
    T: ComplexFloat,
    T::Real: num::FromPrimitive + Clone + Send + Sync + 'static,
    AnyTensor: AnyTensorRef<T::Real>,
{
    let Some(r) = rtol else {
        return Ok(rt::linalg::PinvRtol::None);
    };
    if let Ok(f) = r.extract::<f64>() {
        let f = T::Real::from_f64(f)
            .ok_or_else(|| PyTypeError::new_err("pinv: rtol is not representable in the input's real dtype"))?;
        return Ok(rt::linalg::PinvRtol::Float(f));
    }
    match r.cast::<NativeArray>()?.borrow().t.tensor_ref() {
        Some(t) => Ok(rt::linalg::PinvRtol::Array(t.clone())),
        None => type_err("pinv: rtol must have the input's real floating dtype"),
    }
}

/// Moore-Penrose pseudoinverse; the rust entry's rank output is dropped
/// (array-API `pinv` returns only the array). `rtol` is absent/`None` for the
/// default `max(M, N) * eps`, a Python float for one tolerance on every matrix,
/// or an array broadcast against the batch for one tolerance per matrix.
#[pyfunction]
pub fn linalg_pinv(x: &NativeArray, rtol: Option<&pyo3::Bound<'_, PyAny>>) -> PyResult<NativeArray> {
    let t = match &x.t {
        AnyTensor::F32(a) => any_of(err_py(rt::linalg::pinv_f((a, pinv_rtol::<f32>(rtol)?)))?.pinv),
        AnyTensor::F64(a) => any_of(err_py(rt::linalg::pinv_f((a, pinv_rtol::<f64>(rtol)?)))?.pinv),
        AnyTensor::C32(a) => any_of(err_py(rt::linalg::pinv_f((a, pinv_rtol::<Complex<f32>>(rtol)?)))?.pinv),
        AnyTensor::C64(a) => any_of(err_py(rt::linalg::pinv_f((a, pinv_rtol::<Complex<f64>>(rtol)?)))?.pinv),
        _ => return type_err("linalg: rstsr/faer provides pinv only for float/complex dtypes"),
    };
    Ok(NativeArray { t })
}

/// QR decomposition `(Q, R)`; `mode` is `'reduced'` (default) or `'complete'`.
#[pyfunction]
pub fn linalg_qr(x: &NativeArray, mode: &str) -> PyResult<(NativeArray, NativeArray)> {
    let mode = err_py(rt::linalg::QrMode::try_from(mode))?;
    let (q, r) = unary_fc!(x, |a| {
        let res = err_py(rt::linalg::qr_f((a, mode)))?;
        Ok((any_of(res.q), any_of(res.r)))
    })?;
    Ok((NativeArray { t: q }, NativeArray { t: r }))
}

/// Solve a general linear system `a x = b`.
#[pyfunction]
pub fn linalg_solve(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    let t = bin_fc!(x1, x2, |a, b| any_res(rt::linalg::solve_general_f((a, b))))?;
    Ok(NativeArray { t })
}

/// Singular value decomposition `(U, S, Vh)`; `full_matrices` selects full or
/// reduced matrices.
#[pyfunction]
pub fn linalg_svd(x: &NativeArray, full_matrices: bool) -> PyResult<(NativeArray, NativeArray, NativeArray)> {
    let (u, s, vh) = unary_fc!(x, |a| {
        let r = err_py(rt::linalg::svd_f((a, full_matrices)))?;
        Ok((any_of(r.u), any_of(r.s), any_of(r.vt)))
    })?;
    Ok((NativeArray { t: u }, NativeArray { t: s }, NativeArray { t: vh }))
}

/// Singular values only.
#[pyfunction]
pub fn linalg_svdvals(x: &NativeArray) -> PyResult<NativeArray> {
    let t = unary_fc!(x, |a| any_res(rt::linalg::svdvals_f(a)))?;
    Ok(NativeArray { t })
}

/// Sign and log-absolute-determinant: a 0-d array for a 2-D input, or the
/// batch-shaped arrays for a stack of matrices.
#[pyfunction]
pub fn linalg_slogdet(x: &NativeArray) -> PyResult<(NativeArray, NativeArray)> {
    let (sign, logabsdet) = unary_fc!(x, |a| {
        let r = err_py(rt::linalg::slogdet_f(a))?;
        Ok((any_of(r.sign), any_of(r.logabsdet)))
    })?;
    Ok((NativeArray { t: sign }, NativeArray { t: logabsdet }))
}

// ------------------------------------------------------------- tensor products

/// Mixed-dtype matmul wrapper: the pair promotes to its common dtype
/// ([`DTypePromoteAPI`]), which is also the result dtype, so the dispatch
/// macro lifts the result into the promoted variant. Same-dtype pairs take the
/// same path (promotion is the identity there).
fn op_ext_matmul<T, U>(a: &FTensor<T>, b: &FTensor<U>) -> rt::Result<FTensor<<T as DTypePromoteAPI<U>>::Res>>
where
    T: DTypePromoteAPI<U> + Clone + Send + Sync + 'static,
    U: DTypePromoteAPI<T, Res = <T as DTypePromoteAPI<U>>::Res> + Clone + Send + Sync + 'static,
    <T as DTypePromoteAPI<U>>::Res: Clone + Send + Sync + 'static + PartialEq + Zero + One,
    <T as DTypePromoteAPI<U>>::Res: Add<<T as DTypePromoteAPI<U>>::Res, Output = <T as DTypePromoteAPI<U>>::Res>
        + Mul<<T as DTypePromoteAPI<U>>::Res, Output = <T as DTypePromoteAPI<U>>::Res>,
    DeviceFaer: DeviceRawAPI<<T as DTypePromoteAPI<U>>::Res, Raw = Vec<<T as DTypePromoteAPI<U>>::Res>>
        + DeviceCreationAnyAPI<<T as DTypePromoteAPI<U>>::Res>
        + DeviceRawAPI<MaybeUninit<<T as DTypePromoteAPI<U>>::Res>>
        + DeviceExtMatMulAPI<T, U, <T as DTypePromoteAPI<U>>::Res, IxD, IxD, IxD>,
{
    rt::ext_matmul_f(a, b)
}

/// Matrix product (also the `@` operator), promoting mixed-dtype operands to
/// their common dtype.
#[pyfunction]
pub fn linalg_matmul(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    let t = dispatch_bin_promote_arith!(x1.t, x2.t, "matmul", op_ext_matmul)?;
    Ok(NativeArray { t })
}

/// Mixed-dtype vecdot wrapper: the pair promotes to its common dtype
/// ([`DTypePromoteAPI`]), and the first operand is conjugated in that dtype
/// (the Array-API `vecdot` contract; identity for real dtypes).
fn op_ext_vecdot<T, U>(
    a: &FTensor<T>,
    b: &FTensor<U>,
    axis: isize,
) -> rt::Result<FTensor<<T as DTypePromoteAPI<U>>::Res>>
where
    T: DTypePromoteAPI<U> + Clone + Send + Sync,
    U: Clone + Send + Sync,
    <T as DTypePromoteAPI<U>>::Res: Clone + Send + Sync + Zero + ExtNum,
    <T as DTypePromoteAPI<U>>::Res: Mul<<T as DTypePromoteAPI<U>>::Res, Output = <T as DTypePromoteAPI<U>>::Res>,
    DeviceFaer: DeviceExtVecdotAPI<T, U, <T as DTypePromoteAPI<U>>::Res, IxD, IxD, IxD>
        + DeviceAPI<<T as DTypePromoteAPI<U>>::Res>
        + DeviceCreationAnyAPI<<T as DTypePromoteAPI<U>>::Res>
        + DeviceRawAPI<MaybeUninit<<T as DTypePromoteAPI<U>>::Res>>,
{
    rt::ext_vecdot_f(a, b, axis)
}

/// Vector dot product over `axis` (the first argument is conjugated), promoting
/// mixed-dtype operands to their common dtype.
#[pyfunction]
pub fn linalg_vecdot(x1: &NativeArray, x2: &NativeArray, axis: isize) -> PyResult<NativeArray> {
    let t = dispatch_bin_promote_arith!(x1.t, x2.t, "vecdot", op_ext_vecdot, axis)?;
    Ok(NativeArray { t })
}

/// Mixed-dtype tensordot wrapper: the pair promotes to its common dtype
/// ([`DTypePromoteAPI`]).
fn op_ext_tensordot<T, U>(
    a: &FTensor<T>,
    b: &FTensor<U>,
    axes: &TdAxes,
) -> rt::Result<FTensor<<T as DTypePromoteAPI<U>>::Res>>
where
    T: DTypePromoteAPI<U> + Clone + Send + Sync + 'static,
    U: Clone + Send + Sync + 'static,
    <T as DTypePromoteAPI<U>>::Res: Clone + Send + Sync + 'static + Zero + One,
    <T as DTypePromoteAPI<U>>::Res: Mul<<T as DTypePromoteAPI<U>>::Res, Output = <T as DTypePromoteAPI<U>>::Res>,
    DeviceFaer: DeviceExtTensordotAPI<T, U, <T as DTypePromoteAPI<U>>::Res, IxD, IxD, IxD>
        + DeviceAPI<<T as DTypePromoteAPI<U>>::Res>
        + DeviceCreationAnyAPI<<T as DTypePromoteAPI<U>>::Res>
        + DeviceRawAPI<MaybeUninit<<T as DTypePromoteAPI<U>>::Res>>,
{
    match axes {
        TdAxes::Int(n) => rt::ext_tensordot_f(a, b, *n),
        TdAxes::Pair(va, vb) => rt::ext_tensordot_f(a, b, (va.clone(), vb.clone())),
    }
}

/// Tensor contraction over `axes` — an integer (contract the last `n` axes of
/// `x1` with the first `n` of `x2`) or a pair of per-operand axis sequences,
/// promoting mixed-dtype operands to their common dtype. Top-level array-API
/// function (not part of the `linalg` namespace).
#[pyfunction]
#[pyo3(signature = (x1, x2, axes = None))]
pub fn linalg_tensordot(
    x1: &NativeArray,
    x2: &NativeArray,
    axes: Option<&pyo3::Bound<'_, pyo3::PyAny>>,
) -> PyResult<NativeArray> {
    let axes = parse_tensordot_axes(axes)?;
    let t = dispatch_bin_promote_arith!(x1.t, x2.t, "tensordot", op_ext_tensordot, &axes)?;
    Ok(NativeArray { t })
}

/// Mixed-dtype outer-product wrapper: the pair promotes to its common dtype
/// ([`DTypePromoteAPI`]), which is also the result dtype.
fn op_ext_outer<T, U>(a: &FTensor<T>, b: &FTensor<U>) -> rt::Result<FTensor<<T as DTypePromoteAPI<U>>::Res>>
where
    T: DTypePromoteAPI<U> + Clone + Send + Sync,
    U: Clone + Send + Sync,
    <T as DTypePromoteAPI<U>>::Res: Clone + Send + Sync,
    <T as DTypePromoteAPI<U>>::Res: Mul<<T as DTypePromoteAPI<U>>::Res, Output = <T as DTypePromoteAPI<U>>::Res>,
    DeviceFaer: DeviceExtOuterAPI<T, U, <T as DTypePromoteAPI<U>>::Res>
        + DeviceAPI<<T as DTypePromoteAPI<U>>::Res>
        + DeviceCreationAnyAPI<<T as DTypePromoteAPI<U>>::Res>
        + DeviceRawAPI<MaybeUninit<<T as DTypePromoteAPI<U>>::Res>>,
{
    // the core entry is rank-2 (`Ix2`); the shim's handles are dynamic (`IxD`)
    rt::ext_outer_f(a, b).map(|t| t.into_dim::<IxD>())
}

/// Outer product of two one-dimensional arrays, promoting mixed-dtype operands
/// to their common dtype.
#[pyfunction]
pub fn linalg_outer(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    let t = dispatch_bin_promote_arith!(x1.t, x2.t, "outer", op_ext_outer)?;
    Ok(NativeArray { t })
}

/// Mixed-dtype cross-product wrapper: each product pair promotes to its common
/// dtype ([`DTypePromoteAPI`]), which is also the result dtype.
fn op_ext_cross<T, U>(
    a: &FTensor<T>,
    b: &FTensor<U>,
    axis: isize,
) -> rt::Result<FTensor<<T as DTypePromoteAPI<U>>::Res>>
where
    T: DTypePromoteAPI<U> + Clone + Send + Sync,
    U: Clone + Send + Sync,
    <T as DTypePromoteAPI<U>>::Res: Clone + Send + Sync,
    <T as DTypePromoteAPI<U>>::Res: Mul<<T as DTypePromoteAPI<U>>::Res, Output = <T as DTypePromoteAPI<U>>::Res>
        + Sub<<T as DTypePromoteAPI<U>>::Res, Output = <T as DTypePromoteAPI<U>>::Res>,
    DeviceFaer: DeviceExtCrossAPI<T, U, <T as DTypePromoteAPI<U>>::Res, IxD>
        + DeviceAPI<<T as DTypePromoteAPI<U>>::Res>
        + DeviceCreationAnyAPI<<T as DTypePromoteAPI<U>>::Res>
        + DeviceRawAPI<MaybeUninit<<T as DTypePromoteAPI<U>>::Res>>,
{
    // the shim's handles are dynamic (`IxD`); the core entry keeps the broadcast rank
    rt::ext_cross_f(a, b, axis).map(|t| t.into_dim::<IxD>())
}

/// Cross product of 3-element vectors along `axis` (default the last axis),
/// promoting mixed-dtype operands to their common dtype.
#[pyfunction]
pub fn linalg_cross(x1: &NativeArray, x2: &NativeArray, axis: isize) -> PyResult<NativeArray> {
    let t = dispatch_bin_promote_arith!(x1.t, x2.t, "cross", op_ext_cross, axis)?;
    Ok(NativeArray { t })
}

enum TdAxes {
    Int(isize),
    Pair(Vec<isize>, Vec<isize>),
}

fn side_to_vec(o: &pyo3::Bound<'_, pyo3::PyAny>) -> PyResult<Vec<isize>> {
    if let Ok(n) = o.extract::<isize>() {
        return Ok(vec![n]);
    }
    o.extract::<Vec<isize>>()
        .map_err(|_| PyTypeError::new_err("tensordot: each axes entry must be an int or a sequence of ints"))
}

fn parse_tensordot_axes(axes: Option<&pyo3::Bound<'_, pyo3::PyAny>>) -> PyResult<TdAxes> {
    let axes = match axes {
        None => return Ok(TdAxes::Int(2)),
        Some(a) if a.is_none() => return Ok(TdAxes::Int(2)),
        Some(a) => a,
    };
    if let Ok(n) = axes.extract::<isize>() {
        return Ok(TdAxes::Int(n));
    }
    let seq = axes
        .cast::<PySequence>()
        .map_err(|_| PyTypeError::new_err("tensordot: `axes` must be an int or a pair of sequences"))?;
    if seq.len()? != 2 {
        return Err(PyTypeError::new_err("tensordot: `axes` tuple must have length 2"));
    }
    let va = side_to_vec(&seq.get_item(0)?)?;
    let vb = side_to_vec(&seq.get_item(1)?)?;
    Ok(TdAxes::Pair(va, vb))
}

/// Transpose the last two axes (a stack of matrices transposed independently).
#[pyfunction]
pub fn linalg_matrix_transpose(x: &NativeArray) -> PyResult<NativeArray> {
    let t = unary_all!(x, |a| err_py(rt::matrix_transpose_f(a)).map(|v| any_of(v.into_owned())))?;
    Ok(NativeArray { t })
}

/// The `offset` diagonal of the innermost two axes (array-API convention: the
/// diagonal is appended as the last axis).
#[pyfunction]
pub fn linalg_diagonal(x: &NativeArray, offset: isize) -> PyResult<NativeArray> {
    let t =
        unary_all!(x, |a| { err_py(rt::diagonal_f(a, (offset, -2isize, -1isize))).map(|v| any_of(v.into_owned())) })?;
    Ok(NativeArray { t })
}

/// Trace with the output dtype pinned to the input dtype (the `TOut = T`
/// reduction, the same shape as `op_sum_axes`): the Python layer has already
/// cast the input to the accumulation dtype, so the sum must not widen it
/// again (see `DTYPE-CASTS.md`).
fn op_trace<T>(t: &FTensor<T>, offset: isize) -> rt::Result<FTensor<T>>
where
    DeviceFaer: DeviceRawAPI<T, Raw = Vec<T>> + OpSumAPI<T, IxD, TOut = T>,
{
    rt::trace_f(t, offset)
}

/// Sum along the `offset` diagonal of the innermost two axes (a stack of
/// matrices, one sum per matrix; the two innermost axes are dropped).
#[pyfunction]
pub fn linalg_trace(x: &NativeArray, offset: isize) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_t_numeric_same!(x.t, "trace", op_trace(offset))? })
}
