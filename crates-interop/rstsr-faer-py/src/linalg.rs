//! linalg surface (the array-API `linalg` extension) over rstsr's existing
//! entries.
//!
//! Wrapper-only: each function marshals a `NativeArray` through an rstsr entry
//! that already exists on the faer device — the tensor-tier products
//! (`matmul`, `vecdot`, `matrix_transpose`, `diagonal`) or the faer
//! factorizations (`cholesky`, `det`, `eigh`, `eigvalsh`, `inv`, `pinv`,
//! `solve`, `svd`, `svdvals`). Every dispatch body is written once against a
//! concrete dtype arm, so no generic trait bound is stated here and no faer
//! type is named.
//!
//! Names rstsr/faer does not provide (`qr`, `slogdet`, `eig`, `matrix_norm`,
//! `matrix_power`, `matrix_rank`, `tensordot`, `outer`, `cross`, `trace`, the
//! general-`ord` norms) are absent from the namespace, never stubbed — they
//! are rust-side gaps.

use pyo3::prelude::*;
use pyo3::exceptions::PyTypeError;
use pyo3::types::{PySequence, PySequenceMethods};
use rstsr::prelude::rt;
use rstsr::prelude::*;

use crate::any_tensor::{any_of, device_faer, err_py, type_err, AnyTensor, FTensor, IntoAnyTensor, NativeArray};
use crate::creation::dim_from;

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

/// Same-dtype binary dispatch over the twelve numeric dtypes (matmul, vecdot);
/// bool is declined (rstsr's product kernels are not defined on it).
macro_rules! bin_numeric {
    ($x1:expr, $x2:expr, |$a:ident, $b:ident| $body:expr) => {
        match (&$x1.t, &$x2.t) {
            (AnyTensor::I8($a), AnyTensor::I8($b)) => $body,
            (AnyTensor::I16($a), AnyTensor::I16($b)) => $body,
            (AnyTensor::I32($a), AnyTensor::I32($b)) => $body,
            (AnyTensor::I64($a), AnyTensor::I64($b)) => $body,
            (AnyTensor::U8($a), AnyTensor::U8($b)) => $body,
            (AnyTensor::U16($a), AnyTensor::U16($b)) => $body,
            (AnyTensor::U32($a), AnyTensor::U32($b)) => $body,
            (AnyTensor::U64($a), AnyTensor::U64($b)) => $body,
            (AnyTensor::F32($a), AnyTensor::F32($b)) => $body,
            (AnyTensor::F64($a), AnyTensor::F64($b)) => $body,
            (AnyTensor::C32($a), AnyTensor::C32($b)) => $body,
            (AnyTensor::C64($a), AnyTensor::C64($b)) => $body,
            (AnyTensor::Bool(_), AnyTensor::Bool(_)) => type_err("matmul/vecdot: bool dtype is not defined"),
            _ => type_err("matmul/vecdot: operands must share one dtype (rstsr gap G-009); cast first"),
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

/// Determinant as a 0-d array.
#[pyfunction]
pub fn linalg_det(x: &NativeArray) -> PyResult<NativeArray> {
    let t = unary_fc!(x, |a| {
        let d = err_py(rt::linalg::det_f(a))?;
        any_res(rt::asarray_f((vec![d], dim_from(&[]), device_faer())))
    })?;
    Ok(NativeArray { t })
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

/// Matrix inverse.
#[pyfunction]
pub fn linalg_inv(x: &NativeArray) -> PyResult<NativeArray> {
    let t = unary_fc!(x, |a| any_res(rt::linalg::inv_f(a)))?;
    Ok(NativeArray { t })
}

/// Moore-Penrose pseudoinverse; the rust entry's rank output is dropped
/// (array-API `pinv` returns only the array).
#[pyfunction]
pub fn linalg_pinv(x: &NativeArray, rtol: Option<f64>) -> PyResult<NativeArray> {
    let t = match &x.t {
        AnyTensor::F32(a) => {
            let r = match rtol {
                None => err_py(rt::linalg::pinv_f(a))?.pinv,
                Some(v) => err_py(rt::linalg::pinv_f((a, 0.0f32, v as f32)))?.pinv,
            };
            any_of(r)
        },
        AnyTensor::F64(a) => {
            let r = match rtol {
                None => err_py(rt::linalg::pinv_f(a))?.pinv,
                Some(v) => err_py(rt::linalg::pinv_f((a, 0.0f64, v)))?.pinv,
            };
            any_of(r)
        },
        AnyTensor::C32(a) => {
            let r = match rtol {
                None => err_py(rt::linalg::pinv_f(a))?.pinv,
                Some(v) => err_py(rt::linalg::pinv_f((a, 0.0f32, v as f32)))?.pinv,
            };
            any_of(r)
        },
        AnyTensor::C64(a) => {
            let r = match rtol {
                None => err_py(rt::linalg::pinv_f(a))?.pinv,
                Some(v) => err_py(rt::linalg::pinv_f((a, 0.0f64, v)))?.pinv,
            };
            any_of(r)
        },
        _ => return type_err("linalg: rstsr/faer provides pinv only for float/complex dtypes"),
    };
    Ok(NativeArray { t })
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

/// Sign and log-absolute-determinant, each as a 0-d array.
#[pyfunction]
pub fn linalg_slogdet(x: &NativeArray) -> PyResult<(NativeArray, NativeArray)> {
    let (sign, logabsdet) = unary_fc!(x, |a| {
        let r = err_py(rt::linalg::slogdet_f(a))?;
        Ok((any_of(r.sign), any_of(r.logabsdet)))
    })?;
    Ok((NativeArray { t: sign }, NativeArray { t: logabsdet }))
}

// ------------------------------------------------------------- tensor products

/// Matrix product (also the `@` operator).
#[pyfunction]
pub fn linalg_matmul(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    let t = bin_numeric!(x1, x2, |a, b| any_res(rt::matmul_f(a, b)))?;
    Ok(NativeArray { t })
}

/// Vector dot product over `axis` (the first argument is conjugated).
#[pyfunction]
pub fn linalg_vecdot(x1: &NativeArray, x2: &NativeArray, axis: isize) -> PyResult<NativeArray> {
    let t = bin_numeric!(x1, x2, |a, b| any_res(rt::vecdot_f(a, b, axis)))?;
    Ok(NativeArray { t })
}

/// Tensor contraction over `axes` — an integer (contract the last `n` axes of
/// `x1` with the first `n` of `x2`) or a pair of per-operand axis sequences.
/// Top-level array-API function (not part of the `linalg` namespace).
#[pyfunction]
#[pyo3(signature = (x1, x2, axes = None))]
pub fn linalg_tensordot(
    x1: &NativeArray,
    x2: &NativeArray,
    axes: Option<&pyo3::Bound<'_, pyo3::PyAny>>,
) -> PyResult<NativeArray> {
    let axes = parse_tensordot_axes(axes)?;
    let t = bin_numeric!(x1, x2, |a, b| match &axes {
        TdAxes::Int(n) => any_res(rt::tensordot_f(a, b, *n)),
        TdAxes::Pair(va, vb) => any_res(rt::tensordot_f(a, b, (va.clone(), vb.clone()))),
    })?;
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
