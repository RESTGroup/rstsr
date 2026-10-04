//! Operation surface (S1 vertical slice): elementwise arithmetic, comparisons,
//! predicates, whole-array reductions, reshape/transpose. Same-dtype only —
//! rstsr arithmetic is same-dtype at the device level, and cross-dtype
//! promotion is a registered gap (G-009).
//!
//! Dispatch macros take generic fn items (never closures): a captured
//! closure is type-checked once across all 13 expansion points, while fn
//! items instantiate per arm.

use num::Complex;
use pyo3::prelude::*;
use rstsr::prelude::*;
use rstsr::prelude::rt;
use core::mem::MaybeUninit;
use rstsr_core::storage::exports::{DeviceCreationAnyAPI, DeviceRawAPI};
use rstsr_common::layout::exports::Indexer;
use rstsr_core::operators::assignment::OpAssignAPI;
use rstsr_core::tensor::operators::exports::{
    TensorAddAPI, TensorDivAPI, TensorEqualAPI, TensorGreaterAPI, TensorGreaterEqualAPI,
    TensorLessAPI, TensorLessEqualAPI, TensorMulAPI, TensorNegAPI, TensorNotEqualAPI,
    TensorSubAPI,
};

use crate::any_tensor::{
    device_faer, dispatch_bin, dispatch_bin_numeric_self, dispatch_t, dispatch_t_bool, dispatch_t_signed, err_py,
    lift, type_err, AnyTensor, FTensor, NativeArray,
};
use crate::creation::dim_from;

/// Whole-array boolean reduction rebuilt as a 0-d array. rstsr's `all`/`any`
/// reductions are bool-typed only (OpAllAPI), so truthiness is derived as
/// `x != 0` first — values are exact, no algorithm added (gap G-017).
fn op_all<T>(t: &FTensor<T>) -> rt::Result<FTensor<bool>>
where
    T: Default + PartialEq + Clone,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T>,
    for<'x> &'x FTensor<T>: TensorNotEqualAPI<&'x FTensor<T>, Output = FTensor<bool>>,
{
    let zero: FTensor<T> = rt::asarray_f((vec![T::default()], device_faer()))?;
    let truthy = rt::not_equal_f(t, &zero)?;
    let s: bool = rt::all_f(&truthy)?;
    rt::asarray_f((vec![s], dim_from(&[]), device_faer()))
}

fn op_any<T>(t: &FTensor<T>) -> rt::Result<FTensor<bool>>
where
    T: Default + PartialEq + Clone,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T>,
    for<'x> &'x FTensor<T>: TensorNotEqualAPI<&'x FTensor<T>, Output = FTensor<bool>>,
{
    let zero: FTensor<T> = rt::asarray_f((vec![T::default()], device_faer()))?;
    let truthy = rt::not_equal_f(t, &zero)?;
    let s: bool = rt::any_f(&truthy)?;
    rt::asarray_f((vec![s], dim_from(&[]), device_faer()))
}

fn op_neg<T>(t: &FTensor<T>) -> rt::Result<FTensor<T>>
where
    for<'a> &'a FTensor<T>: TensorNegAPI<Output = FTensor<T>>,
{
    rt::neg_f(t)
}

// isnan/isfinite/isinf: rstsr's is_nan_f/is_finite_f/is_inf_f exist only for
// float/complex tensors (TensorIsNanAPI etc. have no int/bool impls). The
// standard defines them on every dtype — ints/bool yield constant arrays
// (false/true/false). Float/complex go through rstsr's own ops (G-017).

fn const_bool(n: usize, v: bool) -> rt::Result<FTensor<bool>> {
    rt::asarray_f((vec![v; n], device_faer()))
}

fn op_isnan_f<T>(t: &FTensor<T>) -> rt::Result<FTensor<bool>>
where
    for<'a> &'a FTensor<T>: TensorIsNanAPI<Output = FTensor<bool>>,
{
    rt::is_nan_f(t)
}

fn op_isfinite_f<T>(t: &FTensor<T>) -> rt::Result<FTensor<bool>>
where
    for<'a> &'a FTensor<T>: TensorIsFiniteAPI<Output = FTensor<bool>>,
{
    rt::is_finite_f(t)
}

fn op_isinf_f<T>(t: &FTensor<T>) -> rt::Result<FTensor<bool>>
where
    for<'a> &'a FTensor<T>: TensorIsInfAPI<Output = FTensor<bool>>,
{
    rt::is_inf_f(t)
}

fn op_reshape<T>(t: &FTensor<T>, shape: Vec<isize>) -> rt::Result<FTensor<T>>
where
    T: Clone + Send + Sync,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, IxD>,
{
    let cow = rt::reshape_f(t, shape)?;
    Ok(cow.into_owned())
}

fn op_transpose<T>(t: &FTensor<T>, axes: Option<Vec<isize>>) -> rt::Result<FTensor<T>>
where
    T: Clone + Send + Sync,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, IxD>,
{
    // axes=None: full axis reversal (spec `.T` semantics)
    let reversed: Vec<isize> = (0..t.ndim()).rev().map(|i| i as isize).collect();
    let view = rt::transpose_f(t, axes.unwrap_or(reversed))?;
    Ok(view.into_owned())
}

// ------------------------------------------------------------ bin wrappers --

fn op_add<T>(a: &FTensor<T>, b: &FTensor<T>) -> rt::Result<FTensor<T>>
where
    for<'x> &'x FTensor<T>: TensorAddAPI<&'x FTensor<T>, Output = FTensor<T>>,
{
    rt::add_f(a, b)
}

fn op_sub<T>(a: &FTensor<T>, b: &FTensor<T>) -> rt::Result<FTensor<T>>
where
    for<'x> &'x FTensor<T>: TensorSubAPI<&'x FTensor<T>, Output = FTensor<T>>,
{
    rt::sub_f(a, b)
}

fn op_mul<T>(a: &FTensor<T>, b: &FTensor<T>) -> rt::Result<FTensor<T>>
where
    for<'x> &'x FTensor<T>: TensorMulAPI<&'x FTensor<T>, Output = FTensor<T>>,
{
    rt::mul_f(a, b)
}

fn op_div<T>(a: &FTensor<T>, b: &FTensor<T>) -> rt::Result<FTensor<T>>
where
    for<'x> &'x FTensor<T>: TensorDivAPI<&'x FTensor<T>, Output = FTensor<T>>,
{
    rt::div_f(a, b)
}

fn op_equal<T>(a: &FTensor<T>, b: &FTensor<T>) -> rt::Result<FTensor<bool>>
where
    for<'x> &'x FTensor<T>: TensorEqualAPI<&'x FTensor<T>, Output = FTensor<bool>>,
{
    rt::equal_f(a, b)
}

fn op_not_equal<T>(a: &FTensor<T>, b: &FTensor<T>) -> rt::Result<FTensor<bool>>
where
    for<'x> &'x FTensor<T>: TensorNotEqualAPI<&'x FTensor<T>, Output = FTensor<bool>>,
{
    rt::not_equal_f(a, b)
}

fn op_less<T>(a: &FTensor<T>, b: &FTensor<T>) -> rt::Result<FTensor<bool>>
where
    for<'x> &'x FTensor<T>: TensorLessAPI<&'x FTensor<T>, Output = FTensor<bool>>,
{
    rt::less_f(a, b)
}

fn op_less_equal<T>(a: &FTensor<T>, b: &FTensor<T>) -> rt::Result<FTensor<bool>>
where
    for<'x> &'x FTensor<T>: TensorLessEqualAPI<&'x FTensor<T>, Output = FTensor<bool>>,
{
    rt::less_equal_f(a, b)
}

fn op_greater<T>(a: &FTensor<T>, b: &FTensor<T>) -> rt::Result<FTensor<bool>>
where
    for<'x> &'x FTensor<T>: TensorGreaterAPI<&'x FTensor<T>, Output = FTensor<bool>>,
{
    rt::greater_f(a, b)
}

fn op_greater_equal<T>(a: &FTensor<T>, b: &FTensor<T>) -> rt::Result<FTensor<bool>>
where
    for<'x> &'x FTensor<T>: TensorGreaterEqualAPI<&'x FTensor<T>, Output = FTensor<bool>>,
{
    rt::greater_equal_f(a, b)
}

// ------------------------------------------------------------ arithmetic ----

#[pyfunction]
pub fn add(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray {
        t: dispatch_bin_numeric_self!(x1.t, x2.t, "add", op_add)?,
    })
}

#[pyfunction]
pub fn subtract(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray {
        t: dispatch_bin_numeric_self!(x1.t, x2.t, "subtract", op_sub)?,
    })
}

#[pyfunction]
pub fn multiply(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray {
        t: dispatch_bin_numeric_self!(x1.t, x2.t, "multiply", op_mul)?,
    })
}

#[pyfunction]
pub fn divide(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray {
        t: dispatch_bin_numeric_self!(x1.t, x2.t, "divide", op_div)?,
    })
}

/// negative: signed numeric dtypes only (bool/unsigned rejected; unsigned
/// wrap semantics are not defined by the standard).
#[pyfunction]
pub fn negative(x: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray {
        t: dispatch_t_signed!(x.t, op_neg())?,
    })
}

/// abs: numeric only; complex abs yields a REAL tensor (device TOut is the
/// component float), so the output variant changes for C32/C64.
#[pyfunction]
pub fn abs(x: &NativeArray) -> PyResult<NativeArray> {
    let t: AnyTensor = match &x.t {
        AnyTensor::Bool(_) => return type_err("abs: not defined for bool dtype"),
        AnyTensor::I8(t) => lift(rt::abs_f(t), AnyTensor::I8)?,
        AnyTensor::I16(t) => lift(rt::abs_f(t), AnyTensor::I16)?,
        AnyTensor::I32(t) => lift(rt::abs_f(t), AnyTensor::I32)?,
        AnyTensor::I64(t) => lift(rt::abs_f(t), AnyTensor::I64)?,
        AnyTensor::U8(t) => lift(rt::abs_f(t), AnyTensor::U8)?,
        AnyTensor::U16(t) => lift(rt::abs_f(t), AnyTensor::U16)?,
        AnyTensor::U32(t) => lift(rt::abs_f(t), AnyTensor::U32)?,
        AnyTensor::U64(t) => lift(rt::abs_f(t), AnyTensor::U64)?,
        AnyTensor::F32(t) => lift(rt::abs_f(t), AnyTensor::F32)?,
        AnyTensor::F64(t) => lift(rt::abs_f(t), AnyTensor::F64)?,
        AnyTensor::C32(t) => lift(rt::abs_f(t), AnyTensor::F32)?,
        AnyTensor::C64(t) => lift(rt::abs_f(t), AnyTensor::F64)?,
    };
    Ok(NativeArray { t })
}

// ------------------------------------------------------------ comparisons ---

#[pyfunction]
pub fn equal(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray {
        t: dispatch_bin!(x1.t, x2.t, "equal", op_equal, Bool)?,
    })
}

#[pyfunction]
pub fn not_equal(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray {
        t: dispatch_bin!(x1.t, x2.t, "not_equal", op_not_equal, Bool)?,
    })
}

/// Ordering comparisons reject complex dtypes (spec: ordering is real-valued
/// only; rstsr has no PartialOrd for complex); output is always bool.
macro_rules! dispatch_bin_real {
    ($a:expr, $b:expr, $opname:literal, $f:ident, $rv:ident) => {
        match (&$a, &$b) {
            (AnyTensor::C32(_), _) | (_, AnyTensor::C32(_)) | (AnyTensor::C64(_), _)
            | (_, AnyTensor::C64(_)) => type_err(format!(
                "{}: ordering comparison is not defined for complex dtypes",
                $opname
            )),
            (AnyTensor::Bool(a), AnyTensor::Bool(b)) => lift(($f::<bool>)(a, b), AnyTensor::$rv),
            (AnyTensor::I8(a), AnyTensor::I8(b)) => lift(($f::<i8>)(a, b), AnyTensor::$rv),
            (AnyTensor::I16(a), AnyTensor::I16(b)) => lift(($f::<i16>)(a, b), AnyTensor::$rv),
            (AnyTensor::I32(a), AnyTensor::I32(b)) => lift(($f::<i32>)(a, b), AnyTensor::$rv),
            (AnyTensor::I64(a), AnyTensor::I64(b)) => lift(($f::<i64>)(a, b), AnyTensor::$rv),
            (AnyTensor::U8(a), AnyTensor::U8(b)) => lift(($f::<u8>)(a, b), AnyTensor::$rv),
            (AnyTensor::U16(a), AnyTensor::U16(b)) => lift(($f::<u16>)(a, b), AnyTensor::$rv),
            (AnyTensor::U32(a), AnyTensor::U32(b)) => lift(($f::<u32>)(a, b), AnyTensor::$rv),
            (AnyTensor::U64(a), AnyTensor::U64(b)) => lift(($f::<u64>)(a, b), AnyTensor::$rv),
            (AnyTensor::F32(a), AnyTensor::F32(b)) => lift(($f::<f32>)(a, b), AnyTensor::$rv),
            (AnyTensor::F64(a), AnyTensor::F64(b)) => lift(($f::<f64>)(a, b), AnyTensor::$rv),
            _ => type_err(format!(
                "{}: mixed-dtype operands require type promotion (rstsr gap); use astype()",
                $opname
            )),
        }
    };
}

#[pyfunction]
pub fn less(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray {
        t: dispatch_bin_real!(x1.t, x2.t, "less", op_less, Bool)?,
    })
}

#[pyfunction]
pub fn less_equal(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray {
        t: dispatch_bin_real!(x1.t, x2.t, "less_equal", op_less_equal, Bool)?,
    })
}

#[pyfunction]
pub fn greater(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray {
        t: dispatch_bin_real!(x1.t, x2.t, "greater", op_greater, Bool)?,
    })
}

#[pyfunction]
pub fn greater_equal(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray {
        t: dispatch_bin_real!(x1.t, x2.t, "greater_equal", op_greater_equal, Bool)?,
    })
}

// --------------------------------------------------- predicates & logicals --

#[pyfunction]
pub fn all(x: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray {
        t: dispatch_t_bool!(x.t, op_all())?,
    })
}

#[pyfunction]
pub fn any(x: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray {
        t: dispatch_t_bool!(x.t, op_any())?,
    })
}

#[pyfunction]
pub fn isnan(x: &NativeArray) -> PyResult<NativeArray> {
    let t: AnyTensor = match &x.t {
        AnyTensor::F32(v) => lift(rt::is_nan_f(v), AnyTensor::Bool)?,
        AnyTensor::F64(v) => lift(rt::is_nan_f(v), AnyTensor::Bool)?,
        AnyTensor::C32(v) => lift(rt::is_nan_f(v), AnyTensor::Bool)?,
        AnyTensor::C64(v) => lift(rt::is_nan_f(v), AnyTensor::Bool)?,
        other => lift(const_bool(other.size(), false), AnyTensor::Bool)?,
    };
    Ok(NativeArray { t })
}

#[pyfunction]
pub fn isfinite(x: &NativeArray) -> PyResult<NativeArray> {
    let t: AnyTensor = match &x.t {
        AnyTensor::F32(v) => lift(rt::is_finite_f(v), AnyTensor::Bool)?,
        AnyTensor::F64(v) => lift(rt::is_finite_f(v), AnyTensor::Bool)?,
        AnyTensor::C32(v) => lift(rt::is_finite_f(v), AnyTensor::Bool)?,
        AnyTensor::C64(v) => lift(rt::is_finite_f(v), AnyTensor::Bool)?,
        other => lift(const_bool(other.size(), true), AnyTensor::Bool)?,
    };
    Ok(NativeArray { t })
}

#[pyfunction]
pub fn isinf(x: &NativeArray) -> PyResult<NativeArray> {
    let t: AnyTensor = match &x.t {
        AnyTensor::F32(v) => lift(rt::is_inf_f(v), AnyTensor::Bool)?,
        AnyTensor::F64(v) => lift(rt::is_inf_f(v), AnyTensor::Bool)?,
        AnyTensor::C32(v) => lift(rt::is_inf_f(v), AnyTensor::Bool)?,
        AnyTensor::C64(v) => lift(rt::is_inf_f(v), AnyTensor::Bool)?,
        other => lift(const_bool(other.size(), false), AnyTensor::Bool)?,
    };
    Ok(NativeArray { t })
}

// ------------------------------------------------------------ manipulation --

#[pyfunction]
pub fn reshape(x: &NativeArray, shape: Vec<isize>) -> PyResult<NativeArray> {
    Ok(NativeArray {
        t: dispatch_t!(x.t, op_reshape(shape.clone()))?,
    })
}

#[pyfunction]
pub fn transpose(x: &NativeArray, axes: Option<Vec<isize>>) -> PyResult<NativeArray> {
    Ok(NativeArray {
        t: dispatch_t!(x.t, op_transpose(axes.clone()))?,
    })
}

/// Integer index on axis 0, spec semantics (drops the axis). Copy, not a
/// view — aliasing/sharing is unimplemented (register G-036).
fn op_index<T>(t: &FTensor<T>, idx: usize) -> rt::Result<FTensor<T>>
where
    T: Clone + Send + Sync,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, IxD>,
{
    let view = t.i(Indexer::from(idx));
    Ok(view.into_owned())
}

#[pyfunction]
pub fn getitem_int(x: &NativeArray, idx: usize) -> PyResult<NativeArray> {
    Ok(NativeArray {
        t: dispatch_t!(x.t, op_index(idx))?,
    })
}

// Complex referenced by generated turbofish instantiations in macros.
#[allow(unused)]
fn _complex_used(_c: Complex<f64>) {}

/// Whole-array sum returned as a 0-d array (rstsr's `sum` yields the scalar).
fn op_sum<T>(t: &FTensor<T>) -> rt::Result<FTensor<T>>
where
    T: Clone + Send + Sync + core::ops::Add<Output = T> + num::Zero,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>>
        + DeviceCreationAnyAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>,
{
    let s: T = rt::sum_f(t)?;
    rt::asarray_f((vec![s], dim_from(&[]), device_faer()))
}

#[pyfunction]
pub fn sum(x: &NativeArray) -> PyResult<NativeArray> {
    macro_rules! arms {
        ($($dv:ident);* $(;)?) => {
            match &x.t {
                AnyTensor::Bool(_) => type_err("sum: bool dtype not supported yet (gap)"),
                $(AnyTensor::$dv(t) => Ok(NativeArray { t: AnyTensor::$dv(err_py(op_sum(t))?) }),)*
            }
        };
    }
    arms!(I8; I16; I32; I64; U8; U16; U32; U64; F32; F64; C32; C64)
}

fn op_broadcast_to<T>(t: &FTensor<T>, shape: Vec<usize>) -> rt::Result<FTensor<T>>
where
    T: Clone + Send + Sync,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T>,
{
    let v = rt::broadcast_to_f(t, shape)?;
    Ok(v.into_owned())
}

#[pyfunction]
pub fn broadcast_to(x: &NativeArray, shape: Vec<usize>) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_t!(x.t, op_broadcast_to(shape.clone()))? })
}


