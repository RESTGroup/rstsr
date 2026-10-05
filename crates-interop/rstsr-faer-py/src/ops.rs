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
use rstsr_dtype_traits::{DTypeIntoFloatAPI, DTypePromoteAPI};
use rstsr_core::tensor::operators::exports::{
    TensorAddAPI, TensorATan2API, TensorBitAndAPI, TensorBitOrAPI, TensorBitXorAPI, TensorCopySignAPI,
    TensorDivAPI, TensorEqualAPI, TensorFloorDivideAPI, TensorGreaterAPI, TensorGreaterEqualAPI,
    TensorHypotAPI, TensorLessAPI, TensorLessEqualAPI, TensorLogAddExpAPI, TensorMaximumAPI, TensorMinimumAPI,
    TensorMulAPI, TensorNegAPI, TensorNextAfterAPI, TensorNotEqualAPI, TensorPositiveAPI, TensorReciprocalAPI,
    TensorRemAPI, TensorShlAPI, TensorShrAPI, TensorSquareAPI, TensorSubAPI,
};

use crate::any_tensor::{
    device_faer, dispatch_bin_bool_self, dispatch_bin_int_bool_self, dispatch_bin_int_self, dispatch_bin_numeric_self,
    dispatch_bin_promote, dispatch_bin_promote_eq, dispatch_t, dispatch_t_bool, dispatch_t_float_complex_same, dispatch_t_into_float, dispatch_t_no_complex,
    dispatch_t_numeric_same, dispatch_t_real_float_same, dispatch_t_signed, err_py, lift, type_err, AnyTensor,
    FTensor, NativeArray,
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

fn op_equal<T, U>(a: &FTensor<T>, b: &FTensor<U>) -> rt::Result<FTensor<bool>>
where
    for<'x> &'x FTensor<T>: TensorEqualAPI<&'x FTensor<U>, Output = FTensor<bool>>,
{
    rt::equal_f(a, b)
}

fn op_not_equal<T, U>(a: &FTensor<T>, b: &FTensor<U>) -> rt::Result<FTensor<bool>>
where
    for<'x> &'x FTensor<T>: TensorNotEqualAPI<&'x FTensor<U>, Output = FTensor<bool>>,
{
    rt::not_equal_f(a, b)
}

fn op_less<T, U>(a: &FTensor<T>, b: &FTensor<U>) -> rt::Result<FTensor<bool>>
where
    for<'x> &'x FTensor<T>: TensorLessAPI<&'x FTensor<U>, Output = FTensor<bool>>,
{
    rt::less_f(a, b)
}

fn op_less_equal<T, U>(a: &FTensor<T>, b: &FTensor<U>) -> rt::Result<FTensor<bool>>
where
    for<'x> &'x FTensor<T>: TensorLessEqualAPI<&'x FTensor<U>, Output = FTensor<bool>>,
{
    rt::less_equal_f(a, b)
}

fn op_greater<T, U>(a: &FTensor<T>, b: &FTensor<U>) -> rt::Result<FTensor<bool>>
where
    for<'x> &'x FTensor<T>: TensorGreaterAPI<&'x FTensor<U>, Output = FTensor<bool>>,
{
    rt::greater_f(a, b)
}

fn op_greater_equal<T, U>(a: &FTensor<T>, b: &FTensor<U>) -> rt::Result<FTensor<bool>>
where
    for<'x> &'x FTensor<T>: TensorGreaterEqualAPI<&'x FTensor<U>, Output = FTensor<bool>>,
{
    rt::greater_equal_f(a, b)
}

// ------------------------------------------------- W2 elementwise surface ---
//
// Binding-only additions: every wrapper is a thin call into `rt::`; dtype
// policy lives in the dispatch macros of any_tensor.rs. Divergences that a
// binding cannot fix (integer inputs to the dtype-preserving rounding family,
// integer/complex kernels rstsr lacks, mixed-dtype arithmetic) are declined
// with a register reference instead of worked around.

/// Output-type shape is taken from rstsr's own dtype traits (`FloatType`,
/// promoted `Res`), so no dtype table is duplicated in the shim.
macro_rules! unary_wrapper {
    ($wrapper:ident, $rt:ident, $Trait:ident) => {
        fn $wrapper<T>(t: &FTensor<T>) -> rt::Result<FTensor<T::FloatType>>
        where
            T: DTypeIntoFloatAPI,
            for<'x> &'x FTensor<T>: $Trait<Output = FTensor<T::FloatType>>,
        {
            rt::$rt(t)
        }
    };
}

macro_rules! unary_wrapper_same {
    ($wrapper:ident, $rt:ident, $Trait:ident) => {
        fn $wrapper<T>(t: &FTensor<T>) -> rt::Result<FTensor<T>>
        where
            for<'x> &'x FTensor<T>: $Trait<Output = FTensor<T>>,
        {
            rt::$rt(t)
        }
    };
}

// transcendental family (bool rejected; integers -> float64)
unary_wrapper!(op_acos, acos_f, TensorAcosAPI);
unary_wrapper!(op_acosh, acosh_f, TensorAcoshAPI);
unary_wrapper!(op_asin, asin_f, TensorAsinAPI);
unary_wrapper!(op_asinh, asinh_f, TensorAsinhAPI);
unary_wrapper!(op_atan, atan_f, TensorAtanAPI);
unary_wrapper!(op_atanh, atanh_f, TensorAtanhAPI);
unary_wrapper!(op_cos, cos_f, TensorCosAPI);
unary_wrapper!(op_cosh, cosh_f, TensorCoshAPI);
unary_wrapper!(op_exp, exp_f, TensorExpAPI);
unary_wrapper!(op_log, log_f, TensorLogAPI);
unary_wrapper!(op_log2, log2_f, TensorLog2API);
unary_wrapper!(op_log10, log10_f, TensorLog10API);
unary_wrapper!(op_reciprocal, reciprocal_f, TensorReciprocalAPI);
unary_wrapper!(op_sin, sin_f, TensorSinAPI);
unary_wrapper!(op_sinh, sinh_f, TensorSinhAPI);
unary_wrapper!(op_sqrt, sqrt_f, TensorSqrtAPI);
unary_wrapper!(op_tan, tan_f, TensorTanAPI);
unary_wrapper!(op_tanh, tanh_f, TensorTanhAPI);
// real-only kernels
unary_wrapper!(op_expm1, expm1_f, TensorExpm1API);
unary_wrapper_same!(op_ceil, ceil_f, TensorCeilAPI);
unary_wrapper_same!(op_floor, floor_f, TensorFloorAPI);
unary_wrapper_same!(op_trunc, trunc_f, TensorTruncAPI);
unary_wrapper_same!(op_round, round_f, TensorRoundAPI);
// dtype-preserving numeric kernels
unary_wrapper_same!(op_square, square_f, TensorSquareAPI);
unary_wrapper_same!(op_sign, sign_f, TensorSignAPI);
unary_wrapper_same!(op_conj, conj_f, TensorConjAPI);

macro_rules! py_unary_into_float {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x: &NativeArray) -> PyResult<NativeArray> {
                Ok(NativeArray { t: dispatch_t_into_float!(x.t, $wrapper())? })
            }
        )*
    };
}

macro_rules! py_unary_no_complex {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x: &NativeArray) -> PyResult<NativeArray> {
                Ok(NativeArray { t: dispatch_t_no_complex!(x.t, $wrapper())? })
            }
        )*
    };
}

macro_rules! py_unary_real_float_same {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x: &NativeArray) -> PyResult<NativeArray> {
                Ok(NativeArray { t: dispatch_t_real_float_same!(x.t, stringify!($pyname), $wrapper())? })
            }
        )*
    };
}

macro_rules! py_unary_float_complex_same {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x: &NativeArray) -> PyResult<NativeArray> {
                Ok(NativeArray {
                    t: dispatch_t_float_complex_same!(x.t, stringify!($pyname), $wrapper())?,
                })
            }
        )*
    };
}

macro_rules! py_unary_numeric_same {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x: &NativeArray) -> PyResult<NativeArray> {
                Ok(NativeArray { t: dispatch_t_numeric_same!(x.t, stringify!($pyname), $wrapper())? })
            }
        )*
    };
}

py_unary_into_float!(
    acos => op_acos,
    acosh => op_acosh,
    asin => op_asin,
    asinh => op_asinh,
    atan => op_atan,
    atanh => op_atanh,
    cos => op_cos,
    cosh => op_cosh,
    exp => op_exp,
    log => op_log,
    log2 => op_log2,
    log10 => op_log10,
    reciprocal => op_reciprocal,
    sin => op_sin,
    sinh => op_sinh,
    sqrt => op_sqrt,
    tan => op_tan,
    tanh => op_tanh,
);
py_unary_no_complex!(expm1 => op_expm1);
py_unary_real_float_same!(
    ceil => op_ceil,
    floor => op_floor,
    trunc => op_trunc,
    round => op_round,
);
/// `positive`: identity function, routed through rstsr's `TensorPositiveAPI`
/// (rust-side trait added 2026-10-05; no operator trait bound on the dtype).
fn op_positive<T>(t: &FTensor<T>) -> rt::Result<FTensor<T>>
where
    for<'a> &'a FTensor<T>: TensorPositiveAPI<Output = FTensor<T>>,
{
    rt::positive_f(t)
}

py_unary_numeric_same!(
    square => op_square,
    sign => op_sign,
);
py_unary_numeric_same!(positive => op_positive);

py_unary_float_complex_same!(conj => op_conj);

/// `signbit` is present in rstsr but its kernel writes `is_positive()` —
/// the inverse of the standard's sign-bit test (verified: `signbit(-2.0)` is
/// False). The shim declines instead of returning wrong values; rust-side fix
/// requested (register G-054).
#[pyfunction]
pub fn signbit(_x: &NativeArray) -> PyResult<NativeArray> {
    type_err(
        "signbit: rstsr's kernel returns is_positive (inverted sign-bit semantics) — \
         rust-side fix required (register G-054)",
    )
}

#[pyfunction]
pub fn real(x: &NativeArray) -> PyResult<NativeArray> {
    let t: AnyTensor = match &x.t {
        AnyTensor::C32(v) => lift(rt::real_f(v), AnyTensor::F32)?,
        AnyTensor::C64(v) => lift(rt::real_f(v), AnyTensor::F64)?,
        _ => return type_err("real: only complex dtypes are allowed"),
    };
    Ok(NativeArray { t })
}

#[pyfunction]
pub fn imag(x: &NativeArray) -> PyResult<NativeArray> {
    let t: AnyTensor = match &x.t {
        AnyTensor::C32(v) => lift(rt::imag_f(v), AnyTensor::F32)?,
        AnyTensor::C64(v) => lift(rt::imag_f(v), AnyTensor::F64)?,
        _ => return type_err("imag: only complex dtypes are allowed"),
    };
    Ok(NativeArray { t })
}

/// `logical_not` (bool) and `bitwise_invert` (integer/bool): same dtype.
#[pyfunction]
pub fn invert(x: &NativeArray) -> PyResult<NativeArray> {
    let t: AnyTensor = match &x.t {
        AnyTensor::Bool(v) => lift(rt::not_f(v), AnyTensor::Bool)?,
        AnyTensor::I8(v) => lift(rt::not_f(v), AnyTensor::I8)?,
        AnyTensor::I16(v) => lift(rt::not_f(v), AnyTensor::I16)?,
        AnyTensor::I32(v) => lift(rt::not_f(v), AnyTensor::I32)?,
        AnyTensor::I64(v) => lift(rt::not_f(v), AnyTensor::I64)?,
        AnyTensor::U8(v) => lift(rt::not_f(v), AnyTensor::U8)?,
        AnyTensor::U16(v) => lift(rt::not_f(v), AnyTensor::U16)?,
        AnyTensor::U32(v) => lift(rt::not_f(v), AnyTensor::U32)?,
        AnyTensor::U64(v) => lift(rt::not_f(v), AnyTensor::U64)?,
        _ => return type_err("invert: only integer or boolean dtypes are allowed"),
    };
    Ok(NativeArray { t })
}

// --------------------------------------------------- binary (W2) ------------

/// Mixed-dtype wrappers for kernels whose promoted result keeps the promoted
/// dtype (`maximum`-family) ...
macro_rules! bin_promote_wrapper {
    ($wrapper:ident, $rt:ident, $Trait:ident) => {
        fn $wrapper<T, U>(a: &FTensor<T>, b: &FTensor<U>) -> rt::Result<FTensor<<T as DTypePromoteAPI<U>>::Res>>
        where
            T: DTypePromoteAPI<U>,
            for<'x> &'x FTensor<T>: $Trait<&'x FTensor<U>, Output = FTensor<<T as DTypePromoteAPI<U>>::Res>>,
        {
            rt::$rt(a, b)
        }
    };
}

/// ... and for kernels that promote first and then map to the float type
/// (`atan2`-family: `TOut = Res::FloatType`).
macro_rules! bin_promote_float_wrapper {
    ($wrapper:ident, $rt:ident, $Trait:ident) => {
        fn $wrapper<T, U>(a: &FTensor<T>, b: &FTensor<U>)
            -> rt::Result<FTensor<<<T as DTypePromoteAPI<U>>::Res as DTypeIntoFloatAPI>::FloatType>>
        where
            T: DTypePromoteAPI<U>,
            <T as DTypePromoteAPI<U>>::Res: DTypeIntoFloatAPI,
            for<'x> &'x FTensor<T>: $Trait<
                &'x FTensor<U>,
                Output = FTensor<<<T as DTypePromoteAPI<U>>::Res as DTypeIntoFloatAPI>::FloatType>,
            >,
        {
            rt::$rt(a, b)
        }
    };
}

bin_promote_wrapper!(op_maximum, maximum_f, TensorMaximumAPI);
bin_promote_wrapper!(op_minimum, minimum_f, TensorMinimumAPI);
bin_promote_wrapper!(op_floor_divide, floor_divide_f, TensorFloorDivideAPI);
bin_promote_float_wrapper!(op_atan2, atan2_f, TensorATan2API);
bin_promote_float_wrapper!(op_copysign, copysign_f, TensorCopySignAPI);
bin_promote_float_wrapper!(op_hypot, hypot_f, TensorHypotAPI);
bin_promote_float_wrapper!(op_nextafter, nextafter_f, TensorNextAfterAPI);
bin_promote_float_wrapper!(op_logaddexp, log_add_exp_f, TensorLogAddExpAPI);

macro_rules! py_bin_promote {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
                Ok(NativeArray {
                    t: dispatch_bin_promote!(x1.t, x2.t, stringify!($pyname), $wrapper)?,
                })
            }
        )*
    };
}

py_bin_promote!(
    maximum => op_maximum,
    minimum => op_minimum,
    floor_divide => op_floor_divide,
    atan2 => op_atan2,
    copysign => op_copysign,
    hypot => op_hypot,
    nextafter => op_nextafter,
    logaddexp => op_logaddexp,
);

/// Same-dtype integer/bitwise binary ops (`remainder`, `bitwise_*`, shifts).
macro_rules! bin_self_wrapper {
    ($wrapper:ident, $rt:ident, $Trait:ident) => {
        fn $wrapper<T>(a: &FTensor<T>, b: &FTensor<T>) -> rt::Result<FTensor<T>>
        where
            for<'x> &'x FTensor<T>: $Trait<&'x FTensor<T>, Output = FTensor<T>>,
        {
            rt::$rt(a, b)
        }
    };
}

bin_self_wrapper!(op_remainder, rem_f, TensorRemAPI);
bin_self_wrapper!(op_bitwise_and, bitand_f, TensorBitAndAPI);
bin_self_wrapper!(op_bitwise_or, bitor_f, TensorBitOrAPI);
bin_self_wrapper!(op_bitwise_xor, bitxor_f, TensorBitXorAPI);
bin_self_wrapper!(op_bitwise_left_shift, shl_f, TensorShlAPI);
bin_self_wrapper!(op_bitwise_right_shift, shr_f, TensorShrAPI);

macro_rules! py_bin_self {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
                Ok(NativeArray {
                    t: dispatch_bin_numeric_self!(x1.t, x2.t, stringify!($pyname), $wrapper)?,
                })
            }
        )*
    };
}

py_bin_self!(remainder => op_remainder);

macro_rules! py_bin_int_bool {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
                Ok(NativeArray {
                    t: dispatch_bin_int_bool_self!(x1.t, x2.t, stringify!($pyname), $wrapper)?,
                })
            }
        )*
    };
}

py_bin_int_bool!(
    bitwise_and => op_bitwise_and,
    bitwise_or => op_bitwise_or,
    bitwise_xor => op_bitwise_xor,
);

macro_rules! py_bin_int {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
                Ok(NativeArray {
                    t: dispatch_bin_int_self!(x1.t, x2.t, stringify!($pyname), $wrapper)?,
                })
            }
        )*
    };
}

py_bin_int!(
    bitwise_left_shift => op_bitwise_left_shift,
    bitwise_right_shift => op_bitwise_right_shift,
);

/// `pow`: same-dtype only, floats — rstsr's `Pow` bound needs an unsigned
/// exponent for integer bases and `num` provides no `Complex: Pow<Complex>`,
/// so integer and complex pow are rust-side gaps (registered G-053).
#[pyfunction]
pub fn pow(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    let t: AnyTensor = match (&x1.t, &x2.t) {
        (AnyTensor::F32(a), AnyTensor::F32(b)) => lift(rt::pow_f(a, b), AnyTensor::F32)?,
        (AnyTensor::F64(a), AnyTensor::F64(b)) => lift(rt::pow_f(a, b), AnyTensor::F64)?,
        (AnyTensor::F32(_) | AnyTensor::F64(_), _) | (_, AnyTensor::F32(_) | AnyTensor::F64(_)) => {
            return type_err("pow: mixed-dtype operands are not provided by rstsr (gap G-009)")
        }
        _ => return type_err("pow: integer/bool/complex bases are not provided by rstsr (gap G-053)"),
    };
    Ok(NativeArray { t })
}

/// `logical_and/or/xor`: boolean inputs only, boolean output.
macro_rules! py_logical {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
                Ok(NativeArray {
                    t: dispatch_bin_bool_self!(x1.t, x2.t, stringify!($pyname), $wrapper)?,
                })
            }
        )*
    };
}

py_logical!(
    logical_and => op_bitwise_and,
    logical_or => op_bitwise_or,
    logical_xor => op_bitwise_xor,
);

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
        t: dispatch_bin_promote_eq!(x1.t, x2.t, "equal", op_equal)?,
    })
}

#[pyfunction]
pub fn not_equal(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray {
        t: dispatch_bin_promote_eq!(x1.t, x2.t, "not_equal", op_not_equal)?,
    })
}

#[pyfunction]
pub fn less(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray {
        t: dispatch_bin_promote!(x1.t, x2.t, "less", op_less)?,
    })
}

#[pyfunction]
pub fn less_equal(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray {
        t: dispatch_bin_promote!(x1.t, x2.t, "less_equal", op_less_equal)?,
    })
}

#[pyfunction]
pub fn greater(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray {
        t: dispatch_bin_promote!(x1.t, x2.t, "greater", op_greater)?,
    })
}

#[pyfunction]
pub fn greater_equal(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray {
        t: dispatch_bin_promote!(x1.t, x2.t, "greater_equal", op_greater_equal)?,
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


