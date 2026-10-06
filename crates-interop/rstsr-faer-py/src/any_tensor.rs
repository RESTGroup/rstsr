//! Dtype-erased tensor handle: one enum over the 13 canonical dtypes on the
//! faer device, plus the dispatch machinery and Python marshalling helpers.
//!
//! rstsr carries dtype as a static type parameter with no runtime dtype
//! introspection, so this enum is the binding's single source of runtime
//! dtype identity.
//!
//! Dispatch: macro_rules cannot expand to multiple match arms, so every
//! macro here writes the whole `match` and calls a duplicated fn item or
//! closure per arm (one textual copy per dtype, monomorphized per arm):
//! - `for_each_item!` — item position (impls, fns), `;`-separated
//! - `dispatch_t!`   — closure over one tensor, arm tail `.map(vok)`
//! - `dispatch_fn!`  — generic fn item + turbofish + extra args
//! - `dispatch_name!`— dtype-name dispatch calling a generic fn item

use num::Complex;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyComplex, PyTuple};
use pyo3::Bound;
use pyo3::IntoPyObjectExt;
use rstsr::prelude::*;
use rstsr_common::error::RSTSRError;

/// Owned, dynamic-dimension tensor on the faer device.
pub type FTensor<T> = Tensor<T, DeviceFaer, IxD>;

/// The single faer device backing every tensor of this module.
pub fn device_faer() -> &'static DeviceFaer {
    static DEV: std::sync::OnceLock<DeviceFaer> = std::sync::OnceLock::new();
    DEV.get_or_init(DeviceFaer::default)
}

/// Canonical (variant, rust type, api name) table for the 13 dtypes.
/// Item-position repetition: `$mac!(Variant, type, "name", $($extra)*);` x 13.
macro_rules! for_each_item {
    ($mac:ident $(, $extra:expr)*) => {
        $mac!(Bool, bool, "bool" $(, $extra)*);
        $mac!(I8, i8, "int8" $(, $extra)*);
        $mac!(I16, i16, "int16" $(, $extra)*);
        $mac!(I32, i32, "int32" $(, $extra)*);
        $mac!(I64, i64, "int64" $(, $extra)*);
        $mac!(U8, u8, "uint8" $(, $extra)*);
        $mac!(U16, u16, "uint16" $(, $extra)*);
        $mac!(U32, u32, "uint32" $(, $extra)*);
        $mac!(U64, u64, "uint64" $(, $extra)*);
        $mac!(F32, f32, "float32" $(, $extra)*);
        $mac!(F64, f64, "float64" $(, $extra)*);
        $mac!(C32, Complex<f32>, "complex64" $(, $extra)*);
        $mac!(C64, Complex<f64>, "complex128" $(, $extra)*);
    };
}

#[derive(Clone)]
pub enum AnyTensor {
    Bool(FTensor<bool>),
    I8(FTensor<i8>),
    I16(FTensor<i16>),
    I32(FTensor<i32>),
    I64(FTensor<i64>),
    U8(FTensor<u8>),
    U16(FTensor<u16>),
    U32(FTensor<u32>),
    U64(FTensor<u64>),
    F32(FTensor<f32>),
    F64(FTensor<f64>),
    C32(FTensor<Complex<f32>>),
    C64(FTensor<Complex<f64>>),
}

/// Opaque tensor handle exposed to Python; all introspection happens
/// through its methods.
#[pyclass]
pub struct NativeArray {
    pub(crate) t: AnyTensor,
}

#[pymethods]
impl NativeArray {
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        PyTuple::new(py, self.t.shape())
    }

    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Py<crate::dtype::Dtype>> {
        crate::dtype_by_name(py, self.t.dtype_name())
    }

    fn ndim(&self) -> usize {
        self.t.ndim()
    }

    fn size(&self) -> usize {
        self.t.size()
    }

    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        self.t.tolist(py)
    }

    fn item<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        self.t.item(py)
    }

    fn copy(&self) -> NativeArray {
        NativeArray { t: self.t.deep_copy() }
    }
}

/// Per-dtype API name, type-driven.
pub trait DtypeName {
    const NAME: &'static str;
}

macro_rules! impl_traits {
    ($v:ident, $t:ty, $name:literal) => {
        impl DtypeName for $t {
            const NAME: &'static str = $name;
        }
    };
}
for_each_item!(impl_traits);

/// rstsr error -> Python exception, matched by rstsr's own error variant
/// (indexing errors surface as IndexError, everything else as ValueError;
/// specific call sites raise TypeError where the cause is an operand
/// mismatch). This keeps the wrapper layer free of re-validation.
pub fn err_py<T>(r: rt::Result<T>) -> PyResult<T> {
    r.map_err(|e| match &e.inner {
        RSTSRError::IndexError(_) | RSTSRError::AxisError { .. } => PyIndexError::new_err(format!("{e}")),
        _ => PyValueError::new_err(format!("{e}")),
    })
}

pub fn type_err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(PyTypeError::new_err(msg.into()))
}

/// Lift a typed tensor result into the erased enum via the arm's variant
/// constructor (the constructor makes R concrete at each instantiation).
pub(crate) fn lift<R>(r: rt::Result<FTensor<R>>, ctor: impl FnOnce(FTensor<R>) -> AnyTensor) -> PyResult<AnyTensor> {
    err_py(r).map(ctor)
}

/// Same for helpers that already produce `PyResult` (creation paths).
pub(crate) fn liftp<R>(r: PyResult<FTensor<R>>, ctor: impl FnOnce(FTensor<R>) -> AnyTensor) -> PyResult<AnyTensor> {
    r.map(ctor)
}

macro_rules! dispatch_t {
    ($scrut:expr, $f:ident ( $($arg:expr),* )) => {
        match &$scrut {
            AnyTensor::Bool(t) => lift(($f::<bool>)(&t, $($arg),*), AnyTensor::Bool),
            AnyTensor::I8(t) => lift(($f::<i8>)(&t, $($arg),*), AnyTensor::I8),
            AnyTensor::I16(t) => lift(($f::<i16>)(&t, $($arg),*), AnyTensor::I16),
            AnyTensor::I32(t) => lift(($f::<i32>)(&t, $($arg),*), AnyTensor::I32),
            AnyTensor::I64(t) => lift(($f::<i64>)(&t, $($arg),*), AnyTensor::I64),
            AnyTensor::U8(t) => lift(($f::<u8>)(&t, $($arg),*), AnyTensor::U8),
            AnyTensor::U16(t) => lift(($f::<u16>)(&t, $($arg),*), AnyTensor::U16),
            AnyTensor::U32(t) => lift(($f::<u32>)(&t, $($arg),*), AnyTensor::U32),
            AnyTensor::U64(t) => lift(($f::<u64>)(&t, $($arg),*), AnyTensor::U64),
            AnyTensor::F32(t) => lift(($f::<f32>)(&t, $($arg),*), AnyTensor::F32),
            AnyTensor::F64(t) => lift(($f::<f64>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::C32(t) => lift(($f::<Complex<f32>>)(&t, $($arg),*), AnyTensor::C32),
            AnyTensor::C64(t) => lift(($f::<Complex<f64>>)(&t, $($arg),*), AnyTensor::C64),
        }
    };
}
pub(crate) use dispatch_t;

/// Dispatch where every arm's fn returns `FTensor<bool>` (predicates,
/// whole-array reductions); ctor is always Bool.
macro_rules! dispatch_t_bool {
    ($scrut:expr, $f:ident ( $($arg:expr),* )) => {
        match &$scrut {
            AnyTensor::Bool(t) => lift(($f::<bool>)(&t, $($arg),*), AnyTensor::Bool),
            AnyTensor::I8(t) => lift(($f::<i8>)(&t, $($arg),*), AnyTensor::Bool),
            AnyTensor::I16(t) => lift(($f::<i16>)(&t, $($arg),*), AnyTensor::Bool),
            AnyTensor::I32(t) => lift(($f::<i32>)(&t, $($arg),*), AnyTensor::Bool),
            AnyTensor::I64(t) => lift(($f::<i64>)(&t, $($arg),*), AnyTensor::Bool),
            AnyTensor::U8(t) => lift(($f::<u8>)(&t, $($arg),*), AnyTensor::Bool),
            AnyTensor::U16(t) => lift(($f::<u16>)(&t, $($arg),*), AnyTensor::Bool),
            AnyTensor::U32(t) => lift(($f::<u32>)(&t, $($arg),*), AnyTensor::Bool),
            AnyTensor::U64(t) => lift(($f::<u64>)(&t, $($arg),*), AnyTensor::Bool),
            AnyTensor::F32(t) => lift(($f::<f32>)(&t, $($arg),*), AnyTensor::Bool),
            AnyTensor::F64(t) => lift(($f::<f64>)(&t, $($arg),*), AnyTensor::Bool),
            AnyTensor::C32(t) => lift(($f::<Complex<f32>>)(&t, $($arg),*), AnyTensor::Bool),
            AnyTensor::C64(t) => lift(($f::<Complex<f64>>)(&t, $($arg),*), AnyTensor::Bool),
        }
    };
}
pub(crate) use dispatch_t_bool;

/// Dispatch over signed numeric dtypes only (bool/unsigned rejected —
/// `Neg` has no unsigned impls; unsigned negative semantics are not
/// defined by the standard).
macro_rules! dispatch_t_signed {
    ($scrut:expr, $f:ident ( $($arg:expr),* )) => {
        match &$scrut {
            AnyTensor::Bool(_) | AnyTensor::U8(_) | AnyTensor::U16(_) | AnyTensor::U32(_)
            | AnyTensor::U64(_) => type_err("negative: not defined for bool/unsigned dtypes"),
            AnyTensor::I8(t) => lift(($f::<i8>)(&t, $($arg),*), AnyTensor::I8),
            AnyTensor::I16(t) => lift(($f::<i16>)(&t, $($arg),*), AnyTensor::I16),
            AnyTensor::I32(t) => lift(($f::<i32>)(&t, $($arg),*), AnyTensor::I32),
            AnyTensor::I64(t) => lift(($f::<i64>)(&t, $($arg),*), AnyTensor::I64),
            AnyTensor::F32(t) => lift(($f::<f32>)(&t, $($arg),*), AnyTensor::F32),
            AnyTensor::F64(t) => lift(($f::<f64>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::C32(t) => lift(($f::<Complex<f32>>)(&t, $($arg),*), AnyTensor::C32),
            AnyTensor::C64(t) => lift(($f::<Complex<f64>>)(&t, $($arg),*), AnyTensor::C64),
        }
    };
}
pub(crate) use dispatch_t_signed;

macro_rules! dispatch_fn {
    ($scrut:expr, $f:ident ( $($arg:expr),* )) => {
        match &$scrut {
            AnyTensor::Bool(t) => ($f::<bool>)(&t, $($arg),*),
            AnyTensor::I8(t) => ($f::<i8>)(&t, $($arg),*),
            AnyTensor::I16(t) => ($f::<i16>)(&t, $($arg),*),
            AnyTensor::I32(t) => ($f::<i32>)(&t, $($arg),*),
            AnyTensor::I64(t) => ($f::<i64>)(&t, $($arg),*),
            AnyTensor::U8(t) => ($f::<u8>)(&t, $($arg),*),
            AnyTensor::U16(t) => ($f::<u16>)(&t, $($arg),*),
            AnyTensor::U32(t) => ($f::<u32>)(&t, $($arg),*),
            AnyTensor::U64(t) => ($f::<u64>)(&t, $($arg),*),
            AnyTensor::F32(t) => ($f::<f32>)(&t, $($arg),*),
            AnyTensor::F64(t) => ($f::<f64>)(&t, $($arg),*),
            AnyTensor::C32(t) => ($f::<Complex<f32>>)(&t, $($arg),*),
            AnyTensor::C64(t) => ($f::<Complex<f64>>)(&t, $($arg),*),
        }
    };
}

/// Numeric same-dtype binary dispatch where the result keeps the input dtype;
/// bool is rejected (rstsr provides no bool arithmetic, matching the standard).
macro_rules! dispatch_bin_numeric_self {
    ($a:expr, $b:expr, $opname:expr, $f:ident) => {
        match (&$a, &$b) {
            (AnyTensor::Bool(_), _) | (_, AnyTensor::Bool(_)) => {
                type_err(format!("{}: not defined for bool dtype", $opname))
            },
            (AnyTensor::I8(a), AnyTensor::I8(b)) => lift(($f::<i8>)(a, b), AnyTensor::I8),
            (AnyTensor::I16(a), AnyTensor::I16(b)) => lift(($f::<i16>)(a, b), AnyTensor::I16),
            (AnyTensor::I32(a), AnyTensor::I32(b)) => lift(($f::<i32>)(a, b), AnyTensor::I32),
            (AnyTensor::I64(a), AnyTensor::I64(b)) => lift(($f::<i64>)(a, b), AnyTensor::I64),
            (AnyTensor::U8(a), AnyTensor::U8(b)) => lift(($f::<u8>)(a, b), AnyTensor::U8),
            (AnyTensor::U16(a), AnyTensor::U16(b)) => lift(($f::<u16>)(a, b), AnyTensor::U16),
            (AnyTensor::U32(a), AnyTensor::U32(b)) => lift(($f::<u32>)(a, b), AnyTensor::U32),
            (AnyTensor::U64(a), AnyTensor::U64(b)) => lift(($f::<u64>)(a, b), AnyTensor::U64),
            (AnyTensor::F32(a), AnyTensor::F32(b)) => lift(($f::<f32>)(a, b), AnyTensor::F32),
            (AnyTensor::F64(a), AnyTensor::F64(b)) => lift(($f::<f64>)(a, b), AnyTensor::F64),
            (AnyTensor::C32(a), AnyTensor::C32(b)) => lift(($f::<Complex<f32>>)(a, b), AnyTensor::C32),
            (AnyTensor::C64(a), AnyTensor::C64(b)) => lift(($f::<Complex<f64>>)(a, b), AnyTensor::C64),
            _ => {
                type_err(format!("{}: mixed-dtype operands require type promotion (rstsr gap); use astype()", $opname))
            },
        }
    };
}
pub(crate) use dispatch_bin_numeric_self;

macro_rules! dispatch_name {
    ($name:expr, $f:ident ( $($arg:expr),* )) => {
        match $name {
            "bool" => liftp(($f::<bool>)($($arg),*), AnyTensor::Bool),
            "int8" => liftp(($f::<i8>)($($arg),*), AnyTensor::I8),
            "int16" => liftp(($f::<i16>)($($arg),*), AnyTensor::I16),
            "int32" => liftp(($f::<i32>)($($arg),*), AnyTensor::I32),
            "int64" => liftp(($f::<i64>)($($arg),*), AnyTensor::I64),
            "uint8" => liftp(($f::<u8>)($($arg),*), AnyTensor::U8),
            "uint16" => liftp(($f::<u16>)($($arg),*), AnyTensor::U16),
            "uint32" => liftp(($f::<u32>)($($arg),*), AnyTensor::U32),
            "uint64" => liftp(($f::<u64>)($($arg),*), AnyTensor::U64),
            "float32" => liftp(($f::<f32>)($($arg),*), AnyTensor::F32),
            "float64" => liftp(($f::<f64>)($($arg),*), AnyTensor::F64),
            "complex64" => liftp(($f::<Complex<f32>>)($($arg),*), AnyTensor::C32),
            "complex128" => liftp(($f::<Complex<f64>>)($($arg),*), AnyTensor::C64),
            _ => type_err(format!("unknown dtype {:?}", $name)),
        }
    };
}
pub(crate) use dispatch_name;

/// Like `dispatch_name!` but without the bool arm — for fn items whose
/// bounds exclude bool (e.g. `num::Num`-gated creation); bool falls to the
/// error arm so guarded call sites can pre-route bool themselves.
macro_rules! dispatch_name_numeric {
    ($name:expr, $f:ident ( $($arg:expr),* )) => {
        match $name {
            "int8" => liftp(($f::<i8>)($($arg),*), AnyTensor::I8),
            "int16" => liftp(($f::<i16>)($($arg),*), AnyTensor::I16),
            "int32" => liftp(($f::<i32>)($($arg),*), AnyTensor::I32),
            "int64" => liftp(($f::<i64>)($($arg),*), AnyTensor::I64),
            "uint8" => liftp(($f::<u8>)($($arg),*), AnyTensor::U8),
            "uint16" => liftp(($f::<u16>)($($arg),*), AnyTensor::U16),
            "uint32" => liftp(($f::<u32>)($($arg),*), AnyTensor::U32),
            "uint64" => liftp(($f::<u64>)($($arg),*), AnyTensor::U64),
            "float32" => liftp(($f::<f32>)($($arg),*), AnyTensor::F32),
            "float64" => liftp(($f::<f64>)($($arg),*), AnyTensor::F64),
            "complex64" => liftp(($f::<Complex<f32>>)($($arg),*), AnyTensor::C32),
            "complex128" => liftp(($f::<Complex<f64>>)($($arg),*), AnyTensor::C64),
            _ => type_err(format!("dtype {:?} is not valid here", $name)),
        }
    };
}
pub(crate) use dispatch_name_numeric;

/// Real-dtypes-only name dispatch (ints + floats; no bool, no complex).
macro_rules! dispatch_name_real {
    ($name:expr, $f:ident ( $($arg:expr),* )) => {
        match $name {
            "int8" => liftp(($f::<i8>)($($arg),*), AnyTensor::I8),
            "int16" => liftp(($f::<i16>)($($arg),*), AnyTensor::I16),
            "int32" => liftp(($f::<i32>)($($arg),*), AnyTensor::I32),
            "int64" => liftp(($f::<i64>)($($arg),*), AnyTensor::I64),
            "uint8" => liftp(($f::<u8>)($($arg),*), AnyTensor::U8),
            "uint16" => liftp(($f::<u16>)($($arg),*), AnyTensor::U16),
            "uint32" => liftp(($f::<u32>)($($arg),*), AnyTensor::U32),
            "uint64" => liftp(($f::<u64>)($($arg),*), AnyTensor::U64),
            "float32" => liftp(($f::<f32>)($($arg),*), AnyTensor::F32),
            "float64" => liftp(($f::<f64>)($($arg),*), AnyTensor::F64),
            _ => type_err(format!("dtype {:?} is not valid here", $name)),
        }
    };
}
pub(crate) use dispatch_name_real;

// ------------------------------------------- W2: dtype-changing dispatch ----

/// Lifts a typed tensor result into the erased enum by its *own* Rust type,
/// so an op whose output dtype differs from its input (e.g. `exp(int) ->
/// float64`, `abs(complex) -> real`) still lands in the right variant.
pub trait IntoAnyTensor {
    fn into_any(self) -> AnyTensor;
}

/// Free-function form of `IntoAnyTensor::into_any` for use as a `lift`
/// constructor: a closure would need its parameter type inferred before
/// method lookup, which fails for macro-generated arms.
pub(crate) fn any_of<T: IntoAnyTensor>(t: T) -> AnyTensor {
    t.into_any()
}

macro_rules! impl_into_any {
    ($v:ident, $t:ty, $name:literal) => {
        impl IntoAnyTensor for FTensor<$t> {
            fn into_any(self) -> AnyTensor {
                AnyTensor::$v(self)
            }
        }
    };
}
for_each_item!(impl_into_any);

/// Unary dispatch for the transcendental family: bool rejected, integers
/// promote to float64 (`T::FloatType`), floats/complex keep their dtype.
macro_rules! dispatch_t_into_float {
    ($scrut:expr, $f:ident ( $($arg:expr),* )) => {
        match &$scrut {
            AnyTensor::Bool(_) => type_err("unary op: not defined for bool dtype"),
            AnyTensor::I8(t) => lift(($f::<i8>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::I16(t) => lift(($f::<i16>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::I32(t) => lift(($f::<i32>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::I64(t) => lift(($f::<i64>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::U8(t) => lift(($f::<u8>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::U16(t) => lift(($f::<u16>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::U32(t) => lift(($f::<u32>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::U64(t) => lift(($f::<u64>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::F32(t) => lift(($f::<f32>)(&t, $($arg),*), AnyTensor::F32),
            AnyTensor::F64(t) => lift(($f::<f64>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::C32(t) => lift(($f::<Complex<f32>>)(&t, $($arg),*), AnyTensor::C32),
            AnyTensor::C64(t) => lift(($f::<Complex<f64>>)(&t, $($arg),*), AnyTensor::C64),
        }
    };
}
pub(crate) use dispatch_t_into_float;

/// Unary dispatch for kernels with a real-only `Float` bound (no complex
/// kernel in rstsr); integers promote to float64 as in `dispatch_t_into_float`.
macro_rules! dispatch_t_no_complex {
    ($scrut:expr, $f:ident ( $($arg:expr),* )) => {
        match &$scrut {
            AnyTensor::Bool(_) => type_err("unary op: not defined for bool dtype"),
            AnyTensor::I8(t) => lift(($f::<i8>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::I16(t) => lift(($f::<i16>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::I32(t) => lift(($f::<i32>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::I64(t) => lift(($f::<i64>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::U8(t) => lift(($f::<u8>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::U16(t) => lift(($f::<u16>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::U32(t) => lift(($f::<u32>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::U64(t) => lift(($f::<u64>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::F32(t) => lift(($f::<f32>)(&t, $($arg),*), AnyTensor::F32),
            AnyTensor::F64(t) => lift(($f::<f64>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::C32(_) | AnyTensor::C64(_) => {
                type_err("unary op: complex inputs are not provided by rstsr (gap)")
            }
        }
    };
}
pub(crate) use dispatch_t_no_complex;

/// Unary dispatch whose spec output dtype equals the input dtype, restricted
/// to real floating dtypes (`f32`/`f64`): bool, integer and complex inputs
/// are declined. Integer inputs are declined because rstsr's kernels promote
/// them to float64 while the spec requires dtype preservation (register
/// G-052: dtype-preserving integer kernels).
macro_rules! dispatch_t_real_float_same {
    ($scrut:expr, $opname:expr, $f:ident ( $($arg:expr),* )) => {
        match &$scrut {
            AnyTensor::Bool(_) => type_err(format!(
                "{}: not defined for bool dtype",
                $opname
            )),
            AnyTensor::I8(_) | AnyTensor::I16(_) | AnyTensor::I32(_) | AnyTensor::I64(_)
            | AnyTensor::U8(_) | AnyTensor::U16(_) | AnyTensor::U32(_) | AnyTensor::U64(_) => type_err(
                format!(
                    "{}: integer inputs are not provided by rstsr (the kernel promotes to float64; \
                     the spec requires dtype preservation) — register G-052",
                    $opname
                ),
            ),
            AnyTensor::C32(_) | AnyTensor::C64(_) => type_err(format!(
                "{}: complex inputs are not provided by rstsr (gap)",
                $opname
            )),
            AnyTensor::F32(t) => lift(($f::<f32>)(&t, $($arg),*), AnyTensor::F32),
            AnyTensor::F64(t) => lift(($f::<f64>)(&t, $($arg),*), AnyTensor::F64),
        }
    };
}
pub(crate) use dispatch_t_real_float_same;

/// Unary dispatch over every numeric dtype but bool, output dtype == input
/// dtype (rstsr `ExtNum` kernels: sign, conj).
macro_rules! dispatch_t_numeric_same {
    ($scrut:expr, $opname:expr, $f:ident ( $($arg:expr),* )) => {
        match &$scrut {
            AnyTensor::Bool(_) => type_err(format!(
                "{}: not defined for bool dtype",
                $opname
            )),
            AnyTensor::I8(t) => lift(($f::<i8>)(&t, $($arg),*), AnyTensor::I8),
            AnyTensor::I16(t) => lift(($f::<i16>)(&t, $($arg),*), AnyTensor::I16),
            AnyTensor::I32(t) => lift(($f::<i32>)(&t, $($arg),*), AnyTensor::I32),
            AnyTensor::I64(t) => lift(($f::<i64>)(&t, $($arg),*), AnyTensor::I64),
            AnyTensor::U8(t) => lift(($f::<u8>)(&t, $($arg),*), AnyTensor::U8),
            AnyTensor::U16(t) => lift(($f::<u16>)(&t, $($arg),*), AnyTensor::U16),
            AnyTensor::U32(t) => lift(($f::<u32>)(&t, $($arg),*), AnyTensor::U32),
            AnyTensor::U64(t) => lift(($f::<u64>)(&t, $($arg),*), AnyTensor::U64),
            AnyTensor::F32(t) => lift(($f::<f32>)(&t, $($arg),*), AnyTensor::F32),
            AnyTensor::F64(t) => lift(($f::<f64>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::C32(t) => lift(($f::<Complex<f32>>)(&t, $($arg),*), AnyTensor::C32),
            AnyTensor::C64(t) => lift(($f::<Complex<f64>>)(&t, $($arg),*), AnyTensor::C64),
        }
    };
}
pub(crate) use dispatch_t_numeric_same;

/// Unary dispatch whose spec output dtype equals the input, restricted to
/// float/complex dtypes: rstsr's `conj` maps integers through the
/// into-float block (`FloatType`), which the spec forbids, so integer inputs
/// are declined (register G-052).
macro_rules! dispatch_t_float_complex_same {
    ($scrut:expr, $opname:expr, $f:ident ( $($arg:expr),* )) => {
        match &$scrut {
            AnyTensor::F32(t) => lift(($f::<f32>)(&t, $($arg),*), AnyTensor::F32),
            AnyTensor::F64(t) => lift(($f::<f64>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::C32(t) => lift(($f::<Complex<f32>>)(&t, $($arg),*), AnyTensor::C32),
            AnyTensor::C64(t) => lift(($f::<Complex<f64>>)(&t, $($arg),*), AnyTensor::C64),
            _ => type_err(format!(
                "{}: only float/complex dtypes are provided by rstsr (integer inputs promote to \
                 float64; the spec requires dtype preservation) — register G-052",
                $opname
            )),
        }
    };
}
pub(crate) use dispatch_t_float_complex_same;
/// Unary dispatch over real numeric dtypes only (ints + floats; no bool, no
/// complex), output dtype == input dtype (`ExtReal`-bound reductions: max,
/// min — inequality comparison of complex numbers is unspecified in the
/// standard and unimplemented in rstsr).
macro_rules! dispatch_t_real_numeric_same {
    ($scrut:expr, $opname:expr, $f:ident ( $($arg:expr),* )) => {
        match &$scrut {
            AnyTensor::Bool(_) => type_err(format!(
                "{}: not defined for bool dtype",
                $opname
            )),
            AnyTensor::I8(t) => lift(($f::<i8>)(&t, $($arg),*), AnyTensor::I8),
            AnyTensor::I16(t) => lift(($f::<i16>)(&t, $($arg),*), AnyTensor::I16),
            AnyTensor::I32(t) => lift(($f::<i32>)(&t, $($arg),*), AnyTensor::I32),
            AnyTensor::I64(t) => lift(($f::<i64>)(&t, $($arg),*), AnyTensor::I64),
            AnyTensor::U8(t) => lift(($f::<u8>)(&t, $($arg),*), AnyTensor::U8),
            AnyTensor::U16(t) => lift(($f::<u16>)(&t, $($arg),*), AnyTensor::U16),
            AnyTensor::U32(t) => lift(($f::<u32>)(&t, $($arg),*), AnyTensor::U32),
            AnyTensor::U64(t) => lift(($f::<u64>)(&t, $($arg),*), AnyTensor::U64),
            AnyTensor::F32(t) => lift(($f::<f32>)(&t, $($arg),*), AnyTensor::F32),
            AnyTensor::F64(t) => lift(($f::<f64>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::C32(_) | AnyTensor::C64(_) => type_err(format!(
                "{}: complex inputs are not provided by rstsr (gap)",
                $opname
            )),
        }
    };
}
pub(crate) use dispatch_t_real_numeric_same;

/// Binary dispatch for ops whose rstsr device kernel promotes mixed dtypes
/// (`DTypePromoteAPI` bound; real dtypes only — bool and complex are out of
/// the spec contract for these ops). Arms mirror the promotion table in
/// `rstsr-dtype-traits/src/promotion.rs`; the result variant is derived from
/// the promoted type, so no promotion table is duplicated here.
macro_rules! dispatch_bin_promote {
    ($a:expr, $b:expr, $opname:expr, $f:ident) => {
        match (&$a, &$b) {
            (AnyTensor::I8(a), AnyTensor::I8(b)) => lift(($f::<i8, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::I16(b)) => lift(($f::<i16, i16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::I32(b)) => lift(($f::<i32, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::I64(b)) => lift(($f::<i64, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::U8(b)) => lift(($f::<u8, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::U16(b)) => lift(($f::<u16, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::U32(b)) => lift(($f::<u32, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::U64(b)) => lift(($f::<u64, u64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::F32(b)) => lift(($f::<f32, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::F64(b)) => lift(($f::<f64, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::F64(b)) => lift(($f::<f32, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::I16(b)) => lift(($f::<f32, i16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::I32(b)) => lift(($f::<f32, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::I64(b)) => lift(($f::<f32, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::I8(b)) => lift(($f::<f32, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::U16(b)) => lift(($f::<f32, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::U32(b)) => lift(($f::<f32, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::U64(b)) => lift(($f::<f32, u64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::U8(b)) => lift(($f::<f32, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::F32(b)) => lift(($f::<f64, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::I16(b)) => lift(($f::<f64, i16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::I32(b)) => lift(($f::<f64, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::I64(b)) => lift(($f::<f64, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::I8(b)) => lift(($f::<f64, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::U16(b)) => lift(($f::<f64, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::U32(b)) => lift(($f::<f64, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::U64(b)) => lift(($f::<f64, u64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::U8(b)) => lift(($f::<f64, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::F32(b)) => lift(($f::<i16, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::F64(b)) => lift(($f::<i16, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::I32(b)) => lift(($f::<i16, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::I64(b)) => lift(($f::<i16, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::I8(b)) => lift(($f::<i16, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::U16(b)) => lift(($f::<i16, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::U32(b)) => lift(($f::<i16, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::U64(b)) => lift(($f::<i16, u64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::U8(b)) => lift(($f::<i16, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::F32(b)) => lift(($f::<i32, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::F64(b)) => lift(($f::<i32, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::I16(b)) => lift(($f::<i32, i16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::I64(b)) => lift(($f::<i32, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::I8(b)) => lift(($f::<i32, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::U16(b)) => lift(($f::<i32, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::U32(b)) => lift(($f::<i32, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::U64(b)) => lift(($f::<i32, u64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::U8(b)) => lift(($f::<i32, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::F32(b)) => lift(($f::<i64, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::F64(b)) => lift(($f::<i64, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::I16(b)) => lift(($f::<i64, i16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::I32(b)) => lift(($f::<i64, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::I8(b)) => lift(($f::<i64, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::U16(b)) => lift(($f::<i64, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::U32(b)) => lift(($f::<i64, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::U64(b)) => lift(($f::<i64, u64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::U8(b)) => lift(($f::<i64, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I8(a), AnyTensor::F32(b)) => lift(($f::<i8, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I8(a), AnyTensor::F64(b)) => lift(($f::<i8, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I8(a), AnyTensor::I32(b)) => lift(($f::<i8, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I8(a), AnyTensor::I64(b)) => lift(($f::<i8, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I8(a), AnyTensor::U16(b)) => lift(($f::<i8, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I8(a), AnyTensor::U32(b)) => lift(($f::<i8, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I8(a), AnyTensor::U64(b)) => lift(($f::<i8, u64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I8(a), AnyTensor::U8(b)) => lift(($f::<i8, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::F32(b)) => lift(($f::<u16, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::F64(b)) => lift(($f::<u16, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::I16(b)) => lift(($f::<u16, i16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::I32(b)) => lift(($f::<u16, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::I64(b)) => lift(($f::<u16, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::I8(b)) => lift(($f::<u16, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::U32(b)) => lift(($f::<u16, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::U64(b)) => lift(($f::<u16, u64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::U8(b)) => lift(($f::<u16, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::F32(b)) => lift(($f::<u32, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::F64(b)) => lift(($f::<u32, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::I16(b)) => lift(($f::<u32, i16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::I32(b)) => lift(($f::<u32, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::I64(b)) => lift(($f::<u32, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::I8(b)) => lift(($f::<u32, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::U16(b)) => lift(($f::<u32, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::U64(b)) => lift(($f::<u32, u64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::U8(b)) => lift(($f::<u32, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::F32(b)) => lift(($f::<u64, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::F64(b)) => lift(($f::<u64, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::I16(b)) => lift(($f::<u64, i16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::I32(b)) => lift(($f::<u64, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::I64(b)) => lift(($f::<u64, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::I8(b)) => lift(($f::<u64, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::U16(b)) => lift(($f::<u64, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::U32(b)) => lift(($f::<u64, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::U8(b)) => lift(($f::<u64, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::F32(b)) => lift(($f::<u8, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::F64(b)) => lift(($f::<u8, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::I16(b)) => lift(($f::<u8, i16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::I32(b)) => lift(($f::<u8, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::I64(b)) => lift(($f::<u8, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::I8(b)) => lift(($f::<u8, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::U16(b)) => lift(($f::<u8, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::U32(b)) => lift(($f::<u8, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::U64(b)) => lift(($f::<u8, u64>)(a, b), crate::any_tensor::any_of),
            // 89 arms

            // 89 pair arms, generated from the DTypePromoteAPI impls of
            // rstsr-dtype-traits/src/promotion.rs (real dtypes only).
            _ => type_err(format!(
                "{}: this dtype pair is not promoted by rstsr (gap G-009); use astype() or matching dtypes",
                $opname
            )),
        }
    };
}
pub(crate) use dispatch_bin_promote;

/// Same-dtype binary dispatch over integer and boolean dtypes only (bitwise
/// family; the spec excludes floats and complexes).
macro_rules! dispatch_bin_int_bool_self {
    ($a:expr, $b:expr, $opname:expr, $f:ident) => {
        match (&$a, &$b) {
            (AnyTensor::Bool(a), AnyTensor::Bool(b)) => lift(($f)(a, b), AnyTensor::Bool),
            (AnyTensor::I8(a), AnyTensor::I8(b)) => lift(($f)(a, b), AnyTensor::I8),
            (AnyTensor::I16(a), AnyTensor::I16(b)) => lift(($f)(a, b), AnyTensor::I16),
            (AnyTensor::I32(a), AnyTensor::I32(b)) => lift(($f)(a, b), AnyTensor::I32),
            (AnyTensor::I64(a), AnyTensor::I64(b)) => lift(($f)(a, b), AnyTensor::I64),
            (AnyTensor::U8(a), AnyTensor::U8(b)) => lift(($f)(a, b), AnyTensor::U8),
            (AnyTensor::U16(a), AnyTensor::U16(b)) => lift(($f)(a, b), AnyTensor::U16),
            (AnyTensor::U32(a), AnyTensor::U32(b)) => lift(($f)(a, b), AnyTensor::U32),
            (AnyTensor::U64(a), AnyTensor::U64(b)) => lift(($f)(a, b), AnyTensor::U64),
            (AnyTensor::F32(_) | AnyTensor::F64(_) | AnyTensor::C32(_) | AnyTensor::C64(_), _)
            | (_, AnyTensor::F32(_) | AnyTensor::F64(_) | AnyTensor::C32(_) | AnyTensor::C64(_)) => {
                type_err(format!("{}: only integer or boolean dtypes are allowed", $opname))
            },
            _ => type_err(format!(
                "{}: mixed-dtype operands require type promotion (rstsr gap G-009); use astype()",
                $opname
            )),
        }
    };
}
pub(crate) use dispatch_bin_int_bool_self;

/// Same-dtype binary dispatch over integer dtypes only (shift family: the
/// spec allows integers only, and `Shl`/`Shr` are undefined for bool).
macro_rules! dispatch_bin_int_self {
    ($a:expr, $b:expr, $opname:expr, $f:ident) => {
        match (&$a, &$b) {
            (AnyTensor::I8(a), AnyTensor::I8(b)) => lift(($f::<i8>)(a, b), AnyTensor::I8),
            (AnyTensor::I16(a), AnyTensor::I16(b)) => lift(($f::<i16>)(a, b), AnyTensor::I16),
            (AnyTensor::I32(a), AnyTensor::I32(b)) => lift(($f::<i32>)(a, b), AnyTensor::I32),
            (AnyTensor::I64(a), AnyTensor::I64(b)) => lift(($f::<i64>)(a, b), AnyTensor::I64),
            (AnyTensor::U8(a), AnyTensor::U8(b)) => lift(($f::<u8>)(a, b), AnyTensor::U8),
            (AnyTensor::U16(a), AnyTensor::U16(b)) => lift(($f::<u16>)(a, b), AnyTensor::U16),
            (AnyTensor::U32(a), AnyTensor::U32(b)) => lift(($f::<u32>)(a, b), AnyTensor::U32),
            (AnyTensor::U64(a), AnyTensor::U64(b)) => lift(($f::<u64>)(a, b), AnyTensor::U64),
            _ => type_err(format!("{}: only integer dtypes of matching kind are allowed", $opname)),
        }
    };
}
pub(crate) use dispatch_bin_int_self;

/// Boolean-only same-dtype binary dispatch (`logical_*`).
macro_rules! dispatch_bin_bool_self {
    ($a:expr, $b:expr, $opname:expr, $f:ident) => {
        match (&$a, &$b) {
            (AnyTensor::Bool(a), AnyTensor::Bool(b)) => lift(($f)(a, b), AnyTensor::Bool),
            _ => type_err(format!("{}: only boolean dtypes are allowed", $opname)),
        }
    };
}
pub(crate) use dispatch_bin_bool_self;

/// Binary dispatch for `equal`/`not_equal`: the `DTypePromoteAPI`-bound pairs
/// plus the bool-numeric pairs (the spec defines equality on every dtype;
/// `bool` promotes to the numeric side in rstsr's promotion table).
macro_rules! dispatch_bin_promote_eq {
    ($a:expr, $b:expr, $opname:expr, $f:ident) => {
        match (&$a, &$b) {
            (AnyTensor::I8(a), AnyTensor::I8(b)) => lift(($f::<i8, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::I16(b)) => lift(($f::<i16, i16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::I32(b)) => lift(($f::<i32, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::I64(b)) => lift(($f::<i64, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::U8(b)) => lift(($f::<u8, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::U16(b)) => lift(($f::<u16, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::U32(b)) => lift(($f::<u32, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::U64(b)) => lift(($f::<u64, u64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::F32(b)) => lift(($f::<f32, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::F64(b)) => lift(($f::<f64, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::C32(a), AnyTensor::C32(b)) => {
                lift(($f::<Complex<f32>, Complex<f32>>)(a, b), crate::any_tensor::any_of)
            },
            (AnyTensor::C64(a), AnyTensor::C64(b)) => {
                lift(($f::<Complex<f64>, Complex<f64>>)(a, b), crate::any_tensor::any_of)
            },
            (AnyTensor::Bool(a), AnyTensor::Bool(b)) => lift(($f::<bool, bool>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::Bool(a), AnyTensor::I8(b)) => lift(($f::<bool, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I8(a), AnyTensor::Bool(b)) => lift(($f::<i8, bool>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::Bool(a), AnyTensor::I16(b)) => lift(($f::<bool, i16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::Bool(b)) => lift(($f::<i16, bool>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::Bool(a), AnyTensor::I32(b)) => lift(($f::<bool, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::Bool(b)) => lift(($f::<i32, bool>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::Bool(a), AnyTensor::I64(b)) => lift(($f::<bool, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::Bool(b)) => lift(($f::<i64, bool>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::Bool(a), AnyTensor::U8(b)) => lift(($f::<bool, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::Bool(b)) => lift(($f::<u8, bool>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::Bool(a), AnyTensor::U16(b)) => lift(($f::<bool, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::Bool(b)) => lift(($f::<u16, bool>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::Bool(a), AnyTensor::U32(b)) => lift(($f::<bool, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::Bool(b)) => lift(($f::<u32, bool>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::Bool(a), AnyTensor::U64(b)) => lift(($f::<bool, u64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::Bool(b)) => lift(($f::<u64, bool>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::Bool(a), AnyTensor::F32(b)) => lift(($f::<bool, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::Bool(b)) => lift(($f::<f32, bool>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::Bool(a), AnyTensor::F64(b)) => lift(($f::<bool, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::Bool(b)) => lift(($f::<f64, bool>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::Bool(a), AnyTensor::C32(b)) => {
                lift(($f::<bool, Complex<f32>>)(a, b), crate::any_tensor::any_of)
            },
            (AnyTensor::C32(a), AnyTensor::Bool(b)) => {
                lift(($f::<Complex<f32>, bool>)(a, b), crate::any_tensor::any_of)
            },
            (AnyTensor::Bool(a), AnyTensor::C64(b)) => {
                lift(($f::<bool, Complex<f64>>)(a, b), crate::any_tensor::any_of)
            },
            (AnyTensor::C64(a), AnyTensor::Bool(b)) => {
                lift(($f::<Complex<f64>, bool>)(a, b), crate::any_tensor::any_of)
            },
            (AnyTensor::C32(a), AnyTensor::C64(b)) => {
                lift(($f::<Complex<f32>, Complex<f64>>)(a, b), crate::any_tensor::any_of)
            },
            (AnyTensor::C32(a), AnyTensor::F32(b)) => lift(($f::<Complex<f32>, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::C32(a), AnyTensor::F64(b)) => lift(($f::<Complex<f32>, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::C32(a), AnyTensor::I16(b)) => lift(($f::<Complex<f32>, i16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::C32(a), AnyTensor::I32(b)) => lift(($f::<Complex<f32>, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::C32(a), AnyTensor::I64(b)) => lift(($f::<Complex<f32>, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::C32(a), AnyTensor::I8(b)) => lift(($f::<Complex<f32>, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::C32(a), AnyTensor::U16(b)) => lift(($f::<Complex<f32>, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::C32(a), AnyTensor::U32(b)) => lift(($f::<Complex<f32>, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::C32(a), AnyTensor::U64(b)) => lift(($f::<Complex<f32>, u64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::C32(a), AnyTensor::U8(b)) => lift(($f::<Complex<f32>, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::C64(a), AnyTensor::C32(b)) => {
                lift(($f::<Complex<f64>, Complex<f32>>)(a, b), crate::any_tensor::any_of)
            },
            (AnyTensor::C64(a), AnyTensor::F32(b)) => lift(($f::<Complex<f64>, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::C64(a), AnyTensor::F64(b)) => lift(($f::<Complex<f64>, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::C64(a), AnyTensor::I16(b)) => lift(($f::<Complex<f64>, i16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::C64(a), AnyTensor::I32(b)) => lift(($f::<Complex<f64>, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::C64(a), AnyTensor::I64(b)) => lift(($f::<Complex<f64>, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::C64(a), AnyTensor::I8(b)) => lift(($f::<Complex<f64>, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::C64(a), AnyTensor::U16(b)) => lift(($f::<Complex<f64>, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::C64(a), AnyTensor::U32(b)) => lift(($f::<Complex<f64>, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::C64(a), AnyTensor::U64(b)) => lift(($f::<Complex<f64>, u64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::C64(a), AnyTensor::U8(b)) => lift(($f::<Complex<f64>, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::C32(b)) => lift(($f::<f32, Complex<f32>>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::C64(b)) => lift(($f::<f32, Complex<f64>>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::F64(b)) => lift(($f::<f32, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::I16(b)) => lift(($f::<f32, i16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::I32(b)) => lift(($f::<f32, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::I64(b)) => lift(($f::<f32, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::I8(b)) => lift(($f::<f32, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::U16(b)) => lift(($f::<f32, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::U32(b)) => lift(($f::<f32, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::U64(b)) => lift(($f::<f32, u64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F32(a), AnyTensor::U8(b)) => lift(($f::<f32, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::C32(b)) => lift(($f::<f64, Complex<f32>>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::C64(b)) => lift(($f::<f64, Complex<f64>>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::F32(b)) => lift(($f::<f64, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::I16(b)) => lift(($f::<f64, i16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::I32(b)) => lift(($f::<f64, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::I64(b)) => lift(($f::<f64, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::I8(b)) => lift(($f::<f64, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::U16(b)) => lift(($f::<f64, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::U32(b)) => lift(($f::<f64, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::U64(b)) => lift(($f::<f64, u64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::F64(a), AnyTensor::U8(b)) => lift(($f::<f64, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::C32(b)) => lift(($f::<i16, Complex<f32>>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::C64(b)) => lift(($f::<i16, Complex<f64>>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::F32(b)) => lift(($f::<i16, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::F64(b)) => lift(($f::<i16, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::I32(b)) => lift(($f::<i16, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::I64(b)) => lift(($f::<i16, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::I8(b)) => lift(($f::<i16, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::U16(b)) => lift(($f::<i16, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::U32(b)) => lift(($f::<i16, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::U64(b)) => lift(($f::<i16, u64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I16(a), AnyTensor::U8(b)) => lift(($f::<i16, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::C32(b)) => lift(($f::<i32, Complex<f32>>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::C64(b)) => lift(($f::<i32, Complex<f64>>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::F32(b)) => lift(($f::<i32, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::F64(b)) => lift(($f::<i32, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::I16(b)) => lift(($f::<i32, i16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::I64(b)) => lift(($f::<i32, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::I8(b)) => lift(($f::<i32, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::U16(b)) => lift(($f::<i32, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::U32(b)) => lift(($f::<i32, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::U64(b)) => lift(($f::<i32, u64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I32(a), AnyTensor::U8(b)) => lift(($f::<i32, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::C32(b)) => lift(($f::<i64, Complex<f32>>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::C64(b)) => lift(($f::<i64, Complex<f64>>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::F32(b)) => lift(($f::<i64, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::F64(b)) => lift(($f::<i64, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::I16(b)) => lift(($f::<i64, i16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::I32(b)) => lift(($f::<i64, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::I8(b)) => lift(($f::<i64, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::U16(b)) => lift(($f::<i64, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::U32(b)) => lift(($f::<i64, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::U64(b)) => lift(($f::<i64, u64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I64(a), AnyTensor::U8(b)) => lift(($f::<i64, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I8(a), AnyTensor::C32(b)) => lift(($f::<i8, Complex<f32>>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I8(a), AnyTensor::C64(b)) => lift(($f::<i8, Complex<f64>>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I8(a), AnyTensor::F32(b)) => lift(($f::<i8, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I8(a), AnyTensor::F64(b)) => lift(($f::<i8, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I8(a), AnyTensor::I32(b)) => lift(($f::<i8, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I8(a), AnyTensor::I64(b)) => lift(($f::<i8, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I8(a), AnyTensor::U16(b)) => lift(($f::<i8, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I8(a), AnyTensor::U32(b)) => lift(($f::<i8, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I8(a), AnyTensor::U64(b)) => lift(($f::<i8, u64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::I8(a), AnyTensor::U8(b)) => lift(($f::<i8, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::C32(b)) => lift(($f::<u16, Complex<f32>>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::C64(b)) => lift(($f::<u16, Complex<f64>>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::F32(b)) => lift(($f::<u16, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::F64(b)) => lift(($f::<u16, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::I16(b)) => lift(($f::<u16, i16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::I32(b)) => lift(($f::<u16, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::I64(b)) => lift(($f::<u16, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::I8(b)) => lift(($f::<u16, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::U32(b)) => lift(($f::<u16, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::U64(b)) => lift(($f::<u16, u64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U16(a), AnyTensor::U8(b)) => lift(($f::<u16, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::C32(b)) => lift(($f::<u32, Complex<f32>>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::C64(b)) => lift(($f::<u32, Complex<f64>>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::F32(b)) => lift(($f::<u32, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::F64(b)) => lift(($f::<u32, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::I16(b)) => lift(($f::<u32, i16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::I32(b)) => lift(($f::<u32, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::I64(b)) => lift(($f::<u32, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::I8(b)) => lift(($f::<u32, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::U16(b)) => lift(($f::<u32, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::U64(b)) => lift(($f::<u32, u64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U32(a), AnyTensor::U8(b)) => lift(($f::<u32, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::C32(b)) => lift(($f::<u64, Complex<f32>>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::C64(b)) => lift(($f::<u64, Complex<f64>>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::F32(b)) => lift(($f::<u64, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::F64(b)) => lift(($f::<u64, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::I16(b)) => lift(($f::<u64, i16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::I32(b)) => lift(($f::<u64, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::I64(b)) => lift(($f::<u64, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::I8(b)) => lift(($f::<u64, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::U16(b)) => lift(($f::<u64, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::U32(b)) => lift(($f::<u64, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U64(a), AnyTensor::U8(b)) => lift(($f::<u64, u8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::C32(b)) => lift(($f::<u8, Complex<f32>>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::C64(b)) => lift(($f::<u8, Complex<f64>>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::F32(b)) => lift(($f::<u8, f32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::F64(b)) => lift(($f::<u8, f64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::I16(b)) => lift(($f::<u8, i16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::I32(b)) => lift(($f::<u8, i32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::I64(b)) => lift(($f::<u8, i64>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::I8(b)) => lift(($f::<u8, i8>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::U16(b)) => lift(($f::<u8, u16>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::U32(b)) => lift(($f::<u8, u32>)(a, b), crate::any_tensor::any_of),
            (AnyTensor::U8(a), AnyTensor::U64(b)) => lift(($f::<u8, u64>)(a, b), crate::any_tensor::any_of),
            _ => dispatch_bin_promote!($a, $b, $opname, $f),
        }
    };
}
pub(crate) use dispatch_bin_promote_eq;

// --------------------------------------------------- typed generic helpers --

pub(crate) fn proj_shape<T>(x: &FTensor<T>) -> Vec<usize>
where
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>>,
{
    AsRef::<[usize]>::as_ref(x.shape()).to_vec()
}

pub(crate) fn proj_ndim<T>(x: &FTensor<T>) -> usize
where
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>>,
{
    x.ndim()
}

pub(crate) fn proj_size<T>(x: &FTensor<T>) -> usize
where
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>>,
{
    x.size()
}

pub(crate) fn proj_name<T: DtypeName>(_: &FTensor<T>) -> &'static str {
    T::NAME
}

/// A Python scalar leaf, normalized to one of four canonical carriers.
#[derive(Clone, Copy, Debug)]
pub enum PyScalar {
    B(bool),
    I(i64),
    F(f64),
    C(Complex<f64>),
}

impl PyScalar {
    /// Kind rank for default-dtype inference: bool < int < float < complex.
    pub fn kind_rank(&self) -> u8 {
        match self {
            PyScalar::B(_) => 0,
            PyScalar::I(_) => 1,
            PyScalar::F(_) => 2,
            PyScalar::C(_) => 3,
        }
    }

    /// Python object conversion for tolist/item.
    pub fn to_py(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let v = match self {
            PyScalar::B(x) => x.into_py_any(py)?,
            PyScalar::I(x) => x.into_py_any(py)?,
            PyScalar::F(x) => x.into_py_any(py)?,
            PyScalar::C(x) => PyComplex::from_doubles(py, x.re, x.im).into_any().unbind(),
        };
        Ok(v)
    }
}

pub fn parse_leaf(el: &Bound<'_, PyAny>) -> PyResult<PyScalar> {
    if el.is_instance_of::<pyo3::types::PyBool>() {
        Ok(PyScalar::B(el.extract::<bool>()?))
    } else if let Ok(c) = el.cast::<PyComplex>() {
        Ok(PyScalar::C(Complex::new(c.real(), c.imag())))
    } else if let Ok(i) = el.extract::<i64>() {
        Ok(PyScalar::I(i))
    } else if let Ok(f) = el.extract::<f64>() {
        Ok(PyScalar::F(f))
    } else {
        type_err(format!("asarray: unsupported leaf type {}", el.get_type().name()?))
    }
}

/// Scalar -> PyScalar conversion: one `From` impl per dtype.
macro_rules! impl_pyscalar_from {
    ($t:ty, $variant:ident) => {
        impl From<$t> for PyScalar {
            fn from(v: $t) -> PyScalar {
                PyScalar::$variant(v.into())
            }
        }
    };
}
impl_pyscalar_from!(bool, B);
impl_pyscalar_from!(i8, I);
impl_pyscalar_from!(i16, I);
impl_pyscalar_from!(i32, I);
impl_pyscalar_from!(i64, I);
impl_pyscalar_from!(u8, I);
impl_pyscalar_from!(u16, I);
impl_pyscalar_from!(u32, I);
// u64 does not fit i64; wrap (spec edge: values > i64::MAX from u64 arrays).
impl From<u64> for PyScalar {
    fn from(v: u64) -> PyScalar {
        PyScalar::I(v as i64)
    }
}
impl_pyscalar_from!(f32, F);
impl_pyscalar_from!(f64, F);

impl From<Complex<f32>> for PyScalar {
    fn from(v: Complex<f32>) -> PyScalar {
        PyScalar::C(Complex::new(v.re as f64, v.im as f64))
    }
}
impl From<Complex<f64>> for PyScalar {
    fn from(v: Complex<f64>) -> PyScalar {
        PyScalar::C(v)
    }
}

/// Build a nested Python list from row-major data (used by tolist).
fn build_nested<T>(
    py: Python<'_>,
    shape: &[usize],
    data: &mut std::vec::IntoIter<T>,
    conv: impl Fn(T, Python<'_>) -> PyResult<Py<PyAny>> + Copy,
) -> PyResult<Py<PyAny>> {
    if shape.is_empty() {
        let v = data.next().expect("tolist: data/shape mismatch");
        return conv(v, py);
    }
    let mut items = Vec::with_capacity(shape[0]);
    for _ in 0..shape[0] {
        items.push(build_nested(py, &shape[1..], data, conv)?);
    }
    items.into_py_any(py)
}

pub(crate) fn tolist_t<T>(x: &FTensor<T>, py: Python<'_>) -> PyResult<Py<PyAny>>
where
    T: Copy + Into<PyScalar>,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>>,
{
    let data: Vec<T> = x.view().iter().copied().collect();
    let shape = AsRef::<[usize]>::as_ref(x.shape()).to_vec();
    build_nested(py, &shape, &mut data.into_iter(), |s, py| {
        let p: PyScalar = s.into();
        p.to_py(py)
    })
}

pub(crate) fn item_t<T>(x: &FTensor<T>, py: Python<'_>) -> PyResult<Py<PyAny>>
where
    T: Copy + Into<PyScalar>,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>>,
{
    let s: T = err_py(x.to_scalar_f())?;
    let p: PyScalar = s.into();
    p.to_py(py)
}

impl AnyTensor {
    pub fn shape(&self) -> Vec<usize> {
        dispatch_fn!(self, proj_shape())
    }

    pub fn ndim(&self) -> usize {
        dispatch_fn!(self, proj_ndim())
    }

    pub fn size(&self) -> usize {
        dispatch_fn!(self, proj_size())
    }

    pub fn dtype_name(&self) -> &'static str {
        dispatch_fn!(self, proj_name())
    }

    pub fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        dispatch_fn!(self, tolist_t(py))
    }

    pub fn item(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        dispatch_fn!(self, item_t(py))
    }

    /// Deep copy (rstsr `Clone` on owned tensors copies the storage).
    pub fn deep_copy(&self) -> AnyTensor {
        self.clone()
    }
}
