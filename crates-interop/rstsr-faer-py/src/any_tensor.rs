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
//! - `dispatch_bin!` — same-dtype pair; mismatch -> TypeError (promotion gap)
//! - `dispatch_name!`— dtype-name dispatch calling a generic fn item

use num::Complex;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::types::{PyAny, PyComplex, PyTuple};
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
        Ok(PyTuple::new(py, self.t.shape())?)
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
        NativeArray {
            t: self.t.deep_copy(),
        }
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
        RSTSRError::IndexError(_) | RSTSRError::AxisError { .. } => {
            PyIndexError::new_err(format!("{e}"))
        }
        _ => PyValueError::new_err(format!("{e}")),
    })
}

pub fn type_err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(PyTypeError::new_err(msg.into()))
}

/// Lift a typed tensor result into the erased enum via the arm's variant
/// constructor (the constructor makes R concrete at each instantiation).
pub(crate) fn lift<R>(
    r: rt::Result<FTensor<R>>,
    ctor: impl FnOnce(FTensor<R>) -> AnyTensor,
) -> PyResult<AnyTensor> {
    err_py(r).map(ctor)
}

/// Same for helpers that already produce `PyResult` (creation paths).
pub(crate) fn liftp<R>(
    r: PyResult<FTensor<R>>,
    ctor: impl FnOnce(FTensor<R>) -> AnyTensor,
) -> PyResult<AnyTensor> {
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

macro_rules! dispatch_bin {
    ($a:expr, $b:expr, $opname:literal, $f:ident, $rv:ident) => {
        match (&$a, &$b) {
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
            (AnyTensor::C32(a), AnyTensor::C32(b)) => lift(($f::<Complex<f32>>)(a, b), AnyTensor::$rv),
            (AnyTensor::C64(a), AnyTensor::C64(b)) => lift(($f::<Complex<f64>>)(a, b), AnyTensor::$rv),
            _ => type_err(format!(
                "{}: mixed-dtype operands require type promotion (rstsr gap); use astype()",
                $opname
            )),
        }
    };
}
pub(crate) use dispatch_bin;

/// Same-dtype binary dispatch over numeric dtypes only (bool is rejected —
/// rstsr provides no bool arithmetic, matching the standard).

/// Numeric same-dtype binary dispatch where the result keeps the input dtype.
macro_rules! dispatch_bin_numeric_self {
    ($a:expr, $b:expr, $opname:literal, $f:ident) => {
        match (&$a, &$b) {
            (AnyTensor::Bool(_), _) | (_, AnyTensor::Bool(_)) => type_err(format!(
                "{}: not defined for bool dtype",
                $opname
            )),
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
            _ => type_err(format!(
                "{}: mixed-dtype operands require type promotion (rstsr gap); use astype()",
                $opname
            )),
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
        type_err(format!(
            "asarray: unsupported leaf type {}",
            el.get_type().name()?
        ))
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
