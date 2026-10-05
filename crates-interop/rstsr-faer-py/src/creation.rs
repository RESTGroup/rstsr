//! Creation functions: asarray (from flattened Python lists), zeros/ones/
//! empty/full/arange, and astype (element cast through DTypeCastAPI — rstsr
//! 0.9.0 has no tensor-level dtype conversion; see gap register G-007).

use num::Complex;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use pyo3::Bound;
use rstsr::prelude::rt;
use rstsr::prelude::*;

use crate::any_tensor::{
    device_faer, dispatch_name, dispatch_name_numeric, dispatch_name_real, err_py, liftp, parse_leaf, type_err,
    AnyTensor, FTensor, NativeArray, PyScalar,
};
use crate::dtype::Dtype;

pub fn dim_from(shape: &[usize]) -> IxD {
    IxD::try_from(shape.to_vec()).expect("shape -> IxD")
}

/// Element cast matrix for astype/asarray(dtype=...). rstsr 0.9.0 has no
/// tensor-level dtype conversion and its element-level DTypeCastAPI covers
/// only a subset of pairs (gap G-007), so this shim carries its own
/// primitive-cast table — same `as` semantics rstsr itself uses, plus
/// numpy conventions for complex (complex->real takes the real part).
/// Rust float->int casts saturate (vs numpy's UB wrap); the standard
/// leaves out-of-range float->int casts undefined, so this is conformant.
pub(crate) trait NumCastShim<To> {
    fn cast_to(self) -> To;
}

// real (non-bool) -> real (non-bool): plain `as`
macro_rules! cast_row {
    ($s:ty) => {
        macro_rules! cast_col {
            ($u:ty) => {
                impl NumCastShim<$u> for $s {
                    fn cast_to(self) -> $u {
                        self as $u
                    }
                }
            };
        }
        cast_col!(i8);
        cast_col!(i16);
        cast_col!(i32);
        cast_col!(i64);
        cast_col!(u8);
        cast_col!(u16);
        cast_col!(u32);
        cast_col!(u64);
        cast_col!(f32);
        cast_col!(f64);
    };
}
cast_row!(i8);
cast_row!(i16);
cast_row!(i32);
cast_row!(i64);
cast_row!(u8);
cast_row!(u16);
cast_row!(u32);
cast_row!(u64);
cast_row!(f32);
cast_row!(f64);

// bool source: bool as float is not a Rust cast; go through u8
macro_rules! cast_row_bool_src {
    ($u:ty) => {
        impl NumCastShim<$u> for bool {
            fn cast_to(self) -> $u {
                self as u8 as $u
            }
        }
    };
}
impl NumCastShim<bool> for bool {
    fn cast_to(self) -> bool {
        self
    }
}
cast_row_bool_src!(i8);
cast_row_bool_src!(i16);
cast_row_bool_src!(i32);
cast_row_bool_src!(i64);
cast_row_bool_src!(u8);
cast_row_bool_src!(u16);
cast_row_bool_src!(u32);
cast_row_bool_src!(u64);
cast_row_bool_src!(f32);
cast_row_bool_src!(f64);

// -> bool target: nonzero is true (numpy convention; NaN casts to true)
macro_rules! cast_col_to_bool {
    ($s:ty) => {
        impl NumCastShim<bool> for $s {
            fn cast_to(self) -> bool {
                self != 0
            }
        }
    };
}
cast_col_to_bool!(i8);
cast_col_to_bool!(i16);
cast_col_to_bool!(i32);
cast_col_to_bool!(i64);
cast_col_to_bool!(u8);
cast_col_to_bool!(u16);
cast_col_to_bool!(u32);
cast_col_to_bool!(u64);
impl NumCastShim<bool> for f32 {
    fn cast_to(self) -> bool {
        self != 0.0
    }
}
impl NumCastShim<bool> for f64 {
    fn cast_to(self) -> bool {
        self != 0.0
    }
}

// real -> complex (and complex -> complex width changes)
macro_rules! cast_row_to_c64 {
    ($s:ty) => {
        impl NumCastShim<Complex<f64>> for $s {
            fn cast_to(self) -> Complex<f64> {
                Complex::new(self as f64, 0.0)
            }
        }
    };
}
impl NumCastShim<Complex<f64>> for bool {
    fn cast_to(self) -> Complex<f64> {
        Complex::new(self as u8 as f64, 0.0)
    }
}
cast_row_to_c64!(i8);
cast_row_to_c64!(i16);
cast_row_to_c64!(i32);
cast_row_to_c64!(i64);
cast_row_to_c64!(u8);
cast_row_to_c64!(u16);
cast_row_to_c64!(u32);
cast_row_to_c64!(u64);
cast_row_to_c64!(f32);
cast_row_to_c64!(f64);
impl NumCastShim<Complex<f64>> for Complex<f32> {
    fn cast_to(self) -> Complex<f64> {
        Complex::new(self.re as f64, self.im as f64)
    }
}
impl NumCastShim<Complex<f64>> for Complex<f64> {
    fn cast_to(self) -> Complex<f64> {
        self
    }
}

macro_rules! cast_row_to_c32 {
    ($s:ty) => {
        impl NumCastShim<Complex<f32>> for $s {
            fn cast_to(self) -> Complex<f32> {
                Complex::new(self as f32, 0.0)
            }
        }
    };
}
impl NumCastShim<Complex<f32>> for bool {
    fn cast_to(self) -> Complex<f32> {
        Complex::new(self as u8 as f32, 0.0)
    }
}
cast_row_to_c32!(i8);
cast_row_to_c32!(i16);
cast_row_to_c32!(i32);
cast_row_to_c32!(i64);
cast_row_to_c32!(u8);
cast_row_to_c32!(u16);
cast_row_to_c32!(u32);
cast_row_to_c32!(u64);
cast_row_to_c32!(f32);
cast_row_to_c32!(f64);
impl NumCastShim<Complex<f32>> for Complex<f64> {
    fn cast_to(self) -> Complex<f32> {
        Complex::new(self.re as f32, self.im as f32)
    }
}
impl NumCastShim<Complex<f32>> for Complex<f32> {
    fn cast_to(self) -> Complex<f32> {
        self
    }
}

// complex -> real: real part (numpy convention)
impl NumCastShim<bool> for Complex<f64> {
    fn cast_to(self) -> bool {
        self.re != 0.0
    }
}
macro_rules! cast_row_from_c64 {
    ($u:ty) => {
        impl NumCastShim<$u> for Complex<f64> {
            fn cast_to(self) -> $u {
                self.re as $u
            }
        }
    };
}
cast_row_from_c64!(i8);
cast_row_from_c64!(i16);
cast_row_from_c64!(i32);
cast_row_from_c64!(i64);
cast_row_from_c64!(u8);
cast_row_from_c64!(u16);
cast_row_from_c64!(u32);
cast_row_from_c64!(u64);
cast_row_from_c64!(f32);
cast_row_from_c64!(f64);

impl NumCastShim<bool> for Complex<f32> {
    fn cast_to(self) -> bool {
        self.re != 0.0
    }
}
macro_rules! cast_row_from_c32 {
    ($u:ty) => {
        impl NumCastShim<$u> for Complex<f32> {
            fn cast_to(self) -> $u {
                self.re as $u
            }
        }
    };
}
cast_row_from_c32!(i8);
cast_row_from_c32!(i16);
cast_row_from_c32!(i32);
cast_row_from_c32!(i64);
cast_row_from_c32!(u8);
cast_row_from_c32!(u16);
cast_row_from_c32!(u32);
cast_row_from_c32!(u64);
cast_row_from_c32!(f32);
cast_row_from_c32!(f64);

/// Canonical-scalar -> dtype-element cast (asarray/full/arange/setitem all
/// funnel here; lossy casts allowed, matching numpy's asarray-with-dtype
/// behavior). Complex leaves are rejected for real targets at runtime — the
/// standard provides no complex->real scalar conversion.
pub(crate) trait ScalarCastTarget: Sized {
    fn from_scalar(s: PyScalar) -> PyResult<Self>;
}

macro_rules! impl_sct_real {
    ($t:ty) => {
        impl ScalarCastTarget for $t {
            fn from_scalar(s: PyScalar) -> PyResult<Self> {
                match s {
                    PyScalar::B(x) => Ok(x.cast_to()),
                    PyScalar::I(x) => Ok(x.cast_to()),
                    PyScalar::F(x) => Ok(x.cast_to()),
                    PyScalar::C(_) => type_err("cannot convert a complex scalar to a real dtype"),
                }
            }
        }
    };
}
impl_sct_real!(bool);
impl_sct_real!(i8);
impl_sct_real!(i16);
impl_sct_real!(i32);
impl_sct_real!(i64);
impl_sct_real!(u8);
impl_sct_real!(u16);
impl_sct_real!(u32);
impl_sct_real!(u64);
impl_sct_real!(f32);
impl_sct_real!(f64);

macro_rules! impl_sct_complex {
    ($t:ty) => {
        impl ScalarCastTarget for $t {
            fn from_scalar(s: PyScalar) -> PyResult<Self> {
                match s {
                    PyScalar::B(x) => Ok(x.cast_to()),
                    PyScalar::I(x) => Ok(x.cast_to()),
                    PyScalar::F(x) => Ok(x.cast_to()),
                    PyScalar::C(x) => Ok(x.cast_to()),
                }
            }
        }
    };
}
impl_sct_complex!(Complex<f32>);
impl_sct_complex!(Complex<f64>);

fn build_vec<T: ScalarCastTarget>(scalars: &[PyScalar]) -> PyResult<Vec<T>> {
    scalars.iter().map(|&s| T::from_scalar(s)).collect()
}

fn build_t<T: ScalarCastTarget>(scalars: &[PyScalar], shape: &[usize]) -> PyResult<FTensor<T>>
where
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>>,
{
    let data = build_vec::<T>(scalars)?;
    err_py(rt::asarray_f((data, dim_from(shape), device_faer())))
}

enum Filler {
    Zeros,
    Ones,
    Empty,
    Full(PyScalar),
}

fn filled_t<T>(shape: &[usize], filler: &Filler) -> PyResult<FTensor<T>>
where
    T: Clone + num::Num + ScalarCastTarget,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>>,
{
    let r: rt::Result<FTensor<T>> = match filler {
        Filler::Zeros => rt::zeros_f((dim_from(shape), device_faer())),
        Filler::Ones => rt::ones_f((dim_from(shape), device_faer())),
        Filler::Empty => rt::zeros_f((dim_from(shape), device_faer())),
        Filler::Full(s) => {
            let val: T = T::from_scalar(*s)?;
            rt::full_f((dim_from(shape), val, device_faer()))
        },
    };
    err_py(r)
}

/// Python-side asarray flattens nested lists; this entry receives the flat
/// leaves, the shape, and the (optional) target dtype.
#[pyfunction]
pub fn asarray_from_flat<'py>(
    flat: &Bound<'py, PyList>,
    shape: Vec<usize>,
    dtype: Option<&Bound<'py, Dtype>>,
    #[allow(unused_variables)] device: &Bound<'py, crate::device::Device>,
) -> PyResult<NativeArray> {
    let expected: usize = shape.iter().product();
    if flat.len() != expected {
        return type_err(format!(
            "asarray: flat data length {} does not match shape {shape:?} ({expected} elements)",
            flat.len()
        ));
    }
    let mut scalars = Vec::with_capacity(flat.len());
    for el in flat.iter() {
        scalars.push(parse_leaf(&el)?);
    }
    let name: &'static str = match dtype {
        Some(d) => d.borrow().name,
        None => {
            let rank = scalars.iter().map(|s| s.kind_rank()).max().unwrap_or(1);
            // spec default-dtype rules from Python native types
            ["bool", "int64", "float64", "complex128"][rank as usize]
        },
    };
    let t = dispatch_name!(name, build_t(&scalars, &shape))?;
    Ok(NativeArray { t })
}

/// bool has no `Num` impl, so the zeros/ones/full tuple API is unavailable
/// for it; assemble through asarray instead (values are exact constants).
fn filled_bool(shape: &[usize], filler: &Filler) -> PyResult<AnyTensor> {
    let n: usize = shape.iter().product();
    let v = match filler {
        Filler::Zeros | Filler::Empty => vec![false; n],
        Filler::Ones => vec![true; n],
        Filler::Full(s) => {
            let b = match *s {
                PyScalar::B(x) => x,
                PyScalar::I(x) => x != 0,
                PyScalar::F(x) => x != 0.0,
                PyScalar::C(x) => x != num::Complex::new(0.0, 0.0),
            };
            vec![b; n]
        },
    };
    err_py(rt::asarray_f((v, dim_from(shape), device_faer()))).map(AnyTensor::Bool)
}

#[pyfunction]
pub fn zeros<'py>(
    shape: Vec<usize>,
    dtype: &Bound<'py, Dtype>,
    #[allow(unused_variables)] device: &Bound<'py, crate::device::Device>,
) -> PyResult<NativeArray> {
    let t = if dtype.borrow().name == "bool" {
        filled_bool(&shape, &Filler::Zeros)?
    } else {
        dispatch_name_numeric!(dtype.borrow().name, filled_t(&shape, &Filler::Zeros))?
    };
    Ok(NativeArray { t })
}

#[pyfunction]
pub fn ones<'py>(
    shape: Vec<usize>,
    dtype: &Bound<'py, Dtype>,
    #[allow(unused_variables)] device: &Bound<'py, crate::device::Device>,
) -> PyResult<NativeArray> {
    let t = if dtype.borrow().name == "bool" {
        filled_bool(&shape, &Filler::Ones)?
    } else {
        dispatch_name_numeric!(dtype.borrow().name, filled_t(&shape, &Filler::Ones))?
    };
    Ok(NativeArray { t })
}

#[pyfunction]
pub fn empty<'py>(
    shape: Vec<usize>,
    dtype: &Bound<'py, Dtype>,
    #[allow(unused_variables)] device: &Bound<'py, crate::device::Device>,
) -> PyResult<NativeArray> {
    let t = if dtype.borrow().name == "bool" {
        filled_bool(&shape, &Filler::Empty)?
    } else {
        dispatch_name_numeric!(dtype.borrow().name, filled_t(&shape, &Filler::Empty))?
    };
    Ok(NativeArray { t })
}

#[pyfunction]
pub fn full<'py>(
    shape: Vec<usize>,
    fill_value: &Bound<'py, PyAny>,
    dtype: &Bound<'py, Dtype>,
    #[allow(unused_variables)] device: &Bound<'py, crate::device::Device>,
) -> PyResult<NativeArray> {
    let s = parse_leaf(fill_value)?;
    let filler = Filler::Full(s);
    let t = if dtype.borrow().name == "bool" {
        filled_bool(&shape, &filler)?
    } else {
        dispatch_name_numeric!(dtype.borrow().name, filled_t(&shape, &filler))?
    };
    Ok(NativeArray { t })
}

/// arange with spec dtype inference: int args -> int64, any float -> float64
/// (bool/complex arguments are invalid per spec).
#[pyfunction]
pub fn arange<'py>(
    start: &Bound<'py, PyAny>,
    stop: Option<&Bound<'py, PyAny>>,
    step: Option<&Bound<'py, PyAny>>,
    dtype: Option<&Bound<'py, Dtype>>,
    #[allow(unused_variables)] device: &Bound<'py, crate::device::Device>,
) -> PyResult<NativeArray> {
    let s0 = parse_leaf(start)?;
    let s1 = stop.map(parse_leaf).transpose()?;
    let s2 = step.map(parse_leaf).transpose()?;
    let name: &'static str = match dtype {
        Some(d) => d.borrow().name,
        None => {
            if matches!(s0, PyScalar::B(_)) || matches!(s1, Some(PyScalar::B(_))) || matches!(s2, Some(PyScalar::B(_)))
            {
                return type_err("arange: boolean arguments are not supported");
            }
            if matches!(s0, PyScalar::C(_)) || matches!(s1, Some(PyScalar::C(_))) || matches!(s2, Some(PyScalar::C(_)))
            {
                return type_err("arange: complex arguments are not supported by the standard");
            }
            let rank = [Some(s0), s1, s2].iter().flatten().map(|s| s.kind_rank()).max().unwrap_or(1);
            ["bool", "int64", "float64"][rank as usize]
        },
    };
    let t = dispatch_name_real!(name, arange_t(s0, s1, s2))?;
    Ok(NativeArray { t })
}

fn arange_t<T>(s0: PyScalar, s1: Option<PyScalar>, s2: Option<PyScalar>) -> PyResult<FTensor<T>>
where
    T: Copy + Clone + num::Num + PartialOrd + ScalarCastTarget + 'static,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>>,
{
    let a0: T = T::from_scalar(s0)?;
    let a1: Option<T> = match s1 {
        Some(s) => Some(T::from_scalar(s)?),
        None => None,
    };
    let a2: Option<T> = match s2 {
        Some(s) => Some(T::from_scalar(s)?),
        None => None,
    };
    // Normalize all argument arities to (start, stop, step). arange(stop) with
    // an explicit step ≡ arange(0, stop, step); dropping that step would
    // materialize `stop` elements (the suite draws stops up to ~9e18).
    let (start, stop, step) = match (a1, a2) {
        (Some(b), Some(c)) => (a0, b, c),
        (Some(b), None) => (a0, b, T::one()),
        (None, Some(c)) => (T::zero(), a0, c),
        (None, None) => (T::zero(), a0, T::one()),
    };
    // A zero step raises InvalidValue; a sign-mismatched range comes back
    // empty from rstsr itself.
    err_py(rt::arange_f((start, stop, step, device_faer())))
}

fn cast_ts<S, U>(t: &FTensor<S>) -> PyResult<FTensor<U>>
where
    S: Copy + NumCastShim<U>,
    DeviceFaer: DeviceAPI<U, Raw = Vec<U>>,
{
    let shape = AsRef::<[usize]>::as_ref(t.shape()).to_vec();
    let data: Vec<U> = t.view().iter().map(|&v| v.cast_to()).collect();
    err_py(rt::asarray_f((data, dim_from(&shape), device_faer())))
}

/// astype: element-wise cast via DTypeCastAPI (rstsr 0.9.0 has no tensor-level
/// dtype conversion, gap G-007). `copy=false` with matching dtype still
/// deep-copies — the handle model has no shared-storage alias (gap G-014).
#[pyfunction]
pub fn astype<'py>(x: &NativeArray, dtype: &Bound<'py, Dtype>, copy: bool) -> PyResult<NativeArray> {
    if x.t.dtype_name() == dtype.borrow().name {
        return Ok(NativeArray { t: x.t.deep_copy() });
    }
    let _ = copy;
    macro_rules! cast_arm {
        ($src:ty, $tval:expr) => {
            |target: &'static str| match target {
                "bool" => liftp(cast_ts::<$src, bool>($tval), AnyTensor::Bool),
                "int8" => liftp(cast_ts::<$src, i8>($tval), AnyTensor::I8),
                "int16" => liftp(cast_ts::<$src, i16>($tval), AnyTensor::I16),
                "int32" => liftp(cast_ts::<$src, i32>($tval), AnyTensor::I32),
                "int64" => liftp(cast_ts::<$src, i64>($tval), AnyTensor::I64),
                "uint8" => liftp(cast_ts::<$src, u8>($tval), AnyTensor::U8),
                "uint16" => liftp(cast_ts::<$src, u16>($tval), AnyTensor::U16),
                "uint32" => liftp(cast_ts::<$src, u32>($tval), AnyTensor::U32),
                "uint64" => liftp(cast_ts::<$src, u64>($tval), AnyTensor::U64),
                "float32" => liftp(cast_ts::<$src, f32>($tval), AnyTensor::F32),
                "float64" => liftp(cast_ts::<$src, f64>($tval), AnyTensor::F64),
                "complex64" => liftp(cast_ts::<$src, Complex<f32>>($tval), AnyTensor::C32),
                "complex128" => liftp(cast_ts::<$src, Complex<f64>>($tval), AnyTensor::C64),
                _ => type_err(format!("unknown dtype {target:?}")),
            }
        };
    }
    // Arm bodies are expression position: per-arm closure invokes the inner
    // name dispatch with the source type fixed by the arm's pattern.
    let t: AnyTensor = match &x.t {
        AnyTensor::Bool(t) => (cast_arm!(bool, t))(dtype.borrow().name)?,
        AnyTensor::I8(t) => (cast_arm!(i8, t))(dtype.borrow().name)?,
        AnyTensor::I16(t) => (cast_arm!(i16, t))(dtype.borrow().name)?,
        AnyTensor::I32(t) => (cast_arm!(i32, t))(dtype.borrow().name)?,
        AnyTensor::I64(t) => (cast_arm!(i64, t))(dtype.borrow().name)?,
        AnyTensor::U8(t) => (cast_arm!(u8, t))(dtype.borrow().name)?,
        AnyTensor::U16(t) => (cast_arm!(u16, t))(dtype.borrow().name)?,
        AnyTensor::U32(t) => (cast_arm!(u32, t))(dtype.borrow().name)?,
        AnyTensor::U64(t) => (cast_arm!(u64, t))(dtype.borrow().name)?,
        AnyTensor::F32(t) => (cast_arm!(f32, t))(dtype.borrow().name)?,
        AnyTensor::F64(t) => (cast_arm!(f64, t))(dtype.borrow().name)?,
        AnyTensor::C32(t) => (cast_arm!(Complex<f32>, t))(dtype.borrow().name)?,
        AnyTensor::C64(t) => (cast_arm!(Complex<f64>, t))(dtype.borrow().name)?,
    };
    Ok(NativeArray { t })
}
