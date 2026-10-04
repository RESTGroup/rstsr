//! rstsr_faer — pyo3 binding of rstsr (faer device) for array-API validation.
//!
//! Layer split (per DECISIONS.md): this crate owns tensor construction,
//! dtype dispatch, and rstsr calls; the Python package `rstsr_faer.api`
//! owns only signatures, protocol objects, and marshalling. No numeric
//! algorithms beyond what rstsr itself provides.

mod any_tensor;
mod creation;
mod device;
mod dlpack;
mod dtype;
mod indexing;
mod info;
mod ops;

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::OnceLock;
use pyo3::types::{PyAny, PyBool, PyComplex};

use any_tensor::NativeArray;
use dtype::Dtype;

/// Name -> dtype singleton registry, filled once at module init.
static DTYPES: OnceLock<HashMap<&'static str, Py<Dtype>>> = OnceLock::new();

pub(crate) fn dtype_by_name(py: Python<'_>, name: &str) -> PyResult<Py<Dtype>> {
    DTYPES
        .get()
        .and_then(|m| m.get(name))
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("unknown dtype {name:?}")))
}

// ------------------------------------------------------------- data types ---

#[pyfunction]
fn finfo<'py>(py: Python<'py>, dtype: &Bound<'py, Dtype>) -> PyResult<info::Finfo> {
    let d = dtype_by_name(py, dtype.borrow().name)?;
    match dtype.borrow().name {
        "float32" => Ok(info::finfo32(d)),
        "float64" => Ok(info::finfo64(d)),
        _ => Err(PyValueError::new_err(
            "finfo: only real floating-point dtypes are allowed",
        )),
    }
}

#[pyfunction]
fn iinfo<'py>(py: Python<'py>, dtype: &Bound<'py, Dtype>) -> PyResult<info::Iinfo> {
    let d = dtype_by_name(py, dtype.borrow().name)?;
    Ok(match dtype.borrow().name {
        "int8" => info::iinfo_i8(d),
        "int16" => info::iinfo_i16(d),
        "int32" => info::iinfo_i32(d),
        "int64" => info::iinfo_i64(d),
        "uint8" => info::iinfo_u8(d),
        "uint16" => info::iinfo_u16(d),
        "uint32" => info::iinfo_u32(d),
        "uint64" => info::iinfo_u64(d),
        _ => {
            return Err(PyValueError::new_err(
                "iinfo: only integral dtypes are allowed",
            ))
        }
    })
}

/// Kind string used by the Python layer for weak-scalar ordering.
#[pyfunction]
fn dtype_kind(dtype: &Bound<'_, Dtype>) -> &'static str {
    match dtype.borrow().name {
        "bool" => "bool",
        "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32" | "uint64" => {
            "integral"
        }
        "float32" | "float64" => "real floating",
        _ => "complex floating",
    }
}

/// Default dtype for a Python scalar (spec: bool/int -> int64, float ->
/// float64, complex -> complex128).
#[pyfunction]
fn default_dtype_for<'py>(py: Python<'py>, value: &Bound<'py, PyAny>) -> PyResult<Py<Dtype>> {
    let name = if value.is_instance_of::<PyBool>() {
        "bool"
    } else if value.cast::<PyComplex>().is_ok() {
        "complex128"
    } else if value.extract::<i64>().is_ok() {
        "int64"
    } else if value.extract::<f64>().is_ok() {
        "float64"
    } else {
        return Err(PyValueError::new_err(format!(
            "not a supported scalar type: {}",
            value.get_type().name()?
        )));
    };
    dtype_by_name(py, name)
}

#[pymodule]
fn rstsr_faer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    dtype::add_dtype_objects(m)?;
    device::add_device_object(m)?;

    // Collect the singletons just registered into the name-keyed registry.
    let mut map: HashMap<&'static str, Py<Dtype>> = HashMap::new();
    macro_rules! reg {
        ($attr:ident) => {{
            let name: &'static str = stringify!($attr);
            let obj: Py<Dtype> = m.getattr(name)?.extract()?;
            map.insert(name, obj);
        }};
    }
    reg!(bool);
    reg!(int8);
    reg!(int16);
    reg!(int32);
    reg!(int64);
    reg!(uint8);
    reg!(uint16);
    reg!(uint32);
    reg!(uint64);
    reg!(float32);
    reg!(float64);
    reg!(complex64);
    reg!(complex128);
    let _ = DTYPES.set(map);

    m.add("e", std::f64::consts::E)?;
    m.add("inf", f64::INFINITY)?;
    m.add("nan", f64::NAN)?;
    m.add("pi", std::f64::consts::PI)?;

    m.add_class::<NativeArray>()?;
    m.add_class::<Dtype>()?;
    m.add_class::<device::Device>()?;
    m.add_class::<info::Finfo>()?;
    m.add_class::<info::Iinfo>()?;

    m.add_function(wrap_pyfunction!(creation::asarray_from_flat, m)?)?;
    m.add_function(wrap_pyfunction!(creation::zeros, m)?)?;
    m.add_function(wrap_pyfunction!(creation::ones, m)?)?;
    m.add_function(wrap_pyfunction!(creation::empty, m)?)?;
    m.add_function(wrap_pyfunction!(creation::full, m)?)?;
    m.add_function(wrap_pyfunction!(creation::arange, m)?)?;
    m.add_function(wrap_pyfunction!(creation::astype, m)?)?;

    m.add_function(wrap_pyfunction!(ops::add, m)?)?;
    m.add_function(wrap_pyfunction!(ops::subtract, m)?)?;
    m.add_function(wrap_pyfunction!(ops::multiply, m)?)?;
    m.add_function(wrap_pyfunction!(ops::divide, m)?)?;
    m.add_function(wrap_pyfunction!(ops::negative, m)?)?;
    m.add_function(wrap_pyfunction!(ops::abs, m)?)?;

    m.add_function(wrap_pyfunction!(ops::equal, m)?)?;
    m.add_function(wrap_pyfunction!(ops::not_equal, m)?)?;
    m.add_function(wrap_pyfunction!(ops::less, m)?)?;
    m.add_function(wrap_pyfunction!(ops::less_equal, m)?)?;
    m.add_function(wrap_pyfunction!(ops::greater, m)?)?;
    m.add_function(wrap_pyfunction!(ops::greater_equal, m)?)?;

    m.add_function(wrap_pyfunction!(ops::all, m)?)?;
    m.add_function(wrap_pyfunction!(ops::any, m)?)?;
    m.add_function(wrap_pyfunction!(ops::isnan, m)?)?;
    m.add_function(wrap_pyfunction!(ops::isfinite, m)?)?;
    m.add_function(wrap_pyfunction!(ops::isinf, m)?)?;

    m.add_function(wrap_pyfunction!(ops::reshape, m)?)?;
    m.add_function(wrap_pyfunction!(ops::transpose, m)?)?;
    m.add_function(wrap_pyfunction!(ops::getitem_int, m)?)?;
    m.add_function(wrap_pyfunction!(ops::sum, m)?)?;
    m.add_function(wrap_pyfunction!(ops::broadcast_to, m)?)?;

    m.add_function(wrap_pyfunction!(indexing::getitem_basic, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::setitem_basic, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::setitem_scalar, m)?)?;

    m.add_function(wrap_pyfunction!(dlpack::dlpack_export, m)?)?;
    m.add_function(wrap_pyfunction!(dlpack::dlpack_import, m)?)?;

    m.add_function(wrap_pyfunction!(finfo, m)?)?;
    m.add_function(wrap_pyfunction!(iinfo, m)?)?;
    m.add_function(wrap_pyfunction!(dtype_kind, m)?)?;
    m.add_function(wrap_pyfunction!(default_dtype_for, m)?)?;

    Ok(())
}
