//! Indexing surface: spec `__getitem__`/`__setitem__` over DeviceFaer.
//!
//! Basic keys only (int/slice/newaxis/ellipsis), riding rstsr's layout
//! slicing (`i_f`/`i_mut_f`) and its assign/fill. Boolean-mask and
//! integer-array indexing are registered rust-side gaps (G-038/G-039) and
//! are declined here per the wrapper-only rule. Results are fresh owned
//! tensors — the handle model has no shared storage (register G-036).

use core::mem::MaybeUninit;
use num::Complex;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyEllipsis, PyNone, PySlice, PyTuple};
use pyo3::Bound;
use rstsr::prelude::rt;
use rstsr::prelude::*;

use rstsr_common::layout::exports::{Indexer, SliceI};
use rstsr_core::operators::assignment::OpAssignAPI;
use rstsr_core::storage::exports::{DeviceCreationAnyAPI, DeviceRawAPI};

use crate::any_tensor::{dispatch_t, err_py, lift, parse_leaf, type_err, AnyTensor, FTensor, NativeArray, PyScalar};
use crate::creation::ScalarCastTarget;

/* #region key parsing */

enum KeyItem {
    Select(isize),
    Slice(Option<isize>, Option<isize>, Option<isize>),
    NewAxis,
    Ellipsis,
}

fn sl(start: Option<isize>, stop: Option<isize>, step: Option<isize>) -> Indexer {
    Indexer::Slice(SliceI::new(start, stop, step))
}

/// Duck-typed key parse: int / slice / None / Ellipsis. Array keys are
/// declined here (mask/fancy indexing is a registered rust-side gap).
fn parse_key<'py>(key: &Bound<'py, PyTuple>) -> PyResult<Vec<KeyItem>> {
    let mut items = Vec::new();
    for item in key.iter() {
        if item.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err("boolean scalar indices are not supported"));
        }
        if let Ok(i) = item.extract::<i64>() {
            items.push(KeyItem::Select(i as isize));
        } else if let Ok(s) = item.cast::<PySlice>() {
            let g = |v: Bound<'_, PyAny>| -> PyResult<Option<isize>> { v.extract() };
            let (a, b, c) = (g(s.getattr("start")?)?, g(s.getattr("stop")?)?, g(s.getattr("step")?)?);
            items.push(KeyItem::Slice(a, b, c));
        } else if item.is_instance_of::<PyNone>() {
            items.push(KeyItem::NewAxis);
        } else if item.is_instance_of::<PyEllipsis>() {
            items.push(KeyItem::Ellipsis);
        } else if item.extract::<PyRef<'py, NativeArray>>().is_ok() {
            return Err(PyTypeError::new_err(
                "array indexing (boolean mask / integer array) is not provided by rstsr (gap)",
            ));
        } else {
            return Err(PyTypeError::new_err(format!("invalid index element of type {}", item.get_type().name()?)));
        }
    }
    Ok(items)
}

fn to_indexers(items: &[KeyItem]) -> Vec<Indexer> {
    items
        .iter()
        .map(|it| match it {
            KeyItem::Select(i) => Indexer::Select(*i),
            KeyItem::Slice(a, b, c) => sl(*a, *b, *c),
            KeyItem::NewAxis => Indexer::Insert,
            KeyItem::Ellipsis => Indexer::Ellipsis,
        })
        .collect()
}

/* #endregion */

/* #region basic */

fn op_getitem_basic<T>(t: &FTensor<T>, idx: &[Indexer]) -> rt::Result<FTensor<T>>
where
    T: Clone + Send + Sync,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T>,
{
    let view = t.i_f(idx)?;
    Ok(view.into_owned())
}

#[pyfunction]
pub fn getitem_basic(x: &NativeArray, key: &Bound<'_, PyTuple>) -> PyResult<NativeArray> {
    let items = parse_key(key)?;
    let idx = to_indexers(&items);
    Ok(NativeArray { t: dispatch_t!(x.t, op_getitem_basic(&idx))? })
}

/* #endregion */

/* #region setitem */

fn op_setitem_basic<T>(t: &mut FTensor<T>, src: &FTensor<T>, idx: &[Indexer]) -> rt::Result<()>
where
    T: Clone + Send + Sync,
    DeviceFaer:
        DeviceAPI<T, Raw = Vec<T>> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    let mut view = t.i_mut_f(idx)?;
    view.assign_f(src.view())
}

impl NativeArray {
    /// Basic-key assignment; `value` is an array of the same dtype.
    pub fn setitem_basic_arr(&mut self, key: &Bound<'_, PyTuple>, value: &NativeArray) -> PyResult<()> {
        let items = parse_key(key)?;
        let idx = to_indexers(&items);
        macro_rules! arms {
            ($($dv:ident);* $(;)?) => {
                match (&mut self.t, &value.t) {
                    $((AnyTensor::$dv(ref mut a), AnyTensor::$dv(b)) =>
                        err_py(op_setitem_basic(a, b, &idx)),)*
                    _ => type_err(
                        "cross-dtype item assignment is not provided (gap); use matching dtypes",
                    ),
                }
            };
        }
        arms!(Bool; I8; I16; I32; I64; U8; U16; U32; U64; F32; F64; C32; C64)
    }

    /// Basic-key assignment with a Python scalar value.
    pub fn setitem_basic_scalar(&mut self, key: &Bound<'_, PyTuple>, value: PyScalar) -> PyResult<()> {
        let items = parse_key(key)?;
        let idx = to_indexers(&items);
        match &mut self.t {
            AnyTensor::Bool(ref mut a) => {
                let v = <bool as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::I8(ref mut a) => {
                let v = <i8 as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::I16(ref mut a) => {
                let v = <i16 as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::I32(ref mut a) => {
                let v = <i32 as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::I64(ref mut a) => {
                let v = <i64 as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::U8(ref mut a) => {
                let v = <u8 as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::U16(ref mut a) => {
                let v = <u16 as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::U32(ref mut a) => {
                let v = <u32 as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::U64(ref mut a) => {
                let v = <u64 as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::F32(ref mut a) => {
                let v = <f32 as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::F64(ref mut a) => {
                let v = <f64 as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::C32(ref mut a) => {
                let v = <Complex<f32> as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::C64(ref mut a) => {
                let v = <Complex<f64> as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
        }
    }
}

/* #endregion */

/* #region module-level mutation entry points */

#[pyfunction]
pub fn setitem_basic(x: &mut NativeArray, key: &Bound<'_, PyTuple>, value: &NativeArray) -> PyResult<()> {
    x.setitem_basic_arr(key, value)
}

#[pyfunction]
pub fn setitem_scalar(x: &mut NativeArray, key: &Bound<'_, PyTuple>, value: &Bound<'_, PyAny>) -> PyResult<()> {
    let s = parse_leaf(value)?;
    x.setitem_basic_scalar(key, s)
}

/* #endregion */
