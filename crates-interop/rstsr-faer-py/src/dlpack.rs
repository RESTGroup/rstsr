//! DLPack capsule bridge over `rt::dlpack` (rstsr-cpu-dlpack).
//!
//! Export: deep copy into a `dltensor_versioned` capsule (IS_COPIED); the
//! capsule destructor runs the producer deleter exactly once.
//! Import: copy-only — the capsule is adopted as rstsr-cpu-dlpack's read-only
//! view (which takes over the deleter), then gathered into an owned tensor.

use std::ffi::CStr;
use std::mem::MaybeUninit;
use std::ptr::NonNull;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods};
use pyo3::Bound;

use rstsr::prelude::rt::dlpack;
use rstsr::prelude::*;

use rstsr_core::operators::exports::OpAssignAPI;
use rstsr_core::storage::exports::DeviceCreationAnyAPI;

use dlpack_ffi::{DLDataTypeCode, DLManagedTensor, DLManagedTensorVersioned};

use crate::any_tensor::{err_py, lift, type_err, AnyTensor, FTensor, NativeArray};

const NAME_VERSIONED: &CStr = c"dltensor_versioned";
const NAME_LEGACY: &CStr = c"dltensor";
const NAME_USED_VERSIONED: &CStr = c"used_dltensor_versioned";
const NAME_USED_LEGACY: &CStr = c"used_dltensor";

// ---------------------------------------------------------------- export ---

fn export_ptr<T>(t: &FTensor<T>) -> rt::Result<*mut DLManagedTensorVersioned>
where
    T: dlpack::DlpackDtype,
    DeviceFaer:
        DeviceAPI<T, Raw = Vec<T>> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    // Copy export: no aliasing contract, IS_COPIED flag set by the crate.
    Ok(dlpack::to_dlpack_copy_f(t)?.into_raw())
}

/// Capsule destructor: canonical DLPack protocol — only an UNCONSUMED capsule
/// (still named `dltensor_versioned`) is cleaned up here. A consumer signals
/// takeover by renaming to `used_dltensor_versioned`, and then owns the
/// deleter call (numpy frees via its internal base capsule); running the
/// deleter in both places is a double free.
unsafe extern "C" fn capsule_drop(capsule: *mut pyo3::ffi::PyObject) {
    unsafe {
        if pyo3::ffi::PyCapsule_IsValid(capsule, NAME_VERSIONED.as_ptr()) == 0 {
            return;
        }
        let p = pyo3::ffi::PyCapsule_GetPointer(capsule, NAME_VERSIONED.as_ptr()) as *mut DLManagedTensorVersioned;
        if p.is_null() {
            return;
        }
        if let Some(d) = (*p).deleter {
            d(p);
        }
    }
}

/// Build the versioned capsule for `__dlpack__`.
#[pyfunction]
pub fn dlpack_export<'py>(py: Python<'py>, t: &NativeArray) -> PyResult<Bound<'py, PyCapsule>> {
    let raw: *mut DLManagedTensorVersioned = match &t.t {
        AnyTensor::Bool(t) => err_py(export_ptr(t))?,
        AnyTensor::I8(t) => err_py(export_ptr(t))?,
        AnyTensor::I16(t) => err_py(export_ptr(t))?,
        AnyTensor::I32(t) => err_py(export_ptr(t))?,
        AnyTensor::I64(t) => err_py(export_ptr(t))?,
        AnyTensor::U8(t) => err_py(export_ptr(t))?,
        AnyTensor::U16(t) => err_py(export_ptr(t))?,
        AnyTensor::U32(t) => err_py(export_ptr(t))?,
        AnyTensor::U64(t) => err_py(export_ptr(t))?,
        AnyTensor::F32(t) => err_py(export_ptr(t))?,
        AnyTensor::F64(t) => err_py(export_ptr(t))?,
        AnyTensor::C32(t) => err_py(export_ptr(t))?,
        AnyTensor::C64(t) => err_py(export_ptr(t))?,
    };
    let ptr = NonNull::new(raw).ok_or_else(|| PyValueError::new_err("DLPack export produced a NULL pointer"))?;
    // SAFETY: `ptr` comes from `DlpackExport::into_raw` (valid heap box,
    // deleter set); the capsule owns it and `capsule_drop` releases it.
    unsafe {
        PyCapsule::new_with_pointer_and_destructor(
            py,
            ptr.cast::<std::ffi::c_void>(),
            NAME_VERSIONED,
            Some(capsule_drop),
        )
    }
}

// ---------------------------------------------------------------- import ---

enum CapPtr {
    Versioned(*mut DLManagedTensorVersioned),
    Legacy(*mut DLManagedTensor),
}

fn import_owned<T>(p: CapPtr) -> rt::Result<FTensor<T>>
where
    T: dlpack::DlpackDtype,
    DeviceFaer:
        DeviceAPI<T, Raw = Vec<T>> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    // The imported view owns the producer deleter; `to_owned` gathers the
    // elements into a fresh owned tensor, and dropping the view runs the
    // deleter — exactly once, since the capsule destructor is neutralized
    // by the caller before this point.
    let td = match p {
        CapPtr::Versioned(p) => unsafe { dlpack::from_dlpack_versioned_f::<T, DeviceFaer, IxD>(p)? },
        CapPtr::Legacy(p) => unsafe { dlpack::from_dlpack_legacy_f::<T, DeviceFaer, IxD>(p)? },
    };
    Ok(td.to_owned())
}

#[pyfunction]
pub fn dlpack_import(capsule: &Bound<'_, PyCapsule>) -> PyResult<NativeArray> {
    let raw_name = unsafe { pyo3::ffi::PyCapsule_GetName(capsule.as_ptr()) };
    if raw_name.is_null() {
        return type_err("from_dlpack: object is not a named DLPack capsule");
    }
    let name = unsafe { CStr::from_ptr(raw_name) };
    let (versioned, check_name) = if name == NAME_VERSIONED {
        (true, NAME_VERSIONED)
    } else if name == NAME_LEGACY {
        (false, NAME_LEGACY)
    } else if name == NAME_USED_VERSIONED || name == NAME_USED_LEGACY {
        return Err(PyValueError::new_err("from_dlpack: DLPack capsule has already been consumed"));
    } else {
        return type_err(format!("from_dlpack: not a DLPack capsule (name {:?})", name.to_string_lossy()));
    };

    let ptr = capsule.pointer_checked(Some(check_name))?;

    // Take ownership away from the capsule: the imported tensor's owner will
    // run the producer deleter on drop, so the capsule must not do it too.
    unsafe {
        pyo3::ffi::PyCapsule_SetDestructor(capsule.as_ptr(), None);
        pyo3::ffi::PyCapsule_SetName(
            capsule.as_ptr(),
            if versioned { NAME_USED_VERSIONED.as_ptr() } else { NAME_USED_LEGACY.as_ptr() },
        );
    }

    let dtype = if versioned {
        unsafe { (*(ptr.as_ptr() as *mut DLManagedTensorVersioned)).dl_tensor.dtype }
    } else {
        unsafe { (*(ptr.as_ptr() as *mut DLManagedTensor)).dl_tensor.dtype }
    };
    if dtype.lanes != 1 {
        return type_err(format!("from_dlpack: dtype lanes = {} is not supported (only scalar lanes)", dtype.lanes));
    }
    let code = dtype.code;
    let bits = dtype.bits;
    let k = |c: DLDataTypeCode| c.0 as u8;
    let t = if versioned {
        let p = ptr.as_ptr() as *mut DLManagedTensorVersioned;
        match (code, bits) {
            (c, 8) if c == k(DLDataTypeCode::kDLBool) => {
                lift(import_owned::<bool>(CapPtr::Versioned(p)), AnyTensor::Bool)
            },
            (c, 8) if c == k(DLDataTypeCode::kDLInt) => lift(import_owned::<i8>(CapPtr::Versioned(p)), AnyTensor::I8),
            (c, 16) if c == k(DLDataTypeCode::kDLInt) => {
                lift(import_owned::<i16>(CapPtr::Versioned(p)), AnyTensor::I16)
            },
            (c, 32) if c == k(DLDataTypeCode::kDLInt) => {
                lift(import_owned::<i32>(CapPtr::Versioned(p)), AnyTensor::I32)
            },
            (c, 64) if c == k(DLDataTypeCode::kDLInt) => {
                lift(import_owned::<i64>(CapPtr::Versioned(p)), AnyTensor::I64)
            },
            (c, 8) if c == k(DLDataTypeCode::kDLUInt) => lift(import_owned::<u8>(CapPtr::Versioned(p)), AnyTensor::U8),
            (c, 16) if c == k(DLDataTypeCode::kDLUInt) => {
                lift(import_owned::<u16>(CapPtr::Versioned(p)), AnyTensor::U16)
            },
            (c, 32) if c == k(DLDataTypeCode::kDLUInt) => {
                lift(import_owned::<u32>(CapPtr::Versioned(p)), AnyTensor::U32)
            },
            (c, 64) if c == k(DLDataTypeCode::kDLUInt) => {
                lift(import_owned::<u64>(CapPtr::Versioned(p)), AnyTensor::U64)
            },
            (c, 16) if c == k(DLDataTypeCode::kDLFloat) => {
                return type_err("from_dlpack: float16 is not represented in this shim (G-033)")
            },
            (c, 32) if c == k(DLDataTypeCode::kDLFloat) => {
                lift(import_owned::<f32>(CapPtr::Versioned(p)), AnyTensor::F32)
            },
            (c, 64) if c == k(DLDataTypeCode::kDLFloat) => {
                lift(import_owned::<f64>(CapPtr::Versioned(p)), AnyTensor::F64)
            },
            (c, 64) if c == k(DLDataTypeCode::kDLComplex) => {
                lift(import_owned::<num::Complex<f32>>(CapPtr::Versioned(p)), AnyTensor::C32)
            },
            (c, 128) if c == k(DLDataTypeCode::kDLComplex) => {
                lift(import_owned::<num::Complex<f64>>(CapPtr::Versioned(p)), AnyTensor::C64)
            },
            (c, b) => return type_err(format!("from_dlpack: unsupported DLPack dtype (code {c}, bits {b})")),
        }
    } else {
        let p = ptr.as_ptr() as *mut DLManagedTensor;
        match (code, bits) {
            (c, 8) if c == k(DLDataTypeCode::kDLBool) => lift(import_owned::<bool>(CapPtr::Legacy(p)), AnyTensor::Bool),
            (c, 8) if c == k(DLDataTypeCode::kDLInt) => lift(import_owned::<i8>(CapPtr::Legacy(p)), AnyTensor::I8),
            (c, 16) if c == k(DLDataTypeCode::kDLInt) => lift(import_owned::<i16>(CapPtr::Legacy(p)), AnyTensor::I16),
            (c, 32) if c == k(DLDataTypeCode::kDLInt) => lift(import_owned::<i32>(CapPtr::Legacy(p)), AnyTensor::I32),
            (c, 64) if c == k(DLDataTypeCode::kDLInt) => lift(import_owned::<i64>(CapPtr::Legacy(p)), AnyTensor::I64),
            (c, 8) if c == k(DLDataTypeCode::kDLUInt) => lift(import_owned::<u8>(CapPtr::Legacy(p)), AnyTensor::U8),
            (c, 16) if c == k(DLDataTypeCode::kDLUInt) => lift(import_owned::<u16>(CapPtr::Legacy(p)), AnyTensor::U16),
            (c, 32) if c == k(DLDataTypeCode::kDLUInt) => lift(import_owned::<u32>(CapPtr::Legacy(p)), AnyTensor::U32),
            (c, 64) if c == k(DLDataTypeCode::kDLUInt) => lift(import_owned::<u64>(CapPtr::Legacy(p)), AnyTensor::U64),
            (c, 16) if c == k(DLDataTypeCode::kDLFloat) => {
                return type_err("from_dlpack: float16 is not represented in this shim (G-033)")
            },
            (c, 32) if c == k(DLDataTypeCode::kDLFloat) => lift(import_owned::<f32>(CapPtr::Legacy(p)), AnyTensor::F32),
            (c, 64) if c == k(DLDataTypeCode::kDLFloat) => lift(import_owned::<f64>(CapPtr::Legacy(p)), AnyTensor::F64),
            (c, 64) if c == k(DLDataTypeCode::kDLComplex) => {
                lift(import_owned::<num::Complex<f32>>(CapPtr::Legacy(p)), AnyTensor::C32)
            },
            (c, 128) if c == k(DLDataTypeCode::kDLComplex) => {
                lift(import_owned::<num::Complex<f64>>(CapPtr::Legacy(p)), AnyTensor::C64)
            },
            (c, b) => return type_err(format!("from_dlpack: unsupported DLPack dtype (code {c}, bits {b})")),
        }
    };
    Ok(NativeArray { t: t? })
}
