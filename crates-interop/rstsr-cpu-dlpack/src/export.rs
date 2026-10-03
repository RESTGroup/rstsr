//! Exporting rstsr tensors to DLPack.

use core::ffi::c_void;
use core::mem;
use core::mem::MaybeUninit;
use core::ptr::{self, NonNull};

use dlpack_ffi::{
    DLDataType, DLDevice, DLManagedTensorVersioned, DLPackVersion, DLTensor, DLPACK_FLAG_BITMASK_IS_COPIED,
    DLPACK_FLAG_BITMASK_READ_ONLY,
};
use rstsr_common::error::{RSTSRResultAPI, Result};
use rstsr_core::operators::exports::OpAssignAPI;
use rstsr_core::prelude::*;
use rstsr_core::storage::exports::{DeviceCreationAnyAPI, Storage};

use crate::device::DeviceDlpackAPI;
use crate::dtype::DlpackDtype;
use crate::repr::TensorDlpackShared;

/// The allocation the consumer's `DLManagedTensorVersioned` points into.
///
/// `managed` must stay the first field: the deleter recovers this struct from
/// the `DLManagedTensorVersioned` pointer (`#[repr(C)]`).
#[repr(C)]
struct DlpackExportInner<K> {
    managed: DLManagedTensorVersioned,
    shape: Box<[i64]>,
    strides: Box<[i64]>,
    keepalive: K,
}

/// Deleter installed in every export; consumes the allocation exactly once.
///
/// # Safety
///
/// `ptr` must come from `Box::into_raw` on a `DlpackExportInner<K>` of the
/// same `K` and must not have been consumed before.
unsafe extern "C" fn dlpack_export_deleter<K>(ptr: *mut DLManagedTensorVersioned) {
    drop(unsafe { Box::from_raw(ptr as *mut DlpackExportInner<K>) });
}

/// Owned guard around an exported `DLManagedTensorVersioned`.
///
/// It keeps the buffer alive until [`DlpackExport::into_raw`] hands the
/// pointer to a consumer. Dropping the guard without handing off frees
/// everything; after a handoff the consumer must call the `deleter` stored in
/// the struct exactly once.
pub struct DlpackExport {
    ptr: NonNull<DLManagedTensorVersioned>,
}

impl DlpackExport {
    /// The managed tensor as the consumer sees it.
    pub fn managed(&self) -> &DLManagedTensorVersioned {
        // SAFETY: `ptr` is a live allocation owned by `self`.
        unsafe { self.ptr.as_ref() }
    }

    /// The data pointer the consumer will read (`dl_tensor.data`).
    pub fn data_ptr(&self) -> *const c_void {
        self.managed().dl_tensor.data
    }

    /// The DLPack flags of the export.
    pub fn flags(&self) -> u64 {
        self.managed().flags
    }

    /// Hand the pointer to a foreign consumer, transferring ownership.
    pub fn into_raw(self) -> *mut DLManagedTensorVersioned {
        let ptr = self.ptr.as_ptr();
        mem::forget(self);
        ptr
    }
}

impl Drop for DlpackExport {
    fn drop(&mut self) {
        // Take the same path a foreign consumer takes, so both are identical.
        if let Some(deleter) = self.managed().deleter {
            // SAFETY: the deleter is ours and `ptr` has not been handed off.
            unsafe { deleter(self.ptr.as_ptr()) };
        }
    }
}

fn build_export<K>(
    keepalive: K,
    data: *mut c_void,
    device: DLDevice,
    dtype: DLDataType,
    shape: Vec<i64>,
    strides: Vec<i64>,
    flags: u64,
) -> DlpackExport {
    let shape = shape.into_boxed_slice();
    let strides = strides.into_boxed_slice();
    let ndim = shape.len() as i32;
    let mut inner = Box::new(DlpackExportInner {
        managed: DLManagedTensorVersioned {
            version: DLPackVersion { major: 1, minor: 0 },
            manager_ctx: ptr::null_mut(),
            deleter: Some(dlpack_export_deleter::<K>),
            flags,
            dl_tensor: DLTensor {
                data,
                device,
                ndim,
                dtype,
                shape: ptr::null_mut(),
                strides: ptr::null_mut(),
                byte_offset: 0,
            },
        },
        shape,
        strides,
        keepalive,
    });
    if ndim > 0 {
        let shape_ptr = inner.shape.as_mut_ptr();
        let strides_ptr = inner.strides.as_mut_ptr();
        inner.managed.dl_tensor.shape = shape_ptr;
        inner.managed.dl_tensor.strides = strides_ptr;
    }
    // SAFETY: `managed` is the first field of a `#[repr(C)]` struct, so the
    // allocation address is the `DLManagedTensorVersioned` address.
    let ptr = unsafe { NonNull::new_unchecked(Box::into_raw(inner) as *mut DLManagedTensorVersioned) };
    DlpackExport { ptr }
}

fn layout_to_i64<D>(layout: &Layout<D>) -> Result<(Vec<i64>, Vec<i64>)>
where
    D: DimAPI,
{
    let shape_slice: &[usize] = layout.shape().as_ref();
    let mut shape = Vec::with_capacity(shape_slice.len());
    for &dim in shape_slice {
        let dim = i64::try_from(dim).map_err(|_| rstsr_error!(ValueOutOfRange, "shape {dim} does not fit into i64"))?;
        shape.push(dim);
    }
    let stride_slice: &[isize] = layout.stride().as_ref();
    let strides = stride_slice.iter().map(|&s| s as i64).collect();
    Ok((shape, strides))
}

/// Shared by all export entry points: build the managed tensor over `storage`.
fn export_from_storage<R, T, B, D>(storage: Storage<R, T, B>, layout: Layout<D>, flags: u64) -> Result<DlpackExport>
where
    T: DlpackDtype,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceDlpackAPI<T>,
    R: DataAPI<Data = Vec<T>>,
    D: DimAPI,
{
    let ndim = layout.ndim();
    if ndim > i32::MAX as usize {
        return rstsr_raise!(ValueOutOfRange, "tensor has {ndim} dimensions; DLPack allows at most {}", i32::MAX);
    }
    let data = if layout.size() == 0 {
        // DLPack: a zero-size tensor carries a NULL data pointer.
        ptr::null_mut()
    } else {
        // `offset` is bounded by the layout's own bounds check, so the
        // element-zero address stays inside the buffer.
        let base = storage.raw().as_ptr();
        unsafe { base.add(layout.offset()) as *mut c_void }
    };
    let device = storage.device().to_dlpack_device();
    let dtype = T::DLTYPE;
    let (shape, strides) = layout_to_i64(&layout)?;
    Ok(build_export(storage, data, device, dtype, shape, strides, flags))
}

/// Export an owned tensor, transferring ownership to the DLPack consumer.
///
/// No copy: the tensor is moved into the export. The consumer becomes the sole
/// owner for the buffer's whole lifetime, so the export carries
/// `DLPACK_FLAG_BITMASK_IS_COPIED` (a consumer may treat it as writeable).
///
/// # Panics
///
/// - Panics if the tensor cannot be described by DLPack: more than `i32::MAX` dimensions, or a
///   shape entry that does not fit into `i64`.
///
/// For a fallible version, use [`into_dlpack_f`].
pub fn into_dlpack<T, B, D>(tensor: Tensor<T, B, D>) -> DlpackExport
where
    T: DlpackDtype,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceDlpackAPI<T>,
    D: DimAPI,
{
    into_dlpack_f(tensor).rstsr_unwrap()
}

/// Export an owned tensor, transferring ownership to the DLPack consumer.
///
/// See also [`into_dlpack`].
pub fn into_dlpack_f<T, B, D>(tensor: Tensor<T, B, D>) -> Result<DlpackExport>
where
    T: DlpackDtype,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceDlpackAPI<T>,
    D: DimAPI,
{
    let (storage, layout) = tensor.into_raw_parts();
    export_from_storage(storage, layout, DLPACK_FLAG_BITMASK_IS_COPIED as u64)
}

/// Export a shared tensor without giving up access to it.
///
/// Each call clones the `Arc` (no data copy) into a fresh managed tensor, so a
/// Python holder may call `__dlpack__` more than once. The export carries
/// `DLPACK_FLAG_BITMASK_READ_ONLY` because the Rust side may keep reading.
///
/// # Panics
///
/// - Panics if the tensor cannot be described by DLPack (see [`into_dlpack`]).
///
/// For a fallible version, use [`to_dlpack_shared_f`].
///
/// # See also
///
/// - [`into_shared_dlpack_f`](crate::repr::into_shared_dlpack_f): move an owned tensor into the
///   shared representation.
pub fn to_dlpack_shared<T, B, D>(tensor: &TensorDlpackShared<T, B, D>) -> DlpackExport
where
    T: DlpackDtype,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceDlpackAPI<T>,
    D: DimAPI,
{
    to_dlpack_shared_f(tensor).rstsr_unwrap()
}

/// Export a shared tensor without giving up access to it.
///
/// See also [`to_dlpack_shared`].
pub fn to_dlpack_shared_f<T, B, D>(tensor: &TensorDlpackShared<T, B, D>) -> Result<DlpackExport>
where
    T: DlpackDtype,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceDlpackAPI<T>,
    D: DimAPI,
{
    let storage = tensor.storage();
    let layout = tensor.layout();
    // Clone of the representation: bumps the `Arc`, no data copy.
    let repr = (*storage.data()).clone();
    let keepalive = Storage::new(repr, (*storage.device()).clone());
    export_from_storage(keepalive, layout.clone(), DLPACK_FLAG_BITMASK_READ_ONLY as u64)
}

/// Export any tensor or view through a deep copy.
///
/// The safe fallback for layouts whose buffer must stay under the caller's
/// control (e.g. a slice of a bigger tensor): one copy, then the consumer owns
/// the copy (`DLPACK_FLAG_BITMASK_IS_COPIED`).
///
/// # Panics
///
/// - Panics if the copy cannot be described by DLPack (see [`into_dlpack`]).
///
/// For a fallible version, use [`to_dlpack_copy_f`].
pub fn to_dlpack_copy<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> DlpackExport
where
    T: DlpackDtype,
    B: DeviceAPI<T, Raw = Vec<T>>
        + DeviceDlpackAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, D>,
    R: DataAPI<Data = Vec<T>> + DataCloneAPI<Data = Vec<T>>,
    D: DimAPI,
{
    to_dlpack_copy_f(tensor).rstsr_unwrap()
}

/// Export any tensor or view through a deep copy.
///
/// See also [`to_dlpack_copy`].
pub fn to_dlpack_copy_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> Result<DlpackExport>
where
    T: DlpackDtype,
    B: DeviceAPI<T, Raw = Vec<T>>
        + DeviceDlpackAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, D>,
    R: DataAPI<Data = Vec<T>> + DataCloneAPI<Data = Vec<T>>,
    D: DimAPI,
{
    // `to_owned` gathers the visible elements into a fresh contiguous tensor,
    // so the exported copy may have a different (compact) layout.
    let owned = tensor.to_owned();
    into_dlpack_f(owned)
}
