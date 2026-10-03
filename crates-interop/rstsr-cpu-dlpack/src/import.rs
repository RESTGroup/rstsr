//! Importing foreign DLPack tensors as read-only rstsr tensors.

use core::mem::ManuallyDrop;
use core::ptr::NonNull;

use dlpack_ffi::{DLManagedTensor, DLManagedTensorVersioned, DLTensor};
use rstsr_common::error::Result;
use rstsr_core::prelude::*;
use rstsr_core::storage::exports::Storage;

use crate::device::check_cpu_device;
use crate::dtype::{DlpackDtype, CODE_BOOL};
use crate::repr::{DataDlpack, DlpackForeignOwner, TensorDlpack};

/// Upper bound on the accepted dimensionality (mirrors NumPy's `NPY_MAXDIMS`).
const MAX_NDIM: i32 = 64;

/// Everything validated before any ownership is taken from the producer.
struct ImportPlan {
    /// Lowest-addressed element of the tensor.
    base: *mut u8,
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
    span_len: usize,
}

/// # Safety
///
/// `dl` must be a `DLTensor` whose `shape`/`strides` pointers are valid for
/// `ndim` entries, as required by the DLPack producer contract.
unsafe fn plan_import<T: DlpackDtype>(dl: &DLTensor) -> Result<ImportPlan> {
    check_cpu_device(dl.device)?;
    T::from_dltype(dl.dtype)?;

    if !(0..=MAX_NDIM).contains(&dl.ndim) {
        return rstsr_raise!(InvalidValue, "DLPack ndim {} is outside the accepted range 0..={MAX_NDIM}", dl.ndim);
    }
    let ndim = dl.ndim as usize;
    if ndim > 0 && dl.shape.is_null() {
        return rstsr_raise!(InvalidValue, "DLPack shape pointer is NULL for a tensor with ndim > 0");
    }

    let mut shape: Vec<usize> = Vec::with_capacity(ndim);
    let mut numel: usize = 1;
    if ndim > 0 {
        let dims = unsafe { core::slice::from_raw_parts(dl.shape, ndim) };
        for &dim in dims {
            if dim < 0 {
                return rstsr_raise!(InvalidValue, "negative DLPack shape entry {dim}");
            }
            let dim = usize::try_from(dim)
                .map_err(|_| rstsr_error!(ValueOutOfRange, "shape entry {dim} does not fit into usize"))?;
            shape.push(dim);
            numel = numel
                .checked_mul(dim)
                .ok_or_else(|| rstsr_error!(ValueOutOfRange, "DLPack shape product overflows usize"))?;
        }
    }

    let mut strides: Vec<isize> = Vec::with_capacity(ndim);
    if ndim > 0 {
        if dl.strides.is_null() {
            // Pre-1.2 producers: NULL means compact row-major.
            let mut acc: isize = 1;
            let mut compact = vec![0isize; ndim];
            for i in (0..ndim).rev() {
                compact[i] = acc;
                let dim = isize::try_from(shape[i])
                    .map_err(|_| rstsr_error!(ValueOutOfRange, "shape entry does not fit into isize"))?;
                acc = acc
                    .checked_mul(dim)
                    .ok_or_else(|| rstsr_error!(ValueOutOfRange, "compact row-major strides overflow isize"))?;
            }
            strides = compact;
        } else {
            let raw = unsafe { core::slice::from_raw_parts(dl.strides, ndim) };
            for &stride in raw {
                strides.push(
                    isize::try_from(stride)
                        .map_err(|_| rstsr_error!(ValueOutOfRange, "stride {stride} does not fit into isize"))?,
                );
            }
        }
    }

    if numel == 0 {
        // DLPack allows (recommends) a NULL data pointer for empty tensors; the fabricated
        // zero-length span still needs a dangling pointer that is aligned for `T`.
        return Ok(ImportPlan {
            base: NonNull::<T>::dangling().as_ptr() as *mut u8,
            shape,
            strides,
            offset: 0,
            span_len: 0,
        });
    }
    if dl.data.is_null() {
        return rstsr_raise!(InvalidValue, "DLPack data pointer is NULL for a non-empty tensor");
    }

    let byte_offset = usize::try_from(dl.byte_offset)
        .map_err(|_| rstsr_error!(ValueOutOfRange, "byte_offset does not fit into usize"))?;
    let start_addr = (dl.data as usize)
        .checked_add(byte_offset)
        .ok_or_else(|| rstsr_error!(ValueOutOfRange, "data + byte_offset overflows usize"))?;
    if start_addr % core::mem::align_of::<T>() != 0 {
        return rstsr_raise!(
            InvalidValue,
            "DLPack data pointer {start_addr:#x} is not aligned for the element type (alignment {})",
            core::mem::align_of::<T>()
        );
    }
    // Advance the pointer itself rather than rebuilding it from `start_addr`, so the span keeps
    // the producer pointer's provenance; the checked address guarantees this cannot wrap.
    let start = dl.data.cast::<u8>().wrapping_add(byte_offset).cast::<T>();

    // Index-space bounds; `min <= 0 <= max` by construction of the sums.
    let mut min: i128 = 0;
    let mut max: i128 = 0;
    for (dim, &stride) in shape.iter().zip(strides.iter()) {
        let extent = *dim as i128 - 1;
        let stride = stride as i128;
        if stride > 0 {
            max += stride * extent;
        } else if stride < 0 {
            min += stride * extent;
        }
    }
    let span_len =
        usize::try_from(max - min + 1).map_err(|_| rstsr_error!(ValueOutOfRange, "DLPack span overflows usize"))?;
    span_len
        .checked_mul(core::mem::size_of::<T>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| rstsr_error!(ValueOutOfRange, "DLPack span is larger than isize::MAX bytes"))?;
    let min_isize =
        isize::try_from(min).map_err(|_| rstsr_error!(ValueOutOfRange, "DLPack lower bound overflows isize"))?;
    let offset = usize::try_from(-min).map_err(|_| rstsr_error!(ValueOutOfRange, "DLPack offset overflows usize"))?;
    // SAFETY: `start + min` is the lowest addressed element of the tensor, so
    // it lies inside the same allocation as every other element.
    let base = unsafe { start.offset(min_isize) } as *mut u8;

    let plan = ImportPlan { base, shape, strides, offset, span_len };
    validate_bool_payload::<T>(&plan)?;
    Ok(plan)
}

/// `kDLBool` promises boolean semantics but not canonical bytes, while Rust's `bool` must be
/// 0 or 1 — so every element readable through the layout is checked while the buffer is still
/// raw bytes (reading it through the tensor would already build an invalid `bool`).
fn validate_bool_payload<T: DlpackDtype>(plan: &ImportPlan) -> Result<()> {
    if T::DLTYPE.code != CODE_BOOL {
        return Ok(());
    }
    let ndim = plan.shape.len();
    let numel: usize = plan.shape.iter().product();
    let mut index = vec![0isize; ndim];
    for _ in 0..numel {
        let elem =
            plan.offset as i128 + index.iter().zip(&plan.strides).map(|(&i, &s)| i as i128 * s as i128).sum::<i128>();
        // SAFETY: `elem` lies inside the span (the plan bounds the layout), and `plan.base` is the
        // lowest addressed element of the producer's allocation, so the byte is readable.
        let byte = unsafe { *plan.base.add(elem as usize) };
        if byte > 1 {
            return rstsr_raise!(
                InvalidValue,
                "kDLBool element {index:?} holds byte {byte}, but a Rust `bool` must be 0 or 1"
            );
        }
        // odometer over the shape (row-major order of the index tuple)
        for dim in (0..ndim).rev() {
            index[dim] += 1;
            if (index[dim] as usize) < plan.shape[dim] {
                break;
            }
            index[dim] = 0;
        }
    }
    Ok(())
}

/// Validate the plan as a rstsr layout **before** ownership is transferred, so
/// that a rejected import leaves the foreign tensor untouched.
fn plan_layout<D>(plan: &ImportPlan) -> Result<Layout<D>>
where
    D: DimAPI,
{
    let ndim = plan.shape.len();
    let shape = <D as TryFrom<Vec<usize>>>::try_from(plan.shape.clone())
        .map_err(|_| rstsr_error!(InvalidValue, "DLPack ndim {ndim} does not match the requested dimensionality"))?;
    let strides = <D::Stride as TryFrom<Vec<isize>>>::try_from(plan.strides.clone())
        .map_err(|_| rstsr_error!(InvalidValue, "DLPack ndim {ndim} does not match the requested dimensionality"))?;
    Layout::new(shape, strides, plan.offset)
}

/// Build the tensor; infallible: the plan and layout were validated above.
fn build_import<T, B, D>(plan: ImportPlan, layout: Layout<D>, owner: DlpackForeignOwner) -> TensorDlpack<T, B, D>
where
    T: DlpackDtype,
    B: DeviceAPI<T, Raw = Vec<T>>,
    D: DimAPI,
{
    let base = plan.base as *mut T;
    // SAFETY: fabricated span over the foreign buffer. It is never dropped or
    // resized (`ManuallyDrop`), the owner keeps it alive, the pointer is
    // aligned, and the span covers `bounds_index()` (checked above and by
    // `Layout::new`, so `TensorDlpack::new` cannot reject it).
    let span = unsafe { ManuallyDrop::new(Vec::from_raw_parts(base, plan.span_len, plan.span_len)) };
    let repr = unsafe { DataDlpack::from_parts(span, owner) };
    let storage = Storage::new(repr, B::default());
    TensorDlpack::<T, B, D>::new(storage, layout)
}

/// Import a `DLManagedTensorVersioned` (DLPack >= 1.0 producer) as a read-only
/// zero-copy tensor.
///
/// On success the returned tensor owns the producer's lifetime: dropping the
/// last handle calls the producer's deleter. On error nothing is freed and the
/// caller (e.g. the capsule holder) still owns the foreign tensor.
///
/// # Safety
///
/// `ptr` must be a valid pointer to a `DLManagedTensorVersioned` produced by a
/// conforming DLPack producer, which the caller must not hand to anything else
/// (in particular it must not call the deleter afterwards). Everything else —
/// version, device, dtype, shape, strides, alignment, bounds, and the payload
/// of `kDLBool` tensors (every element must be 0 or 1) — is validated.
pub unsafe fn from_dlpack_versioned_f<T, B, D>(ptr: *mut DLManagedTensorVersioned) -> Result<TensorDlpack<T, B, D>>
where
    T: DlpackDtype,
    B: DeviceAPI<T, Raw = Vec<T>>,
    D: DimAPI,
{
    let ptr = NonNull::new(ptr).ok_or_else(|| rstsr_error!(InvalidValue, "NULL DLManagedTensorVersioned pointer"))?;
    // SAFETY: caller contract.
    let managed = unsafe { ptr.as_ref() };
    if managed.version.major != 1 {
        return rstsr_raise!(
            UnImplemented,
            "DLPack version {}.{} is not supported (major version must be 1)",
            managed.version.major,
            managed.version.minor
        );
    }
    let dl_tensor = managed.dl_tensor;
    // SAFETY: caller contract on the embedded DLTensor.
    let plan = unsafe { plan_import::<T>(&dl_tensor)? };
    let layout = plan_layout::<D>(&plan)?;
    Ok(build_import::<T, B, D>(plan, layout, DlpackForeignOwner::new_versioned(ptr)))
}

/// Import a legacy `DLManagedTensor` (deprecated, pre-1.0 struct).
///
/// Semantics are the same as [`from_dlpack_versioned_f`], with NumPy's legacy
/// convention: such tensors are always treated as read-only, and a `NULL`
/// strides pointer means compact row-major.
///
/// # Safety
///
/// Same contract as [`from_dlpack_versioned_f`], for `DLManagedTensor`.
pub unsafe fn from_dlpack_legacy_f<T, B, D>(ptr: *mut DLManagedTensor) -> Result<TensorDlpack<T, B, D>>
where
    T: DlpackDtype,
    B: DeviceAPI<T, Raw = Vec<T>>,
    D: DimAPI,
{
    let ptr = NonNull::new(ptr).ok_or_else(|| rstsr_error!(InvalidValue, "NULL DLManagedTensor pointer"))?;
    // SAFETY: caller contract.
    let managed = unsafe { ptr.as_ref() };
    let dl_tensor = managed.dl_tensor;
    // SAFETY: caller contract on the embedded DLTensor.
    let plan = unsafe { plan_import::<T>(&dl_tensor)? };
    let layout = plan_layout::<D>(&plan)?;
    Ok(build_import::<T, B, D>(plan, layout, DlpackForeignOwner::new_legacy(ptr)))
}
