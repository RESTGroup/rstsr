//! Import: validation matrix, deleter discipline, layout normalisation.

use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use dlpack_ffi::{
    DLDataType, DLDevice, DLDeviceType, DLManagedTensor, DLManagedTensorVersioned, DLPackVersion, DLTensor,
    DLPACK_FLAG_BITMASK_READ_ONLY,
};
use rstsr_core::prelude::*;
use rstsr_cpu_dlpack::dtype::CODE_BOOL;
use rstsr_cpu_dlpack::{from_dlpack_legacy_f, from_dlpack_versioned_f, DlpackDtype, TensorDlpack};

/// Owns the producer's buffers; `calls` observes how often the deleter ran.
struct Owner {
    calls: Arc<AtomicUsize>,
    _data: Box<[f64]>,
    _shape: Box<[i64]>,
    _strides: Option<Box<[i64]>>,
}

unsafe extern "C" fn deleter_versioned(ptr: *mut DLManagedTensorVersioned) {
    let managed = unsafe { Box::from_raw(ptr) };
    let owner = unsafe { Box::from_raw(managed.manager_ctx as *mut Owner) };
    owner.calls.fetch_add(1, Ordering::SeqCst);
    drop(owner);
}

unsafe extern "C" fn deleter_legacy(ptr: *mut DLManagedTensor) {
    let managed = unsafe { Box::from_raw(ptr) };
    let owner = unsafe { Box::from_raw(managed.manager_ctx as *mut Owner) };
    owner.calls.fetch_add(1, Ordering::SeqCst);
    drop(owner);
}

/// Recipe for a hand-made producer tensor.
struct Recipe {
    data: Vec<f64>,
    /// Element offset the data pointer points at (simulates strided producers).
    data_elem_offset: usize,
    /// If set, the data pointer is NULL (only valid for empty tensors).
    null_data: bool,
    byte_offset: u64,
    shape: Vec<i64>,
    /// `None` -> NULL strides pointer (legacy "compact row-major" semantics).
    strides: Option<Vec<i64>>,
    dtype: DLDataType,
    device: DLDevice,
    version: DLPackVersion,
    flags: u64,
}

impl Recipe {
    fn contiguous(shape: Vec<i64>) -> Self {
        let numel: i64 = shape.iter().product();
        Self {
            data: (0..numel).map(|i| i as f64).collect(),
            data_elem_offset: 0,
            null_data: false,
            byte_offset: 0,
            shape,
            strides: None,
            dtype: f64::DLTYPE,
            device: DLDevice { device_type: DLDeviceType::kDLCPU, device_id: 0 },
            version: DLPackVersion { major: 1, minor: 0 },
            flags: 0,
        }
    }
}

fn make_versioned(recipe: Recipe, calls: &Arc<AtomicUsize>) -> *mut DLManagedTensorVersioned {
    let data: Box<[f64]> = recipe.data.into_boxed_slice();
    let shape: Box<[i64]> = recipe.shape.into_boxed_slice();
    let strides: Option<Box<[i64]>> = recipe.strides.map(|s| s.into_boxed_slice());
    let owner = Box::new(Owner { calls: Arc::clone(calls), _data: data, _shape: shape, _strides: strides });

    let data_ptr = if recipe.null_data {
        ptr::null_mut()
    } else {
        // SAFETY: the test recipes keep the offset inside the buffer.
        unsafe { owner._data.as_ptr().add(recipe.data_elem_offset) as *mut c_void }
    };
    let dl_tensor = DLTensor {
        data: data_ptr,
        device: recipe.device,
        ndim: owner._shape.len() as i32,
        dtype: recipe.dtype,
        shape: owner._shape.as_ptr() as *mut i64,
        strides: owner._strides.as_ref().map_or(ptr::null_mut(), |s| s.as_ptr() as *mut i64),
        byte_offset: recipe.byte_offset,
    };
    let managed = Box::new(DLManagedTensorVersioned {
        version: recipe.version,
        manager_ctx: Box::into_raw(owner) as *mut c_void,
        deleter: Some(deleter_versioned),
        flags: recipe.flags,
        dl_tensor,
    });
    Box::into_raw(managed)
}

fn make_legacy(recipe: Recipe, calls: &Arc<AtomicUsize>) -> *mut DLManagedTensor {
    let data: Box<[f64]> = recipe.data.into_boxed_slice();
    let shape: Box<[i64]> = recipe.shape.into_boxed_slice();
    let strides: Option<Box<[i64]>> = recipe.strides.map(|s| s.into_boxed_slice());
    let owner = Box::new(Owner { calls: Arc::clone(calls), _data: data, _shape: shape, _strides: strides });

    let dl_tensor = DLTensor {
        data: unsafe { owner._data.as_ptr().add(recipe.data_elem_offset) as *mut c_void },
        device: recipe.device,
        ndim: owner._shape.len() as i32,
        dtype: recipe.dtype,
        shape: owner._shape.as_ptr() as *mut i64,
        strides: owner._strides.as_ref().map_or(ptr::null_mut(), |s| s.as_ptr() as *mut i64),
        byte_offset: recipe.byte_offset,
    };
    let managed = Box::new(DLManagedTensor {
        dl_tensor,
        manager_ctx: Box::into_raw(owner) as *mut c_void,
        deleter: Some(deleter_legacy),
    });
    Box::into_raw(managed)
}

type Imported = TensorDlpack<f64, DeviceCpuSerial, IxD>;

fn import(ptr: *mut DLManagedTensorVersioned) -> rstsr_common::error::Result<Imported> {
    unsafe { from_dlpack_versioned_f::<f64, DeviceCpuSerial, IxD>(ptr) }
}

/// Logical values in row-major order, read through the tensor's own layout
/// (`.raw()` would be address order, not logical order).
fn values(tensor: &Imported) -> Vec<f64> {
    let shape: Vec<usize> = tensor.layout().shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut out = Vec::with_capacity(numel);
    for flat in 0..numel {
        let mut remainder = flat;
        let mut index = vec![0isize; shape.len()];
        for (dim, idx) in shape.iter().zip(index.iter_mut()).rev() {
            *idx = (remainder % dim) as isize;
            remainder /= dim;
        }
        out.push(tensor.storage().get_index(tensor.layout().index(&index)));
    }
    out
}

fn shape_of(tensor: &Imported) -> Vec<usize> {
    tensor.layout().shape().to_vec()
}

fn strides_of(tensor: &Imported) -> Vec<isize> {
    tensor.layout().stride().to_vec()
}

#[test]
fn contiguous_import_reads_and_frees_once() {
    let calls = Arc::new(AtomicUsize::new(0));
    let ptr = make_versioned(Recipe::contiguous(vec![4]), &calls);
    let tensor = import(ptr).unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(tensor.layout().offset(), 0);
    assert_eq!(strides_of(&tensor), vec![1]);
    assert_eq!(shape_of(&tensor), vec![4]);
    assert_eq!(values(&tensor), vec![0.0, 1.0, 2.0, 3.0]);

    let owned = tensor.to_owned(); // deep copy: does not free the producer
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    drop(owned);
    drop(tensor);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn negative_strides_are_normalised() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut recipe = Recipe::contiguous(vec![4]);
    recipe.data_elem_offset = 3; // data points at the highest-addressed element
    recipe.strides = Some(vec![-1]);
    let ptr = make_versioned(recipe, &calls);

    let tensor = import(ptr).unwrap();
    assert_eq!(tensor.layout().offset(), 3);
    assert_eq!(strides_of(&tensor), vec![-1]);
    assert_eq!(tensor.storage().len(), 4);
    assert_eq!(values(&tensor), vec![3.0, 2.0, 1.0, 0.0]);
    drop(tensor);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn owned_copy_of_a_negative_stride_import_preserves_order() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut recipe = Recipe::contiguous(vec![4]);
    recipe.data_elem_offset = 3;
    recipe.strides = Some(vec![-1]);
    let ptr = make_versioned(recipe, &calls);

    let tensor = import(ptr).unwrap();
    // `raw()` is address order (the whole span), the layout is what makes it a reversed view
    assert_eq!(tensor.raw().to_vec(), vec![0.0, 1.0, 2.0, 3.0]);
    assert_eq!(values(&tensor), vec![3.0, 2.0, 1.0, 0.0]);

    // `to_owned` must own a private buffer with unchanged logical content (it may keep the
    // layout over a copied span instead of gathering — both branches are documented)
    let owned = tensor.to_owned();
    assert!(!ptr::eq(owned.raw().as_ptr(), tensor.raw().as_ptr()));
    let owned_logical: Vec<f64> = (0..4isize).map(|i| owned.storage().get_index(owned.layout().index(&[i]))).collect();
    assert_eq!(owned_logical, vec![3.0, 2.0, 1.0, 0.0]);

    drop(owned);
    drop(tensor);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn byte_offset_is_applied_to_the_pointer() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut recipe = Recipe::contiguous(vec![4]);
    recipe.byte_offset = 3 * 8; // start at element 3
    recipe.strides = Some(vec![-1]);
    let ptr = make_versioned(recipe, &calls);

    let tensor = import(ptr).unwrap();
    assert_eq!(values(&tensor), vec![3.0, 2.0, 1.0, 0.0]);
    drop(tensor);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn strided_2d_import() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut recipe = Recipe::contiguous(vec![2, 3]);
    recipe.data = (0..12).map(|i| i as f64).collect();
    recipe.strides = Some(vec![6, 2]); // (i, j) -> i * 6 + j * 2, max address 10
    let ptr = make_versioned(recipe, &calls);

    let tensor = import(ptr).unwrap();
    assert_eq!(strides_of(&tensor), vec![6, 2]);
    assert_eq!(tensor.storage().len(), 11); // span from address 0 to 10
    assert_eq!(values(&tensor), vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
    drop(tensor);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn legacy_import_null_strides_are_row_major() {
    let calls = Arc::new(AtomicUsize::new(0));
    let ptr = make_legacy(Recipe::contiguous(vec![2, 3]), &calls);

    let tensor = unsafe { from_dlpack_legacy_f::<f64, DeviceCpuSerial, IxD>(ptr) }.unwrap();
    assert!(!tensor.storage().data().owner().is_versioned());
    assert_eq!(tensor.storage().data().owner().flags(), None);
    assert_eq!(strides_of(&tensor), vec![3, 1]);
    assert_eq!(values(&tensor), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    drop(tensor);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn scalar_and_empty_imports() {
    let calls = Arc::new(AtomicUsize::new(0));
    let ptr = make_versioned(Recipe::contiguous(vec![]), &calls);
    let tensor = import(ptr).unwrap();
    assert_eq!(tensor.layout().ndim(), 0);
    assert_eq!(tensor.layout().size(), 1);
    drop(tensor);
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let calls = Arc::new(AtomicUsize::new(0));
    let mut recipe = Recipe::contiguous(vec![0]);
    recipe.null_data = true;
    let ptr = make_versioned(recipe, &calls);
    let tensor = import(ptr).unwrap();
    assert_eq!(tensor.layout().size(), 0);
    drop(tensor);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn versioned_flags_are_recorded() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut recipe = Recipe::contiguous(vec![2]);
    recipe.flags = DLPACK_FLAG_BITMASK_READ_ONLY as u64;
    let ptr = make_versioned(recipe, &calls);
    let tensor = import(ptr).unwrap();
    assert_eq!(tensor.storage().data().owner().flags(), Some(DLPACK_FLAG_BITMASK_READ_ONLY as u64));
    let version = tensor.storage().data().owner().version().unwrap();
    assert_eq!((version.major, version.minor), (1, 0));
    drop(tensor);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// Every rejection must leave the foreign tensor under the caller's control:
/// nothing is freed, and the (single) manual deleter call then runs.
fn assert_rejected(recipe: Recipe) {
    let calls = Arc::new(AtomicUsize::new(0));
    let ptr = make_versioned(recipe, &calls);
    let result = import(ptr);
    assert!(
        result.is_err(),
        "expected rejection, got {result:?}",
        result = result.map(|t| t.layout().shape().to_vec())
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0, "a rejected import must not free the foreign tensor");
    let deleter = unsafe { (*ptr).deleter }.unwrap();
    unsafe { deleter(ptr) };
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn error_matrix() {
    // wrong dtype for the requested Rust type
    let mut recipe = Recipe::contiguous(vec![2]);
    recipe.dtype = DLDataType { code: 2, bits: 32, lanes: 1 };
    assert_rejected(recipe);

    // vector lanes
    let mut recipe = Recipe::contiguous(vec![2]);
    recipe.dtype = DLDataType { code: 2, bits: 64, lanes: 2 };
    assert_rejected(recipe);

    // sub-byte float dtype
    let mut recipe = Recipe::contiguous(vec![2]);
    recipe.dtype = DLDataType { code: 10, bits: 8, lanes: 1 };
    assert_rejected(recipe);

    // non-CPU device
    let mut recipe = Recipe::contiguous(vec![2]);
    recipe.device = DLDevice { device_type: DLDeviceType::kDLCUDA, device_id: 0 };
    assert_rejected(recipe);

    // unsupported major version
    let mut recipe = Recipe::contiguous(vec![2]);
    recipe.version = DLPackVersion { major: 2, minor: 0 };
    assert_rejected(recipe);

    // negative shape entry
    let mut recipe = Recipe::contiguous(vec![2]);
    recipe.shape = vec![-2];
    assert_rejected(recipe);

    // NULL data for a non-empty tensor
    let mut recipe = Recipe::contiguous(vec![2]);
    recipe.null_data = true;
    assert_rejected(recipe);

    // unaligned data pointer
    let mut recipe = Recipe::contiguous(vec![2]);
    recipe.byte_offset = 1;
    assert_rejected(recipe);

    // overlapping layout (strides that rstsr forbids)
    let mut recipe = Recipe::contiguous(vec![2, 2]);
    recipe.strides = Some(vec![1, 1]);
    assert_rejected(recipe);

    // stride arithmetic overflow
    let mut recipe = Recipe::contiguous(vec![2]);
    recipe.strides = Some(vec![i64::MAX]);
    assert_rejected(recipe);
}

#[test]
fn view_outlives_nothing_it_should_not() {
    // The imported tensor keeps the producer alive: values are readable right
    // up to the drop that triggers the deleter.
    let calls = Arc::new(AtomicUsize::new(0));
    let ptr = make_versioned(Recipe::contiguous(vec![3]), &calls);
    let tensor = import(ptr).unwrap();
    let owned = tensor.to_owned();
    drop(tensor);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(owned.raw().to_vec(), vec![0.0, 1.0, 2.0]);
}

// ---- `kDLBool`: DLPack promises boolean semantics, Rust `bool` requires 0/1 ----

/// Producer owning byte payloads (`kDLBool` elements are one byte each).
struct OwnerBytes {
    calls: Arc<AtomicUsize>,
    _data: Box<[u8]>,
    _shape: Box<[i64]>,
    _strides: Option<Box<[i64]>>,
}

unsafe extern "C" fn deleter_bytes(ptr: *mut DLManagedTensorVersioned) {
    let managed = unsafe { Box::from_raw(ptr) };
    let owner = unsafe { Box::from_raw(managed.manager_ctx as *mut OwnerBytes) };
    owner.calls.fetch_add(1, Ordering::SeqCst);
    drop(owner);
}

fn make_versioned_bool(
    bytes: Vec<u8>,
    shape: Vec<i64>,
    strides: Option<Vec<i64>>,
    calls: &Arc<AtomicUsize>,
) -> *mut DLManagedTensorVersioned {
    let data: Box<[u8]> = bytes.into_boxed_slice();
    let shape: Box<[i64]> = shape.into_boxed_slice();
    let strides: Option<Box<[i64]>> = strides.map(|s| s.into_boxed_slice());
    let owner = Box::new(OwnerBytes { calls: Arc::clone(calls), _data: data, _shape: shape, _strides: strides });
    let dl_tensor = DLTensor {
        data: owner._data.as_ptr() as *mut c_void,
        device: DLDevice { device_type: DLDeviceType::kDLCPU, device_id: 0 },
        ndim: owner._shape.len() as i32,
        dtype: DLDataType { code: CODE_BOOL, bits: 8, lanes: 1 },
        shape: owner._shape.as_ptr() as *mut i64,
        strides: owner._strides.as_ref().map_or(ptr::null_mut(), |s| s.as_ptr() as *mut i64),
        byte_offset: 0,
    };
    let managed = Box::new(DLManagedTensorVersioned {
        version: DLPackVersion { major: 1, minor: 0 },
        manager_ctx: Box::into_raw(owner) as *mut c_void,
        deleter: Some(deleter_bytes),
        flags: 0,
        dl_tensor,
    });
    Box::into_raw(managed)
}

#[test]
fn canonical_kdlbool_imports_zero_copy() {
    let calls = Arc::new(AtomicUsize::new(0));
    let ptr = make_versioned_bool(vec![1, 0, 1, 1], vec![4], None, &calls);
    let tensor = unsafe { from_dlpack_versioned_f::<bool, DeviceCpuSerial, IxD>(ptr) }.unwrap();
    assert_eq!(tensor.raw().to_vec(), vec![true, false, true, true]);
    drop(tensor);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn non_canonical_kdlbool_payload_is_rejected() {
    // 2/255 are no valid Rust `bool`s; NumPy can produce such arrays (e.g. `.view(np.bool_)`),
    // so the import must reject them instead of materialising invalid values later.
    let calls = Arc::new(AtomicUsize::new(0));
    let ptr = make_versioned_bool(vec![2, 0, 255, 1], vec![4], None, &calls);
    let result = unsafe { from_dlpack_versioned_f::<bool, DeviceCpuSerial, IxD>(ptr) };
    assert!(result.is_err(), "non-canonical kDLBool bytes must be rejected");
    assert_eq!(calls.load(Ordering::SeqCst), 0, "a rejected import must not free the foreign tensor");
    let deleter = unsafe { (*ptr).deleter }.unwrap();
    unsafe { deleter(ptr) };
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn kdlbool_validation_follows_the_layout_not_the_span() {
    // stride 2: the visible bytes are 1, 0, 1; the in-between bytes are not part of the tensor
    // and must not be validated (a span-wide scan would wrongly reject them)
    let calls = Arc::new(AtomicUsize::new(0));
    let ptr = make_versioned_bool(vec![1, 9, 0, 9, 1, 9], vec![3], Some(vec![2]), &calls);
    let tensor = unsafe { from_dlpack_versioned_f::<bool, DeviceCpuSerial, IxD>(ptr) }.unwrap();
    assert_eq!(tensor.layout().stride().to_vec(), vec![2]);
    assert!(!tensor.storage().get_index(tensor.layout().index(&[1isize])));
    drop(tensor);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
