//! Export: field mapping, layout handling, ownership handoff.

use dlpack_ffi::{DLPACK_FLAG_BITMASK_IS_COPIED, DLPACK_FLAG_BITMASK_READ_ONLY};
use rstsr_core::prelude::*;
use rstsr_core::storage::exports::{DataOwned, Storage};
use rstsr_cpu_dlpack::{
    into_dlpack_f, into_shared_dlpack_f, to_dlpack_copy_f, to_dlpack_shared_f, DlpackExport, TensorDlpackShared,
};

fn device() -> DeviceCpuSerial {
    DeviceCpuSerial::default()
}

/// Contiguous `[start..start+len)` one-dimensional tensor of `f64`.
fn owned_1d(values: Vec<f64>) -> Tensor<f64, DeviceCpuSerial, IxD> {
    let len = values.len();
    let storage = Storage::new(DataOwned::from(values), device());
    Tensor::new(storage, Layout::new(vec![len], vec![1], 0).unwrap())
}

/// Read the values an export exposes, through its own `DLTensor` layout.
fn read_export(export: &DlpackExport) -> Vec<f64> {
    let dl = &export.managed().dl_tensor;
    let shape: Vec<usize> = (0..dl.ndim as usize).map(|i| unsafe { *dl.shape.add(i) } as usize).collect();
    let strides: Vec<isize> = (0..dl.ndim as usize).map(|i| unsafe { *dl.strides.add(i) } as isize).collect();
    let numel: usize = shape.iter().product();
    if numel == 0 {
        assert!(dl.data.is_null());
        return Vec::new();
    }
    assert!(!dl.data.is_null());
    let base = dl.data as *const f64;
    let mut out = vec![0.0; numel];
    for (flat, out_value) in out.iter_mut().enumerate() {
        // row-major index decomposition of `flat`
        let mut remainder = flat;
        let mut index = 0isize;
        for (dim, stride) in shape.iter().zip(strides.iter()).rev() {
            index += (*stride) * (remainder % dim) as isize;
            remainder /= dim;
        }
        *out_value = unsafe { *base.offset(index) };
    }
    out
}

#[test]
fn shared_export_is_zero_copy_and_read_only() {
    let tensor = owned_1d(vec![0.0, 1.0, 2.0, 3.0]);
    let base = tensor.raw().as_ptr();
    let shared = into_shared_dlpack_f(tensor).unwrap();

    let export = to_dlpack_shared_f(&shared).unwrap();
    let dl = &export.managed().dl_tensor;
    assert_eq!(export.flags(), DLPACK_FLAG_BITMASK_READ_ONLY as u64);
    assert_eq!(export.managed().version.major, 1);
    assert_eq!(export.managed().version.minor, 0);
    assert_eq!(dl.ndim, 1);
    assert_eq!(dl.dtype.code, 2);
    assert_eq!(dl.dtype.bits, 64);
    assert_eq!(dl.byte_offset, 0);
    assert_eq!(dl.data as *const f64, base); // zero copy
    assert_eq!(export.data_ptr() as *const f64, base);
    assert_eq!(read_export(&export), vec![0.0, 1.0, 2.0, 3.0]);

    // repeat export: independent managed tensors over the same buffer
    let second = to_dlpack_shared_f(&shared).unwrap();
    assert_eq!(second.data_ptr(), export.data_ptr());
    drop(export);
    assert_eq!(read_export(&second), vec![0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn strided_and_negative_layouts_export_verbatim() {
    // elements 1 and 3 of the buffer: shape [2], stride [2], offset 1
    let storage = Storage::new(DataOwned::from(vec![0.0, 1.0, 2.0, 3.0]), device());
    let tensor = Tensor::new(storage, Layout::new(vec![2], vec![2], 1).unwrap());
    let base = tensor.raw().as_ptr();
    let shared = into_shared_dlpack_f(tensor).unwrap();

    let export = to_dlpack_shared_f(&shared).unwrap();
    assert_eq!(unsafe { *export.managed().dl_tensor.strides }, 2);
    assert_eq!(export.data_ptr() as *const f64, unsafe { base.add(1) });
    assert_eq!(read_export(&export), vec![1.0, 3.0]);

    // reversed view over the same buffer: shape [4], stride [-1], offset 3
    let reversed_storage = Storage::new((*shared.storage().data()).clone(), device());
    let reversed: TensorDlpackShared<f64, DeviceCpuSerial, IxD> =
        TensorDlpackShared::new_f(reversed_storage, Layout::new(vec![4], vec![-1], 3).unwrap()).unwrap();
    let export = to_dlpack_shared_f(&reversed).unwrap();
    assert_eq!(unsafe { *export.managed().dl_tensor.strides }, -1);
    assert_eq!(read_export(&export), vec![3.0, 2.0, 1.0, 0.0]);
}

#[test]
fn copy_export_gathers_views() {
    let storage = Storage::new(DataOwned::from(vec![0.0, 1.0, 2.0, 3.0]), device());
    let tensor = Tensor::new(storage, Layout::new(vec![2], vec![2], 1).unwrap());

    let export = to_dlpack_copy_f(&tensor).unwrap();
    assert_eq!(export.flags(), DLPACK_FLAG_BITMASK_IS_COPIED as u64);
    let dl = &export.managed().dl_tensor;
    assert_eq!(unsafe { *dl.strides }, 1); // compact after the gathering copy
    assert_eq!(read_export(&export), vec![1.0, 3.0]);
}

#[test]
fn owned_export_transfers_and_marks_sole_ownership() {
    let tensor = owned_1d(vec![7.0, 8.0]);
    let base = tensor.raw().as_ptr();
    let export = into_dlpack_f(tensor).unwrap();
    assert_eq!(export.flags(), DLPACK_FLAG_BITMASK_IS_COPIED as u64);
    assert_eq!(export.data_ptr() as *const f64, base);

    // consumer side: take the pointer, read through it, call the deleter once
    let raw = export.into_raw();
    let data = unsafe { (*raw).dl_tensor.data } as *const f64;
    assert_eq!(unsafe { *data.add(1) }, 8.0);
    let deleter = unsafe { (*raw).deleter }.expect("export must install a deleter");
    unsafe { deleter(raw) };
}

#[test]
fn zero_size_export_uses_null_data() {
    let storage = Storage::new(DataOwned::from(Vec::<f64>::new()), device());
    let tensor = Tensor::new(storage, Layout::new(vec![0], vec![1], 0).unwrap());
    let export = into_dlpack_f(tensor).unwrap();
    assert!(export.managed().dl_tensor.data.is_null());
    assert_eq!(unsafe { *export.managed().dl_tensor.shape }, 0);
}

#[test]
fn dropped_export_frees_without_handoff() {
    let tensor = owned_1d(vec![1.0, 2.0, 3.0]);
    let export = into_dlpack_f(tensor).unwrap();
    assert_eq!(export.flags(), DLPACK_FLAG_BITMASK_IS_COPIED as u64);
    drop(export); // must not leak or double free (run under miri to check)
}
