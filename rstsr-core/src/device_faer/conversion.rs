//! Conversion to/from Faer

use crate::prelude_dev::*;
use core::mem::ManuallyDrop;
use faer::prelude::*;
use faer_ext::IntoFaer;

/* #region conversion to Faer objects */

impl<'a, T, B> IntoFaer for TensorView<'a, T, B, Ix2>
where
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    type Faer = MatRef<'a, T>;

    fn into_faer(self) -> Self::Faer {
        let [nrows, ncols] = *self.shape();
        let [row_stride, col_stride] = *self.stride();
        let offset = self.offset();
        // SAFETY: `offset` is within the storage by layout validity (checked when the
        // tensor was constructed).
        let ptr = unsafe { self.raw().as_ptr().add(offset) };
        // SAFETY: strides come from the validated 2-D layout, so faer's element access
        // stays within the slice's allocation; faer only reads through `MatRef`.
        unsafe { MatRef::from_raw_parts(ptr, nrows, ncols, row_stride, col_stride) }
    }
}

impl<'a, T, B> IntoFaer for TensorViewMut<'a, T, B, Ix2>
where
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    type Faer = MatMut<'a, T>;

    fn into_faer(mut self) -> Self::Faer {
        let [nrows, ncols] = *self.shape();
        let [row_stride, col_stride] = *self.stride();
        let offset = self.offset();
        // SAFETY: `offset` within the storage by layout validity; the exclusive `&mut`
        // borrow provides write provenance.
        let ptr = unsafe { self.raw_mut().as_mut_ptr().add(offset) };
        // SAFETY: strides from the validated 2-D layout keep all element access within
        // the slice allocation; exclusive `&mut` borrow backs the write access.
        unsafe { MatMut::from_raw_parts_mut(ptr, nrows, ncols, row_stride, col_stride) }
    }
}

/// Converts a faer matrix view into a non-owning RSTSR tensor view (zero-copy).
///
/// The result borrows faer's buffer (rstsr never frees it) and keeps faer's
/// strides; the source must outlive the view. For an owned [`Mat`], the
/// conversion instead copies; see that impl's documentation.
impl<'a, T> IntoRSTSR for MatRef<'a, T> {
    type RSTSR = TensorView<'a, T, DeviceFaer, Ix2>;

    fn into_rstsr(self) -> Self::RSTSR {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let row_stride = self.row_stride();
        let col_stride = self.col_stride();
        let ptr = self.as_ptr();

        let layout = Layout::new([nrows, ncols], [row_stride, col_stride], 0).unwrap();
        let (_, upper_bound) = layout.bounds_index().unwrap();
        // SAFETY: the Vec is a NON-OWNING handle over faer's buffer: wrapped in
        // `ManuallyDrop` + `DataRef` below, it is never deallocated. `upper_bound` is
        // the validated layout bound and faer guarantees the referenced elements are
        // initialized and live for `'a`.
        let raw = unsafe { Vec::from_raw_parts(ptr as *mut T, upper_bound, upper_bound) };
        let data = DataRef::from_manually_drop(ManuallyDrop::new(raw));
        let storage = Storage::new(data, DeviceFaer::default());
        let tensor = unsafe { TensorView::new_unchecked(storage, layout) };
        return tensor;
    }
}

/// Converts an owned faer matrix into an owning RSTSR tensor by **copying** the
/// elements.
///
/// The result is a [`Tensor`] that owns a fresh, contiguous column-major buffer
/// (stride `[1, nrows]`, offset `0`); faer's original allocation is dropped
/// normally. This differs from the reference conversions ([`MatRef`], [`MatMut`],
/// [`ColRef`]), which are zero-copy, non-owning views that keep faer's strides.
///
/// # Why the owned conversion copies
///
/// The faer buffer cannot be re-homed into a [`Vec`] soundly: for the numeric
/// element types (`f32`, `f64`, complex, ...) faer allocates with 64-byte
/// alignment and pads its row capacity, while a `Vec<T>` always deallocates
/// with `align_of::<T>()` and no padding. Deallocating with a different layout
/// than the original allocation is undefined behavior by the allocator
/// contract (benign under mainstream system allocators, but detected by Miri,
/// and by any custom allocator that switches on the layout in `dealloc`).
/// The elements are therefore copied, and faer deallocates its own buffer.
///
/// # Zero-copy alternative
///
/// If an owning tensor is not required, borrow the matrix first and convert the
/// reference instead: the view form maps faer's buffer in place and never frees
/// it (the source must outlive the view).
///
/// ```rust
/// # use rstsr::prelude::*;
/// # use faer::Mat;
/// let mat = Mat::from_fn(2, 3, |i, j| (i * 3 + j) as f64);
/// let view = mat.as_ref().into_rstsr(); // zero-copy: TensorView borrowing `mat`
/// assert_eq!(view[[0, 2]], 2.0);
/// let tensor = mat.into_rstsr(); // owning: fresh contiguous Tensor (copies)
/// assert_eq!(tensor[[1, 0]], 3.0);
/// ```
impl<T> IntoRSTSR for Mat<T>
where
    T: Clone,
{
    type RSTSR = Tensor<T, DeviceFaer, Ix2>;

    fn into_rstsr(self) -> Self::RSTSR {
        let nrows = self.nrows();
        let ncols = self.ncols();
        // Copy column-by-column (`row_stride` of an owned faer matrix is always 1,
        // so each column is a contiguous, initialized slice). The copy is what
        // keeps this conversion sound; see the impl-level documentation for why
        // the faer allocation itself cannot be adopted by a `Vec`.
        let mut vec = Vec::with_capacity(nrows * ncols);
        for j in 0..ncols {
            vec.extend_from_slice(self.col_as_slice(j));
        }
        let layout = Layout::new([nrows, ncols], [1, nrows as isize], 0).rstsr_unwrap();
        let data = DataOwned::from(vec);
        let storage = Storage::new(data, DeviceFaer::default());
        Tensor::new(storage, layout)
    }
}

/// Converts a faer column view into a non-owning RSTSR tensor view (zero-copy).
///
/// The result borrows faer's buffer (rstsr never frees it) and keeps faer's
/// strides; the source must outlive the view.
impl<'a, T> IntoRSTSR for ColRef<'a, T> {
    type RSTSR = TensorView<'a, T, DeviceFaer, Ix1>;

    fn into_rstsr(self) -> Self::RSTSR {
        let nrows = self.nrows();
        let stride = self.row_stride();
        let ptr = self.as_ptr();

        let layout = Layout::new([nrows], [stride], 0).unwrap();
        let (_, upper_bound) = layout.bounds_index().unwrap();
        // SAFETY: non-owning handle over faer's column buffer, as in `MatRef` above;
        // never deallocated (`ManuallyDrop` + `DataRef`).
        let raw = unsafe { Vec::from_raw_parts(ptr as *mut T, upper_bound, upper_bound) };
        let data = DataRef::from_manually_drop(ManuallyDrop::new(raw));
        let storage = Storage::new(data, DeviceFaer::default());
        let tensor = unsafe { TensorView::new_unchecked(storage, layout) };
        return tensor;
    }
}

/// Converts a faer mutable matrix view into a non-owning mutable RSTSR tensor
/// view (zero-copy).
///
/// The result borrows faer's buffer mutably (rstsr never frees it) and keeps
/// faer's strides; the source must outlive the view.
impl<'a, T> IntoRSTSR for MatMut<'a, T> {
    type RSTSR = TensorViewMut<'a, T, DeviceFaer, Ix2>;

    fn into_rstsr(self) -> Self::RSTSR {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let row_stride = self.row_stride();
        let col_stride = self.col_stride();
        let ptr = self.as_ptr();

        let layout = Layout::new([nrows, ncols], [row_stride, col_stride], 0).unwrap();
        let (_, upper_bound) = layout.bounds_index().unwrap();
        // SAFETY: non-owning handle over faer's buffer with mutable access transferred
        // from the exclusive `MatMut`; never deallocated (`ManuallyDrop` + `DataMut`).
        let raw = unsafe { Vec::from_raw_parts(ptr as *mut T, upper_bound, upper_bound) };
        let data = DataMut::from_manually_drop(ManuallyDrop::new(raw));
        let storage = Storage::new(data, DeviceFaer::default());
        let tensor = unsafe { TensorMut::new_unchecked(storage, layout) };
        return tensor;
    }
}

/* #endregion */

/* #region device conversion */

#[duplicate_item(
    DevA DevB;
   [DeviceFaer     ] [DeviceCpuSerial];
   [DeviceCpuSerial] [DeviceFaer     ];
   [DeviceFaer     ] [DeviceFaer     ];
)]
impl<'a, R, T, D> DeviceChangeAPI<'a, DevB, R, T, D> for DevA
where
    T: Clone + Send + Sync + 'a,
    D: DimAPI,
    R: DataCloneAPI<Data = Vec<T>>,
{
    type Repr = R;
    type ReprTo = DataRef<'a, Vec<T>>;

    fn change_device(tensor: TensorAny<R, T, DevA, D>, device: &DevB) -> Result<TensorAny<Self::Repr, T, DevB, D>> {
        let (storage, layout) = tensor.into_raw_parts();
        let (data, _) = storage.into_raw_parts();
        let storage = Storage::new(data, device.clone());
        let tensor = TensorAny::new(storage, layout);
        Ok(tensor)
    }

    fn into_device(
        tensor: TensorAny<R, T, DevA, D>,
        device: &DevB,
    ) -> Result<TensorAny<DataOwned<Vec<T>>, T, DevB, D>> {
        let tensor = tensor.into_owned();
        DeviceChangeAPI::change_device(tensor, device)
    }

    fn to_device(tensor: &'a TensorAny<R, T, DevA, D>, device: &DevB) -> Result<TensorView<'a, T, DevB, D>> {
        let view = tensor.view();
        DeviceChangeAPI::change_device(view, device)
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_device_conversion() {
        let device_serial = DeviceCpuSerial::default();
        let device_faer = DeviceFaer::new(0);
        let a = linspace((1.0, 5.0, 5, &device_faer));
        let b = a.to_device(&device_serial);
        println!("{b:?}");
        let a = linspace((1.0, 5.0, 5, &device_serial));
        let a_view = a.view();
        let b = a_view.to_device(&device_faer);
        println!("{b:?}");
    }

    #[test]
    fn test_self_conversion() {
        let device_a = DeviceFaer::new(1);
        let device_b = DeviceFaer::new(0);
        let a = linspace((1.0, 5.0, 5, &device_b));
        let b = a.to_device(&device_a);
        println!("{b:?}");
        let a = linspace((1.0, 5.0, 5, &device_a));
        let a_view = a.view();
        let b = a_view.to_device(&device_b);
        println!("{b:?}");
    }

    #[test]
    fn test_mat_owned_into_rstsr() {
        // content preserved; the result owns a fresh contiguous column-major buffer
        let mat = Mat::from_fn(3, 4, |i, j| (i * 4 + j) as f64);
        let tensor = mat.into_rstsr();
        assert_eq!(*tensor.shape(), [3, 4]);
        assert_eq!(*tensor.stride(), [1, 3]);
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(tensor[[i, j]], (i * 4 + j) as f64);
            }
        }
        // nrows=5: faer pads its row capacity (to a multiple of 8 for f64); the
        // conversion must copy exactly the logical elements and nothing else
        let mat = Mat::from_fn(5, 1, |i, _j| i as f64);
        let tensor = mat.into_rstsr();
        assert_eq!(*tensor.shape(), [5, 1]);
        assert_eq!(*tensor.stride(), [1, 5]);
        for i in 0..5 {
            assert_eq!(tensor[[i, 0]], i as f64);
        }
        // zero-copy route from an owned matrix: borrow first, then convert
        let mat = Mat::from_fn(2, 2, |i, j| (i * 2 + j) as f64);
        let view = mat.as_ref().into_rstsr();
        assert_eq!(*view.shape(), [2, 2]);
        assert_eq!(view[[0, 1]], 1.0);
        assert_eq!(view[[1, 0]], 2.0);
    }
}
