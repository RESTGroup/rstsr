//! Batch walk for stacked (n-dimensional) linalg entries on BLAS devices.
//!
//! A stacked tensor holds one matrix per batch element: the `inner_ndim`
//! per-matrix axes are the last ones under `RowMajor` and the first ones under
//! `ColMajor` (the device default order); every other axis is a batch axis.
//!
//! The walk is a split plus two indexed accessors rather than a `map` family:
//! [`StackLayout`] holds the batch/inner split, and [`stack_slice_at`] /
//! [`stack_slice_at_mut`] hand out the slice of batch element `i`. An entry
//! zips any arity of inputs and outputs with a plain `for i in 0..stack.len()`
//! loop, so no per-arity walker is needed.

use rstsr_core::prelude_dev::*;

/// Split a stacked shape into its batch shape and the `inner_ndim` per-matrix
/// axes. The matrix axes are last under `RowMajor`, first under `ColMajor`.
pub fn stack_split_shape(shape: &[usize], inner_ndim: usize, order: FlagOrder) -> Result<(Vec<usize>, Vec<usize>)> {
    let ndim = shape.len();
    rstsr_assert!(ndim >= inner_ndim, InvalidLayout, "linalg: expected at least {inner_ndim} dimensions, got {ndim}")?;
    Ok(match order {
        RowMajor => (shape[..ndim - inner_ndim].to_vec(), shape[ndim - inner_ndim..].to_vec()),
        ColMajor => (shape[inner_ndim..].to_vec(), shape[..inner_ndim].to_vec()),
    })
}

/// Shape of a stacked result whose per-slice shape is `inner_shape`.
pub fn stack_shape(batch_shape: &[usize], inner_shape: &[usize], order: FlagOrder) -> Vec<usize> {
    match order {
        RowMajor => [batch_shape, inner_shape].concat(),
        ColMajor => [inner_shape, batch_shape].concat(),
    }
}

/// Split a stacked layout into (batch, inner) layouts, offset unchanged.
pub fn stack_split_layout(
    layout: &Layout<IxD>,
    inner_ndim: usize,
    order: FlagOrder,
) -> Result<(Layout<IxD>, Layout<IxD>)> {
    let axis = inner_ndim as isize;
    match order {
        RowMajor => layout.dim_split_at(-axis),
        ColMajor => {
            let (inner, batch) = layout.dim_split_at(axis)?;
            Ok((batch, inner))
        },
    }
}

/// A stacked layout split into its batch axes and per-matrix axes.
#[derive(Clone, Debug)]
pub struct StackLayout {
    batch: Layout<IxD>,
    inner: Layout<IxD>,
    n_batch: usize,
}

impl StackLayout {
    /// Split a stacked layout of `inner_ndim` per-matrix axes.
    pub fn new(layout: &Layout<IxD>, inner_ndim: usize, order: FlagOrder) -> Result<Self> {
        let (batch, inner) = stack_split_layout(layout, inner_ndim, order)?;
        let n_batch = batch.shape().iter().product();
        Ok(Self { batch, inner, n_batch })
    }

    /// The batch shape.
    pub fn batch_shape(&self) -> &[usize] {
        self.batch.shape().as_ref()
    }

    /// The per-matrix shape.
    pub fn inner_shape(&self) -> &[usize] {
        self.inner.shape().as_ref()
    }

    /// Number of batch elements (product of the batch axes).
    pub fn len(&self) -> usize {
        self.n_batch
    }

    /// Whether the stack is empty (a zero-size batch axis).
    pub fn is_empty(&self) -> bool {
        self.n_batch == 0
    }

    /// Layout of the matrix slice of batch element `i`, offset included.
    pub fn slice_layout(&self, i: usize) -> Result<Layout<IxD>> {
        rstsr_assert!(i < self.len(), ValueOutOfRange, "stack: batch index {i} out of range (len {})", self.len())?;
        let off = self.offsets()?.nth(i).unwrap_or(0);
        let mut inner = self.inner.clone();
        // SAFETY: the offset comes from iterating the batch part of the same
        // validated parent layout, so the inner block stays in-bounds.
        unsafe {
            inner.set_offset(off);
        }
        Ok(inner)
    }

    /// Offsets of the matrix slices, in batch order.
    pub fn offsets(&self) -> Result<impl Iterator<Item = usize> + '_> {
        IterLayout::new(&self.batch, TensorIterOrder::C)
    }
}

/// Read-only view of the matrix slice of batch element `i` of `a`.
///
/// `stack` must be `StackLayout::new(a.layout(), ..)`. The slice's storage is
/// rebuilt from `a` (as `view()` builds it), so any view kind is walked
/// without copies; each yielded borrow ends with the enclosing statement.
pub fn stack_slice_at<'a, T, B>(
    a: TensorView<'a, T, B, IxD>,
    stack: &StackLayout,
    i: usize,
) -> Result<TensorView<'a, T, B, IxD>>
where
    B: DeviceAPI<T>,
{
    let layout = stack.slice_layout(i)?;
    let (storage, _) = a.into_raw_parts();
    // SAFETY: `slice_layout` only advances the offset within the parent buffer.
    Ok(unsafe { TensorView::new_unchecked(storage, layout) })
}

/// Mutable view of the matrix slice of batch element `i` of `a`.
///
/// Takes `a` by value (a view handle); the returned slice borrows the same
/// parent data, so the caller re-creates `a` for the next index.
pub fn stack_slice_at_mut<'a, T, B>(
    a: TensorMut<'a, T, B, IxD>,
    stack: &StackLayout,
    i: usize,
) -> Result<TensorMut<'a, T, B, IxD>>
where
    B: DeviceAPI<T>,
{
    let layout = stack.slice_layout(i)?;
    let (storage, _) = a.into_raw_parts();
    // SAFETY: `slice_layout` only advances the offset within the parent buffer.
    Ok(unsafe { TensorMut::new_unchecked(storage, layout) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstsr_core::prelude::*;

    fn device(order: FlagOrder) -> DeviceCpuSerial {
        let mut device = DeviceCpuSerial::default();
        device.set_default_order(order);
        device
    }

    // flat 0..24 in a (2, 3, 4) row-major stack: batch 1 starts at flat 12
    #[test]
    fn row_major_stack_offsets() {
        let device = device(RowMajor);
        let a: Tensor<f64, _, IxD> =
            rt::asarray(((0..24).map(|i| i as f64).collect::<Vec<_>>(), [2, 3, 4].c(), &device));
        let stack = StackLayout::new(a.layout(), 2, RowMajor).unwrap();
        assert_eq!(stack.batch_shape(), &[2]);
        assert_eq!(stack.inner_shape(), &[3, 4]);
        for i in 0..stack.len() {
            let s = stack_slice_at(a.view(), &stack, i).unwrap().into_dim::<Ix2>();
            assert_eq!(s[[0, 0]], 12.0 * i as f64);
            assert_eq!(s[[2, 3]], 12.0 * i as f64 + 11.0);
        }
    }

    // col-major (3, 4, 2): matrix axes first, batch axis last
    #[test]
    fn col_major_stack_offsets() {
        let device = device(ColMajor);
        let a: Tensor<f64, _, IxD> =
            rt::asarray(((0..24).map(|i| i as f64).collect::<Vec<_>>(), [3, 4, 2].f(), &device));
        assert_eq!(a.layout().stride().to_vec(), vec![1, 3, 12]);
        let stack = StackLayout::new(a.layout(), 2, ColMajor).unwrap();
        assert_eq!(stack.batch_shape(), &[2]);
        assert_eq!(stack.inner_shape(), &[3, 4]);
        for i in 0..stack.len() {
            let s = stack_slice_at(a.view(), &stack, i).unwrap().into_dim::<Ix2>();
            assert_eq!(s[[0, 0]], 12.0 * i as f64);
            assert_eq!(s[[2, 3]], 12.0 * i as f64 + 11.0);
        }
    }

    // a column-strided parent is non-contiguous; offsets must still land right
    #[test]
    fn non_contiguous_parent() {
        let device = device(RowMajor);
        let a: Tensor<f64, _, IxD> =
            rt::asarray(((0..48).map(|i| i as f64).collect::<Vec<_>>(), [2, 3, 8].c(), &device));
        let a = a.i((.., .., SliceI::new(None, None, 2)));
        let stack = StackLayout::new(a.layout(), 2, RowMajor).unwrap();
        assert_eq!(stack.inner_shape(), &[3, 4]);
        let s = stack_slice_at(a.view(), &stack, 1).unwrap().into_dim::<Ix2>();
        // element (2, 3) of batch 1 is flat index 24 + 2 * 8 + 3 * 2 = 46
        assert_eq!(s[[2, 3]], 46.0);
    }

    #[test]
    fn mutable_slices_write_through() {
        let device = device(RowMajor);
        let mut a: Tensor<f64, _, IxD> =
            rt::asarray(((0..12).map(|i| i as f64).collect::<Vec<_>>(), [3, 2, 2].c(), &device));
        let stack = StackLayout::new(a.layout(), 2, RowMajor).unwrap();
        for i in 0..stack.len() {
            let mut s = stack_slice_at_mut(a.view_mut(), &stack, i).unwrap().into_dim::<Ix2>();
            s.assign_f(rt::full(([2, 2], i as f64, &device))).unwrap();
        }
        let want: Vec<f64> = (0..12).map(|i| (i / 4) as f64).collect();
        assert_eq!(a.raw().to_vec(), want);
    }
}
