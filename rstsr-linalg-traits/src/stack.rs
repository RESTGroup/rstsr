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

use num::Num;
use rstsr_blas_traits::prelude::*;
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
    let ndim = layout.ndim();
    rstsr_assert!(ndim >= inner_ndim, InvalidLayout, "linalg: expected at least {inner_ndim} dimensions, got {ndim}")?;
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

/// Split the right-hand side of a batched solve: `b` is `(batch_b, M, K)` or
/// the vector form `(batch_b, M)`, relative to `a`'s rank. Returns the batch
/// shape, the number of right-hand sides (1 for the vector form) and whether
/// `b` is a vector.
fn split_rhs(b: &[usize], ndim_a: usize, m: usize, order: FlagOrder, op: &str) -> Result<(Vec<usize>, usize, bool)> {
    let ndim_b = b.len();
    rstsr_assert!(ndim_b >= 1, InvalidLayout, "{op}: b must have at least 1 dimension")?;
    // the vector form is relative to a: b has one fewer inner axis than a
    let inner_ndim_b = ndim_b.checked_add(2).and_then(|v| v.checked_sub(ndim_a));
    rstsr_assert!(
        matches!(inner_ndim_b, Some(1..=2)),
        InvalidLayout,
        "{op}: b must be (..., M, K) or (..., M) for a's rank, got b of rank {ndim_b} for a of rank {ndim_a}"
    )?;
    let is_vec = inner_ndim_b == Some(1);
    let (batch_b, inner_b) = stack_split_shape(b, 2 - is_vec as usize, order)?;
    rstsr_assert_eq!(inner_b[0], m, InvalidLayout, "{op}: b's row dimension must match a, got {} vs {m}", inner_b[0])?;
    let k = if is_vec { 1 } else { inner_b[1] };
    Ok((batch_b, k, is_vec))
}

/// Broadcast plan for a batched solve: `a` is `(batch_a, M, M)` and `b` is
/// `(batch_b, M, K)` or the vector form `(M,)`; the two batch shapes broadcast
/// to `batch_out`. Matrix axes follow `order` (last two under `RowMajor`,
/// first two under `ColMajor`).
#[derive(Clone, Debug)]
pub struct SolvePlan {
    pub batch_out: Vec<usize>,
    pub m: usize,
    pub k: usize,
    pub is_vec: bool,
}

impl SolvePlan {
    /// Shape of the broadcast solution: `batch_out ++ [M, K]`, or
    /// `batch_out ++ [M]` for the vector form, per `order`.
    pub fn out_shape(&self, order: FlagOrder) -> Vec<usize> {
        let inner = if self.is_vec { vec![self.m] } else { vec![self.m, self.k] };
        stack_shape(&self.batch_out, &inner, order)
    }

    /// Shape of the broadcast matrix stack: `batch_out ++ [M, M]`, per `order`.
    pub fn a_shape(&self, order: FlagOrder) -> Vec<usize> {
        stack_shape(&self.batch_out, &[self.m, self.m], order)
    }
}

/// Validate shapes of a batched solve and plan its broadcast.
pub fn solve_plan(a: &[usize], b: &[usize], order: FlagOrder, op: &str) -> Result<SolvePlan> {
    let ndim_a = a.len();
    rstsr_assert!(ndim_a >= 2, InvalidLayout, "{op}: a must be at least 2-D, got {ndim_a}")?;
    let (batch_a, inner_a) = stack_split_shape(a, 2, order)?;
    rstsr_assert_eq!(
        inner_a[0],
        inner_a[1],
        InvalidLayout,
        "{op}: a must be square, got {}x{}",
        inner_a[0],
        inner_a[1]
    )?;
    let m = inner_a[0];
    let (batch_b, k, is_vec) = split_rhs(b, ndim_a, m, order, op)?;
    let batch_out = broadcast_shapes_f(&[batch_a, batch_b], order)?;
    Ok(SolvePlan { batch_out, m, k, is_vec })
}

/// Broadcast plan for a batched least-squares solve: `a` is `(batch_a, M, N)`
/// and `b` is `(batch_b, M, K)` or the vector form `(batch_b, M)`; the two
/// batch shapes broadcast to `batch_out`. Matrix axes follow `order` (last two
/// under `RowMajor`, first two under `ColMajor`).
#[derive(Clone, Debug)]
pub struct LstsqPlan {
    pub batch_out: Vec<usize>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub is_vec: bool,
}

impl LstsqPlan {
    /// Shape of the broadcast solution: `batch_out ++ [N, K]`, or
    /// `batch_out ++ [N]` for the vector form, per `order`.
    pub fn x_shape(&self, order: FlagOrder) -> Vec<usize> {
        let inner = if self.is_vec { vec![self.n] } else { vec![self.n, self.k] };
        stack_shape(&self.batch_out, &inner, order)
    }

    /// Shape of the singular values: `batch_out ++ [min(M, N)]`, per `order`.
    pub fn s_shape(&self, order: FlagOrder) -> Vec<usize> {
        stack_shape(&self.batch_out, &[self.m.min(self.n)], order)
    }

    /// Shape of the per-slice numerical ranks: the batch shape.
    pub fn rank_shape(&self) -> Vec<usize> {
        self.batch_out.clone()
    }

    /// Shape of the residual sums: `batch_out ++ [K]` for a matrix `b`, the
    /// batch shape for a vector `b`, and empty when `M <= N`.
    pub fn residual_shape(&self, order: FlagOrder) -> Vec<usize> {
        if self.m <= self.n {
            stack_shape(&self.batch_out, &[0], order)
        } else if self.is_vec {
            self.batch_out.clone()
        } else {
            stack_shape(&self.batch_out, &[self.k], order)
        }
    }

    /// Shape of the broadcast matrix stack: `batch_out ++ [M, N]`, per `order`.
    pub fn a_shape(&self, order: FlagOrder) -> Vec<usize> {
        stack_shape(&self.batch_out, &[self.m, self.n], order)
    }

    /// Shape of the broadcast right-hand-side stack: `batch_out ++ [M, K]`, or
    /// `batch_out ++ [M]` for the vector form, per `order`.
    pub fn b_shape(&self, order: FlagOrder) -> Vec<usize> {
        let inner = if self.is_vec { vec![self.m] } else { vec![self.m, self.k] };
        stack_shape(&self.batch_out, &inner, order)
    }
}

/// Validate shapes of a batched least-squares solve and plan its broadcast.
pub fn lstsq_plan(a: &[usize], b: &[usize], order: FlagOrder, op: &str) -> Result<LstsqPlan> {
    let ndim_a = a.len();
    rstsr_assert!(ndim_a >= 2, InvalidLayout, "{op}: a must be at least 2-D, got {ndim_a}")?;
    let (batch_a, inner_a) = stack_split_shape(a, 2, order)?;
    let (m, n) = (inner_a[0], inner_a[1]);
    let (batch_b, k, is_vec) = split_rhs(b, ndim_a, m, order, op)?;
    let batch_out = broadcast_shapes_f(&[batch_a, batch_b], order)?;
    Ok(LstsqPlan { batch_out, m, n, k, is_vec })
}

/// Solve every system of the planned stack: run `kernel` on the matrix slice
/// of `a` (broadcast to the plan) and the matching solution buffer.
///
/// The whole `b` goes through `overwritable_convert_with_order` once: an
/// immutable input gets a freshly allocated solution; a mutable input is
/// solved in place when its batch fits the broadcast batch (the in-place
/// solve cannot grow), or in a contiguous work buffer otherwise. The vector
/// form `(M,)` is solved as one `(M, 1)` column per slice.
pub fn solve_stack_f<'b, T, B, F>(
    a: TensorView<'_, T, B, IxD>,
    b: TensorReference<'b, T, B, IxD>,
    plan: &SolvePlan,
    order: FlagOrder,
    mut kernel: F,
) -> Result<TensorMutable<'b, T, B, IxD>>
where
    T: Num + Clone,
    B: DeviceAPI<T, Raw = Vec<T>>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationNumAPI<T>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
    F: FnMut(TensorView<'_, T, B, Ix2>, TensorMut<'_, T, B, Ix2>) -> Result<()>,
{
    rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch, "solve: a and b are on different devices")?;

    let a_full = a.broadcast_to(plan.a_shape(order));
    let a_stack = StackLayout::new(a_full.layout(), 2, order)?;

    let b_is_ref = b.is_ref();
    if b_is_ref {
        // the solution cannot fit into an immutable input: allocate it and
        // materialize the (broadcast) right-hand side into the buffer — b is
        // both the solve's input and its output
        let mut out: Tensor<T, B, IxD> = zeros_f((plan.out_shape(order), a.device()))?;
        out.assign_f(b.broadcast_to(plan.out_shape(order)))?;
        let out_stack = StackLayout::new(out.layout(), if plan.is_vec { 1 } else { 2 }, order)?;
        for i in 0..a_stack.len() {
            let a_slice = stack_slice_at(a_full.view(), &a_stack, i)?.into_dim::<Ix2>();
            let mut b_slice = stack_slice_at_mut(out.view_mut(), &out_stack, i)?;
            if plan.is_vec {
                kernel(a_slice, b_slice.i_mut((.., None)).into_dim::<Ix2>())?;
            } else {
                kernel(a_slice, b_slice.into_dim::<Ix2>())?;
            }
        }
        Ok(TensorMutable::Owned(out))
    } else {
        let mut converted_b = overwritable_convert_with_order(b, order)?;
        // in place (or work buffer with write-back): b's batch must fit
        let b_inner = if plan.is_vec { 1 } else { 2 };
        let (batch_b, _) = stack_split_shape(converted_b.view().shape(), b_inner, order)?;
        rstsr_assert_eq!(
            batch_b.as_slice(),
            plan.batch_out.as_slice(),
            InvalidLayout,
            "solve: in-place solve cannot grow b's batch to the broadcast shape; pass an immutable b"
        )?;
        let b_stack = StackLayout::new(converted_b.view().layout(), b_inner, order)?;
        for i in 0..a_stack.len() {
            let a_slice = stack_slice_at(a_full.view(), &a_stack, i)?.into_dim::<Ix2>();
            let mut b_slice = stack_slice_at_mut(converted_b.view_mut(), &b_stack, i)?;
            if plan.is_vec {
                kernel(a_slice, b_slice.i_mut((.., None)).into_dim::<Ix2>())?;
            } else {
                kernel(a_slice, b_slice.into_dim::<Ix2>())?;
            }
        }
        Ok(converted_b)
    }
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
