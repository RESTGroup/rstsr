//! Axis-wise iteration over tensor views: [`TensorView::axes_iter`],
//! [`TensorMut::axes_iter_mut`], and their indexed variants.
//!
//! Each step yields a view of the tensor with the iterated axes removed, so
//! iteration along axis `i` of an N-D tensor yields views of dimensionality
//! N-1. `axes_iter` traverses in the iterated axis's own (K) order;
//! [`TensorView::axes_iter_with_order`] pins the order explicitly, and the
//! `indexed_*` variants follow the device default order. See
//! [`order_semantics`](crate::order_semantics).

#![allow(clippy::missing_transmute_annotations)]

use crate::prelude_dev::*;
use core::mem::transmute;

/* #region axes iter view iterator */

/// Iterator yielding immutable views along an axis of a tensor.
pub struct IterAxesView<'a, T, B>
where
    B: DeviceAPI<T>,
{
    axes_iter: IterLayout<IxD>,
    view: TensorView<'a, T, B, IxD>,
}

impl<T, B> IterAxesView<'_, T, B>
where
    B: DeviceAPI<T>,
{
    pub fn update_offset(&mut self, offset: usize) {
        // SAFETY: `offset` comes from the `IterLayout` iterator over the iterated axes
        // of the validated layout; inner layout + this offset addresses only elements
        // of the original (in-bounds) tensor.
        unsafe { self.view.layout.set_offset(offset) };
    }
}

impl<'a, T, B> Iterator for IterAxesView<'a, T, B>
where
    B: DeviceAPI<T>,
{
    type Item = TensorView<'a, T, B, IxD>;

    fn next(&mut self) -> Option<Self::Item> {
        self.axes_iter.next().map(|offset| {
            self.update_offset(offset);
            // SAFETY: lifetime rewrite only — the iterator owns the `'a` shared borrow of
            // the data; yielded views are immutable shares.
            unsafe { transmute(self.view.view()) }
        })
    }
}

impl<T, B> DoubleEndedIterator for IterAxesView<'_, T, B>
where
    B: DeviceAPI<T>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.axes_iter.next_back().map(|offset| {
            self.update_offset(offset);
            // SAFETY: lifetime rewrite only — the iterator owns the `'a` shared borrow of
            // the data; yielded views are immutable shares.
            unsafe { transmute(self.view.view()) }
        })
    }
}

impl<T, B> ExactSizeIterator for IterAxesView<'_, T, B>
where
    B: DeviceAPI<T>,
{
    fn len(&self) -> usize {
        self.axes_iter.len()
    }
}

impl<T, B> IterSplitAtAPI for IterAxesView<'_, T, B>
where
    B: DeviceAPI<T>,
{
    fn split_at(self, index: usize) -> (Self, Self) {
        let (lhs_axes_iter, rhs_axes_iter) = self.axes_iter.split_at(index);
        // SAFETY: lifetime rewrite to `'a`; the split halves share the iterator's
        // immutable data borrow (rayon requires the halves to be `Send`).
        let view_lhs = unsafe { transmute(self.view.view()) };
        let lhs = IterAxesView { axes_iter: lhs_axes_iter, view: view_lhs };
        let rhs = IterAxesView { axes_iter: rhs_axes_iter, view: self.view };
        return (lhs, rhs);
    }
}

impl<'a, T, B, D> TensorView<'a, T, B, D>
where
    D: DimAPI,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    pub fn axes_iter_with_order_f<I>(&self, axes: I, order: TensorIterOrder) -> Result<IterAxesView<'a, T, B>>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        // convert axis to negative indexes and sort
        let ndim: isize = TryInto::<isize>::try_into(self.ndim())?;
        let axes: Vec<isize> = axes
            .try_into()
            .map_err(Into::into)?
            .as_ref()
            .iter()
            .map(|&v| if v >= 0 { v } else { v + ndim })
            .collect::<Vec<isize>>();
        let mut axes_check = axes.clone();
        axes_check.sort();
        // check no two axis are the same, and no negative index too small
        if axes.first().is_some_and(|&v| v < 0) {
            return Err(rstsr_error!(InvalidValue, "Some negative index is too small."));
        }
        // adjacent pairs after sorting; `windows(2)` is empty for 0 or 1 axes,
        // which keeps an empty axes list valid (it iterates once over the
        // whole tensor)
        for w in axes_check.windows(2) {
            rstsr_assert!(w[0] != w[1], InvalidValue, "Same axes is not allowed here.")?;
        }

        // get full layout
        let layout = self.layout().to_dim::<IxD>()?;
        let shape_full = layout.shape();
        let stride_full = layout.stride();
        let offset = layout.offset();

        // get layout for axes_iter
        let mut shape_axes = vec![];
        let mut stride_axes = vec![];
        for &idx in &axes {
            shape_axes.push(shape_full[idx as usize]);
            stride_axes.push(stride_full[idx as usize]);
        }
        // SAFETY: `layout_axes` repeats selected (shape, stride) pairs of the validated
        // full layout with the same offset — a subset of the original element set.
        let layout_axes = unsafe { Layout::new_unchecked(shape_axes, stride_axes, offset) };

        // get layout for inner view
        let mut shape_inner = vec![];
        let mut stride_inner = vec![];
        for idx in 0..ndim {
            if !axes.contains(&idx) {
                shape_inner.push(shape_full[idx as usize]);
                stride_inner.push(stride_full[idx as usize]);
            }
        }
        // SAFETY: `layout_inner` collects the (shape, stride) pairs of the non-iterated
        // axes of the validated full layout — a subset of the original element set.
        let layout_inner = unsafe { Layout::new_unchecked(shape_inner, stride_inner, offset) };

        // create axes iter
        // The inner view carries the view's inner lifetime `'a`: its data
        // reference points into the *owner* the view borrows from, so the
        // iterator (and the views it yields) stay valid as long as the owner,
        // even if the view value itself is a temporary (e.g. `a.t()`).
        let axes_iter = IterLayout::<IxD>::new(&layout_axes, order)?;
        let r = match self.data().try_as_true_ref() {
            Some(r) => r,
            None => rstsr_raise!(
                UnImplemented,
                "axes_iter is not supported on ManuallyDrop-backed views (e.g. views from `asarray` over a raw slice); take a view of an owning tensor first"
            )?,
        };
        let storage = Storage::new(DataRef::TrueRef(r), self.device().clone());
        // SAFETY: `layout_inner` is derived from `self`'s validated layout and
        // the storage references the same data (see SAFETY note above).
        let view = unsafe { TensorBase::new_unchecked(storage, layout_inner) };
        let iter = IterAxesView { axes_iter, view };
        Ok(iter)
    }

    pub fn axes_iter_f<I>(&self, axes: I) -> Result<IterAxesView<'a, T, B>>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        self.axes_iter_with_order_f(axes, TensorIterOrder::default())
    }

    pub fn axes_iter_with_order<I>(&self, axes: I, order: TensorIterOrder) -> IterAxesView<'a, T, B>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        self.axes_iter_with_order_f(axes, order).rstsr_unwrap()
    }

    /// Iterate over views of a tensor view along the given axis.
    ///
    /// Each step yields a view of dimensionality N-1 (the iterated axis is
    /// removed), sharing the original data. The number of steps equals the
    /// length of `axes`; traversal follows the iterated axis's own (K) order,
    /// which for the usual layouts matches the device default order (see
    /// [`order_semantics`](crate::order_semantics)).
    ///
    /// This method is defined on views only. For owned, arc or cow tensors,
    /// take a view first, e.g. `a.view().axes_iter(0)`; the returned iterator
    /// (and the views it yields) borrow the data owner through the view and
    /// may outlive the view value itself.
    ///
    /// # Parameters
    ///
    /// - `axes`: the axis to iterate along; negative values count from the back. Anything that
    ///   converts into [`AxesIndex<isize>`][AxesIndex].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// let a = rt::arange((6, &device)).into_shape([2, 3]);
    /// for v in a.view().axes_iter(0) {
    ///     println!("{v}");
    /// }
    /// // [ 0 1 2]
    /// // [ 3 4 5]
    /// for v in a.view().axes_iter(-1) {
    ///     println!("{v}");
    /// }
    /// // [ 0 3]
    /// // [ 1 4]
    /// // [ 2 5]
    /// # let rows: Vec<_> = a.view().axes_iter(0).map(|v| v.to_vec()).collect();
    /// # assert_eq!(rows, vec![vec![0, 1, 2], vec![3, 4, 5]]);
    /// ```
    ///
    /// # Panics
    ///
    /// - Panics if `axes` is out of range, or if the view is ManuallyDrop-backed (see
    ///   [`TensorView::axes_iter_f`]).
    ///
    /// For a fallible version, use [`TensorView::axes_iter_f`].
    ///
    /// # See also
    ///
    /// ## Variants of this function
    ///
    /// - [`TensorView::axes_iter_f`]: fallible version.
    /// - [`TensorView::axes_iter_with_order`]: explicit traversal order.
    /// - [`TensorMut::axes_iter_mut`]: mutable views.
    /// - [`TensorView::indexed_axes_iter`]: iteration with the axis position.
    /// - [`TensorView::iter`]: element-wise iteration.
    pub fn axes_iter<I>(&self, axes: I) -> IterAxesView<'a, T, B>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        self.axes_iter_f(axes).rstsr_unwrap()
    }
}

/* #endregion */

/* #region axes iter mut iterator */

/// Iterator yielding mutable views along an axis of a tensor.
pub struct IterAxesMut<'a, T, B>
where
    B: DeviceAPI<T>,
{
    axes_iter: IterLayout<IxD>,
    view: TensorMut<'a, T, B, IxD>,
}

impl<T, B> IterAxesMut<'_, T, B>
where
    B: DeviceAPI<T>,
{
    pub fn update_offset(&mut self, offset: usize) {
        // SAFETY: `offset` comes from the `IterLayout` iterator over the iterated axes
        // of the validated layout; inner layout + this offset addresses only elements
        // of the original (in-bounds) tensor.
        unsafe { self.view.layout.set_offset(offset) };
    }
}

impl<'a, T, B> Iterator for IterAxesMut<'a, T, B>
where
    B: DeviceAPI<T>,
{
    type Item = TensorMut<'a, T, B, IxD>;

    fn next(&mut self) -> Option<Self::Item> {
        self.axes_iter.next().map(|offset| {
            self.update_offset(offset);
            // SAFETY: lifetime rewrite to `'a`; the iterator holds the unique `'a` mutable
            // borrow, and successive views cover disjoint elements provided the iterated
            // axes have non-zero strides (stride-0 mutable layouts are not producible by
            // the safe `&mut` constructors used in practice).
            unsafe { transmute(self.view.view_mut()) }
        })
    }
}

impl<T, B> DoubleEndedIterator for IterAxesMut<'_, T, B>
where
    B: DeviceAPI<T>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.axes_iter.next_back().map(|offset| {
            self.update_offset(offset);
            // SAFETY: lifetime rewrite to `'a`; the iterator holds the unique `'a` mutable
            // borrow, and successive views cover disjoint elements provided the iterated
            // axes have non-zero strides (stride-0 mutable layouts are not producible by
            // the safe `&mut` constructors used in practice).
            unsafe { transmute(self.view.view_mut()) }
        })
    }
}

impl<T, B> ExactSizeIterator for IterAxesMut<'_, T, B>
where
    B: DeviceAPI<T>,
{
    fn len(&self) -> usize {
        self.axes_iter.len()
    }
}

impl<T, B> IterSplitAtAPI for IterAxesMut<'_, T, B>
where
    B: DeviceAPI<T>,
{
    fn split_at(mut self, index: usize) -> (Self, Self) {
        let (lhs_axes_iter, rhs_axes_iter) = self.axes_iter.clone().split_at(index);
        // SAFETY: lifetime rewrite to `'a`; the two halves iterate disjoint offset
        // ranges of the same unique mutable borrow.
        let view_lhs = unsafe { transmute(self.view.view_mut()) };
        let lhs = IterAxesMut { axes_iter: lhs_axes_iter, view: view_lhs };
        let rhs = IterAxesMut { axes_iter: rhs_axes_iter, view: self.view };
        return (lhs, rhs);
    }
}

impl<'a, T, B, D> TensorMut<'a, T, B, D>
where
    D: DimAPI,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    pub fn axes_iter_mut_with_order_f<I>(self, axes: I, order: TensorIterOrder) -> Result<IterAxesMut<'a, T, B>>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        // convert axis to negative indexes and sort
        let ndim: isize = TryInto::<isize>::try_into(self.ndim())?;
        let axes: Vec<isize> = axes
            .try_into()
            .map_err(Into::into)?
            .as_ref()
            .iter()
            .map(|&v| if v >= 0 { v } else { v + ndim })
            .collect::<Vec<isize>>();
        let mut axes_check = axes.clone();
        axes_check.sort();
        // check no two axis are the same, and no negative index too small
        if axes.first().is_some_and(|&v| v < 0) {
            return Err(rstsr_error!(InvalidValue, "Some negative index is too small."));
        }
        // adjacent pairs after sorting; `windows(2)` is empty for 0 or 1 axes,
        // which keeps an empty axes list valid (it iterates once over the
        // whole tensor)
        for w in axes_check.windows(2) {
            rstsr_assert!(w[0] != w[1], InvalidValue, "Same axes is not allowed here.")?;
        }

        // get full layout
        let layout = self.layout().to_dim::<IxD>()?;
        let shape_full = layout.shape();
        let stride_full = layout.stride();
        let offset = layout.offset();

        // get layout for axes_iter
        let mut shape_axes = vec![];
        let mut stride_axes = vec![];
        for &idx in &axes {
            shape_axes.push(shape_full[idx as usize]);
            stride_axes.push(stride_full[idx as usize]);
        }
        // SAFETY: `layout_axes` repeats selected (shape, stride) pairs of the validated
        // full layout with the same offset — a subset of the original element set.
        let layout_axes = unsafe { Layout::new_unchecked(shape_axes, stride_axes, offset) };

        // get layout for inner view
        let mut shape_inner = vec![];
        let mut stride_inner = vec![];
        for idx in 0..ndim {
            if !axes.contains(&idx) {
                shape_inner.push(shape_full[idx as usize]);
                stride_inner.push(stride_full[idx as usize]);
            }
        }
        // SAFETY: `layout_inner` collects the (shape, stride) pairs of the non-iterated
        // axes of the validated full layout — a subset of the original element set.
        let layout_inner = unsafe { Layout::new_unchecked(shape_inner, stride_inner, offset) };

        // successive items of the iterator would be live simultaneously; if an
        // iterated axis is broadcast (zero stride), items would alias each other
        // as multiple `&mut` to the same elements
        rstsr_assert!(!layout_axes.is_broadcasted(), InvalidLayout, "cannot iterate mutably along broadcasted axes")?;
        // create axes iter
        // The receiver is consumed to move the view's inner `&'a mut` out,
        // which lets `a.view_mut().axes_iter_mut(0)` work as a one-liner: the
        // iterator (and the views it yields) borrow the data owner for `'a`.
        let axes_iter = IterLayout::<IxD>::new(&layout_axes, order)?;
        let (storage, _) = self.into_raw_parts();
        let (data, device) = storage.into_raw_parts();
        let m = match data.try_into_true_mut() {
            Some(m) => m,
            None => rstsr_raise!(
                UnImplemented,
                "axes_iter_mut is not supported on ManuallyDrop-backed views (e.g. views from `asarray` over a raw mut slice); take a mutable view of an owning tensor first"
            )?,
        };
        let storage = Storage::new(DataMut::TrueRef(m), device);
        // SAFETY: `layout_inner` is derived from `self`'s validated layout and
        // the storage references the same data (see SAFETY note above).
        let view = unsafe { TensorBase::new_unchecked(storage, layout_inner) };
        let iter = IterAxesMut { axes_iter, view };
        Ok(iter)
    }

    pub fn axes_iter_mut_f<I>(self, axes: I) -> Result<IterAxesMut<'a, T, B>>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        self.axes_iter_mut_with_order_f(axes, TensorIterOrder::default())
    }

    pub fn axes_iter_mut_with_order<I>(self, axes: I, order: TensorIterOrder) -> IterAxesMut<'a, T, B>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        self.axes_iter_mut_with_order_f(axes, order).rstsr_unwrap()
    }

    /// Iterate over mutable views of a mutable tensor view along the given
    /// axis; see [`TensorView::axes_iter`].
    ///
    /// This method is defined on mutable views only and consumes the view.
    /// For owned tensors, take a mutable view first, e.g.
    /// `a.view_mut().axes_iter_mut(0)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// let mut b: Tensor<i32, _> = rt::zeros(([2, 3], &device));
    /// for mut v in b.view_mut().axes_iter_mut(0) {
    ///     v += 1;
    /// }
    /// println!("{b}");
    /// // [[ 1 1 1]
    /// //  [ 1 1 1]]
    /// # assert_eq!(format!("{b}"), "[[ 1 1 1]\n [ 1 1 1]]");
    /// ```
    ///
    /// # See also
    ///
    /// [`TensorView::axes_iter`].
    pub fn axes_iter_mut<I>(self, axes: I) -> IterAxesMut<'a, T, B>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        self.axes_iter_mut_f(axes).rstsr_unwrap()
    }
}

/* #endregion */

/* #region indexed axes iter view iterator */

/// Iterator yielding (position, immutable view) pairs along an axis.
pub struct IndexedIterAxesView<'a, T, B>
where
    B: DeviceAPI<T>,
{
    axes_iter: IterLayout<IxD>,
    view: TensorView<'a, T, B, IxD>,
}

impl<T, B> IndexedIterAxesView<'_, T, B>
where
    B: DeviceAPI<T>,
{
    pub fn update_offset(&mut self, offset: usize) {
        // SAFETY: `offset` comes from the `IterLayout` iterator over the iterated axes
        // of the validated layout; inner layout + this offset addresses only elements
        // of the original (in-bounds) tensor.
        unsafe { self.view.layout.set_offset(offset) };
    }
}

impl<'a, T, B> Iterator for IndexedIterAxesView<'a, T, B>
where
    B: DeviceAPI<T>,
{
    type Item = (IxD, TensorView<'a, T, B, IxD>);

    fn next(&mut self) -> Option<Self::Item> {
        let index = match &self.axes_iter {
            IterLayout::ColMajor(iter_inner) => iter_inner.index_start().clone(),
            IterLayout::RowMajor(iter_inner) => iter_inner.index_start().clone(),
        };
        self.axes_iter.next().map(|offset| {
            self.update_offset(offset);
            // SAFETY: lifetime rewrite only — the iterator owns the `'a` shared borrow;
            // yielded views are immutable shares (indexed variant).
            (index, unsafe { transmute(self.view.view()) })
        })
    }
}

impl<T, B> DoubleEndedIterator for IndexedIterAxesView<'_, T, B>
where
    B: DeviceAPI<T>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let index = match &self.axes_iter {
            IterLayout::ColMajor(iter_inner) => iter_inner.index_start().clone(),
            IterLayout::RowMajor(iter_inner) => iter_inner.index_start().clone(),
        };
        self.axes_iter.next_back().map(|offset| {
            self.update_offset(offset);
            // SAFETY: lifetime rewrite only — the iterator owns the `'a` shared borrow;
            // yielded views are immutable shares (indexed variant).
            (index, unsafe { transmute(self.view.view()) })
        })
    }
}

impl<T, B> ExactSizeIterator for IndexedIterAxesView<'_, T, B>
where
    B: DeviceAPI<T>,
{
    fn len(&self) -> usize {
        self.axes_iter.len()
    }
}

impl<T, B> IterSplitAtAPI for IndexedIterAxesView<'_, T, B>
where
    B: DeviceAPI<T>,
{
    fn split_at(self, index: usize) -> (Self, Self) {
        let (lhs_axes_iter, rhs_axes_iter) = self.axes_iter.split_at(index);
        // SAFETY: lifetime rewrite to `'a`; the split halves share the iterator's
        // immutable data borrow (rayon requires the halves to be `Send`).
        let view_lhs = unsafe { transmute(self.view.view()) };
        let lhs = IndexedIterAxesView { axes_iter: lhs_axes_iter, view: view_lhs };
        let rhs = IndexedIterAxesView { axes_iter: rhs_axes_iter, view: self.view };
        return (lhs, rhs);
    }
}

impl<'a, T, B, D> TensorView<'a, T, B, D>
where
    D: DimAPI,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    pub fn indexed_axes_iter_with_order_f<I>(
        &self,
        axes: I,
        order: TensorIterOrder,
    ) -> Result<IndexedIterAxesView<'a, T, B>>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        use TensorIterOrder::*;
        // this function only accepts c/f iter order currently
        match order {
            C | F => (),
            _ => rstsr_invalid!(order, "This function only accepts TensorIterOrder::C|F.",)?,
        };
        // convert axis to negative indexes and sort
        let ndim: isize = TryInto::<isize>::try_into(self.ndim())?;
        let axes: Vec<isize> = axes
            .try_into()
            .map_err(Into::into)?
            .as_ref()
            .iter()
            .map(|&v| if v >= 0 { v } else { v + ndim })
            .collect::<Vec<isize>>();
        let mut axes_check = axes.clone();
        axes_check.sort();
        // check no two axis are the same, and no negative index too small
        if axes.first().is_some_and(|&v| v < 0) {
            return Err(rstsr_error!(InvalidValue, "Some negative index is too small."));
        }
        // adjacent pairs after sorting; `windows(2)` is empty for 0 or 1 axes,
        // which keeps an empty axes list valid (it iterates once over the
        // whole tensor)
        for w in axes_check.windows(2) {
            rstsr_assert!(w[0] != w[1], InvalidValue, "Same axes is not allowed here.")?;
        }

        // get full layout
        let layout = self.layout().to_dim::<IxD>()?;
        let shape_full = layout.shape();
        let stride_full = layout.stride();
        let offset = layout.offset();

        // get layout for axes_iter
        let mut shape_axes = vec![];
        let mut stride_axes = vec![];
        for &idx in &axes {
            shape_axes.push(shape_full[idx as usize]);
            stride_axes.push(stride_full[idx as usize]);
        }
        // SAFETY: `layout_axes` repeats selected (shape, stride) pairs of the validated
        // full layout with the same offset — a subset of the original element set.
        let layout_axes = unsafe { Layout::new_unchecked(shape_axes, stride_axes, offset) };

        // get layout for inner view
        let mut shape_inner = vec![];
        let mut stride_inner = vec![];
        for idx in 0..ndim {
            if !axes.contains(&idx) {
                shape_inner.push(shape_full[idx as usize]);
                stride_inner.push(stride_full[idx as usize]);
            }
        }
        // SAFETY: `layout_inner` collects the (shape, stride) pairs of the non-iterated
        // axes of the validated full layout — a subset of the original element set.
        let layout_inner = unsafe { Layout::new_unchecked(shape_inner, stride_inner, offset) };

        // create axes iter
        // The inner view carries the view's inner lifetime `'a`;
        // see `axes_iter_with_order_f` for the lifetime contract.
        let axes_iter = IterLayout::<IxD>::new(&layout_axes, order)?;
        let r = match self.data().try_as_true_ref() {
            Some(r) => r,
            None => rstsr_raise!(
                UnImplemented,
                "indexed_axes_iter is not supported on ManuallyDrop-backed views (e.g. views from `asarray` over a raw slice); take a view of an owning tensor first"
            )?,
        };
        let storage = Storage::new(DataRef::TrueRef(r), self.device().clone());
        // SAFETY: `layout_inner` is derived from `self`'s validated layout and
        // the storage references the same data (see SAFETY note above).
        let view = unsafe { TensorBase::new_unchecked(storage, layout_inner) };
        let iter = IndexedIterAxesView { axes_iter, view };
        Ok(iter)
    }

    pub fn indexed_axes_iter_f<I>(&self, axes: I) -> Result<IndexedIterAxesView<'a, T, B>>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        let default_order = self.device().default_order();
        let order = match default_order {
            RowMajor => TensorIterOrder::C,
            ColMajor => TensorIterOrder::F,
        };
        self.indexed_axes_iter_with_order_f(axes, order)
    }

    pub fn indexed_axes_iter_with_order<I>(&self, axes: I, order: TensorIterOrder) -> IndexedIterAxesView<'a, T, B>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        self.indexed_axes_iter_with_order_f(axes, order).rstsr_unwrap()
    }

    /// Iterate over (position, view) pairs along the given axis of a tensor
    /// view; see [`TensorView::axes_iter`] for the view-only policy.
    ///
    /// # See also
    ///
    /// [`TensorView::axes_iter`].
    pub fn indexed_axes_iter<I>(&self, axes: I) -> IndexedIterAxesView<'a, T, B>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        self.indexed_axes_iter_f(axes).rstsr_unwrap()
    }
}

/* #endregion */

/* #region axes iter mut iterator */

/// Iterator yielding (position, mutable view) pairs along an axis.
pub struct IndexedIterAxesMut<'a, T, B>
where
    B: DeviceAPI<T>,
{
    axes_iter: IterLayout<IxD>,
    view: TensorMut<'a, T, B, IxD>,
}

impl<T, B> IndexedIterAxesMut<'_, T, B>
where
    B: DeviceAPI<T>,
{
    pub fn update_offset(&mut self, offset: usize) {
        // SAFETY: `offset` comes from the `IterLayout` iterator over the iterated axes
        // of the validated layout; inner layout + this offset addresses only elements
        // of the original (in-bounds) tensor.
        unsafe { self.view.layout.set_offset(offset) };
    }
}

impl<'a, T, B> Iterator for IndexedIterAxesMut<'a, T, B>
where
    B: DeviceAPI<T>,
{
    type Item = (IxD, TensorMut<'a, T, B, IxD>);

    fn next(&mut self) -> Option<Self::Item> {
        let index = match &self.axes_iter {
            IterLayout::ColMajor(iter_inner) => iter_inner.index_start().clone(),
            IterLayout::RowMajor(iter_inner) => iter_inner.index_start().clone(),
        };
        self.axes_iter.next().map(|offset| {
            self.update_offset(offset);
            // SAFETY: lifetime rewrite of the tuple (same layout) to `'a`; see the mutable
            // iterator SAFETY notes above.
            unsafe { transmute((index, self.view.view_mut())) }
        })
    }
}

impl<T, B> DoubleEndedIterator for IndexedIterAxesMut<'_, T, B>
where
    B: DeviceAPI<T>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let index = match &self.axes_iter {
            IterLayout::ColMajor(iter_inner) => iter_inner.index_start().clone(),
            IterLayout::RowMajor(iter_inner) => iter_inner.index_start().clone(),
        };
        self.axes_iter.next_back().map(|offset| {
            self.update_offset(offset);
            // SAFETY: lifetime rewrite of the tuple (same layout) to `'a`; see the mutable
            // iterator SAFETY notes above.
            unsafe { transmute((index, self.view.view_mut())) }
        })
    }
}

impl<T, B> ExactSizeIterator for IndexedIterAxesMut<'_, T, B>
where
    B: DeviceAPI<T>,
{
    fn len(&self) -> usize {
        self.axes_iter.len()
    }
}

impl<T, B> IterSplitAtAPI for IndexedIterAxesMut<'_, T, B>
where
    B: DeviceAPI<T>,
{
    fn split_at(mut self, index: usize) -> (Self, Self) {
        let (lhs_axes_iter, rhs_axes_iter) = self.axes_iter.clone().split_at(index);
        // SAFETY: lifetime rewrite to `'a`; the two halves iterate disjoint offset
        // ranges of the same unique mutable borrow.
        let view_lhs = unsafe { transmute(self.view.view_mut()) };
        let lhs = IndexedIterAxesMut { axes_iter: lhs_axes_iter, view: view_lhs };
        let rhs = IndexedIterAxesMut { axes_iter: rhs_axes_iter, view: self.view };
        return (lhs, rhs);
    }
}

impl<'a, T, B, D> TensorMut<'a, T, B, D>
where
    D: DimAPI,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    pub fn indexed_axes_iter_mut_with_order_f<I>(
        self,
        axes: I,
        order: TensorIterOrder,
    ) -> Result<IndexedIterAxesMut<'a, T, B>>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        // convert axis to negative indexes and sort
        let ndim: isize = TryInto::<isize>::try_into(self.ndim())?;
        let axes: Vec<isize> = axes
            .try_into()
            .map_err(Into::into)?
            .as_ref()
            .iter()
            .map(|&v| if v >= 0 { v } else { v + ndim })
            .collect::<Vec<isize>>();
        let mut axes_check = axes.clone();
        axes_check.sort();
        // check no two axis are the same, and no negative index too small
        if axes.first().is_some_and(|&v| v < 0) {
            return Err(rstsr_error!(InvalidValue, "Some negative index is too small."));
        }
        // adjacent pairs after sorting; `windows(2)` is empty for 0 or 1 axes,
        // which keeps an empty axes list valid (it iterates once over the
        // whole tensor)
        for w in axes_check.windows(2) {
            rstsr_assert!(w[0] != w[1], InvalidValue, "Same axes is not allowed here.")?;
        }

        // get full layout
        let layout = self.layout().to_dim::<IxD>()?;
        let shape_full = layout.shape();
        let stride_full = layout.stride();
        let offset = layout.offset();

        // get layout for axes_iter
        let mut shape_axes = vec![];
        let mut stride_axes = vec![];
        for &idx in &axes {
            shape_axes.push(shape_full[idx as usize]);
            stride_axes.push(stride_full[idx as usize]);
        }
        // SAFETY: `layout_axes` repeats selected (shape, stride) pairs of the validated
        // full layout with the same offset — a subset of the original element set.
        let layout_axes = unsafe { Layout::new_unchecked(shape_axes, stride_axes, offset) };

        // get layout for inner view
        let mut shape_inner = vec![];
        let mut stride_inner = vec![];
        for idx in 0..ndim {
            if !axes.contains(&idx) {
                shape_inner.push(shape_full[idx as usize]);
                stride_inner.push(stride_full[idx as usize]);
            }
        }
        // SAFETY: `layout_inner` collects the (shape, stride) pairs of the non-iterated
        // axes of the validated full layout — a subset of the original element set.
        let layout_inner = unsafe { Layout::new_unchecked(shape_inner, stride_inner, offset) };

        // successive items of the iterator would be live simultaneously; if an
        // iterated axis is broadcast (zero stride), items would alias each other
        // as multiple `&mut` to the same elements
        rstsr_assert!(!layout_axes.is_broadcasted(), InvalidLayout, "cannot iterate mutably along broadcasted axes")?;
        // create axes iter
        // The receiver is consumed to move the view's inner `&'a mut` out;
        // see `axes_iter_mut_with_order_f` for the lifetime contract.
        let axes_iter = IterLayout::<IxD>::new(&layout_axes, order)?;
        let (storage, _) = self.into_raw_parts();
        let (data, device) = storage.into_raw_parts();
        let m = match data.try_into_true_mut() {
            Some(m) => m,
            None => rstsr_raise!(
                UnImplemented,
                "indexed_axes_iter_mut is not supported on ManuallyDrop-backed views (e.g. views from `asarray` over a raw mut slice); take a mutable view of an owning tensor first"
            )?,
        };
        let storage = Storage::new(DataMut::TrueRef(m), device);
        // SAFETY: `layout_inner` is derived from `self`'s validated layout and
        // the storage references the same data (see SAFETY note above).
        let view = unsafe { TensorBase::new_unchecked(storage, layout_inner) };
        let iter = IndexedIterAxesMut { axes_iter, view };
        Ok(iter)
    }

    pub fn indexed_axes_iter_mut_f<I>(self, axes: I) -> Result<IndexedIterAxesMut<'a, T, B>>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        let default_order = self.device().default_order();
        let order = match default_order {
            RowMajor => TensorIterOrder::C,
            ColMajor => TensorIterOrder::F,
        };
        self.indexed_axes_iter_mut_with_order_f(axes, order)
    }

    pub fn indexed_axes_iter_mut_with_order<I>(self, axes: I, order: TensorIterOrder) -> IndexedIterAxesMut<'a, T, B>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        self.indexed_axes_iter_mut_with_order_f(axes, order).rstsr_unwrap()
    }

    /// Mutable iteration over (position, view) pairs along the given axis of
    /// a mutable tensor view; see [`TensorView::indexed_axes_iter`] for the
    /// view-only policy.
    pub fn indexed_axes_iter_mut<I>(self, axes: I) -> IndexedIterAxesMut<'a, T, B>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        self.indexed_axes_iter_mut_f(axes).rstsr_unwrap()
    }
}

/* #endregion */

#[cfg(test)]
mod tests_serial {
    use super::*;

    #[test]
    fn test_axes_iter() {
        let a = arange(120).into_shape([2, 3, 4, 5]);
        let iter = a.view().axes_iter_f([0, 2]).unwrap();

        let res = iter
            .map(|view| {
                println!("{view:3}");
                view[[1, 2]]
            })
            .collect::<Vec<_>>();
        #[cfg(not(feature = "col_major"))]
        {
            // import numpy as np
            // a = np.arange(120).reshape(2, 3, 4, 5)
            // a[:, 1, :, 2].reshape(-1)
            assert_eq!(res, vec![22, 27, 32, 37, 82, 87, 92, 97]);
        }
        #[cfg(feature = "col_major")]
        {
            // a = range(0, 119) |> collect;
            // a = reshape(a, (2, 3, 4, 5));
            // reshape(a[:, 2, :, 3], 8)'
            assert_eq!(res, vec![50, 51, 56, 57, 62, 63, 68, 69]);
        }
    }

    #[test]
    fn test_axes_iter_mut() {
        let mut a = arange(120).into_shape([2, 3, 4, 5]);
        let iter = a.view_mut().axes_iter_mut_with_order_f([0, 2], TensorIterOrder::C).unwrap();

        let res = iter
            .map(|mut view| {
                view += 1;
                println!("{view:3}");
                view[[1, 2]]
            })
            .collect::<Vec<_>>();
        println!("{res:?}");
        #[cfg(not(feature = "col_major"))]
        {
            // import numpy as np
            // a = np.arange(120).reshape(2, 3, 4, 5)
            // a[:, 1, :, 2].reshape(-1) + 1
            assert_eq!(res, vec![23, 28, 33, 38, 83, 88, 93, 98]);
        }
        #[cfg(feature = "col_major")]
        {
            // a = range(0, 119) |> collect;
            // a = reshape(a, (2, 3, 4, 5));
            // reshape(a[:, 2, :, 3]', 8)' .+ 1
            assert_eq!(res, vec![51, 57, 63, 69, 52, 58, 64, 70]);
        }
    }

    #[test]
    fn test_indexed_axes_iter() {
        let a = arange(120).into_shape([2, 3, 4, 5]);
        let iter = a.view().indexed_axes_iter([0, 2]);

        let res = iter
            .map(|(index, view)| {
                println!("{index:?}");
                println!("{view:3}");
                (index, view[[1, 2]])
            })
            .collect::<Vec<_>>();
        #[cfg(not(feature = "col_major"))]
        {
            // import numpy as np
            // a = np.arange(120).reshape(2, 3, 4, 5)
            // a[:, 1, :, 2].reshape(-1)
            assert_eq!(res, vec![
                (vec![0, 0], 22),
                (vec![0, 1], 27),
                (vec![0, 2], 32),
                (vec![0, 3], 37),
                (vec![1, 0], 82),
                (vec![1, 1], 87),
                (vec![1, 2], 92),
                (vec![1, 3], 97)
            ]);
        }
        #[cfg(feature = "col_major")]
        {
            // a = range(0, 119) |> collect;
            // a = reshape(a, (2, 3, 4, 5));
            // reshape(a[:, 2, :, 3], 8)'
            assert_eq!(res, vec![
                (vec![0, 0], 50),
                (vec![1, 0], 51),
                (vec![0, 1], 56),
                (vec![1, 1], 57),
                (vec![0, 2], 62),
                (vec![1, 2], 63),
                (vec![0, 3], 68),
                (vec![1, 3], 69)
            ]);
        }
    }

    #[test]
    fn test_axes_iter_mut_broadcast_err() {
        // items of a mutable axes iterator are live simultaneously; iterating
        // along a broadcast (stride-0) axis would make items alias each other
        // as multiple `&mut` to the same elements, so it is rejected
        let a = arange((3.0, &DeviceCpu::default()));
        let (storage, _) = a.into_raw_parts();
        let mut c = Tensor::new(storage, Layout::new([2, 3], [0, 1], 0).unwrap());
        assert!(c.view_mut().axes_iter_mut_f([0]).is_err());
        // iterating along a normal axis is fine; items then carry the
        // broadcast axis internally (inert views), and writing through an
        // item is caught by the write-path checks (e.g. `fill`)
        let iter = c.view_mut().axes_iter_mut_f([1]).unwrap();
        let mut first = true;
        for mut item in iter {
            assert!(item.fill_f(0.0).is_err());
            if first {
                let v: Vec<_> = item.view().iter().cloned().collect();
                assert_eq!(v, vec![0., 0.]);
                first = false;
            }
        }
        let v: Vec<_> = c.view().iter().cloned().collect();
        assert_eq!(v, vec![0., 1., 2., 0., 1., 2.]);
    }

    #[test]
    fn test_axes_iter_empty_axes() {
        // the product over an empty axes selection is 1, so exactly one
        // whole-tensor view is yielded (NumPy `ndindex()` semantics);
        // `()` and `vec![]` are the two spellings of "no axes"
        let a = arange(24).into_shape([2, 3, 4]);
        for views in [
            a.view().axes_iter_f(()).unwrap().collect::<Vec<_>>(),
            a.view().axes_iter_f(Vec::<isize>::new()).unwrap().collect::<Vec<_>>(),
        ] {
            assert_eq!(views.len(), 1);
            assert_eq!(views[0].shape(), &[2, 3, 4]);
            assert_eq!(views[0].reshape(-1).to_vec(), (0..24).collect::<Vec<_>>());
        }
        // indexed variant yields a single (empty index, whole view) pair
        let res = a
            .view()
            .indexed_axes_iter(Vec::<isize>::new())
            .map(|(index, view)| (index, view.reshape(-1).to_vec()))
            .collect::<Vec<_>>();
        assert_eq!(res, vec![(vec![], (0..24).collect::<Vec<_>>())]);
        // duplicate axes are still rejected (guard unchanged in behavior)
        assert!(a.view().axes_iter_f([1, 1]).is_err());
        assert!(a.view().indexed_axes_iter_f([1, 1]).is_err());
    }

    #[test]
    fn test_axes_iter_mut_empty_axes() {
        // single whole-tensor mut view; sound (one `&mut`, no aliasing)
        // even for stride-0 layouts since nothing is iterated
        let mut a = arange(24).into_shape([2, 3, 4]);
        for mut view in a.view_mut().axes_iter_mut_f(Vec::<isize>::new()).unwrap() {
            view += 1;
        }
        assert_eq!(a.reshape(-1).to_vec(), (1..25).collect::<Vec<_>>());
        // duplicate axes still rejected on the mut path
        assert!(a.view_mut().axes_iter_mut_f([1, 1]).is_err());
    }
}

#[cfg(test)]
#[cfg(feature = "rayon")]
mod tests_parallel {
    use super::*;

    #[test]
    fn test_axes_iter() {
        let mut a = arange(65536).into_shape([16, 16, 16, 16]);
        let iter = a.view_mut().axes_iter_mut([0, 2]);

        let res = iter
            .into_par_iter()
            .map(|mut view| {
                view += 1;
                println!("{view:6}");
                view[[1, 2]]
            })
            .collect::<Vec<_>>();
        println!("{res:?}");
        #[cfg(not(feature = "col_major"))]
        {
            // a = np.arange(65536).reshape(16, 16, 16, 16)
            // a[:, 1, :, 2].reshape(-1)[:17] + 1
            assert_eq!(res[..17], vec![
                259, 275, 291, 307, 323, 339, 355, 371, 387, 403, 419, 435, 451, 467, 483, 499, 4355
            ]);
        }
        #[cfg(feature = "col_major")]
        {
            // a = range(0, 65535) |> collect;
            // a = reshape(a, (16, 16, 16, 16))
            // (reshape(a[:, 2, :, 3], 16 * 16) .+ 1)[1:17]
            assert_eq!(res[..17], vec![
                8209, 8210, 8211, 8212, 8213, 8214, 8215, 8216, 8217, 8218, 8219, 8220, 8221, 8222, 8223, 8224, 8465
            ]);
        }
    }
}
