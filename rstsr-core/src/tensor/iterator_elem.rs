//! Element-wise iteration over tensor views: [`TensorView::iter`],
//! [`TensorMut::iter_mut`], and their indexed variants.
//!
//! Iteration order follows the device default order ([`RowMajor`] iterates
//! C-like, [`ColMajor`] F-like); see [`order_semantics`](crate::order_semantics).

use crate::prelude_dev::*;
use core::mem::transmute;

/* #region elem view iterator */

/// Iterator yielding element references of a tensor, in layout traversal order.
pub struct IterVecView<'a, T, D>
where
    D: DimDevAPI,
{
    layout_iter: IterLayout<D>,
    view: &'a [T],
}

impl<'a, T, D> Iterator for IterVecView<'a, T, D>
where
    D: DimDevAPI,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.layout_iter.next().map(|offset| &self.view[offset])
    }
}

impl<T, D> DoubleEndedIterator for IterVecView<'_, T, D>
where
    D: DimDevAPI,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.layout_iter.next_back().map(|offset| &self.view[offset])
    }
}

impl<T, D> ExactSizeIterator for IterVecView<'_, T, D>
where
    D: DimDevAPI,
{
    fn len(&self) -> usize {
        self.layout_iter.len()
    }
}

impl<T, D> IterSplitAtAPI for IterVecView<'_, T, D>
where
    D: DimDevAPI,
{
    fn split_at(self, mid: usize) -> (Self, Self) {
        let (lhs, rhs) = self.layout_iter.split_at(mid);
        let lhs = IterVecView { layout_iter: lhs, view: self.view };
        let rhs = IterVecView { layout_iter: rhs, view: self.view };
        (lhs, rhs)
    }
}

impl<'a, T, B, D> TensorView<'a, T, B, D>
where
    D: DimAPI,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    pub fn iter_with_order_f(&self, order: TensorIterOrder) -> Result<IterVecView<'a, T, D>> {
        // The returned iterator carries the view's inner lifetime `'a`: the
        // underlying data reference points into the *owner* the view borrows
        // from, so the iterator stays valid as long as the owner, even if the
        // view value itself is a temporary (e.g. `a.t().iter()`).
        let layout_iter = IterLayout::new(self.layout(), order)?;
        let raw = self.data().as_slice_ref();
        Ok(IterVecView { layout_iter, view: raw })
    }

    pub fn iter_with_order(&self, order: TensorIterOrder) -> IterVecView<'a, T, D> {
        self.iter_with_order_f(order).rstsr_unwrap()
    }

    pub fn iter_f(&self) -> Result<IterVecView<'a, T, D>> {
        let default_order = self.device().default_order();
        let order = match default_order {
            RowMajor => TensorIterOrder::C,
            ColMajor => TensorIterOrder::F,
        };
        self.iter_with_order_f(order)
    }

    /// Iterate over the elements of a tensor view by reference.
    ///
    /// Elements are yielded following the device default order: C-like
    /// (row-major) sequence under [`RowMajor`], F-like (column-major) under
    /// [`ColMajor`]; see [`order_semantics`](crate::order_semantics). Use
    /// [`TensorView::iter_with_order`] to pin the traversal order explicitly.
    ///
    /// This method is defined on views only. For owned, arc or cow tensors,
    /// take a view first, e.g. `a.view().iter()` or `a.t().iter()`; the
    /// returned iterator borrows the data owner through the view and may
    /// outlive the view value itself. Accordingly, an iterator cannot
    /// outlive the data owner:
    ///
    /// ```rust,compile_fail
    /// # use rstsr::prelude::*;
    /// // the owned tensor is dropped at the end of this statement, so using
    /// // the iterator afterwards would dangle — rejected at compile time
    /// let it = rt::arange((6, &DeviceCpu::default())).view().iter();
    /// let _v: Vec<i32> = it.cloned().collect();
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// let a = rt::arange((6, &device)).into_shape([2, 3]);
    /// let collected: Vec<i32> = a.view().iter().cloned().collect();
    /// println!("{collected:?}");
    /// // [0, 1, 2, 3, 4, 5]
    /// # assert_eq!(collected, vec![0, 1, 2, 3, 4, 5]);
    /// ```
    ///
    /// # Panics
    ///
    /// - Panics if the iterator cannot be constructed for the given layout.
    ///
    /// For a fallible version, use [`TensorView::iter_f`].
    ///
    /// # See also
    ///
    /// ## Variants of this function
    ///
    /// - [`TensorView::iter_f`]: fallible version.
    /// - [`TensorView::iter_with_order`] / [`TensorView::iter_with_order_f`]: explicit traversal
    ///   order.
    /// - [`TensorMut::iter_mut`]: mutable element iteration.
    /// - [`TensorView::indexed_iter`]: iteration with logical indices.
    pub fn iter(&self) -> IterVecView<'a, T, D> {
        self.iter_f().rstsr_unwrap()
    }
}

/* #endregion */

/* #region elem mut iterator */

/// Iterator yielding mutable element references of a tensor.
pub struct IterVecMut<'a, T, D>
where
    D: DimDevAPI,
{
    layout_iter: IterLayout<D>,
    view: &'a mut [T],
}

impl<'a, T, D> Iterator for IterVecMut<'a, T, D>
where
    D: DimDevAPI,
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        // SAFETY: lifetime rewrite — each yielded `&mut T` points to a distinct
        // element of the iterator's own `&mut [T]` (the standard owning-iterator
        // pattern, cf. `slice::IterMut`).
        self.layout_iter.next().map(|offset| unsafe { transmute(&mut self.view[offset]) })
    }
}

impl<T, D> DoubleEndedIterator for IterVecMut<'_, T, D>
where
    D: DimDevAPI,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        // SAFETY: lifetime rewrite — each yielded `&mut T` points to a distinct
        // element of the iterator's own `&mut [T]` (the standard owning-iterator
        // pattern, cf. `slice::IterMut`).
        self.layout_iter.next_back().map(|offset| unsafe { transmute(&mut self.view[offset]) })
    }
}

impl<T, D> ExactSizeIterator for IterVecMut<'_, T, D>
where
    D: DimDevAPI,
{
    fn len(&self) -> usize {
        self.layout_iter.len()
    }
}

impl<T, D> IterSplitAtAPI for IterVecMut<'_, T, D>
where
    D: DimDevAPI,
{
    fn split_at(self, mid: usize) -> (Self, Self) {
        // we do not split &mut [T], but split the layout iterator
        // so we use unsafe code to generate two same &mut [T] views
        let (lhs, rhs) = self.layout_iter.split_at(mid);
        let cloned_view = unsafe {
            let len = self.view.len();
            let ptr = self.view.as_mut_ptr();
            // SAFETY: duplicates the iterator's `&mut [T]` as a raw slice so both split
            // halves can iterate; the two layout iterators visit disjoint offsets, and
            // each half only accesses elements of its own iteration range.
            core::slice::from_raw_parts_mut(ptr, len)
        };
        let lhs = IterVecMut { layout_iter: lhs, view: cloned_view };
        let rhs = IterVecMut { layout_iter: rhs, view: self.view };
        (lhs, rhs)
    }
}

impl<'a, T, B, D> TensorMut<'a, T, B, D>
where
    D: DimAPI,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    pub fn iter_mut_with_order_f(self, order: TensorIterOrder) -> Result<IterVecMut<'a, T, D>> {
        // successive items of the iterator would be live simultaneously; on a
        // broadcast layout they would alias the same element as multiple `&mut`
        rstsr_assert!(!self.layout().is_broadcasted(), InvalidLayout, "cannot iterate mutably on broadcasted tensor")?;
        // The receiver is consumed to move the view's inner `&'a mut` out,
        // which lets `a.view_mut().iter_mut()` work as a one-liner: the
        // returned iterator borrows the data owner for `'a`.
        let layout_iter = IterLayout::new(self.layout(), order)?;
        let (storage, _) = self.into_raw_parts();
        let (data, _device) = storage.into_raw_parts();
        let raw = data.into_slice_mut();
        let iter = IterVecMut { layout_iter, view: raw };
        Ok(iter)
    }

    pub fn iter_mut_with_order(self, order: TensorIterOrder) -> IterVecMut<'a, T, D> {
        self.iter_mut_with_order_f(order).rstsr_unwrap()
    }

    pub fn iter_mut_f(self) -> Result<IterVecMut<'a, T, D>> {
        let default_order = self.device().default_order();
        let order = match default_order {
            RowMajor => TensorIterOrder::C,
            ColMajor => TensorIterOrder::F,
        };
        self.iter_mut_with_order_f(order)
    }

    /// Iterate over the elements of a mutable tensor view by mutable
    /// reference; see [`TensorView::iter`] for the traversal order.
    ///
    /// This method is defined on mutable views only and consumes the view.
    /// For owned tensors, take a mutable view first, e.g.
    /// `a.view_mut().iter_mut()`; the returned iterator borrows the data
    /// owner and may outlive the view value itself.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// let mut b: Tensor<i32, _> = rt::zeros(([2, 2], &device));
    /// for (i, x) in b.view_mut().iter_mut().enumerate() {
    ///     *x = i as i32;
    /// }
    /// println!("{b}");
    /// // [[ 0 1]
    /// //  [ 2 3]]
    /// # assert_eq!(format!("{b}"), "[[ 0 1]\n [ 2 3]]");
    /// ```
    ///
    /// # See also
    ///
    /// ## Variants of this function
    ///
    /// - [`TensorMut::iter_mut_f`]: fallible version.
    /// - [`TensorMut::iter_mut_with_order`]: explicit traversal order.
    /// - [`TensorView::iter`]: immutable element iteration.
    pub fn iter_mut(self) -> IterVecMut<'a, T, D> {
        self.iter_mut_f().rstsr_unwrap()
    }
}

/* #endregion */

/* #region elem view indexed iterator */

/// Iterator yielding (logical index, element reference) pairs.
pub struct IndexedIterVecView<'a, T, D>
where
    D: DimDevAPI,
{
    layout_iter: IterLayout<D>,
    view: &'a [T],
}

impl<'a, T, D> Iterator for IndexedIterVecView<'a, T, D>
where
    D: DimDevAPI,
{
    type Item = (D, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let index = match &self.layout_iter {
            IterLayout::ColMajor(iter_inner) => iter_inner.index_start().clone(),
            IterLayout::RowMajor(iter_inner) => iter_inner.index_start().clone(),
        };
        self.layout_iter.next().map(|offset| (index, &self.view[offset]))
    }
}

impl<T, D> DoubleEndedIterator for IndexedIterVecView<'_, T, D>
where
    D: DimDevAPI,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let index = match &self.layout_iter {
            IterLayout::ColMajor(iter_inner) => iter_inner.index_start().clone(),
            IterLayout::RowMajor(iter_inner) => iter_inner.index_start().clone(),
        };
        self.layout_iter.next_back().map(|offset| (index, &self.view[offset]))
    }
}

impl<T, D> ExactSizeIterator for IndexedIterVecView<'_, T, D>
where
    D: DimDevAPI,
{
    fn len(&self) -> usize {
        self.layout_iter.len()
    }
}

impl<T, D> IterSplitAtAPI for IndexedIterVecView<'_, T, D>
where
    D: DimDevAPI,
{
    fn split_at(self, mid: usize) -> (Self, Self) {
        let (lhs, rhs) = self.layout_iter.split_at(mid);
        let lhs = IndexedIterVecView { layout_iter: lhs, view: self.view };
        let rhs = IndexedIterVecView { layout_iter: rhs, view: self.view };
        (lhs, rhs)
    }
}

impl<'a, T, B, D> TensorView<'a, T, B, D>
where
    D: DimAPI,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    pub fn indexed_iter_with_order_f(&self, order: TensorIterOrder) -> Result<IndexedIterVecView<'a, T, D>> {
        use TensorIterOrder::*;
        // this function only accepts c/f iter order currently
        match order {
            C | F => (),
            _ => rstsr_invalid!(order, "This function only accepts TensorIterOrder::C|F.",)?,
        };
        // The iterator carries the view's inner lifetime `'a`;
        // see `iter_with_order_f` for the lifetime contract.
        let layout_iter = IterLayout::<D>::new(self.layout(), order)?;
        let raw = self.data().as_slice_ref();
        Ok(IndexedIterVecView { layout_iter, view: raw })
    }

    pub fn indexed_iter_with_order(&self, order: TensorIterOrder) -> IndexedIterVecView<'a, T, D> {
        self.indexed_iter_with_order_f(order).rstsr_unwrap()
    }

    pub fn indexed_iter_f(&self) -> Result<IndexedIterVecView<'a, T, D>> {
        let default_order = self.device().default_order();
        let order = match default_order {
            RowMajor => TensorIterOrder::C,
            ColMajor => TensorIterOrder::F,
        };
        self.indexed_iter_with_order_f(order)
    }

    /// Iterate over (index, element) pairs of a tensor view; see
    /// [`TensorView::iter`] for the traversal order and the view-only
    /// policy.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// let a = rt::arange((6, &device)).into_shape([2, 3]);
    /// let pairs: Vec<_> = a.view().indexed_iter().map(|(idx, v)| (idx.to_vec(), *v)).collect();
    /// println!("{pairs:?}");
    /// // [([0, 0], 0), ([0, 1], 1), ([0, 2], 2), ([1, 0], 3), ([1, 1], 4), ([1, 2], 5)]
    /// ```
    ///
    /// # See also
    ///
    /// [`TensorView::iter`].
    pub fn indexed_iter(&self) -> IndexedIterVecView<'a, T, D> {
        self.indexed_iter_f().rstsr_unwrap()
    }
}

/* #endregion */

/* #region elem mut col iterator */
/// Iterator yielding (logical index, mutable element reference) pairs.
pub struct IndexedIterVecMut<'a, T, D>
where
    D: DimDevAPI,
{
    layout_iter: IterLayout<D>,
    view: &'a mut [T],
}

impl<'a, T, D> Iterator for IndexedIterVecMut<'a, T, D>
where
    D: DimDevAPI,
{
    type Item = (D, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        let index = match &self.layout_iter {
            IterLayout::ColMajor(iter_inner) => iter_inner.index_start().clone(),
            IterLayout::RowMajor(iter_inner) => iter_inner.index_start().clone(),
        };
        // SAFETY: lifetime rewrite only (same type) — each yielded `&mut T` points to
        // a distinct element of the iterator's own `&mut [T]`.
        self.layout_iter.next().map(|offset| (index, unsafe { transmute::<&mut T, &mut T>(&mut self.view[offset]) }))
    }
}

impl<T, D> DoubleEndedIterator for IndexedIterVecMut<'_, T, D>
where
    D: DimDevAPI,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let index = match &self.layout_iter {
            IterLayout::ColMajor(iter_inner) => iter_inner.index_start().clone(),
            IterLayout::RowMajor(iter_inner) => iter_inner.index_start().clone(),
        };
        self.layout_iter
            .next_back()
            // SAFETY: lifetime rewrite only (same type) — each yielded `&mut T` points to
            // a distinct element of the iterator's own `&mut [T]`.
            .map(|offset| (index, unsafe { transmute::<&mut T, &mut T>(&mut self.view[offset]) }))
    }
}

impl<T, D> ExactSizeIterator for IndexedIterVecMut<'_, T, D>
where
    D: DimDevAPI,
{
    fn len(&self) -> usize {
        self.layout_iter.len()
    }
}

impl<T, D> IterSplitAtAPI for IndexedIterVecMut<'_, T, D>
where
    D: DimDevAPI,
{
    fn split_at(self, mid: usize) -> (Self, Self) {
        let (lhs, rhs) = self.layout_iter.split_at(mid);
        let cloned_view = unsafe {
            let len = self.view.len();
            let ptr = self.view.as_mut_ptr();
            // SAFETY: duplicates the iterator's `&mut [T]` as a raw slice so both split
            // halves can iterate; the two layout iterators visit disjoint offsets, and
            // each half only accesses elements of its own iteration range.
            core::slice::from_raw_parts_mut(ptr, len)
        };
        let lhs = IndexedIterVecMut { layout_iter: lhs, view: cloned_view };
        let rhs = IndexedIterVecMut { layout_iter: rhs, view: self.view };
        (lhs, rhs)
    }
}

impl<'a, T, B, D> TensorMut<'a, T, B, D>
where
    D: DimAPI,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    pub fn indexed_iter_mut_with_order_f(self, order: TensorIterOrder) -> Result<IndexedIterVecMut<'a, T, D>> {
        use TensorIterOrder::*;
        // this function only accepts c/f iter order currently
        match order {
            C | F => (),
            _ => rstsr_invalid!(order, "This function only accepts TensorIterOrder::C|F.",)?,
        };
        // successive items of the iterator would be live simultaneously; on a
        // broadcast layout they would alias the same element as multiple `&mut`
        rstsr_assert!(!self.layout().is_broadcasted(), InvalidLayout, "cannot iterate mutably on broadcasted tensor")?;
        // The receiver is consumed to move the view's inner `&'a mut` out;
        // see `iter_mut_with_order_f` for the lifetime contract.
        let layout_iter = IterLayout::<D>::new(self.layout(), order)?;
        let (storage, _) = self.into_raw_parts();
        let (data, _device) = storage.into_raw_parts();
        let raw = data.into_slice_mut();

        let iter = IndexedIterVecMut { layout_iter, view: raw };
        Ok(iter)
    }

    pub fn indexed_iter_mut_with_order(self, order: TensorIterOrder) -> IndexedIterVecMut<'a, T, D> {
        self.indexed_iter_mut_with_order_f(order).rstsr_unwrap()
    }

    pub fn indexed_iter_mut_f(self) -> Result<IndexedIterVecMut<'a, T, D>> {
        let default_order = self.device().default_order();
        let order = match default_order {
            RowMajor => TensorIterOrder::C,
            ColMajor => TensorIterOrder::F,
        };
        self.indexed_iter_mut_with_order_f(order)
    }

    /// Mutable iteration over (index, element) pairs of a mutable tensor
    /// view; see [`TensorView::indexed_iter`] and [`TensorView::iter`] for
    /// the traversal order and the view-only policy.
    ///
    /// # See also
    ///
    /// [`TensorView::indexed_iter`].
    pub fn indexed_iter_mut(self) -> IndexedIterVecMut<'a, T, D> {
        self.indexed_iter_mut_f().rstsr_unwrap()
    }
}

/* #endregion */

#[cfg(test)]
mod tests_serial {
    use super::*;

    #[test]
    fn test_iter_correctness() {
        // Soundness regression guard: `iter` used to `transmute` its borrow to
        // an unrelated lifetime, which made `let it = { let t = tensor;
        // t.iter() };` compile (use-after-free). Iteration is now view-only:
        // the iterator borrows the data owner through the view's inner
        // lifetime, and `iter` on an owned tensor no longer exists.
        let t = arange(6);
        let v: Vec<_> = t.view().iter().cloned().collect();
        assert_eq!(v, vec![0, 1, 2, 3, 4, 5]);

        let v: Vec<_> = t.view().iter().enumerate().map(|(i, x)| *x + i).collect();
        assert_eq!(v, vec![0, 2, 4, 6, 8, 10]);

        let v: Vec<_> = t.view().indexed_iter().map(|(_, x)| *x).collect();
        assert_eq!(v, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_iter_one_liner_on_view_temporary() {
        // The one-liner over a view temporary is sound again (it was a
        // compile error after tying the iterator to `&self`): the iterator
        // borrows the owner `a` through the view's inner lifetime, so the
        // dropped view temporary is irrelevant.
        let a = arange(6).into_shape([3, 2]);
        let it = a.t().iter();
        let v: Vec<_> = it.cloned().collect();
        #[cfg(not(feature = "col_major"))]
        assert_eq!(v, vec![0, 2, 4, 1, 3, 5]);
        #[cfg(feature = "col_major")]
        assert_eq!(v, vec![0, 3, 1, 4, 2, 5]);

        let it = a.view().iter();
        let v: Vec<_> = it.cloned().collect();
        assert_eq!(v, vec![0, 1, 2, 3, 4, 5]);

        let mut b = arange(6usize);
        let it = b.view_mut().iter_mut();
        it.for_each(|x| *x += 1);
        assert_eq!(b.reshape(-1).to_vec(), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_iter_on_manually_drop_backed_view() {
        // `asarray` over a slice produces a view holding a fabricated
        // `Vec` (ManuallyDrop) over the borrowed buffer; iteration must
        // still read the original slice through `'a`.
        let buffer = [0, 1, 2, 3, 4, 5];
        let t: TensorView<i32> = asarray((buffer.as_slice(), &DeviceCpu::default()));
        let v: Vec<_> = t.iter().cloned().collect();
        assert_eq!(v, vec![0, 1, 2, 3, 4, 5]);

        let mut buffer = [0usize, 1, 2, 3];
        {
            let t: TensorMut<usize> = asarray((buffer.as_mut_slice(), &DeviceCpu::default()));
            let it = t.iter_mut();
            it.for_each(|x| *x += 1);
        }
        assert_eq!(buffer, [1, 2, 3, 4]);
    }

    #[test]
    fn test_axes_iter_correctness() {
        let t = arange(6).into_shape([2, 3]);
        let rows: Vec<Vec<_>> = t.view().axes_iter(0).map(|v| v.iter().cloned().collect()).collect();
        #[cfg(not(feature = "col_major"))]
        {
            assert_eq!(rows, vec![vec![0, 1, 2], vec![3, 4, 5]]);
        }
        #[cfg(feature = "col_major")]
        {
            // column-major storage: t[i, j] = arange[i + 2 * j]
            assert_eq!(rows, vec![vec![0, 2, 4], vec![1, 3, 5]]);
        }
    }

    #[test]
    fn test_iter() {
        let a = arange(6).into_shape([3, 2]);
        let iter = a.view().iter();
        let vec = iter.collect::<Vec<_>>();
        assert_eq!(vec, vec![&0, &1, &2, &3, &4, &5]);

        // one-liner over a view temporary: sound, borrows `a` through the
        // view's inner lifetime
        let iter_t = a.t().iter();
        let vec_t = iter_t.collect::<Vec<_>>();
        #[cfg(not(feature = "col_major"))]
        {
            // a = np.arange(6).reshape(3, 2)
            // a.T.reshape(-1)
            assert_eq!(vec_t, vec![&0, &2, &4, &1, &3, &5]);
        }
        #[cfg(feature = "col_major")]
        {
            // a = reshape(range(0, 5), (3, 2));
            // reshape(a', 6)
            assert_eq!(vec_t, vec![&0, &3, &1, &4, &2, &5]);
        }
    }

    #[test]
    fn test_mut_iter() {
        let mut a = arange(6usize).into_shape([3, 2]);
        let iter = a.view_mut().iter_mut();
        iter.for_each(|x| *x = 0);
        let a = a.reshape(-1).to_vec();
        assert_eq!(a, vec![0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_indexed_c_iter() {
        let a = arange(6).into_layout([3, 2].c());
        let iter = a.view().indexed_iter_with_order(TensorIterOrder::C);
        let vec = iter.collect::<Vec<_>>();
        #[cfg(not(feature = "col_major"))]
        assert_eq!(vec, vec![([0, 0], &0), ([0, 1], &1), ([1, 0], &2), ([1, 1], &3), ([2, 0], &4), ([2, 1], &5)]);
        #[cfg(feature = "col_major")]
        assert_eq!(vec, vec![([0, 0], &0), ([0, 1], &3), ([1, 0], &1), ([1, 1], &4), ([2, 0], &2), ([2, 1], &5)]);

        let iter_t = a.t().indexed_iter_with_order(TensorIterOrder::C);
        let vec_t = iter_t.collect::<Vec<_>>();
        #[cfg(not(feature = "col_major"))]
        assert_eq!(vec_t, vec![([0, 0], &0), ([0, 1], &2), ([0, 2], &4), ([1, 0], &1), ([1, 1], &3), ([1, 2], &5)]);
        #[cfg(feature = "col_major")]
        assert_eq!(vec_t, vec![([0, 0], &0), ([0, 1], &1), ([0, 2], &2), ([1, 0], &3), ([1, 1], &4), ([1, 2], &5)]);
    }

    #[test]
    fn test_iter_mut_broadcast_err() {
        // a broadcast (stride-0) layout aliases elements; mutable iteration
        // would yield multiple live `&mut` to the same element and is rejected
        let mut device = DeviceCpu::default();
        device.set_default_order(RowMajor);
        let a = arange((3.0, &device));
        let (storage, _) = a.into_raw_parts();
        let mut c = Tensor::new(storage, Layout::new([2, 3], [0, 1], 0).unwrap());
        // reading a broadcast layout is fine
        let v: Vec<_> = c.view().iter().cloned().collect();
        assert_eq!(v, vec![0., 1., 2., 0., 1., 2.]);
        // mutable iteration is rejected
        assert!(c.view_mut().iter_mut_f().is_err());
        assert!(c.view_mut().indexed_iter_mut_f().is_err());
        let v: Vec<_> = c.view().iter().cloned().collect();
        assert_eq!(v, vec![0., 1., 2., 0., 1., 2.]);
    }
}

#[cfg(test)]
#[cfg(feature = "rayon")]
mod tests_parallel {
    use super::*;

    #[test]
    fn test_iter() {
        let a = arange(16384).into_shape([128, 128]);
        let iter = a.view().iter().into_par_iter();
        let vec = iter.collect::<Vec<_>>();
        assert_eq!(vec[..6], vec![&0, &1, &2, &3, &4, &5]);

        let iter_t = a.t().iter().into_par_iter();
        let vec_t = iter_t.collect::<Vec<_>>();
        // since we only collect the first 6 elements, the order is the same for col and
        // row major however, if more elements are collected, the order will be
        // different
        assert_eq!(vec_t[..6], vec![&0, &128, &256, &384, &512, &640]);
    }

    #[test]
    fn test_mut_iter() {
        let mut a = arange(16384).into_shape([128, 128]);
        let b = &a + 1;

        let iter = a.view_mut().iter_mut().into_par_iter();
        iter.for_each(|x| *x += 1);

        assert_eq!(a.reshape(-1).to_vec(), b.reshape(-1).to_vec());
    }
}
