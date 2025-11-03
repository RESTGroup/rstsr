#![allow(clippy::missing_transmute_annotations)]

use crate::prelude_dev::*;
use core::mem::transmute;

/* #region axes iter view iterator */

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
        let view_lhs = unsafe { transmute(self.view.view()) };
        let lhs = IterAxesView { axes_iter: lhs_axes_iter, view: view_lhs };
        let rhs = IterAxesView { axes_iter: rhs_axes_iter, view: self.view };
        return (lhs, rhs);
    }
}

impl<'a, R, T, B, D> TensorAny<R, T, B, D>
where
    T: Clone,
    R: DataCloneAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T, Raw = Vec<T>> + 'a,
{
    pub fn axes_iter_with_order_f<I>(&self, axes: I, order: TensorIterOrder) -> Result<IterAxesView<'a, T, B>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        // convert axis to negative indexes and sort
        let ndim: isize = TryInto::<isize>::try_into(self.ndim())?;
        let axes: Vec<isize> =
            axes.try_into()?.as_ref().iter().map(|&v| if v >= 0 { v } else { v + ndim }).collect::<Vec<isize>>();
        let mut axes_check = axes.clone();
        axes_check.sort();
        // check no two axis are the same, and no negative index too small
        if axes.first().is_some_and(|&v| v < 0) {
            return Err(rstsr_error!(InvalidValue, "Some negative index is too small."));
        }
        for i in 0..axes_check.len() - 1 {
            rstsr_assert!(axes_check[i] != axes_check[i + 1], InvalidValue, "Same axes is not allowed here.")?;
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
        let layout_inner = unsafe { Layout::new_unchecked(shape_inner, stride_inner, offset) };

        // create axes iter
        let axes_iter = IterLayout::<IxD>::new(&layout_axes, order)?;
        let mut view = self.view().into_dyn();
        view.layout = layout_inner.clone();
        let iter = IterAxesView { axes_iter, view: unsafe { transmute(view) } };
        Ok(iter)
    }

    pub fn axes_iter_f<I>(&self, axes: I) -> Result<IterAxesView<'a, T, B>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        self.axes_iter_with_order_f(axes, TensorIterOrder::default())
    }

    pub fn axes_iter_with_order<I>(&self, axes: I, order: TensorIterOrder) -> IterAxesView<'a, T, B>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        self.axes_iter_with_order_f(axes, order).rstsr_unwrap()
    }

    pub fn axes_iter<I>(&self, axes: I) -> IterAxesView<'a, T, B>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        self.axes_iter_f(axes).rstsr_unwrap()
    }
}

/* #endregion */

/* #region axes iter mut iterator */

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
        let view_lhs = unsafe { transmute(self.view.view_mut()) };
        let lhs = IterAxesMut { axes_iter: lhs_axes_iter, view: view_lhs };
        let rhs = IterAxesMut { axes_iter: rhs_axes_iter, view: self.view };
        return (lhs, rhs);
    }
}

impl<'a, R, T, B, D> TensorAny<R, T, B, D>
where
    T: Clone,
    R: DataMutAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T, Raw = Vec<T>> + 'a,
{
    pub fn axes_iter_mut_with_order_f<I>(&'a mut self, axes: I, order: TensorIterOrder) -> Result<IterAxesMut<'a, T, B>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        // convert axis to negative indexes and sort
        let ndim: isize = TryInto::<isize>::try_into(self.ndim())?;
        let axes: Vec<isize> =
            axes.try_into()?.as_ref().iter().map(|&v| if v >= 0 { v } else { v + ndim }).collect::<Vec<isize>>();
        let mut axes_check = axes.clone();
        axes_check.sort();
        // check no two axis are the same, and no negative index too small
        if axes.first().is_some_and(|&v| v < 0) {
            return Err(rstsr_error!(InvalidValue, "Some negative index is too small."));
        }
        for i in 0..axes_check.len() - 1 {
            rstsr_assert!(axes_check[i] != axes_check[i + 1], InvalidValue, "Same axes is not allowed here.")?;
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
        let layout_inner = unsafe { Layout::new_unchecked(shape_inner, stride_inner, offset) };

        // create axes iter
        let axes_iter = IterLayout::<IxD>::new(&layout_axes, order)?;
        let mut view = self.view_mut().into_dyn();
        view.layout = layout_inner.clone();
        let iter = IterAxesMut { axes_iter, view };
        Ok(iter)
    }

    pub fn axes_iter_mut_f<I>(&'a mut self, axes: I) -> Result<IterAxesMut<'a, T, B>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        self.axes_iter_mut_with_order_f(axes, TensorIterOrder::default())
    }

    pub fn axes_iter_mut_with_order<I>(&'a mut self, axes: I, order: TensorIterOrder) -> IterAxesMut<'a, T, B>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        self.axes_iter_mut_with_order_f(axes, order).rstsr_unwrap()
    }

    pub fn axes_iter_mut<I>(&'a mut self, axes: I) -> IterAxesMut<'a, T, B>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        self.axes_iter_mut_f(axes).rstsr_unwrap()
    }
}

/* #endregion */

/* #region indexed axes iter view iterator */

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
        let view_lhs = unsafe { transmute(self.view.view()) };
        let lhs = IndexedIterAxesView { axes_iter: lhs_axes_iter, view: view_lhs };
        let rhs = IndexedIterAxesView { axes_iter: rhs_axes_iter, view: self.view };
        return (lhs, rhs);
    }
}

impl<'a, R, T, B, D> TensorAny<R, T, B, D>
where
    T: Clone,
    R: DataCloneAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T, Raw = Vec<T>> + 'a,
{
    pub fn indexed_axes_iter_with_order_f<I>(
        &self,
        axes: I,
        order: TensorIterOrder,
    ) -> Result<IndexedIterAxesView<'a, T, B>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        use TensorIterOrder::*;
        // this function only accepts c/f iter order currently
        match order {
            C | F => (),
            _ => rstsr_invalid!(order, "This function only accepts TensorIterOrder::C|F.",)?,
        };
        // convert axis to negative indexes and sort
        let ndim: isize = TryInto::<isize>::try_into(self.ndim())?;
        let axes: Vec<isize> =
            axes.try_into()?.as_ref().iter().map(|&v| if v >= 0 { v } else { v + ndim }).collect::<Vec<isize>>();
        let mut axes_check = axes.clone();
        axes_check.sort();
        // check no two axis are the same, and no negative index too small
        if axes.first().is_some_and(|&v| v < 0) {
            return Err(rstsr_error!(InvalidValue, "Some negative index is too small."));
        }
        for i in 0..axes_check.len() - 1 {
            rstsr_assert!(axes_check[i] != axes_check[i + 1], InvalidValue, "Same axes is not allowed here.")?;
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
        let layout_inner = unsafe { Layout::new_unchecked(shape_inner, stride_inner, offset) };

        // create axes iter
        let axes_iter = IterLayout::<IxD>::new(&layout_axes, order)?;
        let mut view = self.view().into_dyn();
        view.layout = layout_inner.clone();
        let iter = IndexedIterAxesView { axes_iter, view: unsafe { transmute(view) } };
        Ok(iter)
    }

    pub fn indexed_axes_iter_f<I>(&self, axes: I) -> Result<IndexedIterAxesView<'a, T, B>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
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
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        self.indexed_axes_iter_with_order_f(axes, order).rstsr_unwrap()
    }

    pub fn indexed_axes_iter<I>(&self, axes: I) -> IndexedIterAxesView<'a, T, B>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        self.indexed_axes_iter_f(axes).rstsr_unwrap()
    }
}

/* #endregion */

/* #region axes iter mut iterator */

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
        let view_lhs = unsafe { transmute(self.view.view_mut()) };
        let lhs = IndexedIterAxesMut { axes_iter: lhs_axes_iter, view: view_lhs };
        let rhs = IndexedIterAxesMut { axes_iter: rhs_axes_iter, view: self.view };
        return (lhs, rhs);
    }
}

impl<'a, R, T, B, D> TensorAny<R, T, B, D>
where
    T: Clone,
    R: DataMutAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T, Raw = Vec<T>> + 'a,
{
    pub fn indexed_axes_iter_mut_with_order_f<I>(
        &'a mut self,
        axes: I,
        order: TensorIterOrder,
    ) -> Result<IndexedIterAxesMut<'a, T, B>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        // convert axis to negative indexes and sort
        let ndim: isize = TryInto::<isize>::try_into(self.ndim())?;
        let axes: Vec<isize> =
            axes.try_into()?.as_ref().iter().map(|&v| if v >= 0 { v } else { v + ndim }).collect::<Vec<isize>>();
        let mut axes_check = axes.clone();
        axes_check.sort();
        // check no two axis are the same, and no negative index too small
        if axes.first().is_some_and(|&v| v < 0) {
            return Err(rstsr_error!(InvalidValue, "Some negative index is too small."));
        }
        for i in 0..axes_check.len() - 1 {
            rstsr_assert!(axes_check[i] != axes_check[i + 1], InvalidValue, "Same axes is not allowed here.")?;
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
        let layout_inner = unsafe { Layout::new_unchecked(shape_inner, stride_inner, offset) };

        // create axes iter
        let axes_iter = IterLayout::<IxD>::new(&layout_axes, order)?;
        let mut view = self.view_mut().into_dyn();
        view.layout = layout_inner.clone();
        let iter = IndexedIterAxesMut { axes_iter, view };
        Ok(iter)
    }

    pub fn indexed_axes_iter_mut_f<I>(&'a mut self, axes: I) -> Result<IndexedIterAxesMut<'a, T, B>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        let default_order = self.device().default_order();
        let order = match default_order {
            RowMajor => TensorIterOrder::C,
            ColMajor => TensorIterOrder::F,
        };
        self.indexed_axes_iter_mut_with_order_f(axes, order)
    }

    pub fn indexed_axes_iter_mut_with_order<I>(
        &'a mut self,
        axes: I,
        order: TensorIterOrder,
    ) -> IndexedIterAxesMut<'a, T, B>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        self.indexed_axes_iter_mut_with_order_f(axes, order).rstsr_unwrap()
    }

    pub fn indexed_axes_iter_mut<I>(&'a mut self, axes: I) -> IndexedIterAxesMut<'a, T, B>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
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
        let iter = a.axes_iter_f([0, 2]).unwrap();

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
        let iter = a.axes_iter_mut_with_order_f([0, 2], TensorIterOrder::C).unwrap();

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
        let iter = a.indexed_axes_iter([0, 2]);

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
}

#[cfg(test)]
#[cfg(feature = "rayon")]
mod tests_parallel {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn test_axes_iter() {
        let mut a = arange(65536).into_shape([16, 16, 16, 16]);
        let iter = a.axes_iter_mut([0, 2]);

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
