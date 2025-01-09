use crate::prelude_dev::*;
use core::mem::transmute;

/* #region axes iter view iterator */

pub struct IterAxesView<'a, R>
where
    R: DataAPI,
{
    axes_iter: IterLayout<IxD>,
    view: TensorBase<DataRef<'a, R::Data>, IxD>,
}

impl<R> IterAxesView<'_, R>
where
    R: DataAPI,
{
    pub fn update_offset(&mut self, offset: usize) {
        unsafe { self.view.layout.set_offset(offset) };
    }
}

impl<'a, R> Iterator for IterAxesView<'a, R>
where
    R: DataAPI,
{
    type Item = TensorBase<DataRef<'a, R::Data>, IxD>;

    fn next(&mut self) -> Option<Self::Item> {
        self.axes_iter.next().map(|offset| {
            self.update_offset(offset);
            self.view.clone()
        })
    }
}

impl<R> DoubleEndedIterator for IterAxesView<'_, R>
where
    R: DataAPI,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.axes_iter.next_back().map(|offset| {
            self.update_offset(offset);
            self.view.clone()
        })
    }
}

impl<R> ExactSizeIterator for IterAxesView<'_, R>
where
    R: DataAPI,
{
    fn len(&self) -> usize {
        self.axes_iter.len()
    }
}

impl<R> IterSplitAtAPI for IterAxesView<'_, R>
where
    R: DataAPI,
{
    fn split_at(self, index: usize) -> (Self, Self) {
        let (lhs_axes_iter, rhs_axes_iter) = self.axes_iter.split_at(index);
        let lhs = IterAxesView { axes_iter: lhs_axes_iter, view: self.view.clone() };
        let rhs = IterAxesView { axes_iter: rhs_axes_iter, view: self.view };
        return (lhs, rhs);
    }
}

impl<'a, R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T, RawVec = Vec<T>> + 'a,
{
    pub fn axes_iter_with_order_f<I>(
        &self,
        axes: I,
        order: TensorIterOrder,
    ) -> Result<IterAxesView<'a, R>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        // convert axis to negative indexes and sort
        let ndim: isize = TryInto::<isize>::try_into(self.ndim())?;
        let axes: Vec<isize> = axes
            .try_into()?
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
        for i in 0..axes_check.len() - 1 {
            rstsr_assert!(
                axes_check[i] != axes_check[i + 1],
                InvalidValue,
                "Same axes is not allowed here."
            )?;
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
        #[allow(clippy::missing_transmute_annotations)]
        let iter = IterAxesView { axes_iter, view: unsafe { transmute(view) } };
        Ok(iter)
    }

    pub fn axes_iter_f<I>(&self, axes: I) -> Result<IterAxesView<'a, R>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        self.axes_iter_with_order_f(axes, TensorIterOrder::default())
    }

    pub fn axes_iter_with_order<I>(&self, axes: I, order: TensorIterOrder) -> IterAxesView<'a, R>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        self.axes_iter_with_order_f(axes, order).unwrap()
    }

    pub fn axes_iter<I>(&self, axes: I) -> IterAxesView<'a, R>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        self.axes_iter_f(axes).unwrap()
    }
}

/* #endregion */

/* #region axes iter mut iterator */

pub struct IterAxesMut<'a, R>
where
    R: DataAPI,
{
    axes_iter: IterLayout<IxD>,
    view: TensorBase<DataMut<'a, R::Data>, IxD>,
}

impl<R> IterAxesMut<'_, R>
where
    R: DataAPI,
{
    pub fn update_offset(&mut self, offset: usize) {
        unsafe { self.view.layout.set_offset(offset) };
    }

    unsafe fn view_clone(&mut self) -> TensorBase<DataMut<'_, R::Data>, IxD> {
        let layout = self.view.layout().clone();
        let mut_ref = match &mut self.view.data {
            DataMut::TrueRef(storage) => storage,
            DataMut::ManuallyDropOwned(_) => unreachable!(),
        };
        // unsafely clone mut_ref without clone trait
        let mut_other = &mut *(*mut_ref as *mut _);
        let data_other = DataMut::TrueRef(mut_other);
        TensorBase::new_unchecked(data_other, layout)
    }
}

impl<'a, R> Iterator for IterAxesMut<'a, R>
where
    R: DataAPI,
{
    type Item = TensorBase<DataMut<'a, R::Data>, IxD>;

    fn next(&mut self) -> Option<Self::Item> {
        self.axes_iter.next().map(|offset| {
            self.update_offset(offset);
            unsafe { transmute(self.view_clone()) }
        })
    }
}

impl<R> DoubleEndedIterator for IterAxesMut<'_, R>
where
    R: DataAPI,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.axes_iter.next_back().map(|offset| {
            self.update_offset(offset);
            unsafe { transmute(self.view_clone()) }
        })
    }
}

impl<R> ExactSizeIterator for IterAxesMut<'_, R>
where
    R: DataAPI,
{
    fn len(&self) -> usize {
        self.axes_iter.len()
    }
}

impl<R> IterSplitAtAPI for IterAxesMut<'_, R>
where
    R: DataAPI,
{
    fn split_at(mut self, index: usize) -> (Self, Self) {
        let (lhs_axes_iter, rhs_axes_iter) = self.axes_iter.clone().split_at(index);
        #[allow(clippy::missing_transmute_annotations)]
        let view_lhs = unsafe { transmute(self.view_clone()) };
        let lhs = IterAxesMut { axes_iter: lhs_axes_iter, view: view_lhs };
        let rhs = IterAxesMut { axes_iter: rhs_axes_iter, view: self.view };
        return (lhs, rhs);
    }
}

impl<'a, R, T, D, B> TensorBase<R, D>
where
    R: DataMutAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T, RawVec = Vec<T>> + 'a,
{
    /// # Safety
    ///
    /// `iter_a = a.iter_mut` will generate mutable views of tensor,
    /// both `iter_a` and `a` are mutable, which is not safe in rust.
    pub unsafe fn axes_iter_mut_with_order_f<I>(
        &mut self,
        axes: I,
        order: TensorIterOrder,
    ) -> Result<IterAxesMut<'a, R>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        // convert axis to negative indexes and sort
        let ndim: isize = TryInto::<isize>::try_into(self.ndim())?;
        let axes: Vec<isize> = axes
            .try_into()?
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
        for i in 0..axes_check.len() - 1 {
            rstsr_assert!(
                axes_check[i] != axes_check[i + 1],
                InvalidValue,
                "Same axes is not allowed here."
            )?;
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
        #[allow(clippy::missing_transmute_annotations)]
        let iter = IterAxesMut { axes_iter, view: unsafe { transmute(view) } };
        Ok(iter)
    }

    /// # Safety
    ///
    /// `iter_a = a.iter_mut` will generate mutable views of tensor,
    /// both `iter_a` and `a` are mutable, which is not safe in rust.
    pub unsafe fn axes_iter_mut_f<I>(&mut self, axes: I) -> Result<IterAxesMut<'a, R>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        self.axes_iter_mut_with_order_f(axes, TensorIterOrder::default())
    }

    /// # Safety
    ///
    /// `iter_a = a.iter_mut` will generate mutable views of tensor,
    /// both `iter_a` and `a` are mutable, which is not safe in rust.
    pub unsafe fn axes_iter_mut_with_order<I>(
        &mut self,
        axes: I,
        order: TensorIterOrder,
    ) -> IterAxesMut<'a, R>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        self.axes_iter_mut_with_order_f(axes, order).unwrap()
    }

    /// # Safety
    ///
    /// `iter_a = a.iter_mut` will generate mutable views of tensor,
    /// both `iter_a` and `a` are mutable, which is not safe in rust.
    pub unsafe fn axes_iter_mut<I>(&mut self, axes: I) -> IterAxesMut<'a, R>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        self.axes_iter_mut_f(axes).unwrap()
    }
}

/* #endregion */

/* #region indexed axes iter view iterator */

pub struct IndexedIterAxesView<'a, R>
where
    R: DataAPI,
{
    axes_iter: IterLayout<IxD>,
    view: TensorBase<DataRef<'a, R::Data>, IxD>,
}

impl<R> IndexedIterAxesView<'_, R>
where
    R: DataAPI,
{
    pub fn update_offset(&mut self, offset: usize) {
        unsafe { self.view.layout.set_offset(offset) };
    }
}

impl<'a, R> Iterator for IndexedIterAxesView<'a, R>
where
    R: DataAPI,
{
    type Item = (IxD, TensorBase<DataRef<'a, R::Data>, IxD>);

    fn next(&mut self) -> Option<Self::Item> {
        let index = match &self.axes_iter {
            IterLayout::ColMajor(iter_inner) => iter_inner.index_start.clone(),
            IterLayout::RowMajor(iter_inner) => iter_inner.index_start.clone(),
        };
        self.axes_iter.next().map(|offset| {
            self.update_offset(offset);
            (index, self.view.clone())
        })
    }
}

impl<R> DoubleEndedIterator for IndexedIterAxesView<'_, R>
where
    R: DataAPI,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let index = match &self.axes_iter {
            IterLayout::ColMajor(iter_inner) => iter_inner.index_start.clone(),
            IterLayout::RowMajor(iter_inner) => iter_inner.index_start.clone(),
        };
        self.axes_iter.next_back().map(|offset| {
            self.update_offset(offset);
            (index, self.view.clone())
        })
    }
}

impl<R> ExactSizeIterator for IndexedIterAxesView<'_, R>
where
    R: DataAPI,
{
    fn len(&self) -> usize {
        self.axes_iter.len()
    }
}

impl<R> IterSplitAtAPI for IndexedIterAxesView<'_, R>
where
    R: DataAPI,
{
    fn split_at(self, index: usize) -> (Self, Self) {
        let (lhs_axes_iter, rhs_axes_iter) = self.axes_iter.split_at(index);
        let lhs = IndexedIterAxesView { axes_iter: lhs_axes_iter, view: self.view.clone() };
        let rhs = IndexedIterAxesView { axes_iter: rhs_axes_iter, view: self.view };
        return (lhs, rhs);
    }
}

impl<'a, R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T, RawVec = Vec<T>> + 'a,
{
    pub fn indexed_axes_iter_with_order_f<I>(
        &self,
        axes: I,
        order: TensorIterOrder,
    ) -> Result<IndexedIterAxesView<'a, R>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        use TensorIterOrder::*;
        // this function only accepts c/f iter order currently
        match order {
            C | F => (),
            _ => {
                rstsr_invalid!(order, "This function only accepts TensorIterOrder::C|F.",).unwrap()
            },
        };
        // convert axis to negative indexes and sort
        let ndim: isize = TryInto::<isize>::try_into(self.ndim())?;
        let axes: Vec<isize> = axes
            .try_into()?
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
        for i in 0..axes_check.len() - 1 {
            rstsr_assert!(
                axes_check[i] != axes_check[i + 1],
                InvalidValue,
                "Same axes is not allowed here."
            )?;
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
        #[allow(clippy::missing_transmute_annotations)]
        let iter = IndexedIterAxesView { axes_iter, view: unsafe { transmute(view) } };
        Ok(iter)
    }

    pub fn indexed_axes_iter_f<I>(&self, axes: I) -> Result<IndexedIterAxesView<'a, R>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        let order = match TensorOrder::default() {
            TensorOrder::C => TensorIterOrder::C,
            TensorOrder::F => TensorIterOrder::F,
        };
        self.indexed_axes_iter_with_order_f(axes, order)
    }

    pub fn indexed_axes_iter_with_order<I>(
        &self,
        axes: I,
        order: TensorIterOrder,
    ) -> IndexedIterAxesView<'a, R>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        self.indexed_axes_iter_with_order_f(axes, order).unwrap()
    }

    pub fn indexed_axes_iter<I>(&self, axes: I) -> IndexedIterAxesView<'a, R>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        self.indexed_axes_iter_f(axes).unwrap()
    }
}

/* #endregion */

/* #region axes iter mut iterator */

pub struct IndexedIterAxesMut<'a, R>
where
    R: DataAPI,
{
    axes_iter: IterLayout<IxD>,
    view: TensorBase<DataMut<'a, R::Data>, IxD>,
}

impl<R> IndexedIterAxesMut<'_, R>
where
    R: DataAPI,
{
    pub fn update_offset(&mut self, offset: usize) {
        unsafe { self.view.layout.set_offset(offset) };
    }

    unsafe fn view_clone(&mut self) -> TensorBase<DataMut<'_, R::Data>, IxD> {
        let layout = self.view.layout().clone();
        let mut_ref = match &mut self.view.data {
            DataMut::TrueRef(storage) => storage,
            DataMut::ManuallyDropOwned(_) => unreachable!(),
        };
        // unsafely clone mut_ref without clone trait
        let mut_other = &mut *(*mut_ref as *mut _);
        let data_other = DataMut::TrueRef(mut_other);
        TensorBase::new_unchecked(data_other, layout)
    }
}

impl<'a, R> Iterator for IndexedIterAxesMut<'a, R>
where
    R: DataAPI,
{
    type Item = (IxD, TensorBase<DataMut<'a, R::Data>, IxD>);

    fn next(&mut self) -> Option<Self::Item> {
        let index = match &self.axes_iter {
            IterLayout::ColMajor(iter_inner) => iter_inner.index_start.clone(),
            IterLayout::RowMajor(iter_inner) => iter_inner.index_start.clone(),
        };
        self.axes_iter.next().map(|offset| {
            self.update_offset(offset);
            unsafe { transmute((index, self.view_clone())) }
        })
    }
}

impl<R> DoubleEndedIterator for IndexedIterAxesMut<'_, R>
where
    R: DataAPI,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let index = match &self.axes_iter {
            IterLayout::ColMajor(iter_inner) => iter_inner.index_start.clone(),
            IterLayout::RowMajor(iter_inner) => iter_inner.index_start.clone(),
        };
        self.axes_iter.next_back().map(|offset| {
            self.update_offset(offset);
            unsafe { transmute((index, self.view_clone())) }
        })
    }
}

impl<R> ExactSizeIterator for IndexedIterAxesMut<'_, R>
where
    R: DataAPI,
{
    fn len(&self) -> usize {
        self.axes_iter.len()
    }
}

impl<R> IterSplitAtAPI for IndexedIterAxesMut<'_, R>
where
    R: DataAPI,
{
    fn split_at(mut self, index: usize) -> (Self, Self) {
        let (lhs_axes_iter, rhs_axes_iter) = self.axes_iter.clone().split_at(index);
        #[allow(clippy::missing_transmute_annotations)]
        let view_lhs = unsafe { transmute(self.view_clone()) };
        let lhs = IndexedIterAxesMut { axes_iter: lhs_axes_iter, view: view_lhs };
        let rhs = IndexedIterAxesMut { axes_iter: rhs_axes_iter, view: self.view };
        return (lhs, rhs);
    }
}

impl<'a, R, T, D, B> TensorBase<R, D>
where
    R: DataMutAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T, RawVec = Vec<T>> + 'a,
{
    /// # Safety
    ///
    /// `iter_a = a.iter_mut` will generate mutable views of tensor,
    /// both `iter_a` and `a` are mutable, which is not safe in rust.
    pub unsafe fn indexed_axes_iter_mut_with_order_f<I>(
        &mut self,
        axes: I,
        order: TensorIterOrder,
    ) -> Result<IndexedIterAxesMut<'a, R>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        // convert axis to negative indexes and sort
        let ndim: isize = TryInto::<isize>::try_into(self.ndim())?;
        let axes: Vec<isize> = axes
            .try_into()?
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
        for i in 0..axes_check.len() - 1 {
            rstsr_assert!(
                axes_check[i] != axes_check[i + 1],
                InvalidValue,
                "Same axes is not allowed here."
            )?;
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
        #[allow(clippy::missing_transmute_annotations)]
        let iter = IndexedIterAxesMut { axes_iter, view: unsafe { transmute(view) } };
        Ok(iter)
    }

    /// # Safety
    ///
    /// `iter_a = a.iter_mut` will generate mutable views of tensor,
    /// both `iter_a` and `a` are mutable, which is not safe in rust.
    pub unsafe fn indexed_axes_iter_mut_f<I>(
        &mut self,
        axes: I,
    ) -> Result<IndexedIterAxesMut<'a, R>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        let order = match TensorOrder::default() {
            TensorOrder::C => TensorIterOrder::C,
            TensorOrder::F => TensorIterOrder::F,
        };
        self.indexed_axes_iter_mut_with_order_f(axes, order)
    }

    /// # Safety
    ///
    /// `iter_a = a.iter_mut` will generate mutable views of tensor,
    /// both `iter_a` and `a` are mutable, which is not safe in rust.
    pub unsafe fn indexed_axes_iter_mut_with_order<I>(
        &mut self,
        axes: I,
        order: TensorIterOrder,
    ) -> IndexedIterAxesMut<'a, R>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        self.indexed_axes_iter_mut_with_order_f(axes, order).unwrap()
    }

    /// # Safety
    ///
    /// `iter_a = a.iter_mut` will generate mutable views of tensor,
    /// both `iter_a` and `a` are mutable, which is not safe in rust.
    pub unsafe fn indexed_axes_iter_mut<I>(&mut self, axes: I) -> IndexedIterAxesMut<'a, R>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        self.indexed_axes_iter_mut_f(axes).unwrap()
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
                println!("{:3}", view);
                view[[1, 2]]
            })
            .collect::<Vec<_>>();
        assert_eq!(res, vec![22, 27, 32, 37, 82, 87, 92, 97]);
    }

    #[test]
    fn test_axes_iter_mut() {
        let mut a = arange(120).into_shape([2, 3, 4, 5]);
        let iter = unsafe { a.axes_iter_mut_with_order_f([0, 2], TensorIterOrder::C).unwrap() };

        let res = iter
            .map(|mut view| {
                view += 1;
                println!("{:3}", view);
                view[[1, 2]]
            })
            .collect::<Vec<_>>();
        println!("{:?}", res);
        assert_eq!(res, vec![23, 28, 33, 38, 83, 88, 93, 98]);
    }

    #[test]
    fn test_indexed_axes_iter() {
        let a = arange(120).into_shape([2, 3, 4, 5]);
        let iter = a.indexed_axes_iter([0, 2]);

        let res = iter
            .map(|(index, view)| {
                println!("{:?}", index);
                println!("{:3}", view);
                (index, view[[1, 2]])
            })
            .collect::<Vec<_>>();
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
}

#[cfg(test)]
mod tests_parallel {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn test_axes_iter() {
        let mut a = arange(65536).into_shape([16, 16, 16, 16]);
        let iter = unsafe { a.axes_iter_mut([0, 2]) };

        let res = iter
            .into_par_iter()
            .map(|mut view| {
                view += 1;
                println!("{:6}", view);
                view[[1, 2]]
            })
            .collect::<Vec<_>>();
        println!("{:?}", res);
        assert_eq!(res[..17], vec![
            259, 275, 291, 307, 323, 339, 355, 371, 387, 403, 419, 435, 451, 467, 483, 499, 4355
        ]);
    }
}
