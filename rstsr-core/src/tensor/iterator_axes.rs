use crate::prelude_dev::*;
use core::mem::transmute;

/* #region axes iter view iterator */

pub struct IterAxesView<'a, R>
where
    R: DataAPI,
{
    axes_iter: IterLayout<IxD>,
    layout_inner: Layout<IxD>,
    view: TensorBase<DataRef<'a, R::Data>, IxD>,
}

impl<R> IterAxesView<'_, R>
where
    R: DataAPI,
{
    pub fn update_offset(&mut self, offset: usize) {
        let layout = unsafe { self.layout_inner.set_offset(offset) };
        self.view.layout = layout.clone();
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
        let lhs = IterAxesView {
            axes_iter: lhs_axes_iter,
            layout_inner: self.layout_inner.clone(),
            view: self.view.clone(),
        };
        let rhs = IterAxesView {
            axes_iter: rhs_axes_iter,
            layout_inner: self.layout_inner,
            view: self.view,
        };
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
        #[allow(clippy::missing_transmute_annotations)]
        let iter = IterAxesView {
            axes_iter,
            layout_inner,
            view: unsafe { transmute(self.view().to_dyn()) },
        };
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

#[cfg(test)]
mod test {
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
}
