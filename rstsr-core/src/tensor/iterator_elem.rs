use crate::prelude_dev::*;
use core::mem::transmute;

/* #region elem view iterator */

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

impl<'a, R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T, Raw = Vec<T>> + 'a,
{
    pub fn iter_with_order_f(&self, order: TensorIterOrder) -> Result<IterVecView<'a, T, D>> {
        let layout_iter = IterLayout::new(self.layout(), order)?;
        let raw = self.raw().as_ref();

        // SAFETY: The lifetime of `raw` is guaranteed to be at least `'a`.
        // transmute is to change the lifetime, not for type casting.
        let iter = IterVecView { layout_iter, view: raw };
        Ok(unsafe { transmute(iter) })
    }

    pub fn iter_with_order(&self, order: TensorIterOrder) -> IterVecView<'a, T, D> {
        self.iter_with_order_f(order).unwrap()
    }

    pub fn iter_f(&self) -> Result<IterVecView<'a, T, D>> {
        let order = match TensorOrder::default() {
            TensorOrder::C => TensorIterOrder::C,
            TensorOrder::F => TensorIterOrder::F,
        };
        self.iter_with_order_f(order)
    }

    pub fn iter(&self) -> IterVecView<'a, T, D> {
        self.iter_f().unwrap()
    }
}

/* #endregion */

/* #region elem mut iterator */

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
        self.layout_iter.next().map(|offset| unsafe { transmute(&mut self.view[offset]) })
    }
}

impl<T, D> DoubleEndedIterator for IterVecMut<'_, T, D>
where
    D: DimDevAPI,
{
    fn next_back(&mut self) -> Option<Self::Item> {
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
            core::slice::from_raw_parts_mut(ptr, len)
        };
        let lhs = IterVecMut { layout_iter: lhs, view: cloned_view };
        let rhs = IterVecMut { layout_iter: rhs, view: self.view };
        (lhs, rhs)
    }
}

impl<'a, R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataMutAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T, Raw = Vec<T>> + 'a,
{
    pub fn iter_mut_with_order_f(
        &'a mut self,
        order: TensorIterOrder,
    ) -> Result<IterVecMut<'a, T, D>> {
        let layout_iter = IterLayout::new(self.layout(), order)?;
        let raw = self.raw_mut().as_mut();
        let iter = IterVecMut { layout_iter, view: raw };
        Ok(iter)
    }

    pub fn iter_mut_with_order(&'a mut self, order: TensorIterOrder) -> IterVecMut<'a, T, D> {
        self.iter_mut_with_order_f(order).unwrap()
    }

    pub fn iter_mut_f(&'a mut self) -> Result<IterVecMut<'a, T, D>> {
        let order = match TensorOrder::default() {
            TensorOrder::C => TensorIterOrder::C,
            TensorOrder::F => TensorIterOrder::F,
        };
        self.iter_mut_with_order_f(order)
    }

    pub fn iter_mut(&'a mut self) -> IterVecMut<'a, T, D> {
        self.iter_mut_f().unwrap()
    }
}

/* #endregion */

/* #region elem view indexed iterator */

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
            IterLayout::ColMajor(iter_inner) => iter_inner.index_start.clone(),
            IterLayout::RowMajor(iter_inner) => iter_inner.index_start.clone(),
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
            IterLayout::ColMajor(iter_inner) => iter_inner.index_start.clone(),
            IterLayout::RowMajor(iter_inner) => iter_inner.index_start.clone(),
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

impl<'a, R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T, Raw = Vec<T>> + 'a,
{
    pub fn indexed_iter_with_order_f(
        &self,
        order: TensorIterOrder,
    ) -> Result<IndexedIterVecView<'a, T, D>> {
        use TensorIterOrder::*;
        // this function only accepts c/f iter order currently
        match order {
            C | F => (),
            _ => rstsr_invalid!(order, "This function only accepts TensorIterOrder::C|F.",)?,
        };
        let layout_iter = IterLayout::<D>::new(self.layout(), order)?;
        let raw = self.raw().as_ref();

        // SAFETY: The lifetime of `raw` is guaranteed to be at least `'a`.
        // transmute is to change the lifetime, not for type casting.
        let iter = IndexedIterVecView { layout_iter, view: raw };
        Ok(unsafe { transmute(iter) })
    }

    pub fn indexed_iter_with_order(&self, order: TensorIterOrder) -> IndexedIterVecView<'a, T, D> {
        self.indexed_iter_with_order_f(order).unwrap()
    }

    pub fn indexed_iter_f(&self) -> Result<IndexedIterVecView<'a, T, D>> {
        let order = match TensorOrder::default() {
            TensorOrder::C => TensorIterOrder::C,
            TensorOrder::F => TensorIterOrder::F,
        };
        self.indexed_iter_with_order_f(order)
    }

    pub fn indexed_iter(&self) -> IndexedIterVecView<'a, T, D> {
        self.indexed_iter_f().unwrap()
    }
}

/* #endregion */

/* #region elem mut col iterator */
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
            IterLayout::ColMajor(iter_inner) => iter_inner.index_start.clone(),
            IterLayout::RowMajor(iter_inner) => iter_inner.index_start.clone(),
        };
        self.layout_iter.next().map(|offset| (index, unsafe { transmute(&mut self.view[offset]) }))
    }
}

impl<T, D> DoubleEndedIterator for IndexedIterVecMut<'_, T, D>
where
    D: DimDevAPI,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let index = match &self.layout_iter {
            IterLayout::ColMajor(iter_inner) => iter_inner.index_start.clone(),
            IterLayout::RowMajor(iter_inner) => iter_inner.index_start.clone(),
        };
        self.layout_iter
            .next_back()
            .map(|offset| (index, unsafe { transmute(&mut self.view[offset]) }))
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
            core::slice::from_raw_parts_mut(ptr, len)
        };
        let lhs = IndexedIterVecMut { layout_iter: lhs, view: cloned_view };
        let rhs = IndexedIterVecMut { layout_iter: rhs, view: self.view };
        (lhs, rhs)
    }
}

impl<'a, R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataMutAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T, Raw = Vec<T>> + 'a,
{
    pub fn indexed_iter_mut_with_order_f(
        &'a mut self,
        order: TensorIterOrder,
    ) -> Result<IndexedIterVecMut<'a, T, D>> {
        use TensorIterOrder::*;
        // this function only accepts c/f iter order currently
        match order {
            C | F => (),
            _ => rstsr_invalid!(order, "This function only accepts TensorIterOrder::C|F.",)?,
        };
        let layout_iter = IterLayout::<D>::new(self.layout(), order)?;
        let raw = self.raw_mut().as_mut();

        let iter = IndexedIterVecMut { layout_iter, view: raw };
        Ok(iter)
    }

    pub fn indexed_iter_mut_with_order(
        &'a mut self,
        order: TensorIterOrder,
    ) -> IndexedIterVecMut<'a, T, D> {
        self.indexed_iter_mut_with_order_f(order).unwrap()
    }

    pub fn indexed_iter_mut_f(&'a mut self) -> Result<IndexedIterVecMut<'a, T, D>> {
        let order = match TensorOrder::default() {
            TensorOrder::C => TensorIterOrder::C,
            TensorOrder::F => TensorIterOrder::F,
        };
        self.indexed_iter_mut_with_order_f(order)
    }

    pub fn indexed_iter_mut(&'a mut self) -> IndexedIterVecMut<'a, T, D> {
        self.indexed_iter_mut_f().unwrap()
    }
}

/* #endregion */

#[cfg(test)]
mod tests_serial {
    use super::*;

    #[test]
    fn test_iter() {
        let a = arange(6).into_shape([3, 2]);
        let iter = a.iter();
        let vec = iter.collect::<Vec<_>>();
        assert_eq!(vec, vec![&0, &1, &2, &3, &4, &5]);

        let iter_t = a.t().iter();
        let vec_t = iter_t.collect::<Vec<_>>();
        assert_eq!(vec_t, vec![&0, &2, &4, &1, &3, &5]);
    }

    #[test]
    fn test_mut_iter() {
        let mut a = arange(6usize).into_shape([3, 2]);
        let iter = a.iter_mut();
        iter.for_each(|x| *x = 0);
        let a = a.reshape(-1).to_vec();
        assert_eq!(a, vec![0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_indexed_c_iter() {
        let a = arange(6).into_layout([3, 2].c());
        let iter = a.indexed_iter_with_order(TensorIterOrder::C);
        let vec = iter.collect::<Vec<_>>();
        assert_eq!(vec, vec![
            ([0, 0], &0),
            ([0, 1], &1),
            ([1, 0], &2),
            ([1, 1], &3),
            ([2, 0], &4),
            ([2, 1], &5)
        ]);

        let iter_t = a.t().indexed_iter_with_order(TensorIterOrder::C);
        let vec_t = iter_t.collect::<Vec<_>>();
        assert_eq!(vec_t, vec![
            ([0, 0], &0),
            ([0, 1], &2),
            ([0, 2], &4),
            ([1, 0], &1),
            ([1, 1], &3),
            ([1, 2], &5)
        ]);
    }
}

#[cfg(test)]
#[cfg(feature = "rayon")]
mod tests_parallel {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn test_iter() {
        let a = arange(16384).into_shape([128, 128]);
        let iter = a.iter().into_par_iter();
        let vec = iter.collect::<Vec<_>>();
        assert_eq!(vec[..6], vec![&0, &1, &2, &3, &4, &5]);

        let iter_t = a.t().iter().into_par_iter();
        let vec_t = iter_t.collect::<Vec<_>>();
        assert_eq!(vec_t[..6], vec![&0, &128, &256, &384, &512, &640]);
    }

    #[test]
    fn test_mut_iter() {
        let mut a = arange(16384).into_shape([128, 128]);
        let b = &a + 1;

        let iter = a.iter_mut().into_par_iter();
        iter.for_each(|x| *x += 1);

        assert_eq!(a.reshape(-1).to_vec(), b.reshape(-1).to_vec());
    }
}
