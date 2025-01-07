use crate::prelude_dev::*;

/* #region vec view col iterator */

pub struct IterVecView<'a, T, D>
where
    D: DimDevAPI,
{
    layout_iter: IterLayoutColMajor<D>,
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

impl<'a, R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T, RawVec = Vec<T>> + 'a,
{
    pub fn iter_with_order(&self, order: TensorIterOrder) -> IterVecView<'a, T, D> {
        let layout = translate_to_col_major_unary(self.layout(), order).unwrap();
        let layout_iter = IterLayoutColMajor::new(&layout).unwrap();
        let rawvec = self.rawvec().as_ref();

        // SAFETY: The lifetime of `rawvec` is guaranteed to be at least `'a`.
        // transmute is to change the lifetime, not for type casting.
        let iter = IterVecView { layout_iter, view: rawvec };
        unsafe { core::mem::transmute(iter) }
    }

    pub fn iter(&self) -> IterVecView<'a, T, D> {
        let order = match TensorOrder::default() {
            TensorOrder::C => TensorIterOrder::C,
            TensorOrder::F => TensorIterOrder::F,
        };
        self.iter_with_order(order)
    }
}

/* #endregion */

/* #region vec mut col iterator */

pub struct IterVecMut<'a, T, D>
where
    D: DimDevAPI,
{
    layout_iter: IterLayoutColMajor<D>,
    view: &'a mut [T],
}

impl<'a, T, D> Iterator for IterVecMut<'a, T, D>
where
    D: DimDevAPI,
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        self.layout_iter
            .next()
            .map(|offset| unsafe { core::mem::transmute(&mut self.view[offset]) })
    }
}

impl<T, D> DoubleEndedIterator for IterVecMut<'_, T, D>
where
    D: DimDevAPI,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.layout_iter
            .next_back()
            .map(|offset| unsafe { core::mem::transmute(&mut self.view[offset]) })
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

impl<'a, R, T, D, B> TensorBase<R, D>
where
    R: DataMutAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T, RawVec = Vec<T>> + 'a,
{
    pub fn iter_mut_with_order(&mut self, order: TensorIterOrder) -> IterVecMut<'a, T, D> {
        let layout = translate_to_col_major_unary(self.layout(), order).unwrap();
        let layout_iter = IterLayoutColMajor::new(&layout).unwrap();
        let rawvec = self.rawvec_mut().as_mut();

        // SAFETY: The lifetime of `rawvec` is guaranteed to be at least `'a`.
        // transmute is to change the lifetime, not for type casting.
        let iter = IterVecMut { layout_iter, view: rawvec };
        unsafe { core::mem::transmute(iter) }
    }

    pub fn iter_mut(&mut self) -> IterVecMut<'a, T, D> {
        let order = match TensorOrder::default() {
            TensorOrder::C => TensorIterOrder::C,
            TensorOrder::F => TensorIterOrder::F,
        };
        self.iter_mut_with_order(order)
    }
}

/* #endregion */

/* #region vec view indexed iterator */

macro_rules! impl_indexed_iter {
    ($IndexedIter: ident, $LayoutIter: ident, $indexed_iter: ident) => {
        pub struct $IndexedIter<'a, T, D>
        where
            D: DimDevAPI,
        {
            layout_iter: $LayoutIter<D>,
            view: &'a [T],
        }

        impl<'a, T, D> Iterator for $IndexedIter<'a, T, D>
        where
            D: DimDevAPI,
        {
            type Item = (D, &'a T);

            fn next(&mut self) -> Option<Self::Item> {
                let index = self.layout_iter.index_start.clone();
                self.layout_iter.next().map(|offset| (index, &self.view[offset]))
            }
        }

        impl<T, D> DoubleEndedIterator for $IndexedIter<'_, T, D>
        where
            D: DimDevAPI,
        {
            fn next_back(&mut self) -> Option<Self::Item> {
                let index = self.layout_iter.index_start.clone();
                self.layout_iter.next_back().map(|offset| (index, &self.view[offset]))
            }
        }

        impl<T, D> ExactSizeIterator for $IndexedIter<'_, T, D>
        where
            D: DimDevAPI,
        {
            fn len(&self) -> usize {
                self.layout_iter.len()
            }
        }

        impl<'a, R, T, D, B> TensorBase<R, D>
        where
            R: DataAPI<Data = Storage<T, B>>,
            D: DimAPI,
            B: DeviceAPI<T, RawVec = Vec<T>> + 'a,
        {
            pub fn $indexed_iter(&self) -> $IndexedIter<'a, T, D> {
                let layout_iter = <$LayoutIter<D>>::new(self.layout()).unwrap();
                let rawvec = self.rawvec().as_ref();

                // SAFETY: The lifetime of `rawvec` is guaranteed to be at least `'a`.
                // transmute is to change the lifetime, not for type casting.
                let iter = $IndexedIter { layout_iter, view: rawvec };
                unsafe { core::mem::transmute(iter) }
            }
        }
    };
}

impl_indexed_iter!(IndexedIterVecViewColMajor, IterLayoutColMajor, indexed_f_iter);
impl_indexed_iter!(IndexedIterVecViewRowMajor, IterLayoutRowMajor, indexed_c_iter);

/* #endregion */

/* #region vec mut col iterator */

macro_rules! impl_indexed_iter_mut {
    ($IndexedIter: ident, $LayoutIter: ident, $indexed_iter: ident) => {
        pub struct $IndexedIter<'a, T, D>
        where
            D: DimDevAPI,
        {
            layout_iter: $LayoutIter<D>,
            view: &'a mut [T],
        }

        impl<'a, T, D> Iterator for $IndexedIter<'a, T, D>
        where
            D: DimDevAPI,
        {
            type Item = (D, &'a mut T);

            fn next(&mut self) -> Option<Self::Item> {
                let index = self.layout_iter.index_start.clone();
                self.layout_iter.next().map(|offset| {
                    (index, unsafe {
                        core::mem::transmute::<&mut T, &mut T>(&mut self.view[offset])
                    })
                })
            }
        }

        impl<T, D> DoubleEndedIterator for $IndexedIter<'_, T, D>
        where
            D: DimDevAPI,
        {
            fn next_back(&mut self) -> Option<Self::Item> {
                let index = self.layout_iter.index_start.clone();
                self.layout_iter.next_back().map(|offset| {
                    (index, unsafe {
                        core::mem::transmute::<&mut T, &mut T>(&mut self.view[offset])
                    })
                })
            }
        }

        impl<T, D> ExactSizeIterator for $IndexedIter<'_, T, D>
        where
            D: DimDevAPI,
        {
            fn len(&self) -> usize {
                self.layout_iter.len()
            }
        }

        impl<'a, R, T, D, B> TensorBase<R, D>
        where
            R: DataMutAPI<Data = Storage<T, B>>,
            D: DimAPI,
            B: DeviceAPI<T, RawVec = Vec<T>> + 'a,
        {
            pub fn $indexed_iter(&mut self) -> $IndexedIter<'a, T, D> {
                let layout_iter = <$LayoutIter<D>>::new(self.layout()).unwrap();
                let rawvec = self.rawvec_mut().as_mut();

                // SAFETY: The lifetime of `rawvec` is guaranteed to be at least `'a`.
                // transmute is to change the lifetime, not for type casting.
                let iter = $IndexedIter { layout_iter, view: rawvec };
                unsafe { core::mem::transmute(iter) }
            }
        }
    };
}

impl_indexed_iter_mut!(IndexedIterVecMutColMajor, IterLayoutColMajor, indexed_f_iter_mut);
impl_indexed_iter_mut!(IndexedIterVecMutRowMajor, IterLayoutRowMajor, indexed_c_iter_mut);

/* #endregion */

#[cfg(test)]
mod tests {
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
        let a = arange(6).into_shape([3, 2]);
        let iter = a.indexed_c_iter();
        let vec = iter.collect::<Vec<_>>();
        assert_eq!(vec, vec![
            ([0, 0], &0),
            ([0, 1], &1),
            ([1, 0], &2),
            ([1, 1], &3),
            ([2, 0], &4),
            ([2, 1], &5)
        ]);

        let iter_t = a.t().indexed_c_iter();
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
