//! Layout parallel iterator

use crate::prelude_dev::*;
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::prelude::*;

/* #region template for parallel iterator in RSTSR */

pub struct ParIterRSTSR<It> {
    pub iter: It,
}

impl<It> Producer for ParIterRSTSR<It>
where
    It: Iterator + DoubleEndedIterator + ExactSizeIterator + IterSplitAtAPI + Send,
    It::Item: Send,
{
    type Item = It::Item;
    type IntoIter = It;

    fn into_iter(self) -> Self::IntoIter {
        self.iter
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (lhs, rhs) = self.iter.split_at(index);
        let lhs = ParIterRSTSR::<It> { iter: lhs };
        let rhs = ParIterRSTSR::<It> { iter: rhs };
        return (lhs, rhs);
    }
}

impl<It> ParallelIterator for ParIterRSTSR<It>
where
    It: Iterator + DoubleEndedIterator + ExactSizeIterator + IterSplitAtAPI + Send,
    It::Item: Send,
{
    type Item = It::Item;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.iter.len())
    }
}

impl<It> IndexedParallelIterator for ParIterRSTSR<It>
where
    It: Iterator + DoubleEndedIterator + ExactSizeIterator + IterSplitAtAPI + Send,
    It::Item: Send,
{
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        callback.callback(self)
    }
}

/* #endregion */

/* #region layout iterator */

macro_rules! impl_par_iter_layout {
    ($IterLayout: ident) => {
        impl<D> IntoParallelIterator for $IterLayout<D>
        where
            D: DimDevAPI,
        {
            type Item = usize;
            type Iter = ParIterRSTSR<Self>;

            fn into_par_iter(self) -> Self::Iter {
                Self::Iter { iter: self }
            }
        }
    };
}

impl_par_iter_layout!(IterLayoutColMajor);
impl_par_iter_layout!(IterLayoutRowMajor);
impl_par_iter_layout!(IterLayout);

/* #endregion */

/* #region tensor iterator */

macro_rules! impl_par_iter_tensor {
    ($IterTensor: ident, $item_type: ty) => {
        impl<'a, T, D> IntoParallelIterator for $IterTensor<'a, T, D>
        where
            D: DimDevAPI,
            T: Send + Sync,
        {
            type Item = $item_type;
            type Iter = ParIterRSTSR<Self>;

            fn into_par_iter(self) -> Self::Iter {
                Self::Iter { iter: self }
            }
        }
    };
}

impl_par_iter_tensor!(IterVecView, &'a T);
impl_par_iter_tensor!(IterVecMut, &'a mut T);
impl_par_iter_tensor!(IndexedIterVecView, (D, &'a T));
impl_par_iter_tensor!(IndexedIterVecMut, (D, &'a mut T));

macro_rules! impl_par_axes_iter_tensor {
    ($IterTensor: ident, $TraitBound: ident, $item_type: ty) => {
        impl<'a, R> IntoParallelIterator for $IterTensor<'a, R>
        where
            R: $TraitBound,
            R::Data: Send,
        {
            type Item = $item_type;
            type Iter = ParIterRSTSR<Self>;

            fn into_par_iter(self) -> Self::Iter {
                Self::Iter { iter: self }
            }
        }
    };
}

impl_par_axes_iter_tensor!(IterAxesView, DataAPI, TensorBase<DataRef<'a, R::Data>, IxD>);
impl_par_axes_iter_tensor!(IterAxesMut, DataMutAPI, TensorBase<DataMut<'a, R::Data>, IxD>);
impl_par_axes_iter_tensor!(
    IndexedIterAxesView,
    DataAPI,
    (IxD, TensorBase<DataRef<'a, R::Data>, IxD>)
);
impl_par_axes_iter_tensor!(
    IndexedIterAxesMut,
    DataMutAPI,
    (IxD, TensorBase<DataMut<'a, R::Data>, IxD>)
);

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_col_major() {
        let layout = [10, 10, 10].c();
        let iter_ser = IterLayoutColMajor::new(&layout).unwrap();
        let iter_par = IterLayoutColMajor::new(&layout).unwrap().into_par_iter();
        let vec_ser: Vec<usize> = iter_ser.collect();
        let mut vec_par = vec![];
        iter_par.collect_into_vec(&mut vec_par);
        assert_eq!(vec_ser, vec_par);
    }

    #[test]
    fn test_row_major() {
        let layout = [10, 10, 10].c();
        let iter_ser = IterLayoutRowMajor::new(&layout).unwrap();
        let iter_par = IterLayoutRowMajor::new(&layout).unwrap().into_par_iter();
        let vec_ser: Vec<usize> = iter_ser.collect();
        let mut vec_par = vec![];
        iter_par.collect_into_vec(&mut vec_par);
        assert_eq!(vec_ser, vec_par);
    }
}
