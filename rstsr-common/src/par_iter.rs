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

#[duplicate_item(IterLayout; [IterLayoutColMajor]; [IterLayoutRowMajor]; [IterLayout]; [IndexedIterLayout])]
impl<D> IntoParallelIterator for IterLayout<D>
where
    D: DimDevAPI,
{
    type Item = <IterLayout<D> as Iterator>::Item;
    type Iter = ParIterRSTSR<Self>;

    fn into_par_iter(self) -> Self::Iter {
        Self::Iter { iter: self }
    }
}

/* #endregion */
