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

/* #region tensor iterator */

#[duplicate_item(IterTensor; [IterVecView]; [IterVecMut]; [IndexedIterVecView]; [IndexedIterVecMut])]
impl<'a, T, D> IntoParallelIterator for IterTensor<'a, T, D>
where
    D: DimDevAPI,
    T: Send + Sync,
{
    type Item = <IterTensor<'a, T, D> as Iterator>::Item;
    type Iter = ParIterRSTSR<Self>;

    fn into_par_iter(self) -> Self::Iter {
        Self::Iter { iter: self }
    }
}

#[duplicate_item(IterTensor; [IterAxesView]; [IterAxesMut]; [IndexedIterAxesView]; [IndexedIterAxesMut])]
impl<'a, T, B> IntoParallelIterator for IterTensor<'a, T, B>
where
    T: Send + Sync,
    B::Raw: Send,
    B: DeviceAPI<T> + Send,
{
    type Item = <IterTensor<'a, T, B> as Iterator>::Item;
    type Iter = ParIterRSTSR<Self>;

    fn into_par_iter(self) -> Self::Iter {
        Self::Iter { iter: self }
    }
}

/* #endregion */

/* #region col-major layout dim dispatch */

pub fn layout_col_major_dim_dispatch_par_1<D, F>(la: &Layout<D>, f: F) -> Result<()>
where
    D: DimAPI,
    F: Fn(usize) + Send + Sync,
{
    #[cfg(feature = "dispatch_dim_layout_iter")]
    {
        macro_rules! dispatch {
            ($dim: ident) => {{
                let iter_a = IterLayoutColMajor::new(&la.to_dim::<$dim>()?)?;
                iter_a.into_par_iter().for_each(f);
            }};
        }
        match la.ndim() {
            0 => f(la.offset()),
            1 => dispatch!(Ix1),
            2 => dispatch!(Ix2),
            3 => dispatch!(Ix3),
            4 => dispatch!(Ix4),
            5 => dispatch!(Ix5),
            6 => dispatch!(Ix6),
            _ => {
                let iter_a = IterLayoutColMajor::new(la)?;
                iter_a.into_par_iter().for_each(f);
            },
        }
    }

    #[cfg(not(feature = "dispatch_dim_layout_iter"))]
    {
        let iter_a = IterLayoutColMajor::new(la)?;
        iter_a.into_par_iter().for_each(f);
    }
    Ok(())
}

pub fn layout_col_major_dim_dispatch_par_2<D, F>(la: &Layout<D>, lb: &Layout<D>, f: F) -> Result<()>
where
    D: DimAPI,
    F: Fn((usize, usize)) + Send + Sync,
{
    rstsr_assert_eq!(la.ndim(), lb.ndim(), RuntimeError)?;

    #[cfg(feature = "dispatch_dim_layout_iter")]
    {
        macro_rules! dispatch {
            ($dim: ident) => {{
                let iter_a = IterLayoutColMajor::new(&la.to_dim::<$dim>()?)?;
                let iter_b = IterLayoutColMajor::new(&lb.to_dim::<$dim>()?)?;
                (iter_a, iter_b).into_par_iter().for_each(f);
            }};
        }
        match la.ndim() {
            0 => f((la.offset(), lb.offset())),
            1 => dispatch!(Ix1),
            2 => dispatch!(Ix2),
            3 => dispatch!(Ix3),
            4 => dispatch!(Ix4),
            5 => dispatch!(Ix5),
            6 => dispatch!(Ix6),
            _ => {
                let iter_a = IterLayoutColMajor::new(la)?;
                let iter_b = IterLayoutColMajor::new(lb)?;
                (iter_a, iter_b).into_par_iter().for_each(f);
            },
        }
    }

    #[cfg(not(feature = "dispatch_dim_layout_iter"))]
    {
        let iter_a = IterLayoutColMajor::new(la)?;
        let iter_b = IterLayoutColMajor::new(lb)?;
        (iter_a, iter_b).into_par_iter().for_each(f);
    }
    Ok(())
}

pub fn layout_col_major_dim_dispatch_par_3<D, F>(
    la: &Layout<D>,
    lb: &Layout<D>,
    lc: &Layout<D>,
    f: F,
) -> Result<()>
where
    D: DimAPI,
    F: Fn((usize, usize, usize)) + Send + Sync,
{
    rstsr_assert_eq!(la.ndim(), lb.ndim(), RuntimeError)?;
    rstsr_assert_eq!(la.ndim(), lc.ndim(), RuntimeError)?;

    #[cfg(feature = "dispatch_dim_layout_iter")]
    {
        macro_rules! dispatch {
            ($dim: ident) => {{
                let iter_a = IterLayoutColMajor::new(&la.to_dim::<$dim>()?)?;
                let iter_b = IterLayoutColMajor::new(&lb.to_dim::<$dim>()?)?;
                let iter_c = IterLayoutColMajor::new(&lc.to_dim::<$dim>()?)?;
                (iter_a, iter_b, iter_c).into_par_iter().for_each(f);
            }};
        }
        match la.ndim() {
            0 => f((la.offset(), lb.offset(), lc.offset())),
            1 => dispatch!(Ix1),
            2 => dispatch!(Ix2),
            3 => dispatch!(Ix3),
            4 => dispatch!(Ix4),
            5 => dispatch!(Ix5),
            6 => dispatch!(Ix6),
            _ => {
                let iter_a = IterLayoutColMajor::new(la)?;
                let iter_b = IterLayoutColMajor::new(lb)?;
                let iter_c = IterLayoutColMajor::new(lc)?;
                (iter_a, iter_b, iter_c).into_par_iter().for_each(f);
            },
        }
    }

    #[cfg(not(feature = "dispatch_dim_layout_iter"))]
    {
        let iter_a = IterLayoutColMajor::new(la)?;
        let iter_b = IterLayoutColMajor::new(lb)?;
        let iter_c = IterLayoutColMajor::new(lc)?;
        (iter_a, iter_b, iter_c).into_par_iter().for_each(f);
    }
    Ok(())
}

pub fn layout_col_major_dim_dispatch_par_2diff<DA, DB, F>(
    la: &Layout<DA>,
    lb: &Layout<DB>,
    f: F,
) -> Result<()>
where
    DA: DimAPI,
    DB: DimAPI,
    F: Fn((usize, usize)) + Send + Sync,
{
    #[cfg(feature = "dispatch_dim_layout_iter")]
    {
        macro_rules! dispatch {
            ($dima: ident, $dimb: ident) => {{
                let iter_a = IterLayoutColMajor::new(&la.to_dim::<$dima>()?)?;
                let iter_b = IterLayoutColMajor::new(&lb.to_dim::<$dimb>()?)?;
                (iter_a, iter_b).into_par_iter().for_each(f);
            }};
        }
        match (la.ndim(), lb.ndim()) {
            (0, 0) => f((la.offset(), lb.offset())),
            (1, 1) => dispatch!(Ix1, Ix1),
            (1, 2) => dispatch!(Ix1, Ix2),
            (1, 3) => dispatch!(Ix1, Ix3),
            (1, 4) => dispatch!(Ix1, Ix4),
            (1, 5) => dispatch!(Ix1, Ix5),
            (1, 6) => dispatch!(Ix1, Ix6),
            (2, 1) => dispatch!(Ix2, Ix1),
            (2, 2) => dispatch!(Ix2, Ix2),
            (2, 3) => dispatch!(Ix2, Ix3),
            (2, 4) => dispatch!(Ix2, Ix4),
            (2, 5) => dispatch!(Ix2, Ix5),
            (2, 6) => dispatch!(Ix2, Ix6),
            (3, 1) => dispatch!(Ix3, Ix1),
            (3, 2) => dispatch!(Ix3, Ix2),
            (3, 3) => dispatch!(Ix3, Ix3),
            (3, 4) => dispatch!(Ix3, Ix4),
            (3, 5) => dispatch!(Ix3, Ix5),
            (3, 6) => dispatch!(Ix3, Ix6),
            (4, 1) => dispatch!(Ix4, Ix1),
            (4, 2) => dispatch!(Ix4, Ix2),
            (4, 3) => dispatch!(Ix4, Ix3),
            (4, 4) => dispatch!(Ix4, Ix4),
            (4, 5) => dispatch!(Ix4, Ix5),
            (4, 6) => dispatch!(Ix4, Ix6),
            (5, 1) => dispatch!(Ix5, Ix1),
            (5, 2) => dispatch!(Ix5, Ix2),
            (5, 3) => dispatch!(Ix5, Ix3),
            (5, 4) => dispatch!(Ix5, Ix4),
            (5, 5) => dispatch!(Ix5, Ix5),
            (5, 6) => dispatch!(Ix5, Ix6),
            (6, 1) => dispatch!(Ix6, Ix1),
            (6, 2) => dispatch!(Ix6, Ix2),
            (6, 3) => dispatch!(Ix6, Ix3),
            (6, 4) => dispatch!(Ix6, Ix4),
            (6, 5) => dispatch!(Ix6, Ix5),
            (6, 6) => dispatch!(Ix6, Ix6),
            _ => {
                let iter_a = IterLayoutColMajor::new(la)?;
                let iter_b = IterLayoutColMajor::new(lb)?;
                (iter_a, iter_b).into_par_iter().for_each(f);
            },
        }
    }

    #[cfg(not(feature = "dispatch_dim_layout_iter"))]
    {
        let iter_a = IterLayoutColMajor::new(la)?;
        let iter_b = IterLayoutColMajor::new(lb)?;
        (iter_a, iter_b).into_par_iter().for_each(f);
    }
    Ok(())
}

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
