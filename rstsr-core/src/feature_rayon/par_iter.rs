//! Layout parallel iterator

use crate::prelude_dev::*;
use rayon::prelude::*;

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
