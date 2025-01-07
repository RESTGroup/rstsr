use crate::prelude_dev::*;

pub struct IterVecViewColMajor<'a, T, D>
where
    D: DimDevAPI,
{
    layout_iter: IterLayoutColMajor<D>,
    view: &'a [T],
}

impl<'a, T, D> Iterator for IterVecViewColMajor<'a, T, D>
where
    D: DimDevAPI,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.layout_iter.next().map(|idx| &self.view[idx])
    }
}

impl<T, D> DoubleEndedIterator for IterVecViewColMajor<'_, T, D>
where
    D: DimDevAPI,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.layout_iter.next_back().map(|idx| &self.view[idx])
    }
}

impl<T, D> ExactSizeIterator for IterVecViewColMajor<'_, T, D>
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
    pub fn iter(&self) -> IterVecViewColMajor<'a, T, D> {
        let order = match TensorOrder::default() {
            TensorOrder::C => TensorIterOrder::C,
            TensorOrder::F => TensorIterOrder::F,
        };
        let layout = translate_to_col_major_unary(self.layout(), order).unwrap();
        let layout_iter = IterLayoutColMajor::new(&layout).unwrap();
        let rawvec = self.rawvec().as_ref();

        // SAFETY: The lifetime of `rawvec` is guaranteed to be at least `'a`.
        // transmute is to change the lifetime, not for type casting.
        let iter = IterVecViewColMajor { layout_iter, view: rawvec };
        unsafe { core::mem::transmute(iter) }
    }
}

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
}
