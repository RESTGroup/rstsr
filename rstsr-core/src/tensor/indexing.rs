// Indexing of tensors

use crate::prelude_dev::*;

/* #region slice */

pub trait TensorSliceAPI<'a, Idx> {
    type Out;

    fn slice_f(&'a self, index: Idx) -> Result<Self::Out>;

    fn slice(&'a self, index: Idx) -> Self::Out {
        Self::slice_f(self, index).unwrap()
    }
}

impl<'a, R, D> TensorSliceAPI<'a, &[Indexer]> for TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
    R::Data: 'a,
{
    type Out = TensorBase<DataRef<'a, R::Data>, IxD>;

    fn slice_f(&'a self, index: &[Indexer]) -> Result<Self::Out> {
        let layout = self.layout().dim_slice(index)?;
        let data = self.view().into_data();
        return unsafe { Ok(TensorBase::new_unchecked(data, layout)) };
    }
}

macro_rules! impl_TensorSliceAPI {
    ($($type:ty),*) => {
        $(
            impl<'a, R, D> TensorSliceAPI<'a, $type> for TensorBase<R, D>
            where
                R: DataAPI,
                D: DimAPI + DimSmallerOneAPI,
                R::Data: 'a,
                D::SmallerOne: DimAPI,
            {
                type Out = TensorBase<DataRef<'a, R::Data>, D::SmallerOne>;

                fn slice_f(&'a self, index: $type) -> Result<Self::Out> {
                    let layout = self.layout().dim_select(0, index as isize)?;
                    let data = self.view().into_data();
                    return unsafe { Ok(TensorBase::new_unchecked(data, layout)) };
                }
            }
        )*
    };
}

impl_TensorSliceAPI!(usize, isize, u8, i8, u16, i16, u32, i32, u64, i64, u128, i128);

impl<'a, R, D> TensorSliceAPI<'a, SliceI> for TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
    R::Data: 'a,
{
    type Out = TensorBase<DataRef<'a, R::Data>, D>;

    fn slice_f(&'a self, index: SliceI) -> Result<Self::Out> {
        let layout = self.layout().dim_narrow(0, index)?;
        let data = self.view().into_data();
        return unsafe { Ok(TensorBase::new_unchecked(data, layout)) };
    }
}

/* #endregion */

/* #region slice */

pub trait TensorSliceMutAPI<'a, Idx> {
    type Out;

    fn slice_mut_f(&'a mut self, index: Idx) -> Result<Self::Out>;

    fn slice_mut(&'a mut self, index: Idx) -> Self::Out {
        Self::slice_mut_f(self, index).unwrap()
    }
}

impl<'a, R, D> TensorSliceMutAPI<'a, &[Indexer]> for TensorBase<R, D>
where
    R: DataMutAPI,
    D: DimAPI,
    R::Data: 'a,
{
    type Out = TensorBase<DataMut<'a, R::Data>, IxD>;

    fn slice_mut_f(&'a mut self, index: &[Indexer]) -> Result<Self::Out> {
        let layout = self.layout().dim_slice(index)?;
        let data = self.view_mut().into_data();
        return unsafe { Ok(TensorBase::new_unchecked(data, layout)) };
    }
}

macro_rules! impl_TensorSliceMutAPI {
    ($($type:ty),*) => {
        $(
            impl<'a, R, D> TensorSliceMutAPI<'a, $type> for TensorBase<R, D>
            where
                R: DataMutAPI,
                D: DimAPI + DimSmallerOneAPI,
                R::Data: 'a,
                D::SmallerOne: DimAPI,
            {
                type Out = TensorBase<DataMut<'a, R::Data>, D::SmallerOne>;

                fn slice_mut_f(&'a mut self, index: $type) -> Result<Self::Out> {
                    let layout = self.layout().dim_select(0, index as isize)?;
                    let data = self.view_mut().into_data();
                    return unsafe { Ok(TensorBase::new_unchecked(data, layout)) };
                }
            }
        )*
    };
}

impl_TensorSliceMutAPI!(usize, isize, u8, i8, u16, i16, u32, i32, u64, i64, u128, i128);

impl<'a, R, D> TensorSliceMutAPI<'a, SliceI> for TensorBase<R, D>
where
    R: DataMutAPI,
    D: DimAPI,
    R::Data: 'a,
{
    type Out = TensorBase<DataMut<'a, R::Data>, D>;

    fn slice_mut_f(&'a mut self, index: SliceI) -> Result<Self::Out> {
        let layout = self.layout().dim_narrow(0, index)?;
        let data = self.view_mut().into_data();
        return unsafe { Ok(TensorBase::new_unchecked(data, layout)) };
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_tensor_slice() {
        let tensor = asarray(vec![1, 2, 3, 4, 5]);
        let tensor_slice = tensor.slice(s!(1..4));
        println!("{:?}", tensor_slice);
        let tensor_slice = tensor.slice(s!(1..4, None));
        println!("{:?}", tensor_slice);
        let tensor_slice = tensor.slice(1);
        println!("{:?}", tensor_slice);
        let tensor_slice = tensor.slice(slice!(2, 7, 2));
        println!("{:?}", tensor_slice);

        let mut tensor = asarray(vec![1, 2, 3, 4, 5]);
        let mut tensor_slice = tensor.slice_mut(s!(1..4));
        tensor_slice += 10;
        println!("{:?}", tensor);
        *&mut tensor.slice_mut(s!(1..4)) += 10;
        println!("{:?}", tensor);
    }
}
