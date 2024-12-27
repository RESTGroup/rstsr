// Indexing of tensors

use core::ops::{Index, IndexMut};

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

/* #region indexing */

// It seems that implementing Index for TensorBase is not possible because of
// the lifetime issue. However, directly implementing each struct can avoid such
// problem.

macro_rules! impl_Index_for_Tensor {
    ($tensor_struct: ty) => {
        impl<T, D, B, I> Index<I> for $tensor_struct
        where
            D: DimAPI,
            B: DeviceAPI<T, RawVec = Vec<T>>,
            I: AsRef<[usize]>,
        {
            type Output = T;

            fn index(&self, index: I) -> &Self::Output {
                let index = index.as_ref().iter().map(|&v| v as isize).collect::<Vec<_>>();
                let i = self.layout().index(index.as_ref());
                let rawvec = self.storage().rawvec();
                rawvec.index(i)
            }
        }
    };
}

impl_Index_for_Tensor!(Tensor<T, D, B>);
impl_Index_for_Tensor!(TensorView<'_, T, D, B>);
impl_Index_for_Tensor!(TensorViewMut<'_, T, D, B>);
impl_Index_for_Tensor!(TensorCow<'_, T, D, B>);

macro_rules! impl_IndexMut_for_Tensor {
    ($tensor_struct: ty) => {
        impl<T, D, B, I> IndexMut<I> for $tensor_struct
        where
            D: DimAPI,
            B: DeviceAPI<T, RawVec = Vec<T>>,
            I: AsRef<[usize]>,
        {
            fn index_mut(&mut self, index: I) -> &mut Self::Output {
                let index = index.as_ref().iter().map(|&v| v as isize).collect::<Vec<_>>();
                let i = self.layout().index(index.as_ref());
                let rawvec = self.storage_mut().rawvec_mut();
                rawvec.index_mut(i)
            }
        }
    };
}

impl_IndexMut_for_Tensor!(Tensor<T, D, B>);
impl_IndexMut_for_Tensor!(TensorViewMut<'_, T, D, B>);

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_tensor_slice() {
        let tensor = asarray(vec![1, 2, 3, 4, 5]);
        let tensor_slice = tensor.slice(s![1..4]);
        println!("{:?}", tensor_slice);
        let tensor_slice = tensor.slice(s![1..4, None]);
        println!("{:?}", tensor_slice);
        let tensor_slice = tensor.slice(1);
        println!("{:?}", tensor_slice);
        let tensor_slice = tensor.slice(slice!(2, 7, 2));
        println!("{:?}", tensor_slice);

        let mut tensor = asarray(vec![1, 2, 3, 4, 5]);
        let mut tensor_slice = tensor.slice_mut(s![1..4]);
        tensor_slice += 10;
        println!("{:?}", tensor);
        *&mut tensor.slice_mut(s![1..4]) += 10;
        println!("{:?}", tensor);
    }

    #[test]
    fn test_tensor_index() {
        let mut tensor = asarray(vec![1, 2, 3, 4, 5]);
        let value = tensor[[1]];
        println!("{:?}", value);
        let tensor_view = tensor.view();
        let value = tensor_view[[2]];
        {
            let tensor_view = tensor.view();
            let value = tensor_view[[3]];
            println!("{:?}", value);
            let mut tensor_view_mut = tensor.view_mut();
            tensor_view_mut[[4]] += 1;
        }
        println!("{:?}", value);
        println!("{:?}", tensor);
    }
}
