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

    fn i_f(&'a self, index: Idx) -> Result<Self::Out> {
        Self::slice_f(self, index)
    }

    fn i(&'a self, index: Idx) -> Self::Out {
        Self::slice(self, index)
    }
}

impl<'a, R, D, I> TensorSliceAPI<'a, I> for TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
    R::Data: 'a,
    I: Into<AxesIndex<Indexer>>,
{
    type Out = TensorBase<DataRef<'a, R::Data>, IxD>;

    fn slice_f(&'a self, index: I) -> Result<Self::Out> {
        let index = index.into();
        let layout = self.layout().dim_slice(index.as_ref())?;
        let data = self.view().into_data();
        return unsafe { Ok(TensorBase::new_unchecked(data, layout)) };
    }
}

/* #endregion */

/* #region slice mut */

pub trait TensorSliceMutAPI<'a, Idx> {
    type Out;

    fn slice_mut_f(&'a mut self, index: Idx) -> Result<Self::Out>;

    fn slice_mut(&'a mut self, index: Idx) -> Self::Out {
        Self::slice_mut_f(self, index).unwrap()
    }

    fn i_mut_f(&'a mut self, index: Idx) -> Result<Self::Out> {
        Self::slice_mut_f(self, index)
    }

    fn i_mut(&'a mut self, index: Idx) -> Self::Out {
        Self::slice_mut(self, index)
    }
}

impl<'a, R, D, I> TensorSliceMutAPI<'a, I> for TensorBase<R, D>
where
    R: DataMutAPI,
    D: DimAPI,
    R::Data: 'a,
    I: Into<AxesIndex<Indexer>>,
{
    type Out = TensorBase<DataMut<'a, R::Data>, IxD>;

    fn slice_mut_f(&'a mut self, index: I) -> Result<Self::Out> {
        let index = index.into();
        let layout = self.layout().dim_slice(index.as_ref())?;
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
    fn test_tensor_slice_1d() {
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
    fn test_tensor_nd() {
        let tensor = arange(24.0).into_shape([2, 3, 4]);
        let tensor_slice = tensor.slice(s![1..2, 1..3, 1..4]);
        println!("{:?}", tensor_slice);
        let tensor_slice = tensor.slice(s![1]);
        println!("{:?}", tensor_slice);
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
            *&mut tensor_view_mut.slice_mut(4) += 1;
        }
        println!("{:?}", value);
        println!("{:?}", tensor);
    }
}
