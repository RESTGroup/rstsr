// Indexing of tensors

use core::ops::{Index, IndexMut};

use crate::prelude_dev::*;

/* #region slice */

pub fn into_slice_f<S, D, I>(tensor: TensorBase<S, D>, index: I) -> Result<TensorBase<S, IxD>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<Indexer>>,
    Error: From<I::Error>,
{
    let (data, layout) = tensor.into_raw_parts();
    let index = index.try_into()?;
    let layout = layout.dim_slice(index.as_ref())?;
    return unsafe { Ok(TensorBase::new_unchecked(data, layout)) };
}

pub fn into_slice<S, D, I>(tensor: TensorBase<S, D>, index: I) -> TensorBase<S, IxD>
where
    D: DimAPI,
    I: TryInto<AxesIndex<Indexer>>,
    Error: From<I::Error>,
{
    into_slice_f(tensor, index).unwrap()
}

pub fn slice_f<R, T, B, D, I>(
    tensor: &TensorAny<R, T, B, D>,
    index: I,
) -> Result<TensorView<'_, T, B, IxD>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<Indexer>>,
    Error: From<I::Error>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_slice_f(tensor.view(), index)
}

pub fn slice<R, T, B, D, I>(tensor: &TensorAny<R, T, B, D>, index: I) -> TensorView<'_, T, B, IxD>
where
    D: DimAPI,
    I: TryInto<AxesIndex<Indexer>>,
    Error: From<I::Error>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    slice_f(tensor, index).unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    pub fn into_slice_f<I>(self, index: I) -> Result<TensorAny<R, T, B, IxD>>
    where
        I: TryInto<AxesIndex<Indexer>>,
        Error: From<I::Error>,
    {
        into_slice_f(self, index)
    }

    pub fn into_slice<I>(self, index: I) -> TensorAny<R, T, B, IxD>
    where
        I: TryInto<AxesIndex<Indexer>>,
        Error: From<I::Error>,
    {
        into_slice(self, index)
    }

    pub fn slice_f<I>(&self, index: I) -> Result<TensorView<'_, T, B, IxD>>
    where
        I: TryInto<AxesIndex<Indexer>>,
        Error: From<I::Error>,
    {
        slice_f(self, index)
    }

    pub fn slice<I>(&self, index: I) -> TensorView<'_, T, B, IxD>
    where
        I: TryInto<AxesIndex<Indexer>>,
        Error: From<I::Error>,
    {
        slice(self, index)
    }

    pub fn i_f<I>(&self, index: I) -> Result<TensorView<'_, T, B, IxD>>
    where
        I: TryInto<AxesIndex<Indexer>>,
        Error: From<I::Error>,
    {
        slice_f(self, index)
    }

    pub fn i<I>(&self, index: I) -> TensorView<'_, T, B, IxD>
    where
        I: TryInto<AxesIndex<Indexer>>,
        Error: From<I::Error>,
    {
        slice(self, index)
    }
}

/* #endregion */

/* #region slice mut */

pub fn into_slice_mut_f<S, D, I>(tensor: TensorBase<S, D>, index: I) -> Result<TensorBase<S, IxD>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<Indexer>>,
    Error: From<I::Error>,
{
    let (data, layout) = tensor.into_raw_parts();
    let index = index.try_into()?;
    let layout = layout.dim_slice(index.as_ref())?;
    return unsafe { Ok(TensorBase::new_unchecked(data, layout)) };
}

pub fn into_slice_mut<S, D, I>(tensor: TensorBase<S, D>, index: I) -> TensorBase<S, IxD>
where
    D: DimAPI,
    I: TryInto<AxesIndex<Indexer>>,
    Error: From<I::Error>,
{
    into_slice_mut_f(tensor, index).unwrap()
}

pub fn slice_mut_f<R, T, B, D, I>(
    tensor: &mut TensorAny<R, T, B, D>,
    index: I,
) -> Result<TensorMut<'_, T, B, IxD>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<Indexer>>,
    Error: From<I::Error>,
    R: DataMutAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_slice_mut_f(tensor.view_mut(), index)
}

pub fn slice_mut<R, T, B, D, I>(
    tensor: &mut TensorAny<R, T, B, D>,
    index: I,
) -> TensorMut<'_, T, B, IxD>
where
    D: DimAPI,
    I: TryInto<AxesIndex<Indexer>>,
    Error: From<I::Error>,
    R: DataMutAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    slice_mut_f(tensor, index).unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataMutAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    pub fn into_slice_mut_f<I>(self, index: I) -> Result<TensorAny<R, T, B, IxD>>
    where
        I: TryInto<AxesIndex<Indexer>>,
        Error: From<I::Error>,
    {
        into_slice_mut_f(self, index)
    }

    pub fn into_slice_mut<I>(self, index: I) -> TensorAny<R, T, B, IxD>
    where
        I: TryInto<AxesIndex<Indexer>>,
        Error: From<I::Error>,
    {
        into_slice_mut(self, index)
    }

    pub fn slice_mut_f<I>(&mut self, index: I) -> Result<TensorMut<'_, T, B, IxD>>
    where
        I: TryInto<AxesIndex<Indexer>>,
        Error: From<I::Error>,
    {
        slice_mut_f(self, index)
    }

    pub fn slice_mut<I>(&mut self, index: I) -> TensorMut<'_, T, B, IxD>
    where
        I: TryInto<AxesIndex<Indexer>>,
        Error: From<I::Error>,
    {
        slice_mut(self, index)
    }

    pub fn i_mut_f<I>(&mut self, index: I) -> Result<TensorMut<'_, T, B, IxD>>
    where
        I: TryInto<AxesIndex<Indexer>>,
        Error: From<I::Error>,
    {
        slice_mut_f(self, index)
    }

    pub fn i_mut<I>(&mut self, index: I) -> TensorMut<'_, T, B, IxD>
    where
        I: TryInto<AxesIndex<Indexer>>,
        Error: From<I::Error>,
    {
        slice_mut(self, index)
    }
}

/* #endregion */

/* #region diagonal */

pub struct DiagonalArgs {
    pub offset: Option<isize>,
    pub axis1: Option<isize>,
    pub axis2: Option<isize>,
}

impl From<(isize, isize, isize)> for DiagonalArgs {
    fn from(args: (isize, isize, isize)) -> Self {
        let (offset, axis1, axis2) = args;
        Self { offset: Some(offset), axis1: Some(axis1), axis2: Some(axis2) }
    }
}

impl From<(usize, isize, isize)> for DiagonalArgs {
    fn from(args: (usize, isize, isize)) -> Self {
        let (offset, axis1, axis2) = args;
        Self { offset: Some(offset as isize), axis1: Some(axis1), axis2: Some(axis2) }
    }
}

impl From<(usize, usize, usize)> for DiagonalArgs {
    fn from(args: (usize, usize, usize)) -> Self {
        let (offset, axis1, axis2) = args;
        Self {
            offset: Some(offset as isize),
            axis1: Some(axis1 as isize),
            axis2: Some(axis2 as isize),
        }
    }
}

impl From<isize> for DiagonalArgs {
    fn from(offset: isize) -> Self {
        Self { offset: Some(offset), axis1: None, axis2: None }
    }
}

impl From<usize> for DiagonalArgs {
    fn from(offset: usize) -> Self {
        Self { offset: Some(offset as isize), axis1: None, axis2: None }
    }
}

impl From<()> for DiagonalArgs {
    fn from(_: ()) -> Self {
        Self { offset: None, axis1: None, axis2: None }
    }
}

impl From<Option<isize>> for DiagonalArgs {
    fn from(offset: Option<isize>) -> Self {
        Self { offset, axis1: None, axis2: None }
    }
}

pub fn into_diagonal_f<S, D>(
    tensor: TensorBase<S, D>,
    diagonal_args: impl Into<DiagonalArgs>,
) -> Result<TensorBase<S, D::SmallerOne>>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    let (data, layout) = tensor.into_raw_parts();
    let DiagonalArgs { offset, axis1, axis2 } = diagonal_args.into();
    let layout = layout.diagonal(offset, axis1, axis2)?;
    return unsafe { Ok(TensorBase::new_unchecked(data, layout)) };
}

pub fn into_diagonal<S, D>(
    tensor: TensorBase<S, D>,
    diagonal_args: impl Into<DiagonalArgs>,
) -> TensorBase<S, D::SmallerOne>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    into_diagonal_f(tensor, diagonal_args).unwrap()
}

pub fn diagonal_f<R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    diagonal_args: impl Into<DiagonalArgs>,
) -> Result<TensorView<'_, T, B, D::SmallerOne>>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_diagonal_f(tensor.view(), diagonal_args)
}

pub fn diagonal<R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    diagonal_args: impl Into<DiagonalArgs>,
) -> TensorView<'_, T, B, D::SmallerOne>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    diagonal_f(tensor, diagonal_args).unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    pub fn into_diagonal_f(
        self,
        diagonal_args: impl Into<DiagonalArgs>,
    ) -> Result<TensorAny<R, T, B, D::SmallerOne>> {
        into_diagonal_f(self, diagonal_args)
    }

    pub fn into_diagonal(
        self,
        diagonal_args: impl Into<DiagonalArgs>,
    ) -> TensorAny<R, T, B, D::SmallerOne> {
        into_diagonal(self, diagonal_args)
    }

    pub fn diagonal_f(
        &self,
        diagonal_args: impl Into<DiagonalArgs>,
    ) -> Result<TensorView<'_, T, B, D::SmallerOne>> {
        diagonal_f(self, diagonal_args)
    }

    pub fn diagonal(
        &self,
        diagonal_args: impl Into<DiagonalArgs>,
    ) -> TensorView<'_, T, B, D::SmallerOne> {
        diagonal(self, diagonal_args)
    }
}

/* #endregion */

/* #region diagonal_mut */

pub fn into_diagonal_mut_f<S, D>(
    tensor: TensorBase<S, D>,
    diagonal_args: impl Into<DiagonalArgs>,
) -> Result<TensorBase<S, D::SmallerOne>>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    let (data, layout) = tensor.into_raw_parts();
    let DiagonalArgs { offset, axis1, axis2 } = diagonal_args.into();
    let layout = layout.diagonal(offset, axis1, axis2)?;
    return unsafe { Ok(TensorBase::new_unchecked(data, layout)) };
}

pub fn into_diagonal_mut<S, D>(
    tensor: TensorBase<S, D>,
    diagonal_args: impl Into<DiagonalArgs>,
) -> TensorBase<S, D::SmallerOne>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    into_diagonal_mut_f(tensor, diagonal_args).unwrap()
}

pub fn diagonal_mut_f<R, T, B, D>(
    tensor: &mut TensorAny<R, T, B, D>,
    diagonal_args: impl Into<DiagonalArgs>,
) -> Result<TensorMut<'_, T, B, D::SmallerOne>>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    R: DataMutAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_diagonal_mut_f(tensor.view_mut(), diagonal_args)
}

pub fn diagonal_mut<R, T, B, D>(
    tensor: &mut TensorAny<R, T, B, D>,
    diagonal_args: impl Into<DiagonalArgs>,
) -> TensorMut<'_, T, B, D::SmallerOne>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    R: DataMutAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    diagonal_mut_f(tensor, diagonal_args).unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataMutAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    pub fn into_diagonal_mut_f(
        self,
        diagonal_args: impl Into<DiagonalArgs>,
    ) -> Result<TensorAny<R, T, B, D::SmallerOne>> {
        into_diagonal_mut_f(self, diagonal_args)
    }

    pub fn into_diagonal_mut(
        self,
        diagonal_args: impl Into<DiagonalArgs>,
    ) -> TensorAny<R, T, B, D::SmallerOne> {
        into_diagonal_mut(self, diagonal_args)
    }

    pub fn diagonal_mut_f(
        &mut self,
        diagonal_args: impl Into<DiagonalArgs>,
    ) -> Result<TensorMut<'_, T, B, D::SmallerOne>> {
        diagonal_mut_f(self, diagonal_args)
    }

    pub fn diagonal_mut(
        &mut self,
        diagonal_args: impl Into<DiagonalArgs>,
    ) -> TensorMut<'_, T, B, D::SmallerOne> {
        diagonal_mut(self, diagonal_args)
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
            T: Clone,
            D: DimAPI,
            B: DeviceAPI<T, Raw = Vec<T>>,
            I: AsRef<[usize]>,
        {
            type Output = T;

            #[inline]
            fn index(&self, index: I) -> &Self::Output {
                let index = index.as_ref().iter().map(|&v| v as isize).collect::<Vec<_>>();
                let i = self.layout().index(index.as_ref());
                let raw = self.raw();
                raw.index(i)
            }
        }
    };
}

impl_Index_for_Tensor!(Tensor<T, B, D>);
impl_Index_for_Tensor!(TensorView<'_, T, B, D>);
impl_Index_for_Tensor!(TensorViewMut<'_, T, B, D>);
impl_Index_for_Tensor!(TensorCow<'_, T, B, D>);

macro_rules! impl_IndexMut_for_Tensor {
    ($tensor_struct: ty) => {
        impl<T, D, B, I> IndexMut<I> for $tensor_struct
        where
            T: Clone,
            D: DimAPI,
            B: DeviceAPI<T, Raw = Vec<T>>,
            I: AsRef<[usize]>,
        {
            #[inline]
            fn index_mut(&mut self, index: I) -> &mut Self::Output {
                let index = index.as_ref().iter().map(|&v| v as isize).collect::<Vec<_>>();
                let i = self.layout().index(index.as_ref());
                let raw = self.raw_mut();
                raw.index_mut(i)
            }
        }
    };
}

impl_IndexMut_for_Tensor!(Tensor<T, B, D>);
impl_IndexMut_for_Tensor!(TensorViewMut<'_, T, B, D>);

/* #endregion */

/* #region indexing (unchecked) */

macro_rules! impl_IndexUncheck_for_Tensor {
    ($tensor_struct: ty) => {
        impl<T, B, D> $tensor_struct
        where
            T: Clone,
            D: DimAPI,
            B: DeviceAPI<T, Raw = Vec<T>>,
        {
            /// # Safety
            ///
            /// This function is unsafe because it does not check the validity of the
            /// index.
            #[inline]
            pub unsafe fn index_uncheck<I>(&self, index: I) -> &T
            where
                I: AsRef<[usize]>,
            {
                let index = index.as_ref();
                let i = unsafe { self.layout().index_uncheck(index) } as usize;
                let raw = self.raw();
                raw.index(i)
            }
        }
    };
}

impl_IndexUncheck_for_Tensor!(Tensor<T, B, D>);
impl_IndexUncheck_for_Tensor!(TensorView<'_, T, B, D>);
impl_IndexUncheck_for_Tensor!(TensorViewMut<'_, T, B, D>);
impl_IndexUncheck_for_Tensor!(TensorCow<'_, T, B, D>);

macro_rules! impl_IndexMutUncheck_for_Tensor {
    ($tensor_struct: ty) => {
        impl<T, B, D> $tensor_struct
        where
            T: Clone,
            D: DimAPI,
            B: DeviceAPI<T, Raw = Vec<T>>,
        {
            /// # Safety
            ///
            /// This function is unsafe because it does not check the validity of the
            /// index.
            #[inline]
            pub unsafe fn index_mut_uncheck<I>(&mut self, index: I) -> &mut T
            where
                I: AsRef<[usize]>,
            {
                let index = index.as_ref();
                let i = unsafe { self.layout().index_uncheck(index) } as usize;
                let raw = self.raw_mut();
                raw.index_mut(i)
            }
        }
    };
}
impl_IndexMutUncheck_for_Tensor!(Tensor<T, B, D>);
impl_IndexMutUncheck_for_Tensor!(TensorViewMut<'_, T, B, D>);

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
