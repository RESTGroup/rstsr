//! Cast the most contiguous dimension as array.

use crate::prelude_dev::*;
use core::mem::ManuallyDrop;

/* #region impl directly to PackableArrayAPI */

// DataOwned

impl<T, const N: usize> PackableArrayAPI<T, N> for DataOwned<Vec<T>> {
    type Array = [T; N];
    type ArrayVec = DataOwned<Vec<[T; N]>>;
}

impl<T> PackArrayAPI<T> for DataOwned<Vec<T>> {
    fn pack_array_f<const N: usize>(self) -> Result<<Self as PackableArrayAPI<T, N>>::ArrayVec> {
        let raw = self.into_raw();
        let raw = raw.pack_array_f::<N>()?;
        Ok(DataOwned::from(raw))
    }
}

impl<T, const N: usize> UnpackArrayAPI for DataOwned<Vec<[T; N]>> {
    type Output = DataOwned<Vec<T>>;

    fn unpack_array(self) -> Self::Output {
        let raw = self.into_raw();
        let raw = raw.unpack_array();
        DataOwned::from(raw)
    }
}

// DataRef

impl<'l, T, const N: usize> PackableArrayAPI<T, N> for DataRef<'l, Vec<T>> {
    type Array = [T; N];
    type ArrayVec = DataRef<'l, Vec<[T; N]>>;
}

impl<'l, T> PackArrayAPI<T> for DataRef<'l, Vec<T>> {
    fn pack_array_f<const N: usize>(self) -> Result<<Self as PackableArrayAPI<T, N>>::ArrayVec> {
        let raw = self.raw().as_slice().pack_array_f::<N>()?;
        let vec = unsafe { Vec::from_raw_parts(raw.as_ptr() as *mut [T; N], raw.len(), raw.len()) };
        Ok(DataRef::from_manually_drop(ManuallyDrop::new(vec)))
    }
}

impl<'l, T, const N: usize> UnpackArrayAPI for DataRef<'l, Vec<[T; N]>> {
    type Output = DataRef<'l, Vec<T>>;

    fn unpack_array(self) -> Self::Output {
        let raw = self.raw().as_slice().unpack_array();
        let vec = unsafe { Vec::from_raw_parts(raw.as_ptr() as *mut T, raw.len(), raw.len()) };
        DataRef::from_manually_drop(ManuallyDrop::new(vec))
    }
}

// DataMut

impl<'l, T, const N: usize> PackableArrayAPI<T, N> for DataMut<'l, Vec<T>> {
    type Array = [T; N];
    type ArrayVec = DataMut<'l, Vec<[T; N]>>;
}

impl<'l, T> PackArrayAPI<T> for DataMut<'l, Vec<T>> {
    fn pack_array_f<const N: usize>(self) -> Result<<Self as PackableArrayAPI<T, N>>::ArrayVec> {
        let raw = self.raw().as_slice().pack_array_f::<N>()?;
        let vec = unsafe { Vec::from_raw_parts(raw.as_ptr() as *mut [T; N], raw.len(), raw.len()) };
        Ok(DataMut::from_manually_drop(ManuallyDrop::new(vec)))
    }
}

impl<'l, T, const N: usize> UnpackArrayAPI for DataMut<'l, Vec<[T; N]>> {
    type Output = DataMut<'l, Vec<T>>;

    fn unpack_array(self) -> Self::Output {
        let raw = self.raw().as_slice().unpack_array();
        let vec = unsafe { Vec::from_raw_parts(raw.as_ptr() as *mut T, raw.len(), raw.len()) };
        DataMut::from_manually_drop(ManuallyDrop::new(vec))
    }
}

// DataCow

impl<'l, T, const N: usize> PackableArrayAPI<T, N> for DataCow<'l, Vec<T>> {
    type Array = [T; N];
    type ArrayVec = DataCow<'l, Vec<[T; N]>>;
}

impl<'l, T> PackArrayAPI<T> for DataCow<'l, Vec<T>> {
    fn pack_array_f<const N: usize>(self) -> Result<<Self as PackableArrayAPI<T, N>>::ArrayVec> {
        match self {
            DataCow::Owned(data) => Ok(DataCow::Owned(data.pack_array_f::<N>()?)),
            DataCow::Ref(data) => Ok(DataCow::Ref(data.pack_array_f::<N>()?)),
        }
    }
}

impl<'l, T, const N: usize> UnpackArrayAPI for DataCow<'l, Vec<[T; N]>> {
    type Output = DataCow<'l, Vec<T>>;

    fn unpack_array(self) -> Self::Output {
        match self {
            DataCow::Owned(data) => DataCow::Owned(data.unpack_array()),
            DataCow::Ref(data) => DataCow::Ref(data.unpack_array()),
        }
    }
}

/* #endregion */

/* #region into_pack_array */

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    #[substitute_item(
        ArrayData [<R as PackableArrayAPI<T, N>>::ArrayVec];
        ArrayType [<R as PackableArrayAPI<T, N>>::Array];
    )]
    #[allow(clippy::type_complexity)]
    pub fn into_pack_array_f<const N: usize>(
        self,
        axis: isize,
    ) -> Result<TensorAny<ArrayData, ArrayType, B, D::SmallerOne>>
    where
        B: DeviceAPI<ArrayType>,
        R: PackableArrayAPI<T, N> + PackArrayAPI<T>,
        ArrayData: DataAPI<Data = <B as DeviceRawAPI<ArrayType>>::Raw>,
    {
        // check if the axis is valid
        // dimension check
        let axis = if axis < 0 { self.ndim() as isize + axis } else { axis };
        rstsr_pattern!(axis, 0..self.ndim() as isize, ValueOutOfRange)?;
        let axis = axis as usize;
        rstsr_assert_eq!(self.layout().stride()[axis], 1, InvalidLayout, "The axis must be contiguous")?;
        rstsr_assert_eq!(self.layout().shape()[axis], N, InvalidLayout, "The axis length must be a exactly {N}")?;
        rstsr_assert!(self.layout().offset() % N == 0, InvalidLayout, "The offset must be a multiple of {N}")?;

        let (storage, layout) = self.into_raw_parts();
        let (data, device) = storage.into_raw_parts();
        let data = data.pack_array_f::<N>()?;
        let storage = Storage::new(data, device);
        let layout = layout.dim_select(axis as isize, 0)?;
        let stride = layout
            .stride()
            .as_ref()
            .iter()
            .map(|&s| s / N as isize)
            .collect_vec()
            .try_into()
            .unwrap_or_else(|_| panic!("stride conversion failed"));
        let new_offset = layout.offset() / N;
        let new_layout = unsafe { Layout::new_unchecked(layout.shape().clone(), stride, new_offset) };
        let tensor = unsafe { TensorAny::new_unchecked(storage, new_layout) };
        Ok(tensor)
    }

    #[substitute_item(
        ArrayData [<R as PackableArrayAPI<T, N>>::ArrayVec];
        ArrayType [<R as PackableArrayAPI<T, N>>::Array];
    )]
    #[allow(clippy::type_complexity)]
    pub fn into_pack_array<const N: usize>(self, axis: isize) -> TensorAny<ArrayData, ArrayType, B, D::SmallerOne>
    where
        B: DeviceAPI<ArrayType>,
        R: PackableArrayAPI<T, N> + PackArrayAPI<T>,
        ArrayData: DataAPI<Data = <B as DeviceRawAPI<ArrayType>>::Raw>,
    {
        self.into_pack_array_f::<N>(axis).unwrap()
    }
}

/* #endregion */

/* #region into_unpack_array */

impl<R, T, B, D, const N: usize> TensorAny<R, [T; N], B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<[T; N]>>::Raw>,
    B: DeviceAPI<T> + DeviceAPI<[T; N]>,
    D: DimAPI + DimLargerOneAPI,
    D::LargerOne: DimAPI,
{
    #[substitute_item(ROut [<R as UnpackArrayAPI>::Output])]
    pub fn into_unpack_array_f(self, axis: isize) -> Result<TensorAny<ROut, T, B, D::LargerOne>>
    where
        R: UnpackArrayAPI,
        ROut: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
        B: DeviceAPI<T>,
    {
        // dimension check
        let axis = if axis < 0 { self.ndim() as isize + axis + 1 } else { axis };
        rstsr_pattern!(axis, 0..(self.ndim() + 1) as isize, ValueOutOfRange)?;
        let axis = axis as usize;

        let (storage, layout) = self.into_raw_parts();
        let (data, device) = storage.into_raw_parts();
        let data = data.unpack_array();
        let storage = Storage::new(data, device);

        let mut shape = layout.shape().as_ref().to_vec();
        let mut stride = layout.stride().as_ref().to_vec();
        let mut offset = layout.offset();

        shape.insert(axis, N);
        stride.iter_mut().map(|s| *s *= N as isize).count();
        stride.insert(axis, 1);
        offset *= N;
        let layout = unsafe { Layout::new_unchecked(shape, stride, offset) };
        let layout = layout.into_dim().unwrap();
        let tensor = unsafe { TensorAny::new_unchecked(storage, layout) };
        Ok(tensor)
    }

    #[substitute_item(ROut [<R as UnpackArrayAPI>::Output])]
    pub fn into_unpack_array(self, axis: isize) -> TensorAny<ROut, T, B, D::LargerOne>
    where
        R: UnpackArrayAPI,
        ROut: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
        B: DeviceAPI<T>,
    {
        self.into_unpack_array_f(axis).unwrap()
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_pack_array_owned() {
        let device = DeviceCpuSerial::default();
        let a = asarray((vec![1, 2, 3, 4, 5, 6], [3, 2].c(), &device));
        let b = a.into_pack_array_f::<2>(-1).unwrap();
        println!("{b:?}");
        assert_eq!(b.raw(), &vec![[1, 2], [3, 4], [5, 6]]);

        let c = b.into_unpack_array(-1);
        println!("{c:?}");
        assert_eq!(c.raw(), &vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_pack_array_ref() {
        let device = DeviceCpuSerial::default();
        let vec = vec![1, 2, 3, 4, 5, 6];
        let a = asarray((&vec, [3, 2].c(), &device));
        let b = a.into_pack_array_f::<2>(-1).unwrap();
        println!("{b:?}");
        assert_eq!(b.raw(), &vec![[1, 2], [3, 4], [5, 6]]);

        let c = b.into_unpack_array(-1);
        println!("{c:?}");
        assert_eq!(c.raw(), &vec![1, 2, 3, 4, 5, 6]);
    }
}
