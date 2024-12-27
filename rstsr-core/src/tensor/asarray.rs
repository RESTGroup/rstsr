//! Implementation of function `asarray`.

use core::mem::ManuallyDrop;

use crate::prelude_dev::*;

pub trait AsArrayAPI<Inp>: Sized {
    type Out;

    fn asarray_f(self) -> Result<Self::Out>;

    fn asarray(self) -> Self::Out {
        Self::asarray_f(self).unwrap()
    }
}

/// Convert the input to an array.
///
/// This function takes kinds of input and converts them to an array. Please
/// refer to trait implementations of [`AsArrayAPI`].
///
/// # See also
///
/// [Python array API: `asarray`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.asarray.html)
pub fn asarray<Param, Inp, Rhs>(param: Param) -> Rhs
where
    Param: AsArrayAPI<Inp, Out = Rhs>,
{
    return AsArrayAPI::asarray(param);
}

pub fn asarray_f<Param, Inp, Rhs>(param: Param) -> Result<Rhs>
where
    Param: AsArrayAPI<Inp, Out = Rhs>,
{
    return AsArrayAPI::asarray_f(param);
}

/* #region tensor input */

impl<R, T, D, B> AsArrayAPI<()> for (&TensorBase<R, D>, TensorIterOrder)
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
{
    type Out = Tensor<T, D, B>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, order) = self;
        let layout_a = input.layout();
        let storage_a = input.data().storage();
        let device = input.device();
        let layout_c = layout_for_array_copy(layout_a, order)?;
        let mut storage_c = unsafe { device.empty_impl(layout_c.size())? };
        device.assign(&mut storage_c, &layout_c, storage_a, layout_a)?;
        let data = DataOwned::from(storage_c);
        let tensor = unsafe { Tensor::new_unchecked(data, layout_c) };
        return Ok(tensor);
    }
}

impl<R, T, D, B> AsArrayAPI<()> for &TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
{
    type Out = Tensor<T, D, B>;

    fn asarray_f(self) -> Result<Self::Out> {
        asarray_f((self, TensorIterOrder::default()))
    }
}

impl<T, D, B> AsArrayAPI<()> for (Tensor<T, D, B>, TensorIterOrder)
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
{
    type Out = Tensor<T, D, B>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, order) = self;
        let layout_a = input.layout();
        let storage_a = input.data().storage();
        let device = input.device();
        let layout_c = layout_for_array_copy(layout_a, order)?;
        if layout_c == *layout_a {
            return Ok(input);
        } else {
            let mut storage_c = unsafe { device.empty_impl(layout_c.size())? };
            device.assign(&mut storage_c, &layout_c, storage_a, layout_a)?;
            let data = DataOwned::from(storage_c);
            let tensor = unsafe { Tensor::new_unchecked(data, layout_c) };
            return Ok(tensor);
        }
    }
}

impl<T, D, B> AsArrayAPI<()> for Tensor<T, D, B>
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
{
    type Out = Tensor<T, D, B>;

    fn asarray_f(self) -> Result<Self::Out> {
        asarray_f((self, TensorIterOrder::default()))
    }
}

/* #endregion */

/* #region vec-like input */

impl<T, B> AsArrayAPI<()> for (Vec<T>, &B)
where
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, Ix1, B>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, device) = self;
        let layout = [input.len()].c();
        let storage = device.outof_cpu_vec(input)?;
        let data = DataOwned::from(storage);
        let tensor = unsafe { Tensor::new_unchecked(data, layout) };
        return Ok(tensor);
    }
}

impl<T, B, D, L> AsArrayAPI<D> for (Vec<T>, L, &B)
where
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
    L: Into<Layout<D>>,
{
    type Out = Tensor<T, D, B>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout, device) = self;
        let layout: Layout<D> = layout.into();
        rstsr_assert_eq!(
            layout.bounds_index()?,
            (0, layout.size()),
            InvalidLayout,
            "This constructor assumes compact memory layout."
        )?;
        rstsr_assert_eq!(
            layout.size(),
            input.len(),
            InvalidLayout,
            "This constructor assumes that the layout size is equal to the input size."
        )?;
        let storage = device.outof_cpu_vec(input)?;
        let data = DataOwned::from(storage);
        let tensor = unsafe { Tensor::new_unchecked(data, layout) };
        return Ok(tensor);
    }
}

impl<T> AsArrayAPI<()> for Vec<T>
where
    T: Clone,
{
    type Out = Tensor<T, Ix1, DeviceCpu>;

    fn asarray_f(self) -> Result<Self::Out> {
        asarray_f((self, &DeviceCpu::default()))
    }
}

impl<T, D, L> AsArrayAPI<D> for (Vec<T>, L)
where
    T: Clone,
    D: DimAPI,
    L: Into<Layout<D>>,
{
    type Out = Tensor<T, D, DeviceCpu>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout) = self;
        asarray_f((input, layout, &DeviceCpu::default()))
    }
}

impl<T> From<Vec<T>> for Tensor<T, Ix1, DeviceCpu>
where
    T: Clone,
{
    fn from(input: Vec<T>) -> Self {
        asarray(input)
    }
}

/* #endregion */

/* #region slice-like input */

impl<'a, T, B> AsArrayAPI<()> for (&'a [T], &B)
where
    T: Clone,
    B: DeviceAPI<T, RawVec = Vec<T>> + 'a,
{
    type Out = TensorView<'a, T, Ix1, B>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, device) = self;
        let layout = [input.len()].c();
        let device = device.clone();

        let ptr = input.as_ptr();
        let len = input.len();
        let rawvec: Vec<T> = unsafe {
            let ptr = ptr as *mut T;
            Vec::from_raw_parts(ptr, len, len)
        };
        let storage = ManuallyDrop::new(Storage::new(rawvec, device));
        let data = DataRef::from_manually_drop(storage);
        let tensor = unsafe { TensorView::new_unchecked(data, layout) };
        return Ok(tensor);
    }
}

impl<'a, T> AsArrayAPI<()> for &'a [T]
where
    T: Clone,
{
    type Out = TensorView<'a, T, Ix1, DeviceCpu>;

    fn asarray_f(self) -> Result<Self::Out> {
        asarray_f((self, &DeviceCpu::default()))
    }
}

impl<'a, T, B> AsArrayAPI<()> for (&'a Vec<T>, &B)
where
    T: Clone,
    B: DeviceAPI<T, RawVec = Vec<T>> + 'a,
{
    type Out = TensorView<'a, T, Ix1, B>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, device) = self;
        asarray_f((input.as_slice(), device))
    }
}

impl<'a, T> AsArrayAPI<()> for &'a Vec<T>
where
    T: Clone,
{
    type Out = TensorView<'a, T, Ix1, DeviceCpu>;

    fn asarray_f(self) -> Result<Self::Out> {
        asarray_f((self.as_slice(), &DeviceCpu::default()))
    }
}

impl<'a, T> From<&'a [T]> for TensorView<'a, T, Ix1, DeviceCpu>
where
    T: Clone,
{
    fn from(input: &'a [T]) -> Self {
        asarray(input)
    }
}

impl<'a, T> From<&'a Vec<T>> for TensorView<'a, T, Ix1, DeviceCpu>
where
    T: Clone,
{
    fn from(input: &'a Vec<T>) -> Self {
        asarray(input)
    }
}

/* #endregion */

/* #region slice-like mutable input */

impl<'a, T, B> AsArrayAPI<()> for (&'a mut [T], &B)
where
    T: Clone,
    B: DeviceAPI<T, RawVec = Vec<T>> + 'a,
{
    type Out = TensorMut<'a, T, Ix1, B>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, device) = self;
        let layout = [input.len()].c();
        let device = device.clone();

        let ptr = input.as_ptr();
        let len = input.len();
        let rawvec: Vec<T> = unsafe {
            let ptr = ptr as *mut T;
            Vec::from_raw_parts(ptr, len, len)
        };
        let storage = ManuallyDrop::new(Storage::new(rawvec, device));
        let data = DataMut::from_manually_drop(storage);
        let tensor = unsafe { TensorMut::new_unchecked(data, layout) };
        return Ok(tensor);
    }
}

impl<'a, T> AsArrayAPI<()> for &'a mut [T]
where
    T: Clone,
{
    type Out = TensorMut<'a, T, Ix1, DeviceCpu>;

    fn asarray_f(self) -> Result<Self::Out> {
        asarray_f((self, &DeviceCpu::default()))
    }
}

impl<'a, T, B> AsArrayAPI<()> for (&'a mut Vec<T>, &B)
where
    T: Clone,
    B: DeviceAPI<T, RawVec = Vec<T>> + 'a,
{
    type Out = TensorMut<'a, T, Ix1, B>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, device) = self;
        asarray_f((input.as_mut_slice(), device))
    }
}

impl<'a, T> AsArrayAPI<()> for &'a mut Vec<T>
where
    T: Clone,
{
    type Out = TensorMut<'a, T, Ix1, DeviceCpu>;

    fn asarray_f(self) -> Result<Self::Out> {
        asarray_f((self.as_mut_slice(), &DeviceCpu::default()))
    }
}

impl<'a, T> From<&'a mut [T]> for TensorMut<'a, T, Ix1, DeviceCpu>
where
    T: Clone,
{
    fn from(input: &'a mut [T]) -> Self {
        asarray(input)
    }
}

impl<'a, T> From<&'a mut Vec<T>> for TensorMut<'a, T, Ix1, DeviceCpu>
where
    T: Clone,
{
    fn from(input: &'a mut Vec<T>) -> Self {
        asarray(input)
    }
}

/* #endregion */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asarray() {
        let input = vec![1, 2, 3];
        let tensor = asarray_f(input).unwrap();
        println!("{:?}", tensor);
        let input = [1, 2, 3];
        let tensor = asarray_f(input.as_ref()).unwrap();
        println!("{:?}", tensor);

        let input = vec![1, 2, 3];
        let tensor = asarray_f(&input).unwrap();
        println!("{:?}", tensor.data().storage().rawvec().as_ptr());
        println!("{:?}", tensor);

        let tensor = asarray_f((&tensor, TensorIterOrder::K)).unwrap();
        println!("{:?}", tensor);

        let tensor = asarray_f((tensor, TensorIterOrder::K)).unwrap();
        println!("{:?}", tensor);
    }

    #[test]
    fn vec_cast_to_tensor() {
        use crate::layout::*;
        let a = Tensor::<f64, Ix<2>> {
            data: Storage { rawvec: vec![1.12345, 2.0], device: DeviceCpu::default() }.into(),
            layout: [1, 2].new_c_contig(None),
        };
        println!("{a:6.3?}");
    }
}
