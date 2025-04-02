//! Implementation of function `asarray`.

use crate::prelude_dev::*;
use core::mem::ManuallyDrop;
use num::complex::{Complex32, Complex64};

pub trait AsArrayAPI<Inp> {
    type Out;

    fn asarray_f(self) -> Result<Self::Out>;

    fn asarray(self) -> Self::Out
    where
        Self: Sized,
    {
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
pub fn asarray<Args, Inp>(param: Args) -> Args::Out
where
    Args: AsArrayAPI<Inp>,
{
    return AsArrayAPI::asarray(param);
}

pub fn asarray_f<Args, Inp>(param: Args) -> Result<Args::Out>
where
    Args: AsArrayAPI<Inp>,
{
    return AsArrayAPI::asarray_f(param);
}

/* #region tensor input */

impl<R, T, B, D> AsArrayAPI<()> for (&TensorAny<R, T, B, D>, TensorIterOrder)
where
    R: DataAPI<Data = B::Raw>,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
{
    type Out = Tensor<T, B, D>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, order) = self;
        let device = input.device();
        let layout_a = input.layout();
        let layout_c = layout_for_array_copy(layout_a, order)?;
        let mut storage_c = unsafe { device.empty_impl(layout_c.size())? };
        device.assign(storage_c.raw_mut(), &layout_c, input.raw(), layout_a)?;
        let tensor = unsafe { Tensor::new_unchecked(storage_c, layout_c) };
        return Ok(tensor);
    }
}

impl<R, T, B, D> AsArrayAPI<()> for &TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
{
    type Out = Tensor<T, B, D>;

    fn asarray_f(self) -> Result<Self::Out> {
        asarray_f((self, TensorIterOrder::default()))
    }
}

impl<T, B, D> AsArrayAPI<()> for (Tensor<T, B, D>, TensorIterOrder)
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
{
    type Out = Tensor<T, B, D>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, order) = self;
        let storage_a = input.storage();
        let layout_a = input.layout();
        let device = storage_a.device();
        let layout_c = layout_for_array_copy(layout_a, order)?;
        if layout_c == *layout_a {
            return Ok(input);
        } else {
            let mut storage_c = unsafe { device.empty_impl(layout_c.size())? };
            device.assign(storage_c.raw_mut(), &layout_c, storage_a.raw(), layout_a)?;
            let tensor = unsafe { Tensor::new_unchecked(storage_c, layout_c) };
            return Ok(tensor);
        }
    }
}

impl<T, B, D> AsArrayAPI<()> for Tensor<T, B, D>
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
{
    type Out = Tensor<T, B, D>;

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
    type Out = Tensor<T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, device) = self;
        let layout = vec![input.len()].c();
        let storage = device.outof_cpu_vec(input)?;
        let tensor = unsafe { Tensor::new_unchecked(storage, layout) };
        return Ok(tensor);
    }
}

impl<T, B, D> AsArrayAPI<D> for (Vec<T>, Layout<D>, &B)
where
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout, device) = self;
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
        let tensor = unsafe { Tensor::new_unchecked(storage, layout.into_dim()?) };
        return Ok(tensor);
    }
}

impl<T, B, D> AsArrayAPI<D> for (Vec<T>, D, &B)
where
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, shape, device) = self;
        let default_order = device.default_order();
        let layout = match default_order {
            RowMajor => shape.c(),
            ColMajor => shape.f(),
        };
        asarray_f((input, layout, device))
    }
}

impl<T> AsArrayAPI<()> for Vec<T>
where
    T: Clone,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        asarray_f((self, &DeviceCpu::default()))
    }
}

#[duplicate_item(L; [D]; [Layout<D>])]
impl<T, D> AsArrayAPI<D> for (Vec<T>, L)
where
    T: Clone,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout) = self;
        asarray_f((input, layout, &DeviceCpu::default()))
    }
}

impl<T> From<Vec<T>> for Tensor<T, DeviceCpu, IxD>
where
    T: Clone,
{
    fn from(input: Vec<T>) -> Self {
        asarray_f(input).unwrap()
    }
}

/* #endregion */

/* #region slice-like input */

impl<'a, T, B, D> AsArrayAPI<D> for (&'a [T], Layout<D>, &B)
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>>,
    D: DimAPI,
{
    type Out = TensorView<'a, T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout, device) = self;
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
        let ptr = input.as_ptr();
        let len = input.len();
        let raw = unsafe {
            let ptr = ptr as *mut T;
            Vec::from_raw_parts(ptr, len, len)
        };
        let device = device.clone();
        let data = DataRef::from_manually_drop(ManuallyDrop::new(raw));
        let storage = Storage::new(data, device);
        let tensor = unsafe { TensorView::new_unchecked(storage, layout.into_dim()?) };
        return Ok(tensor);
    }
}

impl<'a, T, B, D> AsArrayAPI<D> for (&'a [T], D, &B)
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>>,
    D: DimAPI,
{
    type Out = TensorView<'a, T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, shape, device) = self;
        let default_order = device.default_order();
        let layout = match default_order {
            RowMajor => shape.c(),
            ColMajor => shape.f(),
        };
        asarray_f((input, layout, device))
    }
}

impl<'a, T, B> AsArrayAPI<()> for (&'a [T], &B)
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    type Out = TensorView<'a, T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, device) = self;
        let layout = vec![input.len()].c();
        let device = device.clone();

        let ptr = input.as_ptr();
        let len = input.len();
        let raw = unsafe {
            let ptr = ptr as *mut T;
            Vec::from_raw_parts(ptr, len, len)
        };
        let data = DataRef::from_manually_drop(ManuallyDrop::new(raw));
        let storage = Storage::new(data, device);
        let tensor = unsafe { TensorView::new_unchecked(storage, layout) };
        return Ok(tensor);
    }
}

#[duplicate_item(L; [D]; [Layout<D>])]
impl<'a, T, D> AsArrayAPI<D> for (&'a [T], L)
where
    T: Clone,
    D: DimAPI,
{
    type Out = TensorView<'a, T, DeviceCpu, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout) = self;
        asarray_f((input, layout, &DeviceCpu::default()))
    }
}

impl<'a, T> AsArrayAPI<()> for &'a [T]
where
    T: Clone,
{
    type Out = TensorView<'a, T, DeviceCpu, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        asarray_f((self, &DeviceCpu::default()))
    }
}

#[duplicate_item(L; [D]; [Layout<D>])]
impl<'a, T, B, D> AsArrayAPI<D> for (&'a Vec<T>, L, &B)
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>> + 'a,
    D: DimAPI,
{
    type Out = TensorView<'a, T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout, device) = self;
        asarray_f((input.as_slice(), layout, device))
    }
}

impl<'a, T, B> AsArrayAPI<()> for (&'a Vec<T>, &B)
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    type Out = TensorView<'a, T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, device) = self;
        asarray_f((input.as_slice(), device))
    }
}

#[duplicate_item(L; [D]; [Layout<D>])]
impl<'a, T, D> AsArrayAPI<D> for (&'a Vec<T>, L)
where
    T: Clone,
    D: DimAPI,
{
    type Out = TensorView<'a, T, DeviceCpu, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout) = self;
        asarray_f((input.as_slice(), layout, &DeviceCpu::default()))
    }
}

impl<'a, T> AsArrayAPI<()> for &'a Vec<T>
where
    T: Clone,
{
    type Out = TensorView<'a, T, DeviceCpu, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        asarray_f((self.as_slice(), &DeviceCpu::default()))
    }
}

impl<'a, T> From<&'a [T]> for TensorView<'a, T, DeviceCpu, IxD>
where
    T: Clone,
{
    fn from(input: &'a [T]) -> Self {
        asarray(input)
    }
}

impl<'a, T> From<&'a Vec<T>> for TensorView<'a, T, DeviceCpu, IxD>
where
    T: Clone,
{
    fn from(input: &'a Vec<T>) -> Self {
        asarray(input)
    }
}

/* #endregion */

/* #region slice-like mutable input */

impl<'a, T, B, D> AsArrayAPI<D> for (&'a mut [T], Layout<D>, &B)
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>>,
    D: DimAPI,
{
    type Out = TensorMut<'a, T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout, device) = self;
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
        let ptr = input.as_ptr();
        let len = input.len();
        let raw = unsafe {
            let ptr = ptr as *mut T;
            Vec::from_raw_parts(ptr, len, len)
        };
        let device = device.clone();
        let data = DataMut::from_manually_drop(ManuallyDrop::new(raw));
        let storage = Storage::new(data, device);
        let tensor = unsafe { TensorMut::new_unchecked(storage, layout.into_dim()?) };
        return Ok(tensor);
    }
}

impl<'a, T, B, D> AsArrayAPI<D> for (&'a mut [T], D, &B)
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>>,
    D: DimAPI,
{
    type Out = TensorMut<'a, T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, shape, device) = self;
        let default_order = device.default_order();
        let layout = match default_order {
            RowMajor => shape.c(),
            ColMajor => shape.f(),
        };
        asarray_f((input, layout, device))
    }
}

impl<'a, T, B> AsArrayAPI<()> for (&'a mut [T], &B)
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    type Out = TensorMut<'a, T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, device) = self;
        let layout = [input.len()].c();
        let device = device.clone();

        let ptr = input.as_ptr();
        let len = input.len();
        let raw = unsafe {
            let ptr = ptr as *mut T;
            Vec::from_raw_parts(ptr, len, len)
        };
        let data = DataMut::from_manually_drop(ManuallyDrop::new(raw));
        let storage = Storage::new(data, device);
        let tensor = unsafe { TensorMut::new_unchecked(storage, layout.into_dim()?) };
        return Ok(tensor);
    }
}

#[duplicate_item(L; [D]; [Layout<D>])]
impl<'a, T, D> AsArrayAPI<D> for (&'a mut [T], L)
where
    T: Clone,
    D: DimAPI,
{
    type Out = TensorMut<'a, T, DeviceCpu, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout) = self;
        asarray_f((input, layout, &DeviceCpu::default()))
    }
}

impl<'a, T> AsArrayAPI<()> for &'a mut [T]
where
    T: Clone,
{
    type Out = TensorMut<'a, T, DeviceCpu, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        asarray_f((self, &DeviceCpu::default()))
    }
}

#[duplicate_item(L; [D]; [Layout<D>])]
impl<'a, T, B, D> AsArrayAPI<D> for (&'a mut Vec<T>, L, &B)
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>>,
    D: DimAPI,
{
    type Out = TensorMut<'a, T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout, device) = self;
        asarray_f((input.as_mut_slice(), layout, device))
    }
}

impl<'a, T, B> AsArrayAPI<()> for (&'a mut Vec<T>, &B)
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    type Out = TensorMut<'a, T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, device) = self;
        asarray_f((input.as_mut_slice(), device))
    }
}

#[duplicate_item(L; [D]; [Layout<D>])]
impl<'a, T, D> AsArrayAPI<D> for (&'a mut Vec<T>, L)
where
    T: Clone,
    D: DimAPI,
{
    type Out = TensorMut<'a, T, DeviceCpu, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout) = self;
        asarray_f((input.as_mut_slice(), layout, &DeviceCpu::default()))
    }
}

impl<'a, T> AsArrayAPI<()> for &'a mut Vec<T>
where
    T: Clone,
{
    type Out = TensorMut<'a, T, DeviceCpu, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        asarray_f((self.as_mut_slice(), &DeviceCpu::default()))
    }
}

impl<'a, T> From<&'a mut [T]> for TensorMut<'a, T, DeviceCpu, IxD>
where
    T: Clone,
{
    fn from(input: &'a mut [T]) -> Self {
        asarray(input)
    }
}

impl<'a, T> From<&'a mut Vec<T>> for TensorMut<'a, T, DeviceCpu, IxD>
where
    T: Clone,
{
    fn from(input: &'a mut Vec<T>) -> Self {
        asarray(input)
    }
}

/* #endregion */

/* #region scalar input */

macro_rules! impl_asarray_scalar {
    ($($t:ty),*) => {
        $(
            impl<B> AsArrayAPI<()> for ($t, &B)
            where
                B: DeviceAPI<$t> + DeviceCreationAnyAPI<$t>,
            {
                type Out = Tensor<$t, B, IxD>;

                fn asarray_f(self) -> Result<Self::Out> {
                    let (input, device) = self;
                    let layout = Layout::new(vec![], vec![], 0)?;
                    let storage = device.outof_cpu_vec(vec![input])?;
                    let tensor = unsafe { Tensor::new_unchecked(storage, layout) };
                    return Ok(tensor);
                }
            }

            impl AsArrayAPI<()> for $t {
                type Out = Tensor<$t, DeviceCpu, IxD>;

                fn asarray_f(self) -> Result<Self::Out> {
                    asarray_f((self, &DeviceCpu::default()))
                }
            }
        )*
    };
}

impl_asarray_scalar!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64, Complex32, Complex64
);

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
        println!("{:?}", tensor.raw().as_ptr());
        println!("{:?}", tensor);

        let tensor = asarray_f((&tensor, TensorIterOrder::K)).unwrap();
        println!("{:?}", tensor);

        let tensor = asarray_f((tensor, TensorIterOrder::K)).unwrap();
        println!("{:?}", tensor);
    }

    #[test]
    fn test_asarray_scalar() {
        let tensor = asarray_f(1).unwrap();
        println!("{:?}", tensor);
        let tensor = asarray_f((Complex64::new(0., 1.), &DeviceCpuSerial::default())).unwrap();
        println!("{:?}", tensor);
    }
}
