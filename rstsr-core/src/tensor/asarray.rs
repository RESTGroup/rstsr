//! Implementation of function `asarray`.

use core::mem::ManuallyDrop;

use crate::prelude_dev::*;

pub trait AsArrayAPI<Param>: Sized {
    fn asarray(param: Param) -> Result<Self>;
}

/// Convert the input to an array.
///
/// This function takes kinds of input and converts them to an array. Please
/// refer to trait implementations of [`AsArrayAPI`].
///
/// # See also
///
/// [Python array API: `asarray`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.asarray.html)
pub fn asarray<Rhs, Param>(param: Param) -> Result<Rhs>
where
    Rhs: AsArrayAPI<Param>,
{
    return Rhs::asarray(param);
}

impl<R, T, D, B> AsArrayAPI<(&TensorBase<R, D>, TensorIterOrder)> for Tensor<T, D, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
{
    fn asarray(param: (&TensorBase<R, D>, TensorIterOrder)) -> Result<Self> {
        let (input, order) = param;
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

impl<T, D, B> AsArrayAPI<(Tensor<T, D, B>, TensorIterOrder)> for Tensor<T, D, B>
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
{
    fn asarray(param: (Tensor<T, D, B>, TensorIterOrder)) -> Result<Self> {
        let (input, order) = param;
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

impl<T, B> AsArrayAPI<(Vec<T>, Option<&B>)> for Tensor<T, Ix1, B>
where
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    fn asarray(param: (Vec<T>, Option<&B>)) -> Result<Self> {
        let (input, device) = param;
        let layout = [input.len()].c();
        let device_binding = B::default();
        let device = device.unwrap_or(&device_binding);
        let storage = device.outof_cpu_vec(input)?;
        let data = DataOwned::from(storage);
        let tensor = unsafe { Tensor::new_unchecked(data, layout) };
        return Ok(tensor);
    }
}

impl<T> AsArrayAPI<Vec<T>> for Tensor<T, Ix1, DeviceCpu>
where
    T: Clone,
{
    fn asarray(input: Vec<T>) -> Result<Self> {
        let layout = [input.len()].c();
        let device = DeviceCpu::default();
        let storage = Storage::new(input, device);
        let data = DataOwned::from(storage);
        let tensor = unsafe { Tensor::new_unchecked(data, layout) };
        return Ok(tensor);
    }
}

impl<T, B, const N: usize> AsArrayAPI<([T; N], Option<&B>)> for Tensor<T, Ix1, B>
where
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    fn asarray(param: ([T; N], Option<&B>)) -> Result<Self> {
        let (input, device) = param;
        let layout = [input.len()].c();
        let device_binding = B::default();
        let device = device.unwrap_or(&device_binding);
        let storage = device.outof_cpu_vec(input.into())?;
        let data = DataOwned::from(storage);
        let tensor = unsafe { Tensor::new_unchecked(data, layout) };
        return Ok(tensor);
    }
}

impl<T, const N: usize> AsArrayAPI<[T; N]> for Tensor<T, Ix1, DeviceCpu>
where
    T: Clone,
{
    fn asarray(input: [T; N]) -> Result<Self> {
        let layout = [input.len()].c();
        let device = DeviceCpu::default();
        let storage = Storage::new(input.into(), device);
        let data = DataOwned::from(storage);
        let tensor = unsafe { Tensor::new_unchecked(data, layout) };
        return Ok(tensor);
    }
}

impl<'a, T> AsArrayAPI<&'a [T]> for TensorView<'a, T, Ix1, DeviceCpu>
where
    T: Clone,
{
    fn asarray(input: &'a [T]) -> Result<Self> {
        let layout = [input.len()].c();
        let device = DeviceCpu::default();

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

/* #region vector casting to tensor */

/// One dimension vector can be simply casted to tensor in CPU.
impl<T> From<Vec<T>> for Tensor<T, Ix1, DeviceCpu>
where
    T: Clone,
{
    fn from(data: Vec<T>) -> Self {
        let size = data.len();
        let device = DeviceCpu::default();
        let storage = Storage { rawvec: data, device };
        let data = DataOwned { storage };
        let layout = [size].into();
        Tensor::new(data, layout).unwrap()
    }
}

/* #endregion */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asarray() {
        let input = vec![1, 2, 3];
        let tensor = Tensor::<_, Ix1, _>::asarray(input).unwrap();
        println!("{:?}", tensor);
        let input = [1, 2, 3];
        let tensor = Tensor::<_, Ix1, _>::asarray(input).unwrap();
        println!("{:?}", tensor);

        let input = vec![1, 2, 3];
        println!("{:?}", input.as_ptr());
        let tensor = TensorView::asarray(&input).unwrap();
        println!("{:?}", tensor.data().storage().rawvec().as_ptr());
        println!("{:?}", tensor);

        let tensor = Tensor::asarray((&tensor, TensorIterOrder::K)).unwrap();
        println!("{:?}", tensor);

        let tensor = Tensor::asarray((tensor, TensorIterOrder::K)).unwrap();
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
