//! Implementation of function `asarray`.

use core::mem::ManuallyDrop;

use crate::prelude_dev::*;

pub trait AsArrayAPI<Param>: Sized {
    fn asarray_f(param: Param) -> Result<Self>;

    fn asarray(param: Param) -> Self {
        Self::asarray_f(param).unwrap()
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
pub fn asarray<Rhs, Param>(param: Param) -> Rhs
where
    Rhs: AsArrayAPI<Param>,
{
    return Rhs::asarray(param);
}

pub fn asarray_f<Rhs, Param>(param: Param) -> Result<Rhs>
where
    Rhs: AsArrayAPI<Param>,
{
    return Rhs::asarray_f(param);
}

impl<R, T, D, B> AsArrayAPI<(&TensorBase<R, D>, TensorIterOrder)> for Tensor<T, D, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
{
    fn asarray_f(param: (&TensorBase<R, D>, TensorIterOrder)) -> Result<Self> {
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
    fn asarray_f(param: (Tensor<T, D, B>, TensorIterOrder)) -> Result<Self> {
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

impl<T, B> AsArrayAPI<(Vec<T>, &B)> for Tensor<T, Ix1, B>
where
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    fn asarray_f(param: (Vec<T>, &B)) -> Result<Self> {
        let (input, device) = param;
        let layout = [input.len()].c();
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
    fn asarray_f(input: Vec<T>) -> Result<Self> {
        Self::asarray_f((input, &DeviceCpu::default()))
    }
}

impl<T, B, const N: usize> AsArrayAPI<([T; N], &B)> for Tensor<T, Ix1, B>
where
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    fn asarray_f(param: ([T; N], &B)) -> Result<Self> {
        let (input, device) = param;
        let layout = [input.len()].c();
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
    fn asarray_f(input: [T; N]) -> Result<Self> {
        Self::asarray_f((input, &DeviceCpu::default()))
    }
}

impl<T, B> AsArrayAPI<(&[Vec<T>], &B)> for Tensor<T, Ix2, B>
where
    T: Clone,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    fn asarray_f(param: (&[Vec<T>], &B)) -> Result<Self> {
        let (input, device) = param;
        // zero rows
        if input.is_empty() {
            return Tensor::<T, Ix1, B>::asarray_f((vec![], device))?
                .into_shape_assume_contig_f([0, 0]);
        }
        // check columns
        let nrow = input.len();
        let ncol = input[0].len();
        for row in input.iter() {
            if row.len() != ncol {
                rstsr_assert_eq!(
                    row.len(),
                    ncol,
                    InvalidLayout,
                    "element numbers in later rows do not match the first row"
                )?;
            }
        }
        let new_vec = input.iter().flatten().cloned().collect::<Vec<_>>();
        let tensor = Tensor::<T, Ix1, B>::asarray_f((new_vec, device))?;
        return tensor.into_shape_assume_contig_f([nrow, ncol]);
    }
}

impl<T> AsArrayAPI<&[Vec<T>]> for Tensor<T, Ix2, DeviceCpu>
where
    T: Clone,
{
    fn asarray_f(input: &[Vec<T>]) -> Result<Self> {
        Self::asarray_f((input, &DeviceCpu::default()))
    }
}

impl<T, B, const N: usize, const M: usize> AsArrayAPI<([[T; M]; N], &B)> for Tensor<T, Ix2, B>
where
    T: Clone,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    fn asarray_f(param: ([[T; M]; N], &B)) -> Result<Self> {
        let (input, device) = param;
        let new_vec = input.iter().flatten().cloned().collect::<Vec<_>>();
        let tensor = Tensor::<T, Ix1, B>::asarray_f((new_vec, device))?;
        return tensor.into_shape_assume_contig_f([N, M]);
    }
}

impl<T, const N: usize, const M: usize> AsArrayAPI<[[T; M]; N]> for Tensor<T, Ix2, DeviceCpu>
where
    T: Clone,
{
    fn asarray_f(input: [[T; M]; N]) -> Result<Self> {
        Self::asarray_f((input, &DeviceCpu::default()))
    }
}

impl<'a, T, B> AsArrayAPI<(&'a [T], &B)> for TensorView<'a, T, Ix1, B>
where
    T: Clone,
    B: DeviceAPI<T, RawVec = Vec<T>>,
{
    fn asarray_f(input: (&'a [T], &B)) -> Result<Self> {
        let (input, device) = input;
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

impl<'a, T> AsArrayAPI<&'a [T]> for TensorView<'a, T, Ix1, DeviceCpu>
where
    T: Clone,
{
    fn asarray_f(input: &'a [T]) -> Result<Self> {
        Self::asarray_f((input, &DeviceCpu::default()))
    }
}

impl<'a, T> AsArrayAPI<&'a Vec<T>> for TensorView<'a, T, Ix1, DeviceCpu>
where
    T: Clone,
{
    fn asarray_f(input: &'a Vec<T>) -> Result<Self> {
        Self::asarray_f((input.as_ref(), &DeviceCpu::default()))
    }
}

impl<T, R, D> From<T> for TensorBase<R, D>
where
    D: DimAPI,
    TensorBase<R, D>: AsArrayAPI<T>,
{
    fn from(input: T) -> Self {
        TensorBase::<R, D>::asarray_f(input).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asarray() {
        let input = vec![1, 2, 3];
        let tensor = Tensor::<_, Ix1, _>::asarray_f(input).unwrap();
        println!("{:?}", tensor);
        let input = [1, 2, 3];
        let tensor = Tensor::<_, Ix1, _>::asarray_f(input).unwrap();
        println!("{:?}", tensor);

        let input = vec![1, 2, 3];
        println!("{:?}", input.as_ptr());
        let tensor = TensorView::asarray_f(&input).unwrap();
        println!("{:?}", tensor.data().storage().rawvec().as_ptr());
        println!("{:?}", tensor);

        let tensor = Tensor::asarray_f((&tensor, TensorIterOrder::K)).unwrap();
        println!("{:?}", tensor);

        let tensor = Tensor::asarray_f((tensor, TensorIterOrder::K)).unwrap();
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
