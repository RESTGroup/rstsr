//! Creation methods for `Tensor` struct.
//!
//! This module relates to the [Python array API standard v2023.12](https://data-apis.org/array-api/2023.12/API_specification/creation_functions.html).
//!
//! Todo list:
//! - [x] `arange`: [`Tensor::arange`], [`Tensor::arange_int`]
//! - [ ] `asarray`
//! - [x] `empty`: [`Tensor::empty`]
//! - [x] `empty_like`: [`Tensor::empty_like`]
//! - [ ] `eye`
//! - [ ] ~`from_dlpack`~
//! - [x] `full`: [`Tensor::full`]
//! - [x] `full_like`: [`Tensor::full_like`]
//! - [x] `linspace`: [`Tensor::linspace`]
//! - [ ] `meshgrid`
//! - [x] `ones`: [`Tensor::ones`]
//! - [x] `ones_like`: [`Tensor::ones_like`]
//! - [ ] `tril`
//! - [ ] `triu`
//! - [x] `zeros`: [`Tensor::zeros`]
//! - [x] `zeros_like`: [`Tensor::zeros_like`]

use crate::prelude_dev::*;
use num::complex::ComplexFloat;
use num::{Float, Num};

/* #region arange */

/// Evenly spaced values within the half-open interval `[start, stop)` as
/// one-dimensional array.
///
/// # See also
///
/// - [Python array API standard: `arange`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.arange.html)
pub fn arange<T, B>(start: T, stop: T, step: T, device: &B) -> Tensor<T, Ix1, B>
where
    T: Float,
    B: DeviceAPI<T> + DeviceCreationFloatAPI<T>,
{
    let data = B::arange_impl(device, start, stop, step).unwrap();
    let layout = [data.len()].into();
    unsafe { Tensor::new_unchecked(data.into(), layout) }
}

impl<T, B> Tensor<T, Ix1, B>
where
    T: Float,
    B: DeviceAPI<T> + DeviceCreationFloatAPI<T>,
{
    /// Evenly spaced values within the half-open interval `[start, stop)` as
    /// one-dimensional array.
    ///
    /// # See also
    ///
    /// [`arange`]
    pub fn arange(start: T, stop: T, step: T, device: &B) -> Tensor<T, Ix1, B> {
        arange(start, stop, step, device)
    }
}

impl<T> Tensor<T, Ix1, DeviceCpu>
where
    T: Float,
    DeviceCpu: DeviceAPI<T> + DeviceCreationFloatAPI<T>,
{
    /// Evenly spaced values within the half-open interval `[start, stop)` as
    /// one-dimensional array.
    ///
    /// # See also
    ///
    /// [`arange`]
    pub fn arange_cpu(start: T, stop: T, step: T) -> Tensor<T, Ix1, DeviceCpu> {
        arange(start, stop, step, &DeviceCpu::default())
    }
}

/* #endregion */

/* #region arange_int */

/// Evenly spaced values within the half-open interval `[0, len)` as
/// one-dimensional array, each step 1.
pub fn arange_int<T, B>(len: usize, device: &B) -> Tensor<T, Ix1, B>
where
    T: Num,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    let data = B::arange_int_impl(device, len).unwrap();
    let layout = [data.len()].into();
    unsafe { Tensor::new_unchecked(data.into(), layout) }
}

impl<T, B> Tensor<T, Ix1, B>
where
    T: Num,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    /// Evenly spaced values within the half-open interval `[0, len)` as
    /// one-dimensional array, each step 1.
    ///
    /// # See also
    ///
    /// [`arange_int`]
    pub fn arange_int(len: usize, device: &B) -> Tensor<T, Ix1, B> {
        arange_int(len, device)
    }
}

impl<T> Tensor<T, Ix1, DeviceCpu>
where
    T: Num,
    DeviceCpu: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    /// Evenly spaced values within the half-open interval `[0, len)` as
    /// one-dimensional array, each step 1.
    ///
    /// # See also
    ///
    /// [`arange_int`]
    pub fn arange_int_cpu(len: usize) -> Tensor<T, Ix1, DeviceCpu> {
        arange_int(len, &DeviceCpu::default())
    }
}

/* #endregion arange_int */

/* #region empty */

/// Uninitialized tensor having a specified shape.
///
/// # Safety
///
/// This function is unsafe because it creates a tensor with uninitialized.
///
/// # See also
///
/// - [Python array API standard: `empty`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.empty.html)
pub unsafe fn empty<T, D, B>(layout: impl Into<Layout<D>>, device: &B) -> Tensor<T, D, B>
where
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    let layout = layout.into();
    let (_, idx_max) = layout.bounds_index().unwrap();
    let data: Storage<T, B> = B::empty_impl(device, idx_max).unwrap();
    unsafe { Tensor::new_unchecked(data.into(), layout) }
}

impl<T, D, B> Tensor<T, D, B>
where
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    /// Uninitialized tensor having a specified shape.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it creates a tensor with uninitialized.
    ///
    /// # See also
    ///
    /// [`empty`]
    pub unsafe fn empty(layout: impl Into<Layout<D>>, device: &B) -> Tensor<T, D, B> {
        empty(layout, device)
    }
}

impl<T, D> Tensor<T, D, DeviceCpu>
where
    T: Clone + Debug,
    D: DimAPI,
{
    /// Uninitialized tensor having a specified shape.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it creates a tensor with uninitialized.
    ///
    /// # See also
    ///
    /// [`empty`]
    pub unsafe fn empty_cpu(layout: impl Into<Layout<D>>) -> Tensor<T, D, DeviceCpu> {
        empty(layout, &DeviceCpu::default())
    }
}

/* #endregion */

/* #region empty_like */

/// Uninitialized tensor with the same shape as an input tensor.
///
/// # Safety
///
/// This function is unsafe because it creates a tensor withuninitialized.
///
/// # See also
///
/// - [Python array API standard: `empty_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.empty_like.html)
pub unsafe fn empty_like<R, T, D, B>(
    tensor: &TensorBase<R, D>,
    order: TensorIterOrder,
) -> Tensor<T, D, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    let layout = layout_for_array_copy(tensor.layout(), order).unwrap();
    let idx_max = layout.size();
    let device = tensor.data().storage().device();
    let data: Storage<T, _> = device.empty_impl(idx_max).unwrap();
    unsafe { Tensor::new_unchecked(data.into(), layout) }
}

impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    /// Uninitialized tensor with the same shape as an input tensor.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it creates a tensor withuninitialized.
    ///
    /// # See also
    ///
    /// [`empty_like`]
    pub unsafe fn empty_like(&self) -> Tensor<T, D, B> {
        empty_like(self, TensorIterOrder::default())
    }
}

/* #endregion */

/* #region eye */

/// Returns a two-dimensional array with ones on the kth diagonal and zeros
/// elsewhere.
///
/// # See also
///
/// - [Python array API standard: `eye`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.eye.html)
pub fn eye<T, B>(
    n_rows: usize,
    n_cols: usize,
    k: isize,
    order: TensorOrder,
    device: &B,
) -> Tensor<T, Ix2, B>
where
    T: Num,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T> + OpAssignAPI<T, Ix1>,
{
    let layout = match order {
        TensorOrder::C => [n_rows, n_cols].c(),
        TensorOrder::F => [n_cols, n_rows].f(),
    };
    let mut data = device.zeros_impl(layout.size()).unwrap();
    let layout_diag = layout.diagonal(Some(k), Some(0), Some(1)).unwrap();
    device.fill(&mut data, &layout_diag, T::one()).unwrap();
    unsafe { Tensor::new_unchecked(data.into(), layout) }
}

impl<T, B> Tensor<T, Ix2, B>
where
    T: Num,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T> + OpAssignAPI<T, Ix1>,
{
    /// Returns a two-dimensional array with ones on the kth diagonal and zeros
    /// elsewhere.
    ///
    /// # See also
    ///
    /// [`eye`]
    pub fn eye(n_rows: usize, device: &B) -> Self {
        eye(n_rows, n_rows, 0, TensorOrder::default(), device)
    }
}

impl<T> Tensor<T, Ix2, DeviceCpu>
where
    T: Num + Clone + Debug,
    DeviceCpu: DeviceAPI<T> + DeviceCreationNumAPI<T> + OpAssignAPI<T, Ix1>,
{
    /// Returns a two-dimensional array with ones on the kth diagonal and zeros
    /// elsewhere.
    ///
    /// # See also
    ///
    /// [`eye`]
    pub fn eye_cpu(n_rows: usize) -> Self {
        eye(n_rows, n_rows, 0, TensorOrder::default(), &DeviceCpu::default())
    }
}

/* #endregion */

/* #region full */

/// New tensor having a specified shape and filled with given value.
///
/// # See also
///
/// - [Python array API standard: `full`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.full.html)
pub fn full<T, D, B>(layout: impl Into<Layout<D>>, fill: T, device: &B) -> Tensor<T, D, B>
where
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    let layout = layout.into();
    let idx_max = layout.size();
    let data: Storage<T, _> = device.full_impl(idx_max, fill).unwrap();
    unsafe { Tensor::new_unchecked(data.into(), layout) }
}

impl<T, D, B> Tensor<T, D, B>
where
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    /// New tensor having a specified shape and filled with given value.
    ///
    /// # See also
    ///
    /// [`full`]
    pub fn full(layout: impl Into<Layout<D>>, fill: T, device: &B) -> Tensor<T, D, B> {
        full(layout, fill, device)
    }
}

impl<T, D> Tensor<T, D, DeviceCpu>
where
    T: Clone + Debug,
    D: DimAPI,
{
    /// New tensor having a specified shape and filled with given value.
    ///
    /// # See also
    ///
    /// [`full`]
    pub fn full_cpu(layout: impl Into<Layout<D>>, fill: T) -> Tensor<T, D, DeviceCpu> {
        full(layout, fill, &DeviceCpu::default())
    }
}

/* #endregion */

/* #region full_like */

/// New tensor filled with given value and having the same shape as an input
/// tensor.
///
/// # See also
///
/// - [Python array API standard: `full_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.full_like.html)
pub fn full_like<R, T, D, B>(
    tensor: &TensorBase<R, D>,
    fill: T,
    order: TensorIterOrder,
) -> Tensor<T, D, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    let layout = layout_for_array_copy(tensor.layout(), order).unwrap();
    let idx_max = layout.size();
    let device = tensor.data().storage().device();
    let data: Storage<T, _> = device.full_impl(idx_max, fill).unwrap();
    unsafe { Tensor::new_unchecked(data.into(), layout) }
}

impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    /// New tensor filled with given value and having the same shape as an input
    /// tensor.
    ///
    /// # See also
    ///
    /// [`full_like`]
    pub fn full_like(&self, fill: T) -> Tensor<T, D, B> {
        full_like(self, fill, TensorIterOrder::default())
    }
}

/* #endregion */

/* #region linspace */

/// Evenly spaced numbers over a specified interval.
///
/// For boundary condition, current implementation is similar to numpy,
/// where `n = 0` will return an empty array, and `n = 1` will return an
/// array with starting value.
///
/// # See also
///
/// - [Python array API standard: `linspace`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.linspace.html)
pub fn linspace<T, B>(start: T, end: T, n: usize, endpoint: bool, device: &B) -> Tensor<T, Ix1, B>
where
    T: ComplexFloat,
    B: DeviceAPI<T> + DeviceCreationComplexFloatAPI<T>,
{
    let data = B::linspace_impl(device, start, end, n, endpoint).unwrap();
    let layout = [data.len()].into();
    unsafe { Tensor::new_unchecked(data.into(), layout) }
}

impl<T, B> Tensor<T, Ix1, B>
where
    T: ComplexFloat,
    B: DeviceAPI<T> + DeviceCreationComplexFloatAPI<T>,
{
    /// Evenly spaced numbers over a specified interval.
    ///
    /// # See also
    ///
    /// [`linspace`]
    pub fn linspace(start: T, end: T, n: usize, device: &B) -> Tensor<T, Ix1, B> {
        linspace(start, end, n, true, device)
    }
}

impl<T> Tensor<T, Ix1, DeviceCpu>
where
    T: ComplexFloat,
    DeviceCpu: DeviceAPI<T> + DeviceCreationComplexFloatAPI<T>,
{
    /// Evenly spaced numbers over a specified interval.
    ///
    /// # See also
    ///
    /// [`linspace`]
    pub fn linspace_cpu(start: T, end: T, n: usize) -> Tensor<T, Ix1, DeviceCpu> {
        linspace(start, end, n, true, &DeviceCpu::default())
    }
}

/* #endregion */

/* #region ones */

/// New tensor filled with ones and having a specified shape.
///
/// # See also
///
/// - [Python array API standard: `ones`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.ones.html)
pub fn ones<T, D, B>(layout: impl Into<Layout<D>>, device: &B) -> Tensor<T, D, B>
where
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    let layout = layout.into();
    let (_, idx_max) = layout.bounds_index().unwrap();
    let data: Storage<T, _> = B::ones_impl(device, idx_max).unwrap();
    unsafe { Tensor::new_unchecked(data.into(), layout) }
}

impl<T, D, B> Tensor<T, D, B>
where
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    /// New tensor filled with ones and having a specified shape.
    ///
    /// # See also
    ///
    /// [`ones`]
    pub fn ones(layout: impl Into<Layout<D>>, device: &B) -> Tensor<T, D, B> {
        ones(layout, device)
    }
}

impl<T, D> Tensor<T, D, DeviceCpu>
where
    T: Num + Clone + Debug,
    D: DimAPI,
{
    /// New tensor filled with ones and having a specified shape.
    ///
    /// # See also
    ///
    /// [`ones`]
    pub fn ones_cpu(layout: impl Into<Layout<D>>) -> Tensor<T, D, DeviceCpu>
    where
        T: Num + Clone + Debug,
        D: DimAPI,
    {
        Tensor::ones(layout, &DeviceCpu::default())
    }
}

/* #endregion */

/* #region ones_like */

/// New tensor filled with ones and having the same shape as an input
/// tensor.
///
/// # See also
///
/// - [Python array API standard: `ones_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.ones_like.html)
pub fn ones_like<R, T, D, B>(tensor: &TensorBase<R, D>, order: TensorIterOrder) -> Tensor<T, D, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    let layout = layout_for_array_copy(tensor.layout(), order).unwrap();
    let idx_max = layout.size();
    let device = tensor.data().storage().device();
    let data: Storage<T, B> = B::ones_impl(device, idx_max).unwrap();
    unsafe { Tensor::new_unchecked(data.into(), layout) }
}

impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    /// New tensor filled with ones and having the same shape as an input
    /// tensor.
    ///
    /// # See also
    ///
    /// [`ones_like`]
    pub fn ones_like(&self) -> Tensor<T, D, B> {
        ones_like(self, TensorIterOrder::default())
    }
}

/* #endregion */

/* #region zeros */

/// New tensor filled with zeros and having a specified shape.
///
/// # See also
///
/// - [Python array API standard: `zeros`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.zeros.html)
pub fn zeros<T, D, B>(layout: impl Into<Layout<D>>, device: &B) -> Tensor<T, D, B>
where
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    let layout = layout.into();
    let (_, idx_max) = layout.bounds_index().unwrap();
    let data: Storage<T, _> = B::zeros_impl(device, idx_max).unwrap();
    unsafe { Tensor::new_unchecked(data.into(), layout) }
}

impl<T, D, B> Tensor<T, D, B>
where
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    /// New tensor filled with zeros and having a specified shape.
    ///
    /// # See also
    ///
    /// [`zeros`]
    pub fn zeros(layout: impl Into<Layout<D>>, device: &B) -> Tensor<T, D, B> {
        zeros(layout, device)
    }
}

impl<T, D> Tensor<T, D, DeviceCpu>
where
    T: Num + Clone + Debug,
    D: DimAPI,
{
    /// New tensor filled with zeros and having a specified shape.
    ///
    /// # See also
    ///
    /// [`zeros`]
    pub fn zeros_cpu(layout: impl Into<Layout<D>>) -> Tensor<T, D, DeviceCpu> {
        zeros(layout, &DeviceCpu::default())
    }
}

/* #endregion */

/* #region zeros_like */

/// New tensor filled with zeros and having the same shape as an input
/// tensor.
///
/// # See also
///
/// - [Python array API standard: `zeros_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.zeros_like.html)
pub fn zeros_like<R, T, D, B>(tensor: &TensorBase<R, D>, order: TensorIterOrder) -> Tensor<T, D, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    let layout = layout_for_array_copy(tensor.layout(), order).unwrap();
    let idx_max = layout.size();
    let device = tensor.data().storage().device();
    let data: Storage<T, _> = B::zeros_impl(device, idx_max).unwrap();
    unsafe { Tensor::new_unchecked(data.into(), layout) }
}

impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    /// New tensor filled with zeros and having the same shape as an input
    /// tensor.
    ///
    /// # See also
    ///
    /// [`zeros_like`]
    pub fn zeros_like(&self) -> Tensor<T, D, B> {
        zeros_like(self, TensorIterOrder::default())
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;
    use num::complex::Complex32;

    #[test]
    fn playground() {
        let a = Tensor::arange_cpu(2.5, 3.2, 0.02);
        println!("{a:6.3?}");
        let a = Tensor::<f64, _>::arange_int_cpu(15);
        println!("{a:6.3?}");
        let a = unsafe { Tensor::<f64, _>::empty_cpu([15, 18].f()) };
        println!("{a:6.3?}");
        let a = unsafe { a.empty_like() };
        println!("{a:6.3?}");
        let a = unsafe { Tensor::empty_like(&a) };
        println!("{a:6.3?}");
        let a = Tensor::<f64, _>::eye_cpu(3);
        println!("{a:6.3?}");
        let a = Tensor::full_cpu([2, 2].f(), 3.16);
        println!("{a:6.3?}");
        let a = Tensor::full_like(&a, 2.71);
        println!("{a:6.3?}");
        let a = a.full_like(2.71);
        println!("{a:6.3?}");
        let a = Tensor::linspace_cpu(3.2, 4.7, 12);
        println!("{a:6.3?}");
        let a = Tensor::linspace_cpu(Complex32::new(1.8, 7.5), Complex32::new(-8.9, 1.6), 12);
        println!("{a:6.3?}");
        let a = Tensor::<f64, _>::ones_cpu([2, 2]);
        println!("{a:6.3?}");
        let a = a.ones_like();
        println!("{a:6.3?}");
        let a = Tensor::<f64, _>::zeros_cpu([2, 2]);
        println!("{a:6.3?}");
        let a = a.zeros_like();
        println!("{a:6.3?}");
    }
}
