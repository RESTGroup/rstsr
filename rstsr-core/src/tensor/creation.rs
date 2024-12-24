//! Creation methods for `Tensor` struct.
//!
//! This module relates to the [Python array API standard v2023.12](https://data-apis.org/array-api/2023.12/API_specification/creation_functions.html).
//!
//! Todo list:
//! - [x] `arange`: [`Tensor::arange`]
//! - [x] `asarray`: [`Tensor::asarray`] (defined elsewhere)
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
use num::Num;

/* #region arange */

pub trait ArrangeAPI<Param>: Sized {
    fn arange_f(param: Param) -> Result<Self>;

    fn arange(param: Param) -> Self {
        Self::arange_f(param).unwrap()
    }
}

/// Evenly spaced values within the half-open interval `[start, stop)` as
/// one-dimensional array.
///
/// # See also
///
/// - [Python array API standard: `arange`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.arange.html)
pub fn arange<Rhs, Param>(param: Param) -> Rhs
where
    Rhs: ArrangeAPI<Param>,
{
    return Rhs::arange(param);
}

pub fn arange_f<Rhs, Param>(param: Param) -> Result<Rhs>
where
    Rhs: ArrangeAPI<Param>,
{
    return Rhs::arange_f(param);
}

impl<T, B> ArrangeAPI<(T, T, T, &B)> for Tensor<T, Ix1, B>
where
    T: Num + PartialOrd,
    B: DeviceAPI<T> + DeviceCreationPartialOrdNumAPI<T>,
{
    fn arange_f(param: (T, T, T, &B)) -> Result<Self> {
        // full implementation
        let (start, stop, step, device) = param;
        let data = device.arange_impl(start, stop, step)?;
        let layout = [data.len()].into();
        unsafe { Ok(Tensor::new_unchecked(data.into(), layout)) }
    }
}

impl<T, B> ArrangeAPI<(T, T, &B)> for Tensor<T, Ix1, B>
where
    T: Num + PartialOrd,
    B: DeviceAPI<T> + DeviceCreationPartialOrdNumAPI<T>,
{
    fn arange_f(param: (T, T, &B)) -> Result<Self> {
        // (start, stop, device) -> (start, stop, 1, device)
        let (start, stop, device) = param;
        let step = T::one();
        arange_f((start, stop, step, device))
    }
}

impl<T, B> ArrangeAPI<(T, &B)> for Tensor<T, Ix1, B>
where
    T: Num + PartialOrd,
    B: DeviceAPI<T> + DeviceCreationPartialOrdNumAPI<T>,
{
    fn arange_f(param: (T, &B)) -> Result<Self> {
        // (stop, device) -> (0, stop, 1, device)
        let (stop, device) = param;
        let start = T::zero();
        let step = T::one();
        arange_f((start, stop, step, device))
    }
}

impl<T> ArrangeAPI<(T, T, T)> for Tensor<T, Ix1, DeviceCpu>
where
    T: Num + PartialOrd + Clone,
{
    fn arange_f(param: (T, T, T)) -> Result<Self> {
        // full implementation
        let (start, stop, step) = param;
        arange_f((start, stop, step, &DeviceCpu::default()))
    }
}

impl<T> ArrangeAPI<(T, T)> for Tensor<T, Ix1, DeviceCpu>
where
    T: Num + PartialOrd + Clone,
{
    fn arange_f(param: (T, T)) -> Result<Self> {
        // (start, stop) -> (start, stop, 1)
        let (start, stop) = param;
        arange_f((start, stop, &DeviceCpu::default()))
    }
}

impl<T> ArrangeAPI<T> for Tensor<T, Ix1, DeviceCpu>
where
    T: Num + PartialOrd + Clone,
{
    fn arange_f(stop: T) -> Result<Self> {
        // (stop) -> (0, stop, 1)
        arange_f((T::zero(), stop, &DeviceCpu::default()))
    }
}

/* #endregion */

/* #region empty */

pub trait EmptyAPI<Param>: Sized {
    /// # Safety
    ///
    /// This function is unsafe because it creates a tensor with uninitialized.
    unsafe fn empty_f(param: Param) -> Result<Self>;

    /// # Safety
    ///
    /// This function is unsafe because it creates a tensor with uninitialized.
    unsafe fn empty(param: Param) -> Self {
        Self::empty_f(param).unwrap()
    }
}

/// Uninitialized tensor having a specified shape.
///
/// # Safety
///
/// This function is unsafe because it creates a tensor with uninitialized.
///
/// # See also
///
/// - [Python array API standard: `empty`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.empty.html)
pub unsafe fn empty<Rhs, Param>(param: Param) -> Rhs
where
    Rhs: EmptyAPI<Param>,
{
    return Rhs::empty(param);
}

/// # Safety
///
/// This function is unsafe because it creates a tensor with uninitialized.
pub unsafe fn empty_f<Rhs, Param>(param: Param) -> Result<Rhs>
where
    Rhs: EmptyAPI<Param>,
{
    return Rhs::empty_f(param);
}

impl<T, D, B, L> EmptyAPI<(L, &B)> for Tensor<T, D, B>
where
    L: Into<Layout<D>>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    unsafe fn empty_f(param: (L, &B)) -> Result<Self> {
        let (layout, device) = param;
        let layout = layout.into();
        let (_, idx_max) = layout.bounds_index()?;
        let data: Storage<T, B> = B::empty_impl(device, idx_max)?;
        unsafe { Ok(Tensor::new_unchecked(data.into(), layout)) }
    }
}

impl<T, D, L> EmptyAPI<L> for Tensor<T, D, DeviceCpu>
where
    L: Into<Layout<D>>,
    D: DimAPI,
    T: Clone,
{
    unsafe fn empty_f(layout: L) -> Result<Self> {
        empty_f((layout, &DeviceCpu::default()))
    }
}

/* #endregion */

/* #region empty_like */

pub trait EmptyLikeAPI: Sized {
    type Out;

    /// # Safety
    ///
    /// This function is unsafe because it creates a tensor with uninitialized.
    unsafe fn empty_like_f(self) -> Result<Self::Out>;

    /// # Safety
    ///
    /// This function is unsafe because it creates a tensor with uninitialized.
    unsafe fn empty_like(self) -> Self::Out {
        Self::empty_like_f(self).unwrap()
    }
}

/// Uninitialized tensor with the same shape as an input tensor.
///
/// # Safety
///
/// This function is unsafe because it creates a tensor with uninitialized.
///
/// # See also
///
/// - [Python array API standard: `empty_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.empty_like.html)
pub unsafe fn empty_like<Param, Rhs>(param: Param) -> Rhs
where
    Param: EmptyLikeAPI<Out = Rhs>,
{
    return EmptyLikeAPI::empty_like(param);
}

/// # Safety
///
/// This function is unsafe because it creates a tensor with uninitialized.
pub unsafe fn empty_like_f<Param, Rhs>(param: Param) -> Result<Rhs>
where
    Param: EmptyLikeAPI<Out = Rhs>,
{
    return EmptyLikeAPI::empty_like_f(param);
}

impl<R, T, D, B> EmptyLikeAPI for (&TensorBase<R, D>, TensorIterOrder, &B)
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, D, B>;

    unsafe fn empty_like_f(self) -> Result<Self::Out> {
        let (tensor, order, device) = self;
        let layout = layout_for_array_copy(tensor.layout(), order)?;
        let idx_max = layout.size();
        let data: Storage<T, _> = device.empty_impl(idx_max)?;
        unsafe { Ok(Tensor::new_unchecked(data.into(), layout)) }
    }
}

impl<R, T, D, B> EmptyLikeAPI for (&TensorBase<R, D>, &B)
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, D, B>;

    unsafe fn empty_like_f(self) -> Result<Self::Out> {
        let (tensor, device) = self;
        (tensor, TensorIterOrder::default(), device).empty_like_f()
    }
}

impl<R, T, D, B> EmptyLikeAPI for (&TensorBase<R, D>, TensorIterOrder)
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, D, B>;

    unsafe fn empty_like_f(self) -> Result<Self::Out> {
        let (tensor, order) = self;
        let device = tensor.device();
        (tensor, order, device).empty_like_f()
    }
}

impl<R, T, D, B> EmptyLikeAPI for &TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, D, B>;

    unsafe fn empty_like_f(self) -> Result<Self::Out> {
        let device = self.device();
        (self, TensorIterOrder::default(), device).empty_like_f()
    }
}

/* #endregion */

/* #region eye */

pub trait EyeAPI<Param>: Sized {
    fn eye_f(param: Param) -> Result<Self>;

    fn eye(param: Param) -> Self {
        Self::eye_f(param).unwrap()
    }
}

/// Returns a two-dimensional array with ones on the kth diagonal and zeros
/// elsewhere.
///
/// # See also
///
/// - [Python array API standard: `eye`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.eye.html)
pub fn eye<Rhs, Param>(param: Param) -> Rhs
where
    Rhs: EyeAPI<Param>,
{
    return Rhs::eye(param);
}

pub fn eye_f<Rhs, Param>(param: Param) -> Result<Rhs>
where
    Rhs: EyeAPI<Param>,
{
    return Rhs::eye_f(param);
}

impl<T, B> EyeAPI<(usize, usize, isize, TensorOrder, &B)> for Tensor<T, Ix2, B>
where
    T: Num,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T> + OpAssignAPI<T, Ix1>,
{
    fn eye_f(param: (usize, usize, isize, TensorOrder, &B)) -> Result<Self> {
        let (n_rows, n_cols, k, order, device) = param;
        let layout = match order {
            TensorOrder::C => [n_rows, n_cols].c(),
            TensorOrder::F => [n_cols, n_rows].f(),
        };
        let mut data = device.zeros_impl(layout.size())?;
        let layout_diag = layout.diagonal(Some(k), Some(0), Some(1))?;
        device.fill(&mut data, &layout_diag, T::one())?;
        unsafe { Ok(Tensor::new_unchecked(data.into(), layout)) }
    }
}

impl<T, B> EyeAPI<(usize, usize, isize, &B)> for Tensor<T, Ix2, B>
where
    T: Num,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T> + OpAssignAPI<T, Ix1>,
{
    fn eye_f(param: (usize, usize, isize, &B)) -> Result<Self> {
        // (n_rows, n_cols, k, device) -> (n_rows, n_cols, k, C, device)
        let (n_rows, n_cols, k, device) = param;
        eye_f((n_rows, n_cols, k, TensorOrder::default(), device))
    }
}

impl<T, B> EyeAPI<(usize, &B)> for Tensor<T, Ix2, B>
where
    T: Num,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T> + OpAssignAPI<T, Ix1>,
{
    fn eye_f(param: (usize, &B)) -> Result<Self> {
        // (n_rows, n_cols, k, device) -> (n_rows, n_cols, k, C, device)
        let (n_rows, device) = param;
        eye_f((n_rows, n_rows, 0, TensorOrder::default(), device))
    }
}

impl<T> EyeAPI<(usize, usize, isize, TensorOrder)> for Tensor<T, Ix2, DeviceCpu>
where
    T: Num + Clone + Send + Sync,
{
    fn eye_f(param: (usize, usize, isize, TensorOrder)) -> Result<Self> {
        let (n_rows, n_cols, k, order) = param;
        eye_f((n_rows, n_cols, k, order, &DeviceCpu::default()))
    }
}

impl<T> EyeAPI<(usize, usize, isize)> for Tensor<T, Ix2, DeviceCpu>
where
    T: Num + Clone + Send + Sync,
{
    fn eye_f(param: (usize, usize, isize)) -> Result<Self> {
        // (n_rows, n_cols, k) -> (n_rows, n_cols, k, C)
        let (n_rows, n_cols, k) = param;
        eye_f((n_rows, n_cols, k, TensorOrder::default(), &DeviceCpu::default()))
    }
}

impl<T> EyeAPI<usize> for Tensor<T, Ix2, DeviceCpu>
where
    T: Num + Clone + Send + Sync,
{
    fn eye_f(n_rows: usize) -> Result<Self> {
        // n_rows -> (n_rows, n_rows, 0, C)
        eye_f((n_rows, n_rows, 0, TensorOrder::default(), &DeviceCpu::default()))
    }
}

/* #endregion */

/* #region full */

pub trait FullAPI<Param>: Sized {
    fn full_f(param: Param) -> Result<Self>;

    fn full(param: Param) -> Self {
        Self::full_f(param).unwrap()
    }
}

/// New tensor having a specified shape and filled with given value.
///
/// # See also
///
/// - [Python array API standard: `full`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.full.html)
pub fn full<Rhs, Param>(param: Param) -> Rhs
where
    Rhs: FullAPI<Param>,
{
    return Rhs::full(param);
}

pub fn full_f<Rhs, Param>(param: Param) -> Result<Rhs>
where
    Rhs: FullAPI<Param>,
{
    return Rhs::full_f(param);
}

impl<T, D, B, L> FullAPI<(L, T, &B)> for Tensor<T, D, B>
where
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
    L: Into<Layout<D>>,
{
    fn full_f(param: (L, T, &B)) -> Result<Self> {
        let (layout, fill, device) = param;
        let layout = layout.into();
        let idx_max = layout.size();
        let data: Storage<T, _> = device.full_impl(idx_max, fill)?;
        unsafe { Ok(Tensor::new_unchecked(data.into(), layout)) }
    }
}

impl<T, D> FullAPI<(Layout<D>, T)> for Tensor<T, D, DeviceCpu>
where
    D: DimAPI,
    T: Clone,
{
    fn full_f(param: (Layout<D>, T)) -> Result<Self> {
        let (layout, fill) = param;
        full_f((layout, fill, &DeviceCpu::default()))
    }
}

/* #endregion */

/* #region full_like */

pub trait FullLikeAPI<Param>: Sized {
    fn full_like_f(param: Param) -> Result<Self>;

    fn full_like(param: Param) -> Self {
        Self::full_like_f(param).unwrap()
    }
}

/// New tensor filled with given value and having the same shape as an input
/// tensor.
///
/// # See also
///
/// - [Python array API standard: `full_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.full_like.html)
pub fn full_like<Rhs, Param>(param: Param) -> Rhs
where
    Rhs: FullLikeAPI<Param>,
{
    return Rhs::full_like(param);
}

pub fn full_like_f<Rhs, Param>(param: Param) -> Result<Rhs>
where
    Rhs: FullLikeAPI<Param>,
{
    return Rhs::full_like_f(param);
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
        full_like((self, fill, TensorIterOrder::default()))
    }
}

impl<R, T, D, B> FullLikeAPI<(&TensorBase<R, D>, T, TensorIterOrder, &B)> for Tensor<T, D, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    fn full_like_f(param: (&TensorBase<R, D>, T, TensorIterOrder, &B)) -> Result<Self> {
        let (tensor, fill, order, device) = param;
        let layout = layout_for_array_copy(tensor.layout(), order)?;
        let idx_max = layout.size();
        let data: Storage<T, _> = device.full_impl(idx_max, fill)?;
        unsafe { Ok(Tensor::new_unchecked(data.into(), layout)) }
    }
}

impl<R, T, D, B> FullLikeAPI<(&TensorBase<R, D>, T, &B)> for Tensor<T, D, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    fn full_like_f(param: (&TensorBase<R, D>, T, &B)) -> Result<Self> {
        let (tensor, fill, device) = param;
        full_like_f((tensor, fill, TensorIterOrder::default(), device))
    }
}

impl<R, T, D, B> FullLikeAPI<(&TensorBase<R, D>, T, TensorIterOrder)> for Tensor<T, D, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    fn full_like_f(param: (&TensorBase<R, D>, T, TensorIterOrder)) -> Result<Self> {
        let (tensor, fill, order) = param;
        let device = tensor.device();
        full_like_f((tensor, fill, order, device))
    }
}

impl<R, T, D, B> FullLikeAPI<(&TensorBase<R, D>, T)> for Tensor<T, D, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    fn full_like_f(param: (&TensorBase<R, D>, T)) -> Result<Self> {
        let (tensor, fill) = param;
        let device = tensor.device();
        full_like_f((tensor, fill, TensorIterOrder::default(), device))
    }
}

/* #endregion */

/* #region linspace */

pub trait LinspaceAPI<Param>: Sized {
    fn linspace_f(param: Param) -> Result<Self>;

    fn linspace(param: Param) -> Self {
        Self::linspace_f(param).unwrap()
    }
}

/// Evenly spaced numbers over a specified interval.
///
/// For boundary condition, current implementation is similar to numpy,
/// where `n = 0` will return an empty array, and `n = 1` will return an
/// array with starting value.
///
/// # See also
///
/// - [Python array API standard: `linspace`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.linspace.html)
pub fn linspace<Rhs, Param>(param: Param) -> Rhs
where
    Rhs: LinspaceAPI<Param>,
{
    return Rhs::linspace(param);
}

pub fn linspace_f<Rhs, Param>(param: Param) -> Result<Rhs>
where
    Rhs: LinspaceAPI<Param>,
{
    return Rhs::linspace_f(param);
}

impl<T, B> LinspaceAPI<(T, T, usize, bool, &B)> for Tensor<T, Ix1, B>
where
    T: ComplexFloat,
    B: DeviceAPI<T> + DeviceCreationComplexFloatAPI<T>,
{
    fn linspace_f(param: (T, T, usize, bool, &B)) -> Result<Self> {
        let (start, end, n, endpoint, device) = param;
        let data = B::linspace_impl(device, start, end, n, endpoint)?;
        let layout = [data.len()].into();
        unsafe { Ok(Tensor::new_unchecked(data.into(), layout)) }
    }
}

impl<T, B> LinspaceAPI<(T, T, usize, &B)> for Tensor<T, Ix1, B>
where
    T: ComplexFloat,
    B: DeviceAPI<T> + DeviceCreationComplexFloatAPI<T>,
{
    fn linspace_f(param: (T, T, usize, &B)) -> Result<Self> {
        // (start, end, n, device) -> (start, end, n, true, device)
        let (start, end, n, device) = param;
        linspace_f((start, end, n, true, device))
    }
}

impl<T> LinspaceAPI<(T, T, usize, bool)> for Tensor<T, Ix1, DeviceCpu>
where
    T: ComplexFloat + Send + Sync,
{
    fn linspace_f(param: (T, T, usize, bool)) -> Result<Self> {
        // (start, end, n, endpoint) -> (start, end, n, endpoint, device)
        let (start, end, n, endpoint) = param;
        linspace_f((start, end, n, endpoint, &DeviceCpu::default()))
    }
}

impl<T> LinspaceAPI<(T, T, usize)> for Tensor<T, Ix1, DeviceCpu>
where
    T: ComplexFloat + Send + Sync,
{
    fn linspace_f(param: (T, T, usize)) -> Result<Self> {
        // (start, end, n) -> (start, end, n, true, device)
        let (start, end, n) = param;
        linspace_f((start, end, n, true, &DeviceCpu::default()))
    }
}

/* #endregion */

/* #region ones */

pub trait OnesAPI<Param>: Sized {
    fn ones_f(param: Param) -> Result<Self>;

    fn ones(param: Param) -> Self {
        Self::ones_f(param).unwrap()
    }
}

/// New tensor filled with ones and having a specified shape.
///
/// # See also
///
/// - [Python array API standard: `ones`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.ones.html)
pub fn ones<Rhs, Param>(param: Param) -> Rhs
where
    Rhs: OnesAPI<Param>,
{
    return Rhs::ones(param);
}

pub fn ones_f<Rhs, Param>(param: Param) -> Result<Rhs>
where
    Rhs: OnesAPI<Param>,
{
    return Rhs::ones_f(param);
}

impl<T, D, B, L> OnesAPI<(L, &B)> for Tensor<T, D, B>
where
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
    L: Into<Layout<D>>,
{
    fn ones_f(param: (L, &B)) -> Result<Self> {
        let (layout, device) = param;
        let layout = layout.into();
        let (_, idx_max) = layout.bounds_index()?;
        let data: Storage<T, _> = device.ones_impl(idx_max)?;
        unsafe { Ok(Tensor::new_unchecked(data.into(), layout)) }
    }
}

impl<T, D, L> OnesAPI<L> for Tensor<T, D, DeviceCpu>
where
    T: Num + Clone,
    D: DimAPI,
    L: Into<Layout<D>>,
{
    fn ones_f(layout: L) -> Result<Self> {
        ones_f((layout, &DeviceCpu::default()))
    }
}

/* #endregion */

/* #region ones_like */

pub trait OnesLikeAPI<Param>: Sized {
    fn ones_like_f(param: Param) -> Result<Self>;

    fn ones_like(param: Param) -> Self {
        Self::ones_like_f(param).unwrap()
    }
}

/// New tensor filled with ones and having the same shape as an input
/// tensor.
///
/// # See also
///
/// - [Python array API standard: `ones_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.ones_like.html)
pub fn ones_like<Rhs, Param>(param: Param) -> Rhs
where
    Rhs: OnesLikeAPI<Param>,
{
    return Rhs::ones_like(param);
}

pub fn ones_like_f<Rhs, Param>(param: Param) -> Result<Rhs>
where
    Rhs: OnesLikeAPI<Param>,
{
    return Rhs::ones_like_f(param);
}

impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    T: Num,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    /// New tensor filled with ones and having the same shape as an input
    /// tensor.
    ///
    /// # See also
    ///
    /// [`ones_like`]
    pub fn ones_like(&self) -> Tensor<T, D, B> {
        ones_like((self, TensorIterOrder::default(), self.device()))
    }

    pub fn ones_like_f(&self) -> Result<Tensor<T, D, B>> {
        ones_like_f((self, TensorIterOrder::default(), self.device()))
    }
}

impl<R, T, D, B> OnesLikeAPI<(&TensorBase<R, D>, TensorIterOrder, &B)> for Tensor<T, D, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    fn ones_like_f(param: (&TensorBase<R, D>, TensorIterOrder, &B)) -> Result<Self> {
        let (tensor, order, device) = param;
        let layout = layout_for_array_copy(tensor.layout(), order)?;
        let idx_max = layout.size();
        let data: Storage<T, _> = device.ones_impl(idx_max)?;
        unsafe { Ok(Tensor::new_unchecked(data.into(), layout)) }
    }
}

impl<R, T, D, B> OnesLikeAPI<(&TensorBase<R, D>, &B)> for Tensor<T, D, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    fn ones_like_f(param: (&TensorBase<R, D>, &B)) -> Result<Self> {
        let (tensor, device) = param;
        ones_like_f((tensor, TensorIterOrder::default(), device))
    }
}

impl<R, T, D, B> OnesLikeAPI<(&TensorBase<R, D>, TensorIterOrder)> for Tensor<T, D, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    fn ones_like_f(param: (&TensorBase<R, D>, TensorIterOrder)) -> Result<Self> {
        let (tensor, order) = param;
        let device = tensor.device();
        ones_like_f((tensor, order, device))
    }
}

impl<R, T, D, B> OnesLikeAPI<&TensorBase<R, D>> for Tensor<T, D, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    fn ones_like_f(tensor: &TensorBase<R, D>) -> Result<Self> {
        let device = tensor.device();
        ones_like_f((tensor, TensorIterOrder::default(), device))
    }
}

/* #endregion */

/* #region zeros */

pub trait ZerosAPI<Param>: Sized {
    fn zeros_f(param: Param) -> Result<Self>;

    fn zeros(param: Param) -> Self {
        Self::zeros_f(param).unwrap()
    }
}

/// New tensor filled with zeros and having a specified shape.
///
/// # See also
///
/// - [Python array API standard: `zeros`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.zeros.html)
pub fn zeros<Rhs, Param>(param: Param) -> Rhs
where
    Rhs: ZerosAPI<Param>,
{
    return Rhs::zeros(param);
}

pub fn zeros_f<Rhs, Param>(param: Param) -> Result<Rhs>
where
    Rhs: ZerosAPI<Param>,
{
    return Rhs::zeros_f(param);
}

impl<T, D, B, L> ZerosAPI<(L, &B)> for Tensor<T, D, B>
where
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
    L: Into<Layout<D>>,
{
    fn zeros_f(param: (L, &B)) -> Result<Self> {
        let (layout, device) = param;
        let layout = layout.into();
        let (_, idx_max) = layout.bounds_index()?;
        let data: Storage<T, _> = B::zeros_impl(device, idx_max)?;
        unsafe { Ok(Tensor::new_unchecked(data.into(), layout)) }
    }
}

impl<T, D, L> ZerosAPI<L> for Tensor<T, D, DeviceCpu>
where
    T: Num + Clone,
    D: DimAPI,
    L: Into<Layout<D>>,
{
    fn zeros_f(layout: L) -> Result<Self> {
        zeros_f((layout, &DeviceCpu::default()))
    }
}

/* #endregion */

/* #region zeros_like */

pub trait ZerosLikeAPI<Param>: Sized {
    fn zeros_like_f(param: Param) -> Result<Self>;

    fn zeros_like(param: Param) -> Self {
        Self::zeros_like_f(param).unwrap()
    }
}

/// New tensor filled with zeros and having the same shape as an input
/// tensor.
///
/// # See also
///
/// - [Python array API standard: `zeros_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.zeros_like.html)
pub fn zeros_like<Rhs, Param>(param: Param) -> Rhs
where
    Rhs: ZerosLikeAPI<Param>,
{
    return Rhs::zeros_like(param);
}

pub fn zeros_like_f<Rhs, Param>(param: Param) -> Result<Rhs>
where
    Rhs: ZerosLikeAPI<Param>,
{
    return Rhs::zeros_like_f(param);
}

impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    T: Num,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    /// New tensor filled with zeros and having the same shape as an input
    /// tensor.
    ///
    /// # See also
    ///
    /// [`zeros_like`]
    pub fn zeros_like(&self) -> Tensor<T, D, B> {
        zeros_like((self, TensorIterOrder::default(), self.device()))
    }

    pub fn zeros_like_f(&self) -> Result<Tensor<T, D, B>> {
        zeros_like_f((self, TensorIterOrder::default(), self.device()))
    }
}

impl<R, T, D, B> ZerosLikeAPI<(&TensorBase<R, D>, TensorIterOrder, &B)> for Tensor<T, D, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    fn zeros_like_f(param: (&TensorBase<R, D>, TensorIterOrder, &B)) -> Result<Self> {
        let (tensor, order, device) = param;
        let layout = layout_for_array_copy(tensor.layout(), order)?;
        let idx_max = layout.size();
        let data: Storage<T, _> = B::zeros_impl(device, idx_max)?;
        unsafe { Ok(Tensor::new_unchecked(data.into(), layout)) }
    }
}

impl<R, T, D, B> ZerosLikeAPI<(&TensorBase<R, D>, &B)> for Tensor<T, D, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    fn zeros_like_f(param: (&TensorBase<R, D>, &B)) -> Result<Self> {
        let (tensor, device) = param;
        zeros_like_f((tensor, TensorIterOrder::default(), device))
    }
}

impl<R, T, D, B> ZerosLikeAPI<(&TensorBase<R, D>, TensorIterOrder)> for Tensor<T, D, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    fn zeros_like_f(param: (&TensorBase<R, D>, TensorIterOrder)) -> Result<Self> {
        let (tensor, order) = param;
        let device = tensor.device();
        zeros_like_f((tensor, order, device))
    }
}

impl<R, T, D, B> ZerosLikeAPI<&TensorBase<R, D>> for Tensor<T, D, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    fn zeros_like_f(tensor: &TensorBase<R, D>) -> Result<Self> {
        zeros_like_f((tensor, TensorIterOrder::default(), tensor.device()))
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;
    use num::complex::Complex32;

    #[test]
    fn playground() {
        let a = Tensor::arange((2.5, 3.2, 0.02));
        println!("{a:6.3?}");
        let a = Tensor::<f64, _>::arange(15.0);
        println!("{a:6.3?}");
        let a = unsafe { Tensor::<f64, _>::empty([15, 18].f()) };
        println!("{a:6.3?}");
        let a = unsafe { a.empty_like() };
        println!("{a:6.3?}");
        let a = unsafe { empty_like((&a, TensorIterOrder::C)) };
        println!("{a:6.3?}");
        let a = Tensor::<f64, _>::eye(3);
        println!("{a:6.3?}");
        let a = Tensor::full(([2, 2].f(), 3.16));
        println!("{a:6.3?}");
        let a = Tensor::full_like(&a, 2.71);
        println!("{a:6.3?}");
        let a = a.full_like(2.71);
        println!("{a:6.3?}");
        let a = Tensor::linspace((3.2, 4.7, 12));
        println!("{a:6.3?}");
        let a = Tensor::linspace((Complex32::new(1.8, 7.5), Complex32::new(-8.9, 1.6), 12));
        println!("{a:6.3?}");
        let a = Tensor::<f64, _>::ones([2, 2]);
        println!("{a:6.3?}");
        let a = a.ones_like();
        println!("{a:6.3?}");
        let a = Tensor::<f64, _>::zeros([2, 2]);
        println!("{a:6.3?}");
        let a = a.zeros_like();
        println!("{a:6.3?}");
    }
}
