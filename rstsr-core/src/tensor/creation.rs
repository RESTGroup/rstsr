//! Creation methods for `Tensor` struct.
//!
//! This module relates to the [Python array API standard v2023.12](https://data-apis.org/array-api/2023.12/API_specification/creation_functions.html).
//!
//! Todo list:
//! - [x] `arange`: [`arange`]
//! - [x] `asarray`: [`asarray`] (defined elsewhere)
//! - [x] `empty`: [`empty`]
//! - [x] `empty_like`: [`empty_like`]
//! - [x] `eye`: [`eye`]
//! - [ ] ~`from_dlpack`~
//! - [x] `full`: [`full`]
//! - [x] `full_like`: [`full_like`]
//! - [x] `linspace`: [`linspace`]
//! - [ ] `meshgrid`
//! - [x] `ones`: [`ones`]
//! - [x] `ones_like`: [`ones_like`]
//! - [ ] `tril`
//! - [ ] `triu`
//! - [x] `zeros`: [`zeros`]
//! - [x] `zeros_like`: [`zeros_like`]

use crate::prelude_dev::*;
use num::complex::ComplexFloat;
use num::Num;

/* #region arange */

pub trait ArangeAPI<Inp>: Sized {
    type Out;

    fn arange_f(self) -> Result<Self::Out>;

    fn arange(self) -> Self::Out {
        Self::arange_f(self).unwrap()
    }
}

/// Evenly spaced values within the half-open interval `[start, stop)` as
/// one-dimensional array.
///
/// # See also
///
/// - [Python array API standard: `arange`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.arange.html)
pub fn arange<Param, Inp, Rhs>(param: Param) -> Rhs
where
    Param: ArangeAPI<Inp, Out = Rhs>,
{
    return ArangeAPI::arange(param);
}

pub fn arange_f<Param, Inp, Rhs>(param: Param) -> Result<Rhs>
where
    Param: ArangeAPI<Inp, Out = Rhs>,
{
    return ArangeAPI::arange_f(param);
}

impl<T, B> ArangeAPI<(T, B)> for (T, T, T, &B)
where
    T: Num + PartialOrd,
    B: DeviceAPI<T> + DeviceCreationPartialOrdNumAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    fn arange_f(self) -> Result<Self::Out> {
        // full implementation
        let (start, stop, step, device) = self;
        let data = device.arange_impl(start, stop, step)?;
        let layout = vec![data.len()].into();
        unsafe { Ok(Tensor::new_unchecked(data.into(), layout)) }
    }
}

impl<T, B> ArangeAPI<(T, B)> for (T, T, &B)
where
    T: Num + PartialOrd,
    B: DeviceAPI<T> + DeviceCreationPartialOrdNumAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    fn arange_f(self) -> Result<Self::Out> {
        // (start, stop, device) -> (start, stop, 1, device)
        let (start, stop, device) = self;
        let step = T::one();
        arange_f((start, stop, step, device))
    }
}

impl<T, B> ArangeAPI<(T, B)> for (T, &B)
where
    T: Num + PartialOrd,
    B: DeviceAPI<T> + DeviceCreationPartialOrdNumAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    fn arange_f(self) -> Result<Self::Out> {
        // (stop, device) -> (0, stop, 1, device)
        let (stop, device) = self;
        let start = T::zero();
        let step = T::one();
        arange_f((start, stop, step, device))
    }
}

impl<T> ArangeAPI<T> for (T, T, T)
where
    T: Num + PartialOrd + Clone + Send + Sync,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    fn arange_f(self) -> Result<Self::Out> {
        // full implementation
        let (start, stop, step) = self;
        arange_f((start, stop, step, &DeviceCpu::default()))
    }
}

impl<T> ArangeAPI<T> for (T, T)
where
    T: Num + PartialOrd + Clone,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    fn arange_f(self) -> Result<Self::Out> {
        // (start, stop) -> (start, stop, 1)
        let (start, stop) = self;
        arange_f((start, stop, &DeviceCpu::default()))
    }
}

impl<T> ArangeAPI<T> for T
where
    T: Num + PartialOrd + Clone,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    fn arange_f(self) -> Result<Self::Out> {
        // (stop) -> (0, stop, 1)
        arange_f((T::zero(), self, &DeviceCpu::default()))
    }
}

/* #endregion */

/* #region empty */

pub trait EmptyAPI<Inp>: Sized {
    type Out;

    /// # Safety
    ///
    /// This function is unsafe because it creates a tensor with uninitialized.
    unsafe fn empty_f(self) -> Result<Self::Out>;

    /// # Safety
    ///
    /// This function is unsafe because it creates a tensor with uninitialized.
    unsafe fn empty(self) -> Self::Out {
        Self::empty_f(self).unwrap()
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
pub unsafe fn empty<Param, Inp, Rhs>(param: Param) -> Rhs
where
    Param: EmptyAPI<Inp, Out = Rhs>,
{
    return EmptyAPI::empty(param);
}

/// # Safety
///
/// This function is unsafe because it creates a tensor with uninitialized.
pub unsafe fn empty_f<Param, Inp, Rhs>(param: Param) -> Result<Rhs>
where
    Param: EmptyAPI<Inp, Out = Rhs>,
{
    return EmptyAPI::empty_f(param);
}

impl<T, D, B, L> EmptyAPI<(T, D)> for (L, &B)
where
    L: Into<Layout<D>>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    unsafe fn empty_f(self) -> Result<Self::Out> {
        let (layout, device) = self;
        let layout = layout.into();
        let (_, idx_max) = layout.bounds_index()?;
        let storage = B::empty_impl(device, idx_max)?;
        unsafe { Ok(Tensor::new_unchecked(storage, layout.into_dim()?)) }
    }
}

impl<T, D, L> EmptyAPI<(T, D)> for L
where
    L: Into<Layout<D>>,
    D: DimAPI,
    T: Clone,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    unsafe fn empty_f(self) -> Result<Self::Out> {
        empty_f((self, &DeviceCpu::default()))
    }
}

/* #endregion */

/* #region empty_like */

pub trait EmptyLikeAPI<Inp>: Sized {
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
pub unsafe fn empty_like<Param, Inp, Rhs>(param: Param) -> Rhs
where
    Param: EmptyLikeAPI<Inp, Out = Rhs>,
{
    return EmptyLikeAPI::empty_like(param);
}

/// # Safety
///
/// This function is unsafe because it creates a tensor with uninitialized.
pub unsafe fn empty_like_f<Param, Inp, Rhs>(param: Param) -> Result<Rhs>
where
    Param: EmptyLikeAPI<Inp, Out = Rhs>,
{
    return EmptyLikeAPI::empty_like_f(param);
}

impl<R, T, B, D> EmptyLikeAPI<()> for (&TensorAny<R, T, B, D>, TensorIterOrder, &B)
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, D>;

    unsafe fn empty_like_f(self) -> Result<Self::Out> {
        let (tensor, order, device) = self;
        let layout = layout_for_array_copy(tensor.layout(), order)?;
        let idx_max = layout.size();
        let storage = device.empty_impl(idx_max)?;
        unsafe { Ok(Tensor::new_unchecked(storage.into(), layout)) }
    }
}

impl<R, T, B, D> EmptyLikeAPI<()> for (&TensorAny<R, T, B, D>, &B)
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, D>;

    unsafe fn empty_like_f(self) -> Result<Self::Out> {
        let (tensor, device) = self;
        empty_like_f((tensor, TensorIterOrder::default(), device))
    }
}

impl<R, T, B, D> EmptyLikeAPI<()> for (&TensorAny<R, T, B, D>, TensorIterOrder)
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, D>;

    unsafe fn empty_like_f(self) -> Result<Self::Out> {
        let (tensor, order) = self;
        let device = tensor.device();
        empty_like_f((tensor, order, device))
    }
}

impl<R, T, B, D> EmptyLikeAPI<()> for &TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, D>;

    unsafe fn empty_like_f(self) -> Result<Self::Out> {
        let device = self.device();
        empty_like_f((self, TensorIterOrder::default(), device))
    }
}

/* #endregion */

/* #region eye */

pub trait EyeAPI<Inp>: Sized {
    type Out;

    fn eye_f(self) -> Result<Self::Out>;

    fn eye(self) -> Self::Out {
        Self::eye_f(self).unwrap()
    }
}

/// Returns a two-dimensional array with ones on the kth diagonal and zeros
/// elsewhere.
///
/// # See also
///
/// - [Python array API standard: `eye`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.eye.html)
pub fn eye<Param, Inp, Rhs>(param: Param) -> Rhs
where
    Param: EyeAPI<Inp, Out = Rhs>,
{
    return EyeAPI::eye(param);
}

pub fn eye_f<Param, Inp, Rhs>(param: Param) -> Result<Rhs>
where
    Param: EyeAPI<Inp, Out = Rhs>,
{
    return EyeAPI::eye_f(param);
}

impl<T, B> EyeAPI<(T, B)> for (usize, usize, isize, TensorOrder, &B)
where
    T: Num,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T> + OpAssignAPI<T, Ix1>,
{
    type Out = Tensor<T, B, IxD>;

    fn eye_f(self) -> Result<Self::Out> {
        let (n_rows, n_cols, k, order, device) = self;
        let layout = match order {
            TensorOrder::C => [n_rows, n_cols].c(),
            TensorOrder::F => [n_cols, n_rows].f(),
        };
        let mut storage = device.zeros_impl(layout.size())?;
        let layout_diag = layout.diagonal(Some(k), Some(0), Some(1))?;
        device.fill(storage.raw_mut(), &layout_diag, T::one())?;
        unsafe { Ok(Tensor::new_unchecked(storage.into(), layout.into_dim()?)) }
    }
}

impl<T, B> EyeAPI<(T, B)> for (usize, usize, isize, &B)
where
    T: Num,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T> + OpAssignAPI<T, Ix1>,
{
    type Out = Tensor<T, B, IxD>;

    fn eye_f(self) -> Result<Self::Out> {
        // (n_rows, n_cols, k, device) -> (n_rows, n_cols, k, C, device)
        let (n_rows, n_cols, k, device) = self;
        eye_f((n_rows, n_cols, k, TensorOrder::default(), device))
    }
}

impl<T, B> EyeAPI<(T, B)> for (usize, &B)
where
    T: Num,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T> + OpAssignAPI<T, Ix1>,
{
    type Out = Tensor<T, B, IxD>;

    fn eye_f(self) -> Result<Self::Out> {
        // (n_rows, n_cols, k, device) -> (n_rows, n_cols, k, C, device)
        let (n_rows, device) = self;
        eye_f((n_rows, n_rows, 0, TensorOrder::default(), device))
    }
}

impl<T> EyeAPI<T> for (usize, usize, isize, TensorOrder)
where
    T: Num + Clone + Send + Sync,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    fn eye_f(self) -> Result<Self::Out> {
        let (n_rows, n_cols, k, order) = self;
        eye_f((n_rows, n_cols, k, order, &DeviceCpu::default()))
    }
}

impl<T> EyeAPI<T> for (usize, usize, isize)
where
    T: Num + Clone + Send + Sync,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    fn eye_f(self) -> Result<Self::Out> {
        // (n_rows, n_cols, k) -> (n_rows, n_cols, k, C)
        let (n_rows, n_cols, k) = self;
        eye_f((n_rows, n_cols, k, TensorOrder::default(), &DeviceCpu::default()))
    }
}

impl<T> EyeAPI<T> for usize
where
    T: Num + Clone + Send + Sync,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    fn eye_f(self) -> Result<Self::Out> {
        // n_rows -> (n_rows, n_rows, 0, C)
        eye_f((self, self, 0, TensorOrder::default(), &DeviceCpu::default()))
    }
}

/* #endregion */

/* #region full */

pub trait FullAPI<Inp>: Sized {
    type Out;

    fn full_f(self) -> Result<Self::Out>;

    fn full(self) -> Self::Out {
        Self::full_f(self).unwrap()
    }
}

/// New tensor having a specified shape and filled with given value.
///
/// # See also
///
/// - [Python array API standard: `full`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.full.html)
pub fn full<Param, Inp, Rhs>(param: Param) -> Rhs
where
    Param: FullAPI<Inp, Out = Rhs>,
{
    return FullAPI::full(param);
}

pub fn full_f<Param, Inp, Rhs>(param: Param) -> Result<Rhs>
where
    Param: FullAPI<Inp, Out = Rhs>,
{
    return FullAPI::full_f(param);
}

impl<T, D, B, L> FullAPI<(T, D)> for (L, T, &B)
where
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
    L: Into<Layout<D>>,
{
    type Out = Tensor<T, B, IxD>;

    fn full_f(self) -> Result<Self::Out> {
        let (layout, fill, device) = self;
        let layout = layout.into();
        let idx_max = layout.size();
        let storage = device.full_impl(idx_max, fill)?;
        unsafe { Ok(Tensor::new_unchecked(storage, layout.into_dim()?)) }
    }
}

impl<T, D, L> FullAPI<(T, D)> for (L, T)
where
    D: DimAPI,
    T: Clone,
    L: Into<Layout<D>>,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    fn full_f(self) -> Result<Self::Out> {
        let (layout, fill) = self;
        full_f((layout, fill, &DeviceCpu::default()))
    }
}

/* #endregion */

/* #region full_like */

pub trait FullLikeAPI<Inp>: Sized {
    type Out;

    fn full_like_f(self) -> Result<Self::Out>;

    fn full_like(self) -> Self::Out {
        Self::full_like_f(self).unwrap()
    }
}

/// New tensor filled with given value and having the same shape as an input
/// tensor.
///
/// # See also
///
/// - [Python array API standard: `full_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.full_like.html)
pub fn full_like<Param, Inp, Rhs>(param: Param) -> Rhs
where
    Param: FullLikeAPI<Inp, Out = Rhs>,
{
    return FullLikeAPI::full_like(param);
}

pub fn full_like_f<Param, Inp, Rhs>(param: Param) -> Result<Rhs>
where
    Param: FullLikeAPI<Inp, Out = Rhs>,
{
    return FullLikeAPI::full_like_f(param);
}

impl<R, T, B, D> FullLikeAPI<()> for (&TensorAny<R, T, B, D>, T, TensorIterOrder, &B)
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, D>;

    fn full_like_f(self) -> Result<Self::Out> {
        let (tensor, fill, order, device) = self;
        let layout = layout_for_array_copy(tensor.layout(), order)?;
        let idx_max = layout.size();
        let storage = device.full_impl(idx_max, fill)?;
        unsafe { Ok(Tensor::new_unchecked(storage, layout)) }
    }
}

impl<R, T, B, D> FullLikeAPI<()> for (&TensorAny<R, T, B, D>, T, &B)
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, D>;

    fn full_like_f(self) -> Result<Self::Out> {
        let (tensor, fill, device) = self;
        full_like_f((tensor, fill, TensorIterOrder::default(), device))
    }
}

impl<R, T, B, D> FullLikeAPI<()> for (&TensorAny<R, T, B, D>, T, TensorIterOrder)
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, D>;

    fn full_like_f(self) -> Result<Self::Out> {
        let (tensor, fill, order) = self;
        let device = tensor.device();
        full_like_f((tensor, fill, order, device))
    }
}

impl<R, T, B, D> FullLikeAPI<()> for (&TensorAny<R, T, B, D>, T)
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, D>;

    fn full_like_f(self) -> Result<Self::Out> {
        let (tensor, fill) = self;
        let device = tensor.device();
        full_like_f((tensor, fill, TensorIterOrder::default(), device))
    }
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    pub fn full_like(&self, fill: T) -> Tensor<T, B, D> {
        full_like((self, fill))
    }

    pub fn full_like_f(&self, fill: T) -> Result<Tensor<T, B, D>> {
        full_like_f((self, fill))
    }
}

/* #endregion */

/* #region linspace */

pub trait LinspaceAPI<Inp>: Sized {
    type Out;

    fn linspace_f(self) -> Result<Self::Out>;

    fn linspace(self) -> Self::Out {
        Self::linspace_f(self).unwrap()
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
pub fn linspace<Param, Inp, Rhs>(param: Param) -> Rhs
where
    Param: LinspaceAPI<Inp, Out = Rhs>,
{
    return LinspaceAPI::linspace(param);
}

pub fn linspace_f<Param, Inp, Rhs>(param: Param) -> Result<Rhs>
where
    Param: LinspaceAPI<Inp, Out = Rhs>,
{
    return LinspaceAPI::linspace_f(param);
}

impl<T, B> LinspaceAPI<(T, B)> for (T, T, usize, bool, &B)
where
    T: ComplexFloat,
    B: DeviceAPI<T> + DeviceCreationComplexFloatAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    fn linspace_f(self) -> Result<Self::Out> {
        let (start, end, n, endpoint, device) = self;
        let data = B::linspace_impl(device, start, end, n, endpoint)?;
        let layout = vec![data.len()].into();
        unsafe { Ok(Tensor::new_unchecked(data.into(), layout)) }
    }
}

impl<T, B> LinspaceAPI<(T, B)> for (T, T, usize, &B)
where
    T: ComplexFloat,
    B: DeviceAPI<T> + DeviceCreationComplexFloatAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    fn linspace_f(self) -> Result<Self::Out> {
        // (start, end, n, device) -> (start, end, n, true, device)
        let (start, end, n, device) = self;
        linspace_f((start, end, n, true, device))
    }
}

impl<T> LinspaceAPI<T> for (T, T, usize, bool)
where
    T: ComplexFloat + Send + Sync,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    fn linspace_f(self) -> Result<Self::Out> {
        // (start, end, n, endpoint) -> (start, end, n, endpoint, device)
        let (start, end, n, endpoint) = self;
        linspace_f((start, end, n, endpoint, &DeviceCpu::default()))
    }
}

impl<T> LinspaceAPI<T> for (T, T, usize)
where
    T: ComplexFloat + Send + Sync,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    fn linspace_f(self) -> Result<Self::Out> {
        // (start, end, n) -> (start, end, n, true, device)
        let (start, end, n) = self;
        linspace_f((start, end, n, true, &DeviceCpu::default()))
    }
}

/* #endregion */

/* #region ones */

pub trait OnesAPI<Inp>: Sized {
    type Out;

    fn ones_f(self) -> Result<Self::Out>;

    fn ones(self) -> Self::Out {
        Self::ones_f(self).unwrap()
    }
}

/// New tensor filled with ones and having a specified shape.
///
/// # See also
///
/// - [Python array API standard: `ones`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.ones.html)
pub fn ones<Param, Inp, Rhs>(param: Param) -> Rhs
where
    Param: OnesAPI<Inp, Out = Rhs>,
{
    return OnesAPI::ones(param);
}

pub fn ones_f<Param, Inp, Rhs>(param: Param) -> Result<Rhs>
where
    Param: OnesAPI<Inp, Out = Rhs>,
{
    return OnesAPI::ones_f(param);
}

impl<T, D, B, L> OnesAPI<(T, D)> for (L, &B)
where
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
    L: Into<Layout<D>>,
{
    type Out = Tensor<T, B, IxD>;

    fn ones_f(self) -> Result<Self::Out> {
        let (layout, device) = self;
        let layout = layout.into();
        let (_, idx_max) = layout.bounds_index()?;
        let storage = device.ones_impl(idx_max)?;
        unsafe { Ok(Tensor::new_unchecked(storage, layout.into_dim()?)) }
    }
}

impl<T, D, L> OnesAPI<(T, D)> for L
where
    T: Num + Clone,
    D: DimAPI,
    L: Into<Layout<D>>,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    fn ones_f(self) -> Result<Self::Out> {
        ones_f((self, &DeviceCpu::default()))
    }
}

/* #endregion */

/* #region ones_like */

pub trait OnesLikeAPI<Inp>: Sized {
    type Out;

    fn ones_like_f(self) -> Result<Self::Out>;

    fn ones_like(self) -> Self::Out {
        Self::ones_like_f(self).unwrap()
    }
}

/// New tensor filled with ones and having the same shape as an input
/// tensor.
///
/// # See also
///
/// - [Python array API standard: `ones_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.ones_like.html)
pub fn ones_like<Param, Inp, Rhs>(param: Param) -> Rhs
where
    Param: OnesLikeAPI<Inp, Out = Rhs>,
{
    return OnesLikeAPI::ones_like(param);
}

pub fn ones_like_f<Param, Inp, Rhs>(param: Param) -> Result<Rhs>
where
    Param: OnesLikeAPI<Inp, Out = Rhs>,
{
    return OnesLikeAPI::ones_like_f(param);
}

impl<R, T, B, D> OnesLikeAPI<()> for (&TensorAny<R, T, B, D>, TensorIterOrder, &B)
where
    R: DataAPI<Data = B::Raw>,
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    type Out = Tensor<T, B, D>;

    fn ones_like_f(self) -> Result<Self::Out> {
        let (tensor, order, device) = self;
        let layout = layout_for_array_copy(tensor.layout(), order)?;
        let idx_max = layout.size();
        let storage = device.ones_impl(idx_max)?;
        unsafe { Ok(Tensor::new_unchecked(storage, layout)) }
    }
}

impl<R, T, B, D> OnesLikeAPI<()> for (&TensorAny<R, T, B, D>, &B)
where
    R: DataAPI<Data = B::Raw>,
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    type Out = Tensor<T, B, D>;

    fn ones_like_f(self) -> Result<Self::Out> {
        let (tensor, device) = self;
        ones_like_f((tensor, TensorIterOrder::default(), device))
    }
}

impl<R, T, B, D> OnesLikeAPI<()> for (&TensorAny<R, T, B, D>, TensorIterOrder)
where
    R: DataAPI<Data = B::Raw>,
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    type Out = Tensor<T, B, D>;

    fn ones_like_f(self) -> Result<Self::Out> {
        let (tensor, order) = self;
        let device = tensor.device();
        ones_like_f((tensor, order, device))
    }
}

impl<R, T, B, D> OnesLikeAPI<()> for &TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    type Out = Tensor<T, B, D>;

    fn ones_like_f(self) -> Result<Self::Out> {
        let device = self.device();
        ones_like_f((self, TensorIterOrder::default(), device))
    }
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
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
    pub fn ones_like(&self) -> Tensor<T, B, D> {
        ones_like((self, TensorIterOrder::default(), self.device()))
    }

    pub fn ones_like_f(&self) -> Result<Tensor<T, B, D>> {
        ones_like_f((self, TensorIterOrder::default(), self.device()))
    }
}

/* #endregion */

/* #region zeros */

pub trait ZerosAPI<Inp>: Sized {
    type Out;

    fn zeros_f(self) -> Result<Self::Out>;

    fn zeros(self) -> Self::Out {
        Self::zeros_f(self).unwrap()
    }
}

/// New tensor filled with zeros and having a specified shape.
///
/// # See also
///
/// - [Python array API standard: `zeros`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.zeros.html)
pub fn zeros<Param, Inp, Rhs>(param: Param) -> Rhs
where
    Param: ZerosAPI<Inp, Out = Rhs>,
{
    return ZerosAPI::zeros(param);
}

pub fn zeros_f<Param, Inp, Rhs>(param: Param) -> Result<Rhs>
where
    Param: ZerosAPI<Inp, Out = Rhs>,
{
    return ZerosAPI::zeros_f(param);
}

impl<T, D, B, L> ZerosAPI<(T, D)> for (L, &B)
where
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
    L: Into<Layout<D>>,
{
    type Out = Tensor<T, B, IxD>;

    fn zeros_f(self) -> Result<Self::Out> {
        let (layout, device) = self;
        let layout: Layout<D> = layout.into();
        let (_, idx_max) = layout.bounds_index()?;
        let storage = B::zeros_impl(device, idx_max)?;
        unsafe { Ok(Tensor::new_unchecked(storage, layout.into_dim()?)) }
    }
}

impl<T, D, L> ZerosAPI<(T, D)> for L
where
    T: Num + Clone,
    D: DimAPI,
    L: Into<Layout<D>>,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    fn zeros_f(self) -> Result<Self::Out> {
        zeros_f((self, &DeviceCpu::default()))
    }
}

/* #endregion */

/* #region zeros_like */

pub trait ZerosLikeAPI<Inp>: Sized {
    type Out;

    fn zeros_like_f(self) -> Result<Self::Out>;

    fn zeros_like(self) -> Self::Out {
        Self::zeros_like_f(self).unwrap()
    }
}

/// New tensor filled with zeros and having the same shape as an input
/// tensor.
///
/// # See also
///
/// - [Python array API standard: `zeros_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.zeros_like.html)
pub fn zeros_like<Param, Inp, Rhs>(param: Param) -> Rhs
where
    Param: ZerosLikeAPI<Inp, Out = Rhs>,
{
    return ZerosLikeAPI::zeros_like(param);
}

pub fn zeros_like_f<Param, Inp, Rhs>(param: Param) -> Result<Rhs>
where
    Param: ZerosLikeAPI<Inp, Out = Rhs>,
{
    return ZerosLikeAPI::zeros_like_f(param);
}

impl<R, T, B, D> ZerosLikeAPI<()> for (&TensorAny<R, T, B, D>, TensorIterOrder, &B)
where
    R: DataAPI<Data = B::Raw>,
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    type Out = Tensor<T, B, D>;

    fn zeros_like_f(self) -> Result<Self::Out> {
        let (tensor, order, device) = self;
        let layout = layout_for_array_copy(tensor.layout(), order)?;
        let idx_max = layout.size();
        let storage = B::zeros_impl(device, idx_max)?;
        unsafe { Ok(Tensor::new_unchecked(storage, layout)) }
    }
}

impl<R, T, B, D> ZerosLikeAPI<()> for (&TensorAny<R, T, B, D>, &B)
where
    R: DataAPI<Data = B::Raw>,
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    type Out = Tensor<T, B, D>;

    fn zeros_like_f(self) -> Result<Self::Out> {
        let (tensor, device) = self;
        zeros_like_f((tensor, TensorIterOrder::default(), device))
    }
}

impl<R, T, B, D> ZerosLikeAPI<()> for (&TensorAny<R, T, B, D>, TensorIterOrder)
where
    R: DataAPI<Data = B::Raw>,
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    type Out = Tensor<T, B, D>;

    fn zeros_like_f(self) -> Result<Self::Out> {
        let (tensor, order) = self;
        let device = tensor.device();
        zeros_like_f((tensor, order, device))
    }
}

impl<R, T, B, D> ZerosLikeAPI<()> for &TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    type Out = Tensor<T, B, D>;

    fn zeros_like_f(self) -> Result<Self::Out> {
        zeros_like_f((self, TensorIterOrder::default(), self.device()))
    }
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
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
    pub fn zeros_like(&self) -> Tensor<T, B, D> {
        zeros_like((self, TensorIterOrder::default(), self.device()))
    }

    pub fn zeros_like_f(&self) -> Result<Tensor<T, B, D>> {
        zeros_like_f((self, TensorIterOrder::default(), self.device()))
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;
    use num::complex::Complex32;

    #[test]
    fn playground() {
        let a = arange((2.5, 3.2, 0.02));
        println!("{a:6.3?}");
        let a = arange(15.0);
        println!("{a:6.3?}");
        let a = arange((15.0, &DeviceCpu::default()));
        println!("{a:6.3?}");
        let a: Tensor<f64, _> = unsafe { empty(([15, 18].f(), &DeviceCpuSerial)) };
        println!("{a:6.3?}");
        let a = unsafe { a.empty_like() };
        println!("{a:6.3?}");
        let a = unsafe { empty_like((&a, TensorIterOrder::C)) };
        println!("{a:6.3?}");
        let a: Tensor<f64, _> = eye(3);
        println!("{a:6.3?}");
        let a = full(([2, 2].f(), 3.16));
        println!("{a:6.3?}");
        let a = full_like((&a, 2.71));
        println!("{a:6.3?}");
        let a = a.full_like(2.71);
        println!("{a:6.3?}");
        let a = linspace((3.2, 4.7, 12));
        println!("{a:6.3?}");
        let a = linspace((Complex32::new(1.8, 7.5), Complex32::new(-8.9, 1.6), 12));
        println!("{a:6.3?}");
        let a: Tensor<f64> = ones(vec![2, 2]);
        println!("{a:6.3?}");
        let a = a.ones_like();
        println!("{a:6.3?}");
        let a: Tensor<f64> = zeros([2, 2]);
        println!("{a:6.3?}");
        let a: Tensor<f64, _> = zeros(([2, 2], &DeviceCpuSerial));
        println!("{a:6.3?}");
        let a = a.zeros_like();
        println!("{a:6.3?}");
    }
}
