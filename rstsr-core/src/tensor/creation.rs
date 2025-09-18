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
//! - [x] `meshgrid`
//! - [x] `ones`: [`ones`]
//! - [x] `ones_like`: [`ones_like`]
//! - [x] `tril`
//! - [x] `triu`
//! - [x] `zeros`: [`zeros`]
//! - [x] `zeros_like`: [`zeros_like`]

use crate::prelude_dev::*;
use num::complex::ComplexFloat;
use num::Num;

/* #region arange */

pub trait ArangeAPI<Inp> {
    type Out;

    fn arange_f(self) -> Result<Self::Out>;

    fn arange(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::arange_f(self).unwrap()
    }
}

/// Evenly spaced values within the half-open interval `[start, stop)` as
/// one-dimensional array.
///
/// # See also
///
/// - [Python array API standard: `arange`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.arange.html)
pub fn arange<Args, Inp>(param: Args) -> Args::Out
where
    Args: ArangeAPI<Inp>,
{
    return ArangeAPI::arange(param);
}

pub fn arange_f<Args, Inp>(param: Args) -> Result<Args::Out>
where
    Args: ArangeAPI<Inp>,
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
        unsafe { Ok(Tensor::new_unchecked(data, layout)) }
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

pub trait EmptyAPI<Inp> {
    type Out;

    /// # Safety
    ///
    /// This function is unsafe because it creates a tensor with uninitialized.
    unsafe fn empty_f(self) -> Result<Self::Out>;

    /// # Safety
    ///
    /// This function is unsafe because it creates a tensor with uninitialized.
    unsafe fn empty(self) -> Self::Out
    where
        Self: Sized,
    {
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
pub unsafe fn empty<Args, Inp>(param: Args) -> Args::Out
where
    Args: EmptyAPI<Inp>,
{
    return EmptyAPI::empty(param);
}

/// # Safety
///
/// This function is unsafe because it creates a tensor with uninitialized.
pub unsafe fn empty_f<Args, Inp>(param: Args) -> Result<Args::Out>
where
    Args: EmptyAPI<Inp>,
{
    return EmptyAPI::empty_f(param);
}

impl<T, D, B> EmptyAPI<(T, D)> for (Layout<D>, &B)
where
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    unsafe fn empty_f(self) -> Result<Self::Out> {
        let (layout, device) = self;
        let (_, idx_max) = layout.bounds_index()?;
        let storage = B::empty_impl(device, idx_max)?;
        unsafe { Ok(Tensor::new_unchecked(storage, layout.into_dim()?)) }
    }
}

impl<T, D, B> EmptyAPI<(T, D)> for (D, FlagOrder, &B)
where
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    unsafe fn empty_f(self) -> Result<Self::Out> {
        let (shape, order, device) = self;
        let layout = shape.new_contig(None, order);
        empty_f((layout, device))
    }
}

impl<T, D, B> EmptyAPI<(T, D)> for (D, &B)
where
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    unsafe fn empty_f(self) -> Result<Self::Out> {
        let (shape, device) = self;
        let default_order = device.default_order();
        let layout = shape.new_contig(None, default_order);
        empty_f((layout, device))
    }
}

impl<T, D> EmptyAPI<(T, D)> for (D, FlagOrder)
where
    D: DimAPI,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    unsafe fn empty_f(self) -> Result<Self::Out> {
        let (shape, order) = self;
        empty_f((shape, order, &DeviceCpu::default()))
    }
}

#[duplicate_item(L; [D]; [Layout<D>])]
impl<T, D> EmptyAPI<(T, D)> for L
where
    D: DimAPI,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    unsafe fn empty_f(self) -> Result<Self::Out> {
        empty_f((self, &DeviceCpu::default()))
    }
}

/* #endregion */

/* #region empty_like */

pub trait EmptyLikeAPI<Inp> {
    type Out;

    /// # Safety
    ///
    /// This function is unsafe because it creates a tensor with uninitialized.
    unsafe fn empty_like_f(self) -> Result<Self::Out>;

    /// # Safety
    ///
    /// This function is unsafe because it creates a tensor with uninitialized.
    unsafe fn empty_like(self) -> Self::Out
    where
        Self: Sized,
    {
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
pub unsafe fn empty_like<Args, Inp>(param: Args) -> Args::Out
where
    Args: EmptyLikeAPI<Inp>,
{
    return EmptyLikeAPI::empty_like(param);
}

/// # Safety
///
/// This function is unsafe because it creates a tensor with uninitialized.
pub unsafe fn empty_like_f<Args, Inp>(param: Args) -> Result<Args::Out>
where
    Args: EmptyLikeAPI<Inp>,
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
        unsafe { Ok(Tensor::new_unchecked(storage, layout)) }
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

pub trait EyeAPI<Inp> {
    type Out;

    fn eye_f(self) -> Result<Self::Out>;

    fn eye(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::eye_f(self).unwrap()
    }
}

/// Returns a two-dimensional array with ones on the kth diagonal and zeros
/// elsewhere.
///
/// # See also
///
/// - [Python array API standard: `eye`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.eye.html)
pub fn eye<Args, Inp>(param: Args) -> Args::Out
where
    Args: EyeAPI<Inp>,
{
    return EyeAPI::eye(param);
}

pub fn eye_f<Args, Inp>(param: Args) -> Result<Args::Out>
where
    Args: EyeAPI<Inp>,
{
    return EyeAPI::eye_f(param);
}

impl<T, B> EyeAPI<(T, B)> for (usize, usize, isize, FlagOrder, &B)
where
    T: Num,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T> + OpAssignAPI<T, Ix1>,
{
    type Out = Tensor<T, B, IxD>;

    fn eye_f(self) -> Result<Self::Out> {
        let (n_rows, n_cols, k, order, device) = self;
        let layout = match order {
            RowMajor => [n_rows, n_cols].c(),
            ColMajor => [n_cols, n_rows].f(),
        };
        let mut storage = device.zeros_impl(layout.size())?;
        let layout_diag = layout.diagonal(Some(k), Some(0), Some(1))?;
        device.fill(storage.raw_mut(), &layout_diag, T::one())?;
        unsafe { Ok(Tensor::new_unchecked(storage, layout.into_dim()?)) }
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
        let default_order = device.default_order();
        eye_f((n_rows, n_cols, k, default_order, device))
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
        let default_order = device.default_order();
        eye_f((n_rows, n_rows, 0, default_order, device))
    }
}

impl<T> EyeAPI<T> for (usize, usize, isize, FlagOrder)
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
        let device = DeviceCpu::default();
        let default_order = device.default_order();
        eye_f((n_rows, n_cols, k, default_order, &device))
    }
}

impl<T> EyeAPI<T> for usize
where
    T: Num + Clone + Send + Sync,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    fn eye_f(self) -> Result<Self::Out> {
        // n_rows -> (n_rows, n_rows, 0, C)
        let device = DeviceCpu::default();
        let default_order = device.default_order();
        eye_f((self, self, 0, default_order, &device))
    }
}

/* #endregion */

/* #region full */

pub trait FullAPI<Inp> {
    type Out;

    fn full_f(self) -> Result<Self::Out>;

    fn full(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::full_f(self).unwrap()
    }
}

/// New tensor having a specified shape and filled with given value.
///
/// # See also
///
/// - [Python array API standard: `full`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.full.html)
pub fn full<Args, Inp>(param: Args) -> Args::Out
where
    Args: FullAPI<Inp>,
{
    return FullAPI::full(param);
}

pub fn full_f<Args, Inp>(param: Args) -> Result<Args::Out>
where
    Args: FullAPI<Inp>,
{
    return FullAPI::full_f(param);
}

impl<T, D, B> FullAPI<(T, D)> for (Layout<D>, T, &B)
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    fn full_f(self) -> Result<Self::Out> {
        let (layout, fill, device) = self;
        let idx_max = layout.size();
        let storage = device.full_impl(idx_max, fill)?;
        unsafe { Ok(Tensor::new_unchecked(storage, layout.into_dim()?)) }
    }
}

impl<T, D, B> FullAPI<(T, D)> for (D, T, FlagOrder, &B)
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    fn full_f(self) -> Result<Self::Out> {
        let (shape, fill, order, device) = self;
        let layout = shape.new_contig(None, order);
        full_f((layout, fill, device))
    }
}

impl<T, D, B> FullAPI<(T, D)> for (D, T, &B)
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    fn full_f(self) -> Result<Self::Out> {
        let (shape, fill, device) = self;
        let default_order = device.default_order();
        let layout = shape.new_contig(None, default_order);
        full_f((layout, fill, device))
    }
}

impl<T, D> FullAPI<(T, D)> for (D, T, FlagOrder)
where
    T: Clone,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    fn full_f(self) -> Result<Self::Out> {
        let (shape, fill, order) = self;
        full_f((shape, fill, order, &DeviceCpu::default()))
    }
}

#[duplicate_item(L; [D]; [Layout<D>])]
impl<T, D> FullAPI<(T, D)> for (L, T)
where
    T: Clone,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    fn full_f(self) -> Result<Self::Out> {
        let (shape, fill) = self;
        full_f((shape, fill, &DeviceCpu::default()))
    }
}

/* #endregion */

/* #region full_like */

pub trait FullLikeAPI<Inp> {
    type Out;

    fn full_like_f(self) -> Result<Self::Out>;

    fn full_like(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::full_like_f(self).unwrap()
    }
}

/// New tensor filled with given value and having the same shape as an input
/// tensor.
///
/// # See also
///
/// - [Python array API standard: `full_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.full_like.html)
pub fn full_like<Args, Inp>(param: Args) -> Args::Out
where
    Args: FullLikeAPI<Inp>,
{
    return FullLikeAPI::full_like(param);
}

pub fn full_like_f<Args, Inp>(param: Args) -> Result<Args::Out>
where
    Args: FullLikeAPI<Inp>,
{
    return FullLikeAPI::full_like_f(param);
}

impl<R, T, B, D> FullLikeAPI<()> for (&TensorAny<R, T, B, D>, T, TensorIterOrder, &B)
where
    T: Clone,
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
    T: Clone,
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
    T: Clone,
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
    T: Clone,
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

pub trait LinspaceAPI<Inp> {
    type Out;

    fn linspace_f(self) -> Result<Self::Out>;

    fn linspace(self) -> Self::Out
    where
        Self: Sized,
    {
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
pub fn linspace<Args, Inp>(param: Args) -> Args::Out
where
    Args: LinspaceAPI<Inp>,
{
    return LinspaceAPI::linspace(param);
}

pub fn linspace_f<Args, Inp>(param: Args) -> Result<Args::Out>
where
    Args: LinspaceAPI<Inp>,
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
        unsafe { Ok(Tensor::new_unchecked(data, layout)) }
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

pub trait OnesAPI<Inp> {
    type Out;

    fn ones_f(self) -> Result<Self::Out>;

    fn ones(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::ones_f(self).unwrap()
    }
}

/// New tensor filled with ones and having a specified shape.
///
/// # See also
///
/// - [Python array API standard: `ones`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.ones.html)
pub fn ones<Args, Inp>(param: Args) -> Args::Out
where
    Args: OnesAPI<Inp>,
{
    return OnesAPI::ones(param);
}

pub fn ones_f<Args, Inp>(param: Args) -> Result<Args::Out>
where
    Args: OnesAPI<Inp>,
{
    return OnesAPI::ones_f(param);
}

impl<T, D, B> OnesAPI<(T, D)> for (Layout<D>, &B)
where
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    fn ones_f(self) -> Result<Self::Out> {
        let (layout, device) = self;
        let (_, idx_max) = layout.bounds_index()?;
        let storage = device.ones_impl(idx_max)?;
        unsafe { Ok(Tensor::new_unchecked(storage, layout.into_dim()?)) }
    }
}

impl<T, D, B> OnesAPI<(T, D)> for (D, FlagOrder, &B)
where
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    fn ones_f(self) -> Result<Self::Out> {
        let (shape, order, device) = self;
        let layout = shape.new_contig(None, order);
        ones_f((layout, device))
    }
}

impl<T, D, B> OnesAPI<(T, D)> for (D, &B)
where
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    fn ones_f(self) -> Result<Self::Out> {
        let (shape, device) = self;
        let default_order = device.default_order();
        let layout = shape.new_contig(None, default_order);
        ones_f((layout, device))
    }
}

impl<T, D> OnesAPI<(T, D)> for (D, FlagOrder)
where
    T: Num + Clone,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    fn ones_f(self) -> Result<Self::Out> {
        let (shape, order) = self;
        ones_f((shape, order, &DeviceCpu::default()))
    }
}

#[duplicate_item(L; [D]; [Layout<D>])]
impl<T, D> OnesAPI<(T, D)> for L
where
    T: Num + Clone,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    fn ones_f(self) -> Result<Self::Out> {
        ones_f((self, &DeviceCpu::default()))
    }
}

/* #endregion */

/* #region ones_like */

pub trait OnesLikeAPI<Inp> {
    type Out;

    fn ones_like_f(self) -> Result<Self::Out>;

    fn ones_like(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::ones_like_f(self).unwrap()
    }
}

/// New tensor filled with ones and having the same shape as an input
/// tensor.
///
/// # See also
///
/// - [Python array API standard: `ones_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.ones_like.html)
pub fn ones_like<Args, Inp>(param: Args) -> Args::Out
where
    Args: OnesLikeAPI<Inp>,
{
    return OnesLikeAPI::ones_like(param);
}

pub fn ones_like_f<Args, Inp>(param: Args) -> Result<Args::Out>
where
    Args: OnesLikeAPI<Inp>,
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

/* #region uninit */

pub trait UninitAPI<Inp> {
    type Out;

    fn uninit_f(self) -> Result<Self::Out>;

    fn uninit(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::uninit_f(self).unwrap()
    }
}

/// New tensor filled with uninitialized values and having a specified shape.
pub fn uninit<Args, Inp>(param: Args) -> Args::Out
where
    Args: UninitAPI<Inp>,
{
    return UninitAPI::uninit(param);
}

pub fn uninit_f<Args, Inp>(param: Args) -> Result<Args::Out>
where
    Args: UninitAPI<Inp>,
{
    return UninitAPI::uninit_f(param);
}

impl<T, D, B> UninitAPI<(T, D)> for (Layout<D>, &B)
where
    T: Num,
    D: DimAPI,
    B: DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<MaybeUninit<T>, B, IxD>;

    fn uninit_f(self) -> Result<Self::Out> {
        let (layout, device) = self;
        let (_, idx_max) = layout.bounds_index()?;
        let storage = B::uninit_impl(device, idx_max)?;
        unsafe { Ok(Tensor::new_unchecked(storage, layout.into_dim()?)) }
    }
}

impl<T, D, B> UninitAPI<(T, D)> for (D, FlagOrder, &B)
where
    T: Num,
    D: DimAPI,
    B: DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<MaybeUninit<T>, B, IxD>;

    fn uninit_f(self) -> Result<Self::Out> {
        let (shape, order, device) = self;
        let layout = shape.new_contig(None, order);
        uninit_f((layout, device))
    }
}

impl<T, D, B> UninitAPI<(T, D)> for (D, &B)
where
    T: Num,
    D: DimAPI,
    B: DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<MaybeUninit<T>, B, IxD>;

    fn uninit_f(self) -> Result<Self::Out> {
        let (shape, device) = self;
        let default_order = device.default_order();
        let layout = shape.new_contig(None, default_order);
        uninit_f((layout, device))
    }
}

impl<T, D> UninitAPI<(T, D)> for (D, FlagOrder)
where
    T: Num + Clone,
    D: DimAPI,
{
    type Out = Tensor<MaybeUninit<T>, DeviceCpu, IxD>;

    fn uninit_f(self) -> Result<Self::Out> {
        let (shape, order) = self;
        uninit_f((shape, order, &DeviceCpu::default()))
    }
}

#[duplicate_item(L; [D]; [Layout<D>])]
impl<T, D> UninitAPI<(T, D)> for L
where
    T: Num + Clone,
    D: DimAPI,
{
    type Out = Tensor<MaybeUninit<T>, DeviceCpu, IxD>;

    fn uninit_f(self) -> Result<Self::Out> {
        uninit_f((self, &DeviceCpu::default()))
    }
}

/* #endregion */

/* #region assume_init */

/// Converts a tensor with uninitialized values into a tensor with initialized values.
///
/// # Safety
///
/// This function is unsafe because it assumes that all elements in the input tensor are properly
/// initialized.
pub unsafe fn assume_init_f<T, B, D>(tensor: Tensor<MaybeUninit<T>, B, D>) -> Result<Tensor<T, B, D>>
where
    D: DimAPI,
    B: DeviceAPI<T> + DeviceAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T>,
{
    let (storage, layout) = tensor.into_raw_parts();
    let storage = B::assume_init_impl(storage)?;
    unsafe { Ok(Tensor::new_unchecked(storage, layout)) }
}

/// Converts a tensor with uninitialized values into a tensor with initialized values.
///
/// # Safety
///
/// This function is unsafe because it assumes that all elements in the input tensor are properly
/// initialized.
pub unsafe fn assume_init<T, B, D>(tensor: Tensor<MaybeUninit<T>, B, D>) -> Tensor<T, B, D>
where
    D: DimAPI,
    B: DeviceAPI<T> + DeviceAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T>,
{
    unsafe { assume_init_f(tensor).unwrap() }
}

/* #endregion */

/* #region zeros */

pub trait ZerosAPI<Inp> {
    type Out;

    fn zeros_f(self) -> Result<Self::Out>;

    fn zeros(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::zeros_f(self).unwrap()
    }
}

/// New tensor filled with zeros and having a specified shape.
///
/// # See also
///
/// - [Python array API standard: `zeros`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.zeros.html)
pub fn zeros<Args, Inp>(param: Args) -> Args::Out
where
    Args: ZerosAPI<Inp>,
{
    return ZerosAPI::zeros(param);
}

pub fn zeros_f<Args, Inp>(param: Args) -> Result<Args::Out>
where
    Args: ZerosAPI<Inp>,
{
    return ZerosAPI::zeros_f(param);
}

impl<T, D, B> ZerosAPI<(T, D)> for (Layout<D>, &B)
where
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    fn zeros_f(self) -> Result<Self::Out> {
        let (layout, device) = self;
        let (_, idx_max) = layout.bounds_index()?;
        let storage = B::zeros_impl(device, idx_max)?;
        unsafe { Ok(Tensor::new_unchecked(storage, layout.into_dim()?)) }
    }
}

impl<T, D, B> ZerosAPI<(T, D)> for (D, FlagOrder, &B)
where
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    fn zeros_f(self) -> Result<Self::Out> {
        let (shape, order, device) = self;
        let layout = shape.new_contig(None, order);
        zeros_f((layout, device))
    }
}

impl<T, D, B> ZerosAPI<(T, D)> for (D, &B)
where
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    fn zeros_f(self) -> Result<Self::Out> {
        let (shape, device) = self;
        let default_order = device.default_order();
        let layout = shape.new_contig(None, default_order);
        zeros_f((layout, device))
    }
}

impl<T, D> ZerosAPI<(T, D)> for (D, FlagOrder)
where
    T: Num + Clone,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    fn zeros_f(self) -> Result<Self::Out> {
        let (shape, order) = self;
        zeros_f((shape, order, &DeviceCpu::default()))
    }
}

#[duplicate_item(L; [D]; [Layout<D>])]
impl<T, D> ZerosAPI<(T, D)> for L
where
    T: Num + Clone,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    fn zeros_f(self) -> Result<Self::Out> {
        zeros_f((self, &DeviceCpu::default()))
    }
}

/* #endregion */

/* #region zeros_like */

pub trait ZerosLikeAPI<Inp> {
    type Out;

    fn zeros_like_f(self) -> Result<Self::Out>;

    fn zeros_like(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::zeros_like_f(self).unwrap()
    }
}

/// New tensor filled with zeros and having the same shape as an input
/// tensor.
///
/// # See also
///
/// - [Python array API standard: `zeros_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.zeros_like.html)
pub fn zeros_like<Args, Inp>(param: Args) -> Args::Out
where
    Args: ZerosLikeAPI<Inp>,
{
    return ZerosLikeAPI::zeros_like(param);
}

pub fn zeros_like_f<Args, Inp>(param: Args) -> Result<Args::Out>
where
    Args: ZerosLikeAPI<Inp>,
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

/* #region tril */

pub trait TrilAPI<Inp> {
    type Out;

    fn tril_f(self) -> Result<Self::Out>;

    fn tril(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::tril_f(self).unwrap()
    }
}

/// Returns the lower triangular part of a matrix (or a stack of matrices) x.
///
/// # See also
///
/// - [Python array API standard: `tril`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.tril.html)
pub fn tril<Args, Inp>(param: Args) -> Args::Out
where
    Args: TrilAPI<Inp>,
{
    return TrilAPI::tril(param);
}

pub fn tril_f<Args, Inp>(param: Args) -> Result<Args::Out>
where
    Args: TrilAPI<Inp>,
{
    return TrilAPI::tril_f(param);
}

impl<T, D, B> TrilAPI<()> for (TensorView<'_, T, B, D>, isize)
where
    T: Num + Clone,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceCreationTriAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, D, D>
        + OpAssignAPI<T, D>,
    B::Raw: Clone,
{
    type Out = Tensor<T, B, D>;

    fn tril_f(self) -> Result<Self::Out> {
        let (x, k) = self;
        let default_order = x.device().default_order();
        let mut x = x.into_contig_f(default_order)?;
        let device = x.device().clone();
        let layout = x.layout().clone();
        device.tril_impl(x.raw_mut(), &layout, k)?;
        Ok(x)
    }
}

impl<T, D, B> TrilAPI<()> for TensorView<'_, T, B, D>
where
    T: Num + Clone,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceCreationTriAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, D, D>
        + OpAssignAPI<T, D>,
    B::Raw: Clone,
{
    type Out = Tensor<T, B, D>;

    fn tril_f(self) -> Result<Self::Out> {
        tril_f((self, 0))
    }
}

impl<'a, T, D, B> TrilAPI<()> for (TensorMut<'a, T, B, D>, isize)
where
    T: Num + Clone,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceCreationTriAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, D, D>
        + OpAssignAPI<T, D>,
{
    type Out = TensorMut<'a, T, B, D>;

    fn tril_f(self) -> Result<Self::Out> {
        let (mut x, k) = self;
        let device = x.device().clone();
        let layout = x.layout().clone();
        device.tril_impl(x.raw_mut(), &layout, k)?;
        Ok(x)
    }
}

impl<'a, T, D, B> TrilAPI<()> for TensorMut<'a, T, B, D>
where
    T: Num + Clone,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceCreationTriAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, D, D>
        + OpAssignAPI<T, D>,
{
    type Out = TensorMut<'a, T, B, D>;

    fn tril_f(self) -> Result<Self::Out> {
        tril_f((self, 0))
    }
}

impl<T, D, B> TrilAPI<()> for (Tensor<T, B, D>, isize)
where
    T: Num + Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, D>;

    fn tril_f(self) -> Result<Self::Out> {
        let (mut x, k) = self;
        let device = x.device().clone();
        let layout = x.layout().clone();
        device.tril_impl(x.raw_mut(), &layout, k)?;
        Ok(x)
    }
}

impl<T, D, B> TrilAPI<()> for Tensor<T, B, D>
where
    T: Num + Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, D>;

    fn tril_f(self) -> Result<Self::Out> {
        tril_f((self, 0))
    }
}

impl<R, T, D, B> TrilAPI<()> for (&TensorAny<R, T, B, D>, isize)
where
    R: DataAPI<Data = B::Raw>,
    T: Num + Clone,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceCreationTriAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, D, D>
        + OpAssignAPI<T, D>,
    B::Raw: Clone,
{
    type Out = Tensor<T, B, D>;

    fn tril_f(self) -> Result<Self::Out> {
        let (x, k) = self;
        tril_f((x.view(), k))
    }
}

impl<R, T, D, B> TrilAPI<()> for &TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    T: Num + Clone,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceCreationTriAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, D, D>
        + OpAssignAPI<T, D>,
    B::Raw: Clone,
{
    type Out = Tensor<T, B, D>;

    fn tril_f(self) -> Result<Self::Out> {
        tril_f((self.view(), 0))
    }
}

/* #endregion */

/* #region triu */

pub trait TriuAPI<Inp> {
    type Out;

    fn triu_f(self) -> Result<Self::Out>;

    fn triu(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::triu_f(self).unwrap()
    }
}

/// Returns the upper triangular part of a matrix (or a stack of matrices) x.
///
/// # See also
///
/// - [Python array API standard: `triu`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.triu.html)
pub fn triu<Args, Inp>(param: Args) -> Args::Out
where
    Args: TriuAPI<Inp>,
{
    return TriuAPI::triu(param);
}

pub fn triu_f<Args, Inp>(param: Args) -> Result<Args::Out>
where
    Args: TriuAPI<Inp>,
{
    return TriuAPI::triu_f(param);
}

impl<T, D, B> TriuAPI<()> for (TensorView<'_, T, B, D>, isize)
where
    T: Num + Clone,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceCreationTriAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, D, D>
        + OpAssignAPI<T, D>,
    B::Raw: Clone,
{
    type Out = Tensor<T, B, D>;

    fn triu_f(self) -> Result<Self::Out> {
        let (x, k) = self;
        let default_order = x.device().default_order();
        let mut x = x.into_contig_f(default_order)?;
        let device = x.device().clone();
        let layout = x.layout().clone();
        device.triu_impl(x.raw_mut(), &layout, k)?;
        Ok(x)
    }
}

impl<T, D, B> TriuAPI<()> for TensorView<'_, T, B, D>
where
    T: Num + Clone,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceCreationTriAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, D, D>
        + OpAssignAPI<T, D>,
    B::Raw: Clone,
{
    type Out = Tensor<T, B, D>;

    fn triu_f(self) -> Result<Self::Out> {
        triu_f((self, 0))
    }
}

impl<'a, T, D, B> TriuAPI<()> for (TensorMut<'a, T, B, D>, isize)
where
    T: Num + Clone,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceCreationTriAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, D, D>
        + OpAssignAPI<T, D>,
{
    type Out = TensorMut<'a, T, B, D>;

    fn triu_f(self) -> Result<Self::Out> {
        let (mut x, k) = self;
        let device = x.device().clone();
        let layout = x.layout().clone();
        device.triu_impl(x.raw_mut(), &layout, k)?;
        Ok(x)
    }
}

impl<'a, T, D, B> TriuAPI<()> for TensorMut<'a, T, B, D>
where
    T: Num + Clone,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceCreationTriAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, D, D>
        + OpAssignAPI<T, D>,
{
    type Out = TensorMut<'a, T, B, D>;

    fn triu_f(self) -> Result<Self::Out> {
        triu_f((self, 0))
    }
}

impl<T, D, B> TriuAPI<()> for (Tensor<T, B, D>, isize)
where
    T: Num + Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, D>;

    fn triu_f(self) -> Result<Self::Out> {
        let (mut x, k) = self;
        let device = x.device().clone();
        let layout = x.layout().clone();
        device.triu_impl(x.raw_mut(), &layout, k)?;
        Ok(x)
    }
}

impl<T, D, B> TriuAPI<()> for Tensor<T, B, D>
where
    T: Num + Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, D>;

    fn triu_f(self) -> Result<Self::Out> {
        triu_f((self, 0))
    }
}

impl<R, T, D, B> TriuAPI<()> for (&TensorAny<R, T, B, D>, isize)
where
    R: DataAPI<Data = B::Raw>,
    T: Num + Clone,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceCreationTriAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, D, D>
        + OpAssignAPI<T, D>,
    B::Raw: Clone,
{
    type Out = Tensor<T, B, D>;

    fn triu_f(self) -> Result<Self::Out> {
        let (x, k) = self;
        triu_f((x.view(), k))
    }
}

impl<R, T, D, B> TriuAPI<()> for &TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    T: Num + Clone,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceCreationTriAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, D, D>
        + OpAssignAPI<T, D>,
    B::Raw: Clone,
{
    type Out = Tensor<T, B, D>;

    fn triu_f(self) -> Result<Self::Out> {
        triu_f((self.view(), 0))
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
        let a: Tensor<f64, _> = unsafe { empty(([15, 18].f(), &DeviceCpuSerial::default())) };
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
        let a: Tensor<f64, _> = zeros(([2, 2], &DeviceCpuSerial::default()));
        println!("{a:6.3?}");
        let a = a.zeros_like();
        println!("{a:6.3?}");
    }

    #[test]
    fn test_tril() {
        let a = arange((1, 10)).into_shape((3, 3));
        let b = a.view().tril();
        println!("{b:6.3?}");
        let b = triu((a, 1));
        println!("{b:6.3?}");
    }
}
