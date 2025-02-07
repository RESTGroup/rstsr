//! This module handles tensor data manipulation.

use crate::prelude_dev::*;

/* #region broadcast_arrays */

/// Broadcasts any number of arrays against each other.
///
/// # See also
///
/// [Python Array API standard: `broadcast_arrays`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.broadcast_arrays.html)
pub fn broadcast_arrays<S>(tensors: Vec<TensorBase<S, IxD>>) -> Vec<TensorBase<S, IxD>> {
    broadcast_arrays_f(tensors).unwrap()
}

pub fn broadcast_arrays_f<S>(tensors: Vec<TensorBase<S, IxD>>) -> Result<Vec<TensorBase<S, IxD>>> {
    // fast return if there is only zero/one tensor
    if tensors.len() <= 1 {
        return Ok(tensors);
    }
    let mut shape_b = tensors[0].shape().clone();
    for tensor in tensors.iter().skip(1) {
        let shape = tensor.shape();
        let (shape, _, _) = broadcast_shape(shape, &shape_b)?;
        shape_b = shape;
    }
    let mut tensors_new = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let tensor = into_broadcast_f(tensor, shape_b.clone())?;
        tensors_new.push(tensor);
    }
    return Ok(tensors_new);
}

/* #endregion */

/* #region broadcast_to */

pub fn into_broadcast_f<S, D, D2>(tensor: TensorBase<S, D>, shape: D2) -> Result<TensorBase<S, D2>>
where
    D: DimAPI + DimMaxAPI<D2, Max = D2>,
    D2: DimAPI,
{
    let shape1 = tensor.shape();
    let shape2 = &shape;
    let (shape, tp1, _) = broadcast_shape(shape1, shape2)?;
    let (storage, layout) = tensor.into_raw_parts();
    let layout = update_layout_by_shape(&layout, &shape, &tp1)?;
    unsafe { Ok(TensorBase::new_unchecked(storage, layout)) }
}

/// Broadcasts an array to a specified shape.
///
/// # See also
///
/// [Python Array API standard: `broadcast_to`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.broadcast_to.html)
pub fn to_broadcast<R, T, B, D, D2>(
    tensor: &TensorAny<R, T, B, D>,
    shape: D2,
) -> TensorView<'_, T, B, D2>
where
    D: DimAPI + DimMaxAPI<D2, Max = D2>,
    D2: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_broadcast_f(tensor.view(), shape).unwrap()
}

pub fn to_broadcast_f<R, T, B, D, D2>(
    tensor: &TensorAny<R, T, B, D>,
    shape: D2,
) -> Result<TensorView<'_, T, B, D2>>
where
    D: DimAPI + DimMaxAPI<D2, Max = D2>,
    D2: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_broadcast_f(tensor.view(), shape)
}

pub fn into_broadcast<S, D, D2>(tensor: TensorBase<S, D>, shape: D2) -> TensorBase<S, D2>
where
    D: DimAPI + DimMaxAPI<D2, Max = D2>,
    D2: DimAPI,
{
    into_broadcast_f(tensor, shape).unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Broadcasts an array to a specified shape.
    ///
    /// # See also
    ///
    /// [`to_broadcast`]
    pub fn to_broadcast<D2>(&self, shape: D2) -> TensorView<'_, T, B, D2>
    where
        D2: DimAPI,
        D: DimMaxAPI<D2, Max = D2>,
    {
        to_broadcast(self, shape)
    }

    pub fn to_broadcast_f<D2>(&self, shape: D2) -> Result<TensorView<'_, T, B, D2>>
    where
        D2: DimAPI,
        D: DimMaxAPI<D2, Max = D2>,
    {
        to_broadcast_f(self, shape)
    }

    /// Broadcasts an array to a specified shape.
    ///
    /// # See also
    ///
    /// [`to_broadcast`]
    pub fn into_broadcast<D2>(self, shape: D2) -> TensorAny<R, T, B, D2>
    where
        D2: DimAPI,
        D: DimMaxAPI<D2, Max = D2>,
    {
        into_broadcast(self, shape)
    }

    pub fn into_broadcast_f<D2>(self, shape: D2) -> Result<TensorAny<R, T, B, D2>>
    where
        D2: DimAPI,
        D: DimMaxAPI<D2, Max = D2>,
    {
        into_broadcast_f(self, shape)
    }
}

/* #endregion */

/* #region expand_dims */

pub fn into_expand_dims_f<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> Result<TensorBase<S, IxD>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
{
    // convert axis to negative indexes and sort
    let ndim: isize = TryInto::<isize>::try_into(tensor.ndim())?;
    let (storage, layout) = tensor.into_raw_parts();
    let mut layout = layout.into_dim::<IxD>()?;
    let mut axes: Vec<isize> = axes
        .try_into()?
        .as_ref()
        .iter()
        .map(|&v| if v >= 0 { v - ndim - 1 } else { v })
        .collect::<Vec<isize>>();
    axes.sort();
    for &axis in axes.iter() {
        layout = layout.dim_insert(axis)?;
    }
    unsafe { Ok(TensorBase::new_unchecked(storage, layout)) }
}

/// Expands the shape of an array by inserting a new axis (dimension) of size
/// one at the position specified by `axis`.
///
/// # Panics
///
/// - If `axis` is greater than the number of axes in the original tensor.
///
/// # See also
///
/// [Python Array API standard: `expand_dims`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.expand_dims.html)
pub fn expand_dims<I, R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    axes: I,
) -> TensorView<'_, T, B, IxD>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_expand_dims_f(tensor.view(), axes).unwrap()
}

pub fn expand_dims_f<I, R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    axes: I,
) -> Result<TensorView<'_, T, B, IxD>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_expand_dims_f(tensor.view(), axes)
}

pub fn into_expand_dims<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> TensorBase<S, IxD>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
{
    into_expand_dims_f(tensor, axes).unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Expands the shape of an array by inserting a new axis (dimension) of
    /// size one at the position specified by `axis`.
    ///
    /// # See also
    ///
    /// [`expand_dims`]
    pub fn expand_dims<I>(&self, axes: I) -> TensorView<'_, T, B, IxD>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        into_expand_dims(self.view(), axes)
    }

    pub fn expand_dims_f<I>(&self, axes: I) -> Result<TensorView<'_, T, B, IxD>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        into_expand_dims_f(self.view(), axes)
    }

    /// Expands the shape of an array by inserting a new axis (dimension) of
    /// size one at the position specified by `axis`.
    ///
    /// # See also
    ///
    /// [`expand_dims`]
    pub fn into_expand_dims<I>(self, axes: I) -> TensorAny<R, T, B, IxD>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        into_expand_dims(self, axes)
    }

    pub fn into_expand_dims_f<I>(self, axes: I) -> Result<TensorAny<R, T, B, IxD>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        into_expand_dims_f(self, axes)
    }
}

/* #endregion */

/* #region flip */

pub fn into_flip_f<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> Result<TensorBase<S, D>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
{
    let (storage, mut layout) = tensor.into_raw_parts();
    let axes = axes.try_into()?;
    match axes {
        AxesIndex::Val(axis) => {
            layout = layout.dim_narrow(axis, slice!(None, None, -1))?;
        },
        AxesIndex::Vec(axes) => {
            for &axis in axes.iter() {
                layout = layout.dim_narrow(axis, slice!(None, None, -1))?;
            }
        },
    }
    unsafe { Ok(TensorBase::new_unchecked(storage, layout)) }
}

/// Reverses the order of elements in an array along the given axis.
///
/// # Panics
///
/// - If some index in `axis` is greater than the number of axes in the original
///   tensor.
///
/// # See also
///
/// [Python array API standard: `flip`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.flip.html)
pub fn flip<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axes: I) -> TensorView<'_, T, B, D>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_flip_f(tensor.view(), axes).unwrap()
}

pub fn flip_f<I, R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    axes: I,
) -> Result<TensorView<'_, T, B, D>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_flip_f(tensor.view(), axes)
}

pub fn into_flip<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> TensorBase<S, D>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
{
    into_flip_f(tensor, axes).unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Reverses the order of elements in an array along the given axis.
    ///
    /// # See also
    ///
    /// [`flip`]
    pub fn flip<I>(&self, axis: I) -> TensorView<'_, T, B, D>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        flip(self, axis)
    }

    pub fn flip_f<I>(&self, axis: I) -> Result<TensorView<'_, T, B, D>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        flip_f(self, axis)
    }

    /// Reverses the order of elements in an array along the given axis.
    ///
    /// # See also
    ///
    /// [`flip`]
    pub fn into_flip<I>(self, axis: I) -> TensorAny<R, T, B, D>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        into_flip(self, axis)
    }

    pub fn into_flip_f<I>(self, axis: I) -> Result<TensorAny<R, T, B, D>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        into_flip_f(self, axis)
    }
}

/* #endregion */

/* #region permute_dims */

pub fn into_transpose_f<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> Result<TensorBase<S, D>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
{
    let axes = axes.try_into()?;
    if axes.as_ref().is_empty() {
        return Ok(into_reverse_axes(tensor));
    }
    let (storage, layout) = tensor.into_raw_parts();
    let layout = layout.transpose(axes.as_ref())?;
    unsafe { Ok(TensorBase::new_unchecked(storage, layout)) }
}

/// Permutes the axes (dimensions) of an array `x`.
///
/// # See also
///
/// - [Python array API standard: `permute_dims`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.permute_dims.html)
pub fn transpose<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axes: I) -> TensorView<'_, T, B, D>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_transpose_f(tensor.view(), axes).unwrap()
}

pub fn transpose_f<I, R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    axes: I,
) -> Result<TensorView<'_, T, B, D>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_transpose_f(tensor.view(), axes)
}

pub fn into_transpose<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> TensorBase<S, D>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
{
    into_transpose_f(tensor, axes).unwrap()
}

pub use into_transpose as into_permute_dims;
pub use into_transpose_f as into_permute_dims_f;
pub use transpose as permute_dims;
pub use transpose_f as permute_dims_f;

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Permutes the axes (dimensions) of an array `x`.
    ///
    /// # See also
    ///
    /// [`transpose`]
    pub fn transpose<I>(&self, axes: I) -> TensorView<'_, T, B, D>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        transpose(self, axes)
    }

    pub fn transpose_f<I>(&self, axes: I) -> Result<TensorView<'_, T, B, D>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        transpose_f(self, axes)
    }

    /// Permutes the axes (dimensions) of an array `x`.
    ///
    /// # See also
    ///
    /// [`transpose`]
    pub fn into_transpose<I>(self, axes: I) -> TensorAny<R, T, B, D>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        into_transpose(self, axes)
    }

    pub fn into_transpose_f<I>(self, axes: I) -> Result<TensorAny<R, T, B, D>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        into_transpose_f(self, axes)
    }

    /// Permutes the axes (dimensions) of an array `x`.
    ///
    /// # See also
    ///
    /// [`transpose`]
    pub fn permute_dims<I>(&self, axes: I) -> TensorView<'_, T, B, D>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        transpose(self, axes)
    }

    pub fn permute_dims_f<I>(&self, axes: I) -> Result<TensorView<'_, T, B, D>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        transpose_f(self, axes)
    }

    /// Permutes the axes (dimensions) of an array `x`.
    ///
    /// # See also
    ///
    /// [`transpose`]
    pub fn into_permute_dims<I>(self, axes: I) -> TensorAny<R, T, B, D>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        into_transpose(self, axes)
    }

    pub fn into_permute_dims_f<I>(self, axes: I) -> Result<TensorAny<R, T, B, D>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        into_transpose_f(self, axes)
    }
}

/* #endregion */

/* #region reverse_axes */

pub fn into_reverse_axes<S, D>(tensor: TensorBase<S, D>) -> TensorBase<S, D>
where
    D: DimAPI,
{
    let (storage, layout) = tensor.into_raw_parts();
    let layout = layout.reverse_axes();
    unsafe { TensorBase::new_unchecked(storage, layout) }
}

/// Reverse the order of elements in an array along the given axis.
pub fn reverse_axes<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> TensorView<'_, T, B, D>
where
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_reverse_axes(tensor.view())
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Reverse the order of elements in an array along the given axis.
    ///
    /// # See also
    ///
    /// [`reverse_axes`]
    pub fn reverse_axes(&self) -> TensorView<'_, T, B, D> {
        into_reverse_axes(self.view())
    }

    /// Reverse the order of elements in an array along the given axis.
    ///
    /// # See also
    ///
    /// [`reverse_axes`]
    pub fn into_reverse_axes(self) -> TensorAny<R, T, B, D> {
        into_reverse_axes(self)
    }

    /// Reverse the order of elements in an array along the given axis.
    ///
    /// # See also
    ///
    /// [`reverse_axes`]
    pub fn t(&self) -> TensorView<'_, T, B, D> {
        into_reverse_axes(self.view())
    }
}

/* #endregion */

/* #region swapaxes */

pub fn into_swapaxes_f<I, S, D>(
    tensor: TensorBase<S, D>,
    axis1: I,
    axis2: I,
) -> Result<TensorBase<S, D>>
where
    D: DimAPI,
    I: TryInto<isize>,
{
    let axis1 = axis1.try_into().map_err(|_| rstsr_error!(TryFromIntError))?;
    let axis2 = axis2.try_into().map_err(|_| rstsr_error!(TryFromIntError))?;
    let (storage, layout) = tensor.into_raw_parts();
    let layout = layout.swapaxes(axis1, axis2)?;
    unsafe { Ok(TensorBase::new_unchecked(storage, layout)) }
}

/// Interchange two axes of an array.
///
/// # See also
///
/// - [numpy `swapaxes`](https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html)
pub fn swapaxes<I, R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    axis1: I,
    axis2: I,
) -> TensorView<'_, T, B, D>
where
    D: DimAPI,
    I: TryInto<isize>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_swapaxes_f(tensor.view(), axis1, axis2).unwrap()
}

pub fn swapaxes_f<I, R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    axis1: I,
    axis2: I,
) -> Result<TensorView<'_, T, B, D>>
where
    D: DimAPI,
    I: TryInto<isize>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_swapaxes_f(tensor.view(), axis1, axis2)
}

pub fn into_swapaxes<I, S, D>(tensor: TensorBase<S, D>, axis1: I, axis2: I) -> TensorBase<S, D>
where
    D: DimAPI,
    I: TryInto<isize>,
{
    into_swapaxes_f(tensor, axis1, axis2).unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Interchange two axes of an array.
    ///
    /// # See also
    ///
    /// [`swapaxes`]
    pub fn swapaxes<I>(&self, axis1: I, axis2: I) -> TensorView<'_, T, B, D>
    where
        I: TryInto<isize>,
    {
        swapaxes(self, axis1, axis2)
    }

    pub fn swapaxes_f<I>(&self, axis1: I, axis2: I) -> Result<TensorView<'_, T, B, D>>
    where
        I: TryInto<isize>,
    {
        swapaxes_f(self, axis1, axis2)
    }

    /// Interchange two axes of an array.
    ///
    /// # See also
    ///
    /// [`swapaxes`]
    pub fn into_swapaxes<I>(self, axis1: I, axis2: I) -> TensorAny<R, T, B, D>
    where
        I: TryInto<isize>,
    {
        into_swapaxes(self, axis1, axis2)
    }

    pub fn into_swapaxes_f<I>(self, axis1: I, axis2: I) -> Result<TensorAny<R, T, B, D>>
    where
        I: TryInto<isize>,
    {
        into_swapaxes_f(self, axis1, axis2)
    }
}

/* #endregion */

/* #region squeeze */

pub fn into_squeeze_f<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> Result<TensorBase<S, IxD>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
{
    // convert axis to positive indexes and (reversed) sort
    let ndim: isize = TryInto::<isize>::try_into(tensor.ndim())?;
    let (storage, layout) = tensor.into_raw_parts();
    let mut layout = layout.into_dim::<IxD>()?;
    let mut axes: Vec<isize> =
        axes.try_into()?.as_ref().iter().map(|&v| if v >= 0 { v } else { v + ndim }).collect::<_>();
    axes.sort_by(|a, b| b.cmp(a));
    if axes.first().is_some_and(|&v| v < 0) {
        return Err(rstsr_error!(InvalidValue, "Some negative index is too small."));
    }
    // check no two axis are the same
    for i in 0..axes.len() - 1 {
        rstsr_assert!(axes[i] != axes[i + 1], InvalidValue, "Same axes is not allowed here.")?;
    }
    // perform squeeze
    for &axis in axes.iter() {
        layout = layout.dim_eliminate(axis)?;
    }
    unsafe { Ok(TensorBase::new_unchecked(storage, layout)) }
}

/// Removes singleton dimensions (axes) from `x`.
///
/// # See also
///
/// [Python array API standard: `squeeze`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.squeeze.html)
pub fn squeeze<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axes: I) -> TensorView<'_, T, B, IxD>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_squeeze_f(tensor.view(), axes).unwrap()
}

pub fn squeeze_f<I, R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    axes: I,
) -> Result<TensorView<'_, T, B, IxD>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_squeeze_f(tensor.view(), axes)
}

pub fn into_squeeze<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> TensorBase<S, IxD>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
{
    into_squeeze_f(tensor, axes).unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Removes singleton dimensions (axes) from `x`.
    ///
    /// # See also
    ///
    /// [`squeeze`]
    pub fn squeeze<I>(&self, axis: I) -> TensorView<'_, T, B, IxD>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        squeeze(self, axis)
    }

    pub fn squeeze_f<I>(&self, axis: I) -> Result<TensorView<'_, T, B, IxD>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        squeeze_f(self, axis)
    }

    /// Removes singleton dimensions (axes) from `x`.
    ///
    /// # See also
    ///
    /// [`squeeze`]
    pub fn into_squeeze<I>(self, axis: I) -> TensorAny<R, T, B, IxD>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        into_squeeze(self, axis)
    }

    pub fn into_squeeze_f<I>(self, axis: I) -> Result<TensorAny<R, T, B, IxD>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        into_squeeze_f(self, axis)
    }
}

/* #endregion */

/* #region into_dim */

pub fn into_dim_f<S, D, D2>(tensor: TensorBase<S, D>) -> Result<TensorBase<S, D2>>
where
    D: DimAPI + DimIntoAPI<D2>,
    D2: DimAPI,
{
    let (storage, layout) = tensor.into_raw_parts();
    let layout = layout.into_dim::<D2>()?;
    unsafe { Ok(TensorBase::new_unchecked(storage, layout)) }
}

/// Convert layout to the other dimension.
///
/// This is mostly used when converting static dimension to dynamic
/// dimension or vice versa.
pub fn to_dim<R, T, B, D, D2>(tensor: &TensorAny<R, T, B, D>) -> TensorView<'_, T, B, D2>
where
    D: DimAPI,
    D2: DimAPI,
    D: DimIntoAPI<D2>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_dim_f(tensor.view()).unwrap()
}

pub fn to_dim_f<R, T, B, D, D2>(tensor: &TensorAny<R, T, B, D>) -> Result<TensorView<'_, T, B, D2>>
where
    D: DimAPI,
    D2: DimAPI,
    D: DimIntoAPI<D2>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_dim_f(tensor.view())
}

pub fn into_dim<S, D, D2>(tensor: TensorBase<S, D>) -> TensorBase<S, D2>
where
    D: DimAPI,
    D2: DimAPI,
    D: DimIntoAPI<D2>,
{
    into_dim_f(tensor).unwrap()
}

pub fn to_dyn<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> TensorView<'_, T, B, IxD>
where
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_dim_f(tensor.view()).unwrap()
}

pub fn into_dyn<S, D>(tensor: TensorBase<S, D>) -> TensorBase<S, IxD>
where
    D: DimAPI,
{
    into_dim_f(tensor).unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    /// Convert layout to the other dimension.
    ///
    /// This is mostly used when converting static dimension to dynamic
    /// dimension or vice versa.
    ///
    /// # See also
    ///
    /// [`into_dim`]
    pub fn to_dim<D2>(&self) -> TensorView<'_, T, B, D2>
    where
        D2: DimAPI,
        D: DimIntoAPI<D2>,
    {
        to_dim(self)
    }

    pub fn to_dim_f<D2>(&self) -> Result<TensorView<'_, T, B, D2>>
    where
        D2: DimAPI,
        D: DimIntoAPI<D2>,
    {
        to_dim_f(self)
    }

    /// Convert layout to another dimension.
    ///
    /// # See also
    ///
    /// [`into_dim`]
    pub fn into_dim<D2>(self) -> TensorAny<R, T, B, D2>
    where
        D2: DimAPI,
        D: DimIntoAPI<D2>,
    {
        into_dim(self)
    }

    pub fn into_dim_f<D2>(self) -> Result<TensorAny<R, T, B, D2>>
    where
        D2: DimAPI,
        D: DimIntoAPI<D2>,
    {
        into_dim_f(self)
    }

    /// Convert layout to dynamic dimension.
    pub fn to_dyn(&self) -> TensorView<'_, T, B, IxD> {
        to_dyn(self)
    }

    /// Convert layout to dynamic dimension.
    pub fn into_dyn(self) -> TensorAny<R, T, B, IxD> {
        into_dyn(self)
    }
}

/* #endregion */

/* #region reshape_assume_contig */

pub fn into_shape_assume_contig_f<S, D, D2>(
    tensor: TensorBase<S, D>,
    shape: D2,
) -> Result<TensorBase<S, D2>>
where
    D: DimAPI,
    D2: DimAPI,
{
    let (storage, layout) = tensor.into_raw_parts();

    rstsr_assert_eq!(
        layout.size(),
        shape.shape_size(),
        InvalidLayout,
        "Number of elements not same."
    )?;

    let new_layout = if TensorOrder::default() == TensorOrder::C && layout.c_contig() {
        shape.new_c_contig(Some(layout.offset))
    } else if TensorOrder::default() == TensorOrder::F && layout.f_contig() {
        shape.new_f_contig(Some(layout.offset))
    } else {
        rstsr_raise!(InvalidLayout, "This array is not contiguous by {:?}", TensorOrder::default())?
    };
    unsafe { Ok(TensorBase::new_unchecked(storage, new_layout)) }
}

/// Assuming contiguous array, reshapes an array without changing its data.
///
/// This function may return c-contiguous or f-contiguous array depending on
/// crate feature `f_prefer`.
///
/// # See also
///
/// [Python array API standard: `reshape`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.reshape.html)
pub fn to_shape_assume_contig<R, T, B, D, D2>(
    tensor: &TensorAny<R, T, B, D>,
    shape: D2,
) -> TensorView<'_, T, B, D2>
where
    D: DimAPI,
    D2: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_shape_assume_contig_f(tensor.view(), shape).unwrap()
}

pub fn to_shape_assume_contig_f<R, T, B, D, D2>(
    tensor: &TensorAny<R, T, B, D>,
    shape: D2,
) -> Result<TensorView<'_, T, B, D2>>
where
    D: DimAPI,
    D2: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_shape_assume_contig_f(tensor.view(), shape)
}

pub fn into_shape_assume_contig<S, D, D2>(tensor: TensorBase<S, D>, shape: D2) -> TensorBase<S, D2>
where
    D: DimAPI,
    D2: DimAPI,
{
    into_shape_assume_contig_f(tensor, shape).unwrap()
}

pub use to_shape_assume_contig as reshape_assume_contig;
pub use to_shape_assume_contig_f as reshape_assume_contig_f;

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Assuming contiguous array, reshapes an array without changing its data.
    ///
    /// # See also
    ///
    /// [`reshape_assume_contig`]
    pub fn reshape_assume_contig<D2>(&self, shape: D2) -> TensorView<'_, T, B, D2>
    where
        D2: DimAPI,
    {
        into_shape_assume_contig(self.view(), shape)
    }

    pub fn reshape_assume_contig_f<D2>(&self, shape: D2) -> Result<TensorView<'_, T, B, D2>>
    where
        D2: DimAPI,
    {
        into_shape_assume_contig_f(self.view(), shape)
    }

    pub fn to_shape_assume_contig<D2>(&self, shape: D2) -> TensorView<'_, T, B, D2>
    where
        D2: DimAPI,
    {
        into_shape_assume_contig(self.view(), shape)
    }

    pub fn to_shape_assume_contig_f<D2>(&self, shape: D2) -> Result<TensorView<'_, T, B, D2>>
    where
        D2: DimAPI,
    {
        into_shape_assume_contig_f(self.view(), shape)
    }

    pub fn into_shape_assume_contig<D2>(self, shape: D2) -> TensorAny<R, T, B, D2>
    where
        D2: DimAPI,
    {
        into_shape_assume_contig(self, shape)
    }

    pub fn into_shape_assume_contig_f<D2>(self, shape: D2) -> Result<TensorAny<R, T, B, D2>>
    where
        D2: DimAPI,
    {
        into_shape_assume_contig_f(self, shape)
    }
}

/* #endregion */

/* #region reshape */

pub fn change_shape_f<'a, I, R, T, B, D>(
    tensor: TensorAny<R, T, B, D>,
    shape: I,
) -> Result<TensorCow<'a, T, B, IxD>>
where
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
    R: DataAPI<Data = B::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    // own shape, this is cheap operation
    let shape_new = reshape_substitute_negatives(shape.try_into()?.as_ref(), tensor.size())?;
    if let Some(layout_new) = layout_reshapeable(&tensor.layout().to_dim()?, &shape_new)? {
        // shape does not need to be changed
        let (storage, _) = tensor.into_raw_parts();
        let layout = layout_new.into_dim::<IxD>()?;
        return unsafe { Ok(TensorBase::new_unchecked(storage, layout).into_cow()) };
    } else {
        // clone underlying data by assign_arbitary
        let (storage, layout) = tensor.into_raw_parts();
        let device = storage.device();
        let layout_new = shape_new.new_contig(None);
        let mut storage_new = unsafe { device.empty_impl(layout_new.size())? };
        device.assign_arbitary(storage_new.raw_mut(), &layout_new, storage.raw(), &layout)?;
        return unsafe { Ok(TensorBase::new_unchecked(storage_new, layout_new).into_cow()) };
    }
}

pub fn change_shape<'a, I, R, T, B, D>(
    tensor: TensorAny<R, T, B, D>,
    shape: I,
) -> TensorCow<'a, T, B, IxD>
where
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
    R: DataAPI<Data = B::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    change_shape_f(tensor, shape).unwrap()
}

pub fn into_shape_f<'a, I, R, T, B, D>(
    tensor: TensorAny<R, T, B, D>,
    shape: I,
) -> Result<Tensor<T, B, IxD>>
where
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
    R: DataAPI<Data = B::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, IxD, D>
        + OpAssignAPI<T, IxD>,
    B::Raw: 'a,
{
    change_shape_f(tensor, shape).map(|v| v.into_owned())
}

pub fn into_shape<'a, I, R, T, B, D>(tensor: TensorAny<R, T, B, D>, shape: I) -> Tensor<T, B, IxD>
where
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
    R: DataAPI<Data = B::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, IxD, D>
        + OpAssignAPI<T, IxD>,
    B::Raw: 'a,
{
    into_shape_f(tensor, shape).unwrap()
}

pub fn to_shape_f<'a, I, R, T, B, D>(
    tensor: &'a TensorAny<R, T, B, D>,
    shape: I,
) -> Result<TensorCow<'a, T, B, IxD>>
where
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
    R: DataAPI<Data = B::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    change_shape_f(tensor.view(), shape)
}

pub fn to_shape<'a, I, R, T, B, D>(
    tensor: &'a TensorAny<R, T, B, D>,
    shape: I,
) -> TensorCow<'a, T, B, IxD>
where
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
    R: DataAPI<Data = B::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    to_shape_f(tensor, shape).unwrap()
}

pub fn reshape_f<'a, I, R, T, B, D>(
    tensor: &'a TensorAny<R, T, B, D>,
    shape: I,
) -> Result<TensorCow<'a, T, B, IxD>>
where
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
    R: DataAPI<Data = B::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    to_shape_f(tensor, shape)
}

pub fn reshape<'a, I, R, T, B, D>(
    tensor: &'a TensorAny<R, T, B, D>,
    shape: I,
) -> TensorCow<'a, T, B, IxD>
where
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
    R: DataAPI<Data = B::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    to_shape(tensor, shape)
}

impl<'a, R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, IxD, D>
        + OpAssignAPI<T, IxD>,
    T: Clone,
{
    pub fn change_shape_f<I>(self, shape: I) -> Result<TensorCow<'a, T, B, IxD>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        change_shape_f(self, shape)
    }

    pub fn change_shape<I>(self, shape: I) -> TensorCow<'a, T, B, IxD>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        change_shape(self, shape)
    }

    pub fn into_shape_f<I>(self, shape: I) -> Result<Tensor<T, B, IxD>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
        B::Raw: 'a,
    {
        into_shape_f(self, shape)
    }

    pub fn into_shape<I>(self, shape: I) -> Tensor<T, B, IxD>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
        B::Raw: 'a,
    {
        into_shape(self, shape)
    }

    pub fn to_shape_f<I>(&'a self, shape: I) -> Result<TensorCow<'a, T, B, IxD>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        self.view().change_shape_f(shape)
    }

    pub fn to_shape<I>(&'a self, shape: I) -> TensorCow<'a, T, B, IxD>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        self.view().change_shape(shape)
    }

    pub fn reshape_f<I>(&'a self, shape: I) -> Result<TensorCow<'a, T, B, IxD>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        self.view().change_shape_f(shape)
    }

    pub fn reshape<I>(&'a self, shape: I) -> TensorCow<'a, T, B, IxD>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        self.view().change_shape(shape)
    }
}

/* #endregion */

/* #region to_layout */

pub fn change_layout_f<'a, R, T, B, D, D2>(
    tensor: TensorAny<R, T, B, D>,
    layout: Layout<D2>,
) -> Result<TensorCow<'a, T, B, D2>>
where
    R: DataAPI<Data = B::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    D2: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D2, D>,
{
    let shape = layout.shape();
    rstsr_assert_eq!(tensor.size(), shape.shape_size(), InvalidLayout)?;
    let same_layout = tensor.layout().to_dim::<IxD>()? == layout.to_dim::<IxD>()?;
    let contig_c = tensor.layout().c_contig()
        && layout.c_contig()
        && tensor.layout().offset() == layout.offset();
    let contig_f = tensor.layout().f_contig()
        && layout.f_contig()
        && tensor.layout().offset() == layout.offset();
    let contig = match TensorOrder::default() {
        TensorOrder::C => contig_c,
        TensorOrder::F => contig_f,
    };
    if same_layout || contig {
        // no data cloned
        let (storage, _) = tensor.into_raw_parts();
        let tensor = unsafe { TensorBase::new_unchecked(storage, layout) };
        return Ok(tensor.into_cow());
    } else {
        // layout changed, or not c and f contiguous with same layout
        // clone data by assign
        let (storage_old, layout_old) = tensor.into_raw_parts();
        let device = storage_old.device();
        let (_, idx_max) = layout.bounds_index()?;
        let mut storage_new = unsafe { device.empty_impl(idx_max)? };
        device.assign_arbitary(storage_new.raw_mut(), &layout, storage_old.raw(), &layout_old)?;
        let tensor = unsafe { TensorBase::new_unchecked(storage_new, layout) };
        return Ok(tensor.into_cow());
    }
}

/// Convert tensor to the other layout.
pub fn to_layout<R, T, D, B, D2>(
    tensor: &TensorAny<R, T, B, D>,
    layout: Layout<D2>,
) -> TensorCow<'_, T, B, D2>
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    D2: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D2, D>,
{
    change_layout_f(tensor.view(), layout).unwrap()
}

pub fn to_layout_f<R, T, D, B, D2>(
    tensor: &TensorAny<R, T, B, D>,
    layout: Layout<D2>,
) -> Result<TensorCow<'_, T, B, D2>>
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    D2: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D2, D>,
{
    change_layout_f(tensor.view(), layout)
}

pub fn into_layout_f<'a, R, T, B, D, D2>(
    tensor: TensorAny<R, T, B, D>,
    layout: Layout<D2>,
) -> Result<Tensor<T, B, D2>>
where
    R: DataAPI<Data = B::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    D2: DimAPI,
    T: Clone,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D2, D> + OpAssignAPI<T, D2>,
    B::Raw: 'a,
{
    change_layout_f(tensor, layout).map(|v| v.into_owned())
}

pub fn into_layout<'a, R, T, B, D, D2>(
    tensor: TensorAny<R, T, B, D>,
    layout: Layout<D2>,
) -> Tensor<T, B, D2>
where
    R: DataAPI<Data = B::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    D2: DimAPI,
    T: Clone,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D2, D> + OpAssignAPI<T, D2>,
    B::Raw: 'a,
{
    into_layout_f(tensor, layout).unwrap()
}

pub fn change_layout<'a, R, T, B, D, D2>(
    tensor: TensorAny<R, T, B, D>,
    layout: Layout<D2>,
) -> TensorCow<'a, T, B, D2>
where
    R: DataAPI<Data = B::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    D2: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D2, D>,
{
    change_layout_f(tensor, layout).unwrap()
}

impl<'a, R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    /// Convert tensor to the other layout.
    ///
    /// # See also
    ///
    /// [`to_layout`]
    pub fn to_layout<D2>(&self, layout: Layout<D2>) -> TensorCow<'_, T, B, D2>
    where
        D2: DimAPI,
        B: OpAssignArbitaryAPI<T, D2, D>,
    {
        to_layout(self, layout)
    }

    pub fn to_layout_f<D2>(&self, layout: Layout<D2>) -> Result<TensorCow<'_, T, B, D2>>
    where
        D2: DimAPI,
        B: OpAssignArbitaryAPI<T, D2, D>,
    {
        to_layout_f(self, layout)
    }

    pub fn into_layout_f<D2>(self, layout: Layout<D2>) -> Result<Tensor<T, B, D2>>
    where
        D2: DimAPI,
        B: OpAssignArbitaryAPI<T, D2, D> + OpAssignAPI<T, D2>,
        B::Raw: 'a,
    {
        into_layout_f(self, layout)
    }

    pub fn into_layout<D2>(self, layout: Layout<D2>) -> Tensor<T, B, D2>
    where
        D2: DimAPI,
        B: OpAssignArbitaryAPI<T, D2, D> + OpAssignAPI<T, D2>,
        B::Raw: 'a,
    {
        into_layout(self, layout)
    }

    pub fn change_layout_f<D2>(self, layout: Layout<D2>) -> Result<TensorCow<'a, T, B, D2>>
    where
        D2: DimAPI,
        B: OpAssignArbitaryAPI<T, D2, D>,
    {
        change_layout_f(self, layout)
    }

    pub fn change_layout<D2>(self, layout: Layout<D2>) -> TensorCow<'a, T, B, D2>
    where
        D2: DimAPI,
        B: OpAssignArbitaryAPI<T, D2, D>,
    {
        change_layout(self, layout)
    }
}

/* #endregion */

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_to_shape_assume_contig() {
        let a = linspace((2.5, 3.2, 16));
        let b = a.to_shape_assume_contig_f([4, 4]).unwrap();
        println!("{:.3?}", b);
    }

    #[test]
    fn test_expand_dims() {
        let a: Tensor<f64, _> = zeros([4, 9, 8]);
        let b = a.expand_dims(2);
        assert_eq!(b.shape(), &[4, 9, 1, 8]);
        let b = a.expand_dims([1, 3]);
        assert_eq!(b.shape(), &[4, 1, 9, 8, 1]);
        let b = a.expand_dims([1, -1]);
        assert_eq!(b.shape(), &[4, 1, 9, 8, 1]);
        let b = a.expand_dims([-1, -4, 1, 0]);
        assert_eq!(b.shape(), &[1, 1, 4, 1, 9, 8, 1]);
    }

    #[test]
    fn test_squeeze() {
        let a: Tensor<f64, _> = zeros([4, 1, 9, 1, 8, 1]);
        let b = a.squeeze(3);
        assert_eq!(b.shape(), &[4, 1, 9, 8, 1]);
        let b = a.squeeze([1, 3]);
        assert_eq!(b.shape(), &[4, 9, 8, 1]);
        let b = a.squeeze([1, -1]);
        assert_eq!(b.shape(), &[4, 9, 1, 8]);
        let b = a.squeeze_f(-7);
        assert!(b.is_err());
    }

    #[test]
    fn test_flip() {
        let a = arange(24.0).into_shape([2, 3, 4]).into_owned();
        println!("{:?}", a);

        let b = a.flip(1);
        println!("{:?}", b);
        assert_eq!(b.shape(), &[2, 3, 4]);
        let c = a.flip([0, -1]);
        println!("{:?}", c);
        assert_eq!(c.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_swapaxes() {
        let a = arange(24.0).into_shape([2, 3, 4]).into_owned();
        println!("{:?}", a);

        let b = a.swapaxes(0, 1);
        println!("{:?}", b);
        assert_eq!(b.shape(), &[3, 2, 4]);
    }

    #[test]
    fn test_to_shape() {
        let a = linspace((0.0, 15.0, 16));
        let mut a = a.to_shape([4, 4]);
        a.layout = Layout::new(vec![2, 2], vec![2, 4], 0).unwrap();
        println!("{:?}", a);
        let b = a.to_shape([2, 2]);
        println!("{:?}", b);

        let c = a.to_shape([2, -1]);
        println!("{:?}", c);
        assert_eq!(c.shape(), &[2, 2]);

        let d = a.to_shape_f([3, -1]);
        assert!(d.is_err());
    }

    #[test]
    fn test_broadcast_to() {
        let a = linspace((0.0, 15.0, 16));
        let a = a.into_shape_assume_contig_f([4, 1, 4]).unwrap();
        let a = a.to_broadcast_f([6, 4, 3, 4]).unwrap();
        assert_eq!(a.layout(), unsafe { &Layout::new_unchecked([6, 4, 3, 4], [0, 4, 0, 1], 0) });
        println!("{:?}", a);
    }

    #[test]
    fn test_to_layout() {
        let a = linspace((0.0, 15.0, 16));
        let a = a.change_shape([4, 4]);
        let a = a.into_layout(Layout::new([2, 8], [12, 120], 8).unwrap());
        println!("{:?}", a);
    }
}
