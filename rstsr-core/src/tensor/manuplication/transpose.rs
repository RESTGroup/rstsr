use crate::prelude_dev::*;

/* #region permute_dims */

pub fn into_transpose_f<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> Result<TensorBase<S, D>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
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
/// - [Python array API standard: `permute_dims`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.permute_dims.html)
pub fn transpose<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axes: I) -> TensorView<'_, T, B, D>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_transpose_f(tensor.view(), axes).unwrap()
}

pub fn transpose_f<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axes: I) -> Result<TensorView<'_, T, B, D>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_transpose_f(tensor.view(), axes)
}

pub fn into_transpose<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> TensorBase<S, D>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
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
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        transpose(self, axes)
    }

    pub fn transpose_f<I>(&self, axes: I) -> Result<TensorView<'_, T, B, D>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
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
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        into_transpose(self, axes)
    }

    pub fn into_transpose_f<I>(self, axes: I) -> Result<TensorAny<R, T, B, D>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
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
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        transpose(self, axes)
    }

    pub fn permute_dims_f<I>(&self, axes: I) -> Result<TensorView<'_, T, B, D>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
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
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        into_transpose(self, axes)
    }

    pub fn into_permute_dims_f<I>(self, axes: I) -> Result<TensorAny<R, T, B, D>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
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

pub fn into_swapaxes_f<I, S, D>(tensor: TensorBase<S, D>, axis1: I, axis2: I) -> Result<TensorBase<S, D>>
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
pub fn swapaxes<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axis1: I, axis2: I) -> TensorView<'_, T, B, D>
where
    D: DimAPI,
    I: TryInto<isize>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_swapaxes_f(tensor.view(), axis1, axis2).unwrap()
}

pub fn swapaxes_f<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axis1: I, axis2: I) -> Result<TensorView<'_, T, B, D>>
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
