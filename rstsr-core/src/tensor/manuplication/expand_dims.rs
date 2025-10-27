use crate::prelude_dev::*;

/* #region expand_dims */

pub fn into_expand_dims_f<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> Result<TensorBase<S, IxD>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
{
    // convert axis to negative indexes and sort
    let ndim: isize = TryInto::<isize>::try_into(tensor.ndim())?;
    let (storage, layout) = tensor.into_raw_parts();
    let mut layout = layout.into_dim::<IxD>()?;
    let mut axes: Vec<isize> =
        axes.try_into()?.as_ref().iter().map(|&v| if v >= 0 { v - ndim - 1 } else { v }).collect::<Vec<isize>>();
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
/// [Python Array API standard: `expand_dims`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.expand_dims.html)
pub fn expand_dims<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axes: I) -> TensorView<'_, T, B, IxD>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_expand_dims_f(tensor.view(), axes).unwrap()
}

pub fn expand_dims_f<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axes: I) -> Result<TensorView<'_, T, B, IxD>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_expand_dims_f(tensor.view(), axes)
}

pub fn into_expand_dims<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> TensorBase<S, IxD>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
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
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        into_expand_dims(self.view(), axes)
    }

    pub fn expand_dims_f<I>(&self, axes: I) -> Result<TensorView<'_, T, B, IxD>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
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
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        into_expand_dims(self, axes)
    }

    pub fn into_expand_dims_f<I>(self, axes: I) -> Result<TensorAny<R, T, B, IxD>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        into_expand_dims_f(self, axes)
    }
}

/* #endregion */
