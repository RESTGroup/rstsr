use crate::prelude_dev::*;

/* #region squeeze */

pub fn into_squeeze_f<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> Result<TensorBase<S, IxD>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
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
/// [Python array API standard: `squeeze`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.squeeze.html)
pub fn squeeze<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axes: I) -> TensorView<'_, T, B, IxD>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_squeeze_f(tensor.view(), axes).rstsr_unwrap()
}

pub fn squeeze_f<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axes: I) -> Result<TensorView<'_, T, B, IxD>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_squeeze_f(tensor.view(), axes)
}

pub fn into_squeeze<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> TensorBase<S, IxD>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
{
    into_squeeze_f(tensor, axes).rstsr_unwrap()
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
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        squeeze(self, axis)
    }

    pub fn squeeze_f<I>(&self, axis: I) -> Result<TensorView<'_, T, B, IxD>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
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
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        into_squeeze(self, axis)
    }

    pub fn into_squeeze_f<I>(self, axis: I) -> Result<TensorAny<R, T, B, IxD>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        into_squeeze_f(self, axis)
    }
}

/* #endregion */
