use crate::prelude_dev::*;

/* #region flip */

pub fn into_flip_f<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> Result<TensorBase<S, D>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
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
/// - If some index in `axis` is greater than the number of axes in the original tensor.
///
/// # See also
///
/// [Python array API standard: `flip`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.flip.html)
pub fn flip<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axes: I) -> TensorView<'_, T, B, D>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_flip_f(tensor.view(), axes).unwrap()
}

pub fn flip_f<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axes: I) -> Result<TensorView<'_, T, B, D>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_flip_f(tensor.view(), axes)
}

pub fn into_flip<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> TensorBase<S, D>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
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
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        flip(self, axis)
    }

    pub fn flip_f<I>(&self, axis: I) -> Result<TensorView<'_, T, B, D>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
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
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        into_flip(self, axis)
    }

    pub fn into_flip_f<I>(self, axis: I) -> Result<TensorAny<R, T, B, D>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        into_flip_f(self, axis)
    }
}

/* #endregion */
