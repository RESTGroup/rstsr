use crate::prelude_dev::*;

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
