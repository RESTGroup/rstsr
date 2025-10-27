use crate::prelude_dev::*;

/* #region broadcast_arrays */

/// Broadcasts any number of arrays against each other.
///
/// # See also
///
/// [Python Array API standard: `broadcast_arrays`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.broadcast_arrays.html)
pub fn broadcast_arrays<R, T, B>(tensors: Vec<TensorAny<R, T, B, IxD>>) -> Vec<TensorAny<R, T, B, IxD>>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    broadcast_arrays_f(tensors).unwrap()
}

pub fn broadcast_arrays_f<R, T, B>(tensors: Vec<TensorAny<R, T, B, IxD>>) -> Result<Vec<TensorAny<R, T, B, IxD>>>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    // fast return if there is only zero/one tensor
    if tensors.len() <= 1 {
        return Ok(tensors);
    }
    let device_b = tensors[0].device().clone();
    let default_order = device_b.default_order();
    let mut shape_b = tensors[0].shape().clone();
    for tensor in tensors.iter().skip(1) {
        rstsr_assert!(device_b.same_device(tensor.device()), DeviceMismatch)?;
        let shape = tensor.shape();
        let (shape, _, _) = broadcast_shape(shape, &shape_b, default_order)?;
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

pub fn into_broadcast_f<R, T, B, D, D2>(tensor: TensorAny<R, T, B, D>, shape: D2) -> Result<TensorAny<R, T, B, D2>>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI + DimMaxAPI<D2, Max = D2>,
    D2: DimAPI,
{
    let shape1 = tensor.shape();
    let shape2 = &shape;
    let default_order = tensor.device().default_order();
    let (shape, tp1, _) = broadcast_shape(shape1, shape2, default_order)?;
    let (storage, layout) = tensor.into_raw_parts();
    let layout = update_layout_by_shape(&layout, &shape, &tp1, default_order)?;
    unsafe { Ok(TensorBase::new_unchecked(storage, layout)) }
}

/// Broadcasts an array to a specified shape.
///
/// # See also
///
/// [Python Array API standard: `broadcast_to`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.broadcast_to.html)
pub fn to_broadcast<R, T, B, D, D2>(tensor: &TensorAny<R, T, B, D>, shape: D2) -> TensorView<'_, T, B, D2>
where
    D: DimAPI + DimMaxAPI<D2, Max = D2>,
    D2: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_broadcast_f(tensor.view(), shape).unwrap()
}

pub fn to_broadcast_f<R, T, B, D, D2>(tensor: &TensorAny<R, T, B, D>, shape: D2) -> Result<TensorView<'_, T, B, D2>>
where
    D: DimAPI + DimMaxAPI<D2, Max = D2>,
    D2: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_broadcast_f(tensor.view(), shape)
}

pub fn into_broadcast<R, T, B, D, D2>(tensor: TensorAny<R, T, B, D>, shape: D2) -> TensorAny<R, T, B, D2>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
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
