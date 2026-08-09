use crate::prelude_dev::*;

/// Moves array axes (dimensions) to new positions, while leaving other axes in their original
/// positions.
///
/// See also [`moveaxis`].
pub fn into_moveaxis_f<IS, ID, S, D>(tensor: TensorBase<S, D>, source: IS, destination: ID) -> Result<TensorBase<S, D>>
where
    D: DimAPI,
    IS: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    ID: TryInto<AxesIndex<isize>, Error: Into<Error>>,
{
    let source = source.try_into().map_err(Into::into)?;
    let destination = destination.try_into().map_err(Into::into)?;

    let ndim = tensor.ndim();

    // Normalize axes
    let source = normalize_axes_index(source, tensor.ndim(), false, false)?;
    let destination = normalize_axes_index(destination, tensor.ndim(), false, false)?;

    // Check that source and destination have the same length
    rstsr_assert_eq!(
        source.len(),
        destination.len(),
        InvalidValue,
        "`source` and `destination` arguments must have the same number of elements"
    )?;

    // Build the permutation order
    // Start with all axes that are not in source
    let mut order: Vec<isize> = (0..ndim as isize).filter(|&i| !source.contains(&i)).collect();

    // Insert source axes at their destination positions
    // Sort pairs by destination to insert in correct order
    let mut pairs: Vec<(isize, isize)> = destination.iter().zip(source.iter()).map(|(&d, &s)| (d, s)).collect();
    pairs.sort_by_key(|&(d, _)| d);

    for (dest, src) in pairs {
        order.insert(dest as usize, src);
    }

    // Apply the transpose
    let (storage, layout) = tensor.into_raw_parts();
    let layout = layout.transpose(&order)?;
    unsafe { Ok(TensorBase::new_unchecked(storage, layout)) }
}

/// Moves array axes (dimensions) to new positions, while leaving other axes in their original
/// positions.
///
/// Returns an array with axes moved to new positions. Other axes remain in their
/// original order. This is a view operation; no data is copied.
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny)
///
///   - The input tensor whose axes are to be moved.
///
/// - `source`: TryInto [`AxesIndex<isize>`]
///
///   - Original positions of the axes to move. These must be unique.
///   - Can be a single axis index or a sequence of axis indices.
///   - Negative values are supported and indicate counting dimensions from the back.
///
/// - `destination`: TryInto [`AxesIndex<isize>`]
///
///   - Destination positions for each of the original axes. These must also be unique.
///   - Can be a single axis index or a sequence of axis indices.
///   - Negative values are supported and indicate counting dimensions from the back.
///   - Must have the same number of elements as `source`.
///
/// # Returns
///
/// - [`TensorView<'_, T, B, D>`](TensorView)
///
///   - A view of the input tensor with moved axes.
///   - No data is copied; only the shape and strides are modified.
///
/// # Examples
///
/// Move a single axis to a new position:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let x: Tensor<f64, _> = rt::zeros(([3, 4, 5], &device));
/// let result = x.moveaxis(0, -1);
/// println!("{:?}", result.shape());
/// // [4, 5, 3]
/// ```
///
/// Move multiple axes to new positions:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let x: Tensor<f64, _> = rt::zeros(([3, 4, 5], &device));
/// let result = x.moveaxis([0, 1], [-1, -2]);
/// println!("{:?}", result.shape());
/// // [5, 4, 3]
/// ```
///
/// Using negative indices:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let x: Tensor<f64, _> = rt::zeros(([3, 4, 5], &device));
/// let result = x.moveaxis(-1, 0);
/// println!("{:?}", result.shape());
/// // [5, 3, 4]
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `moveaxis(x, source, destination, /)` ([`moveaxis`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.moveaxis.html))
/// - NumPy: `moveaxis(a, source, destination)` ([`numpy.moveaxis`](https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html))
/// - RSTSR: `tensor.moveaxis(source, destination)` or `rt::moveaxis(&tensor, source, destination)`
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`transpose`] - General axis permutation
/// - [`swapaxes`] - Swap two specific axes
/// - [`reverse_axes`] - Reverse all axes order
///
/// ## Variants of this function
///
/// - [`moveaxis`] / [`moveaxis_f`]: Returning a view.
/// - [`into_moveaxis`] / [`into_moveaxis_f`]: Consuming version.
///
/// - Associated methods on `TensorAny`:
///
///   - [`TensorAny::moveaxis`] / [`TensorAny::moveaxis_f`]
///   - [`TensorAny::into_moveaxis`] / [`TensorAny::into_moveaxis_f`]
pub fn moveaxis<IS, ID, R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    source: IS,
    destination: ID,
) -> TensorView<'_, T, B, D>
where
    D: DimAPI,
    IS: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    ID: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_moveaxis_f(tensor.view(), source, destination).rstsr_unwrap()
}

/// Moves array axes (dimensions) to new positions, while leaving other axes in their original
/// positions.
///
/// See also [`moveaxis`].
pub fn moveaxis_f<IS, ID, R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    source: IS,
    destination: ID,
) -> Result<TensorView<'_, T, B, D>>
where
    D: DimAPI,
    IS: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    ID: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_moveaxis_f(tensor.view(), source, destination)
}

/// Moves array axes (dimensions) to new positions, while leaving other axes in their original
/// positions.
///
/// See also [`moveaxis`].
pub fn into_moveaxis<IS, ID, S, D>(tensor: TensorBase<S, D>, source: IS, destination: ID) -> TensorBase<S, D>
where
    D: DimAPI,
    IS: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    ID: TryInto<AxesIndex<isize>, Error: Into<Error>>,
{
    into_moveaxis_f(tensor, source, destination).rstsr_unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Moves array axes (dimensions) to new positions, while leaving other axes in their original
    /// positions.
    ///
    /// See also [`moveaxis`].
    pub fn moveaxis<IS, ID>(&self, source: IS, destination: ID) -> TensorView<'_, T, B, D>
    where
        IS: TryInto<AxesIndex<isize>, Error: Into<Error>>,
        ID: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        moveaxis(self, source, destination)
    }

    /// Moves array axes (dimensions) to new positions, while leaving other axes in their original
    /// positions.
    ///
    /// See also [`moveaxis`].
    pub fn moveaxis_f<IS, ID>(&self, source: IS, destination: ID) -> Result<TensorView<'_, T, B, D>>
    where
        IS: TryInto<AxesIndex<isize>, Error: Into<Error>>,
        ID: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        moveaxis_f(self, source, destination)
    }

    /// Moves array axes (dimensions) to new positions, while leaving other axes in their original
    /// positions.
    ///
    /// See also [`moveaxis`].
    pub fn into_moveaxis<IS, ID>(self, source: IS, destination: ID) -> TensorAny<R, T, B, D>
    where
        IS: TryInto<AxesIndex<isize>, Error: Into<Error>>,
        ID: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        into_moveaxis(self, source, destination)
    }

    /// Moves array axes (dimensions) to new positions, while leaving other axes in their original
    /// positions.
    ///
    /// See also [`moveaxis`].
    pub fn into_moveaxis_f<IS, ID>(self, source: IS, destination: ID) -> Result<TensorAny<R, T, B, D>>
    where
        IS: TryInto<AxesIndex<isize>, Error: Into<Error>>,
        ID: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        into_moveaxis_f(self, source, destination)
    }
}
