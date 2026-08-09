use crate::prelude_dev::*;

/* #region into_compatible_shape */

/// Reshapes the given tensor to the specified shape if the layout is compatible.
///
/// See also [`to_compatible_shape`].
pub fn into_compatible_shape_f<R, T, B, D>(
    tensor: TensorAny<R, T, B, D>,
    shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    order: impl Into<Option<FlagOrder>>,
) -> Result<TensorAny<R, T, B, IxD>>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    let shape_new = reshape_substitute_negatives(shape.try_into().map_err(Into::into)?.as_ref(), tensor.size())?;
    let order = order.into().unwrap_or(tensor.device().default_order());
    if let Some(layout_new) = layout_reshapeable(&tensor.layout().to_dim()?, &shape_new, order)? {
        let (storage, _) = tensor.into_raw_parts();
        unsafe { Ok(TensorBase::new_unchecked(storage, layout_new)) }
    } else {
        rstsr_raise!(InvalidLayout, "Cannot reshape {:?} to {shape_new:?} with order {order:?}.", tensor.layout())?
    }
}

/// Reshapes the given tensor to the specified shape if the layout is compatible.
///
/// See also [`to_compatible_shape`].
pub fn into_compatible_shape<R, T, B, D>(
    tensor: TensorAny<R, T, B, D>,
    shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    order: impl Into<Option<FlagOrder>>,
) -> TensorAny<R, T, B, IxD>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    into_compatible_shape_f(tensor, shape, order).rstsr_unwrap()
}

/// Reshapes the given tensor to the specified shape if the layout is compatible.
///
/// See also [`to_compatible_shape`].
pub fn to_compatible_shape_f<R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    order: impl Into<Option<FlagOrder>>,
) -> Result<TensorView<'_, T, B, IxD>>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    into_compatible_shape_f(tensor.view(), shape, order)
}

/// Returns a view of the tensor with the specified shape if the layout is compatible.
///
/// This function takes a reference to the input tensor and returns a view with the new shape.
/// No data is copied; only the layout is modified. If the layout is not compatible,
/// this function will panic.
///
/// <div class="warning">
///
/// **Row/Column Major Notice**
///
/// This function behaves differently on default orders ([`RowMajor`] and [`ColMajor`]) of device.
///
/// </div>
///
/// <div class="warning">
///
/// **Different Signature Convention to [`reshape`]**
///
/// This function does not change the underlying data, only manipulating the layout. So this
/// function returns a view instead of copy-on-write tensor.
///
/// </div>
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny)
///
///   - The input tensor.
///
/// - `shape`: TryInto [`AxesIndex<isize>`]
///
///   - The new shape of the tensor.
///   - Can be a single integer, or a list/tuple of integers.
///   - Negative values are supported and indicate counting dimensions from the back.
///   - Overloads:
///
///     - integer: 1-D shape with a single dimension.
///     - vector/array/tuple of integers: N-D shape with N dimensions. For tuples,
///       mixed-signed/unsigned integers are supported.
///
/// - `order`: Into [`Option<FlagOrder>`]
///
///   - The indexing order for reading and writing the tensor.
///   - [`RowMajor`] and [`ColMajor`] are supported.
///   - By default, the device's default order is used.
///
/// # Returns
///
/// - [`TensorView<'_, T, B, IxD>`]
///
///   - A view of the tensor with the new shape.
///
/// # Examples
///
/// Reshape a tensor when the layout is compatible:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // shape: (4, 6), stride: (6, 1), c-contiguous
/// let a = rt::arange((24, &device)).into_shape([4, 6]);
/// // Split a dimension: (4, 6) -> (2, 2, 6) - layout compatible
/// let b = a.to_compatible_shape([2, 2, 6], RowMajor);
/// assert_eq!(b.shape(), &[2, 2, 6]);
/// ```
///
/// Reshape a tensor when the layout is not compatible will panic (or use the falling variant
/// `to_compatible_shape_f` to handle the error):
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // shape: (4, 6, 9), stride: (72, 9, 1), not c-contiguous
/// let a = rt::arange((288, &device)).into_shape([4, 8, 9]).into_slice((.., 0..6, ..));
/// // layout compatible
/// assert!(a.to_compatible_shape_f([4, 6 * 9], RowMajor).is_ok());
/// // layout incompatible
/// assert!(a.to_compatible_shape_f([4 * 6, 9], RowMajor).is_err());
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `reshape(x, /, shape, *, copy=None)` ([`reshape`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.reshape.html))
/// - NumPy: `reshape(a, /, shape, order='C', *, copy=False)` ([`numpy.reshape`](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html))
/// - RSTSR: `rt::to_compatible_shape(&tensor, shape, order)` or `tensor.to_compatible_shape(shape,
///   order)`
///
/// This function is similar to reshape with `copy=False` in NumPy/array-API, but with explicit
/// order parameter. Unlike [`reshape`], this function never copies data and will panic if the
/// layout is not compatible.
///
/// # Panics
///
/// Panics if the tensor cannot be reshaped to the specified shape with the given order
/// without copying data.
///
/// For a fallible version, use [`to_compatible_shape_f`].
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`reshape`]: Reshapes a tensor, copying data if necessary (returning copy-on-write tensor).
/// - [`reshapeable_without_copy`]: Check if reshape can be done without copying data.
///
/// ## Variants of this function
///
/// - [`to_compatible_shape`] / [`to_compatible_shape_f`]: Returning a view.
/// - [`into_compatible_shape`] / [`into_compatible_shape_f`]: Consuming version.
///
/// - Associated methods on [`TensorAny`]:
///
///   - [`TensorAny::to_compatible_shape`] / [`TensorAny::to_compatible_shape_f`]
///   - [`TensorAny::into_compatible_shape`] / [`TensorAny::into_compatible_shape_f`]
pub fn to_compatible_shape<R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    order: impl Into<Option<FlagOrder>>,
) -> TensorView<'_, T, B, IxD>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    to_compatible_shape_f(tensor, shape, order).rstsr_unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Reshapes the given tensor to the specified shape if the layout is compatible.
    ///
    /// See also [`to_compatible_shape`].
    pub fn into_compatible_shape_f(
        self,
        shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
        order: impl Into<Option<FlagOrder>>,
    ) -> Result<TensorAny<R, T, B, IxD>> {
        into_compatible_shape_f(self, shape, order)
    }

    /// Reshapes the given tensor to the specified shape if the layout is compatible.
    ///
    /// See also [`to_compatible_shape`].
    pub fn into_compatible_shape(
        self,
        shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
        order: impl Into<Option<FlagOrder>>,
    ) -> TensorAny<R, T, B, IxD> {
        into_compatible_shape(self, shape, order)
    }

    /// Reshapes the given tensor to the specified shape if the layout is compatible.
    ///
    /// See also [`to_compatible_shape`].
    pub fn to_compatible_shape_f(
        &self,
        shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
        order: impl Into<Option<FlagOrder>>,
    ) -> Result<TensorView<'_, T, B, IxD>> {
        to_compatible_shape_f(self, shape, order)
    }

    /// Reshapes the given tensor to the specified shape if the layout is compatible.
    ///
    /// See also [`to_compatible_shape`].
    pub fn to_compatible_shape(
        &self,
        shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
        order: impl Into<Option<FlagOrder>>,
    ) -> TensorView<'_, T, B, IxD> {
        to_compatible_shape(self, shape, order)
    }
}

/* #endregion */

/* #region reshape_assume_contig (deprecated) */

/// Assuming contiguous array, reshapes an array without changing its data (falling variant).
///
/// See also [`to_shape_assume_contig`].
#[deprecated(since = "0.6.2", note = "Use `into_compatible_shape_f` instead with explicit order argument")]
pub fn into_shape_assume_contig_f<R, T, B, D, D2>(
    tensor: TensorAny<R, T, B, D>,
    shape: D2,
) -> Result<TensorAny<R, T, B, D2>>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
    D2: DimAPI,
{
    let default_order = tensor.device().default_order();
    let (storage, layout) = tensor.into_raw_parts();

    rstsr_assert_eq!(layout.size(), shape.shape_size(), InvalidLayout, "Number of elements not same.")?;

    let new_layout = {
        if default_order == FlagOrder::C && layout.c_contig() {
            shape.new_c_contig(Some(layout.offset()))
        } else if default_order == FlagOrder::F && layout.f_contig() {
            shape.new_f_contig(Some(layout.offset()))
        } else {
            rstsr_raise!(InvalidLayout, "This array is not contiguous by {:?}", default_order)?
        }
    };
    unsafe { Ok(TensorBase::new_unchecked(storage, new_layout)) }
}

/// Assuming contiguous array, reshapes an array without changing its data.
///
/// Returns a view of the tensor with the new shape, assuming the tensor is contiguous.
/// This function may return c-contiguous or f-contiguous array depending on
/// crate feature `col_major`.
///
/// <div class="warning">
///
/// **Deprecated**
///
/// This function is deprecated since version 0.6.2. Use [`to_compatible_shape`] instead
/// which provides the same functionality with a more consistent API.
///
/// </div>
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny)
///
///   - The input tensor. Must be contiguous (c-contiguous or f-contiguous depending on crate
///     feature `col_major`).
///
/// - `shape`: `D2`
///
///   - The new shape of the tensor.
///   - The number of elements must match the original tensor.
///
/// # Returns
///
/// - [`TensorView<'_, T, B, D2>`]
///
///   - A view of the tensor with the new shape.
///
/// # Panics
///
/// Panics if
///
/// - The number of elements in the new shape does not match the original tensor.
/// - The tensor is not contiguous in the expected order (c-contiguous by default, or f-contiguous
///   if `col_major` feature is enabled).
///
/// # Migration Guide
///
/// ```ignore
/// // Before
/// let view = to_shape_assume_contig(&tensor, shape);
///
/// // After
/// let view = to_compatible_shape(&tensor, shape, RowMajor);
/// // or use the device's default order
/// let view = to_compatible_shape(&tensor, shape, tensor.device().default_order());
/// ```
///
/// # See also
///
/// - [`to_compatible_shape`]: The recommended replacement with explicit order argument.
/// - [Python array API standard: `reshape`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.reshape.html)
#[deprecated(since = "0.6.2", note = "Use `to_compatible_shape` instead with explicit order argument")]
#[allow(deprecated)]
pub fn to_shape_assume_contig<R, T, B, D, D2>(tensor: &TensorAny<R, T, B, D>, shape: D2) -> TensorView<'_, T, B, D2>
where
    D: DimAPI,
    D2: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_shape_assume_contig_f(tensor.view(), shape).rstsr_unwrap()
}

/// Assuming contiguous array, reshapes an array without changing its data (falling variant).
///
/// See also [`to_shape_assume_contig`].
#[deprecated(since = "0.6.2", note = "Use `to_compatible_shape_f` instead with explicit order argument")]
#[allow(deprecated)]
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

/// Assuming contiguous array, reshapes an array without changing its data.
///
/// See also [`to_shape_assume_contig`].
#[deprecated(since = "0.6.2", note = "Use `into_compatible_shape` instead with explicit order argument")]
#[allow(deprecated)]
pub fn into_shape_assume_contig<R, T, B, D, D2>(tensor: TensorAny<R, T, B, D>, shape: D2) -> TensorAny<R, T, B, D2>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
    D2: DimAPI,
{
    into_shape_assume_contig_f(tensor, shape).rstsr_unwrap()
}

#[deprecated(since = "0.6.2", note = "Use `to_compatible_shape` instead with explicit order argument")]
#[allow(deprecated)]
pub use to_shape_assume_contig as reshape_assume_contig;
#[deprecated(since = "0.6.2", note = "Use `to_compatible_shape_f` instead with explicit order argument")]
#[allow(deprecated)]
pub use to_shape_assume_contig_f as reshape_assume_contig_f;

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Assuming contiguous array, reshapes an array without changing its data.
    ///
    /// See also [`to_shape_assume_contig`].
    #[deprecated(since = "0.6.2", note = "Use `to_compatible_shape` instead with explicit order argument")]
    #[allow(deprecated)]
    pub fn reshape_assume_contig<D2>(&self, shape: D2) -> TensorView<'_, T, B, D2>
    where
        D2: DimAPI,
    {
        into_shape_assume_contig(self.view(), shape)
    }

    /// Assuming contiguous array, reshapes an array without changing its data (falling variant).
    ///
    /// See also [`to_shape_assume_contig`].
    #[deprecated(since = "0.6.2", note = "Use `to_compatible_shape_f` instead with explicit order argument")]
    #[allow(deprecated)]
    pub fn reshape_assume_contig_f<D2>(&self, shape: D2) -> Result<TensorView<'_, T, B, D2>>
    where
        D2: DimAPI,
    {
        into_shape_assume_contig_f(self.view(), shape)
    }

    /// Assuming contiguous array, reshapes an array without changing its data.
    ///
    /// See also [`to_shape_assume_contig`].
    #[deprecated(since = "0.6.2", note = "Use `to_compatible_shape` instead with explicit order argument")]
    #[allow(deprecated)]
    pub fn to_shape_assume_contig<D2>(&self, shape: D2) -> TensorView<'_, T, B, D2>
    where
        D2: DimAPI,
    {
        into_shape_assume_contig(self.view(), shape)
    }

    /// Assuming contiguous array, reshapes an array without changing its data (falling variant).
    ///
    /// See also [`to_shape_assume_contig`].
    #[deprecated(since = "0.6.2", note = "Use `to_compatible_shape_f` instead with explicit order argument")]
    #[allow(deprecated)]
    pub fn to_shape_assume_contig_f<D2>(&self, shape: D2) -> Result<TensorView<'_, T, B, D2>>
    where
        D2: DimAPI,
    {
        into_shape_assume_contig_f(self.view(), shape)
    }

    /// Assuming contiguous array, reshapes an array without changing its data.
    ///
    /// See also [`to_shape_assume_contig`].
    #[deprecated(since = "0.6.2", note = "Use `into_compatible_shape` instead with explicit order argument")]
    #[allow(deprecated)]
    pub fn into_shape_assume_contig<D2>(self, shape: D2) -> TensorAny<R, T, B, D2>
    where
        D2: DimAPI,
    {
        into_shape_assume_contig(self, shape)
    }

    /// Assuming contiguous array, reshapes an array without changing its data (falling variant).
    ///
    /// See also [`to_shape_assume_contig`].
    #[deprecated(since = "0.6.2", note = "Use `into_compatible_shape_f` instead with explicit order argument")]
    #[allow(deprecated)]
    pub fn into_shape_assume_contig_f<D2>(self, shape: D2) -> Result<TensorAny<R, T, B, D2>>
    where
        D2: DimAPI,
    {
        into_shape_assume_contig_f(self, shape)
    }
}

/* #endregion */
