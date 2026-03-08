use crate::prelude_dev::*;

/* #region expand_dims */

/// Expands the shape of an array by inserting a new axis (dimension) of size one at the position
/// specified by `axis`.
///
/// # See also
///
/// Refer to [`expand_dims`] for more detailed documentation.
pub fn into_expand_dims_f<S, D>(
    tensor: TensorBase<S, D>,
    axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
) -> Result<TensorBase<S, IxD>>
where
    D: DimAPI,
{
    // convert axis to negative indexes and sort
    let ndim = tensor.ndim();
    let (storage, layout) = tensor.into_raw_parts();
    let mut layout = layout.into_dim::<IxD>()?;
    let axes = axes.try_into().map_err(Into::into)?;
    let len_axes = axes.as_ref().len();
    let axes = normalize_axes_index(axes, ndim + len_axes, false, true)?;
    for axis in axes {
        layout = layout.dim_insert(axis)?;
    }
    unsafe { Ok(TensorBase::new_unchecked(storage, layout)) }
}

/// Expands the shape of an array by inserting a new axis (dimension) of size one at the position
/// specified by `axis`.
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny)
///
///   - The input tensor.
///   - Note on variant [`into_expand_dims`]: This takes ownership [`Tensor<R, T, B, D>`] of input
///     tensor, and will not perform change to underlying data, only layout changes.
///
/// - `axes`: TryInto [`AxesIndex<isize>`]
///
///   - Position in the expanded axes where the new axis (or axes) is placed.
///   - Can be a single integer, or a list/tuple of integers.
///   - Negative values are supported and indicate counting dimensions from the back.
///
/// # Returns
///
/// - [`TensorView<'_, T, B, IxD>`](TensorView)
///
///   - A view of the input tensor with the new axis (or axes) inserted.
///   - If you want to convert the tensor itself (taking the ownership instead of returning view),
///     use [`into_expand_dims`] instead.
///
/// # Panics
///
/// - If `axis` is greater than the number of axes in the original tensor.
/// - If expaneded axis has duplicated values.
///
/// # Examples
///
/// We first initialize a vector of shape (2,):
///
/// ```rust
/// use rstsr::prelude::*;
/// let x = rt::asarray(vec![1, 2]);
/// ```
///
/// Expand dims at axis 0, which is equilvalent to `x.i(None)`:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let x = rt::arange((2, &device));
/// let y = x.expand_dims(0);
/// println!("{y}");
/// // [[ 0 1]]
/// println!("y shape: {:?}", y.shape());
/// // y shape: [1, 2]
/// assert_eq!(x.i(None).shape(), y.shape());
/// ```
///
/// Expand dims at axis -1 (last axis), which is equilvalent to `x.i((Ellipsis, None))`, or in this
/// 1-dimension specific case, `x.i((.., None))`:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // expand dims at axis -1 (last axis)
/// let x = rt::arange((2, &device));
/// let y = x.expand_dims(-1);
/// println!("{y}");
/// // [[ 0]
/// //  [ 1]]
/// println!("y shape: {:?}", y.shape());
/// // y shape: [2, 1]
/// let y_expected = rt::tensor_from_nested!([[0], [1]]);
/// assert_eq!(x.i((Ellipsis, None)).shape(), &[2, 1]);
/// ```
///
/// Expand dims at axes 0 and 1, which is equilvalent to `x.i((None, None))`:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let x = rt::arange((2, &device));
/// let y = x.expand_dims([0, 1]);
/// println!("{y}");
/// // [[[ 0 1]]]
/// println!("y shape: {:?}", y.shape());
/// // y shape: [1, 1, 2]
/// let y_expected = rt::tensor_from_nested!([[[0, 1]]], &device);
/// assert_eq!(x.i((None, None)).shape(), &[1, 1, 2]);
/// ```
///
/// /// Expand dims at axes 0 and 2, which is equilvalent to `x.i((None, Ellipsis, None))`:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let x = rt::arange((2, &device));
/// let y = x.expand_dims([0, 2]);
/// println!("{y}");
/// // [[[ 0]]
/// //  [[ 1]]]
/// println!("y shape: {:?}", y.shape());
/// // y shape: [1, 2, 1]
/// let y_expected = rt::tensor_from_nested!([[[0], [1]]], &device);
/// assert_eq!(x.i((None, Ellipsis, None)).shape(), &[1, 2, 1]);
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `expand_dims(x, /, axis)` ([`expand_dims` in Array-API](https://data-apis.org/array-api/latest/API_specification/generated/array_api.expand_dims.html))
/// - NumPy: `expand_dims(a, axis)` ([`numpy.expand_dims`](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html))
/// - RSTSR: `rt::expand_dims(tensor, axes)`
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`i`](TensorAny::i) or [`slice`](slice()): Basic indexing and slicing of tensors, without
///   modification of the underlying data.
/// - [`squeeze`]: Removes singleton dimensions (axes) from `x`.
///
/// ## Variants of this function
///
/// - [expand_dims] / [`expand_dims_f`]: Returning a view.
/// - [`into_expand_dims`] / [`into_expand_dims_f`]: Consuming version.
/// - [`unsqueeze`] / [`unsqueeze_f`]: Alias of [`expand_dims`] / [`expand_dims_f`].
/// - [`into_unsqueeze`] / [`into_unsqueeze_f`]: Alias of [`into_expand_dims`] /
///   [`into_expand_dims_f`].
///
/// - Associated methods on [`TensorAny`]:
///
///   - [`TensorAny::expand_dims`] / [`TensorAny::expand_dims_f`]
///   - [`TensorAny::into_expand_dims`] / [`TensorAny::into_expand_dims_f`]
///   - [`TensorAny::unsqueeze`] / [`TensorAny::unsqueeze_f`]
///   - [`TensorAny::into_unsqueeze`] / [`TensorAny::into_unsqueeze_f`]
pub fn expand_dims<R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
) -> TensorView<'_, T, B, IxD>
where
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_expand_dims_f(tensor.view(), axes).rstsr_unwrap()
}

/// Expands the shape of an array by inserting a new axis (dimension) of size one at the position
/// specified by `axis`.
///
/// # See also
///
/// Refer to [`expand_dims`] for more detailed documentation.
pub fn expand_dims_f<R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
) -> Result<TensorView<'_, T, B, IxD>>
where
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_expand_dims_f(tensor.view(), axes)
}

/// Expands the shape of an array by inserting a new axis (dimension) of size one at the position
/// specified by `axis`.
///
/// # See also
///
/// Refer to [`expand_dims`] for more detailed documentation.
pub fn into_expand_dims<S, D>(
    tensor: TensorBase<S, D>,
    axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
) -> TensorBase<S, IxD>
where
    D: DimAPI,
{
    into_expand_dims_f(tensor, axes).rstsr_unwrap()
}

pub use expand_dims as unsqueeze;
pub use expand_dims_f as unsqueeze_f;
pub use into_expand_dims as into_unsqueeze;
pub use into_expand_dims_f as into_unsqueeze_f;

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Expands the shape of an array by inserting a new axis (dimension) of size one at the
    /// position specified by `axis`.
    ///
    /// # See also
    ///
    /// Refer to [`expand_dims`] for more detailed documentation.
    pub fn expand_dims(&self, axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>) -> TensorView<'_, T, B, IxD> {
        into_expand_dims(self.view(), axes)
    }

    /// Expands the shape of an array by inserting a new axis (dimension) of size one at the
    /// position specified by `axis`.
    ///
    /// # See also
    ///
    /// Refer to [`expand_dims`] for more detailed documentation.
    pub fn expand_dims_f(
        &self,
        axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    ) -> Result<TensorView<'_, T, B, IxD>> {
        into_expand_dims_f(self.view(), axes)
    }

    /// Expands the shape of an array by inserting a new axis (dimension) of size one at the
    /// position specified by `axis`.
    ///
    /// # See also
    ///
    /// Refer to [`expand_dims`] for more detailed documentation.
    pub fn into_expand_dims(self, axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>) -> TensorAny<R, T, B, IxD> {
        into_expand_dims(self, axes)
    }

    /// Expands the shape of an array by inserting a new axis (dimension) of size one at the
    /// position specified by `axis`.
    ///
    /// # See also
    ///
    /// Refer to [`expand_dims`] for more detailed documentation.
    pub fn into_expand_dims_f(
        self,
        axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    ) -> Result<TensorAny<R, T, B, IxD>> {
        into_expand_dims_f(self, axes)
    }

    /// Expands the shape of an array by inserting a new axis (dimension) of size one at the
    /// position specified by `axis`.
    ///
    /// # See also
    ///
    /// Refer to [`expand_dims`] for more detailed documentation.
    pub fn unsqueeze(&self, axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>) -> TensorView<'_, T, B, IxD> {
        self.expand_dims(axes)
    }

    /// Expands the shape of an array by inserting a new axis (dimension) of size one at the
    /// position specified by `axis`.
    ///
    /// # See also
    ///
    /// Refer to [`expand_dims`] for more detailed documentation.
    pub fn unsqueeze_f(
        &self,
        axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    ) -> Result<TensorView<'_, T, B, IxD>> {
        self.expand_dims_f(axes)
    }

    /// Expands the shape of an array by inserting a new axis (dimension) of size one at the
    /// position specified by `axis`.
    ///
    /// # See also
    ///
    /// Refer to [`expand_dims`] for more detailed documentation.
    pub fn into_unsqueeze(self, axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>) -> TensorAny<R, T, B, IxD> {
        self.into_expand_dims(axes)
    }

    /// Expands the shape of an array by inserting a new axis (dimension) of size one at the
    /// position specified by `axis`.
    ///
    /// # See also
    ///
    /// Refer to [`expand_dims`] for more detailed documentation.
    pub fn into_unsqueeze_f(
        self,
        axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    ) -> Result<TensorAny<R, T, B, IxD>> {
        self.into_expand_dims_f(axes)
    }
}

/* #endregion */
