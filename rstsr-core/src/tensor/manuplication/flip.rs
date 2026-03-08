use crate::prelude_dev::*;

/* #region flip */

/// Reverses the order of elements in an array along the given axis.
///
/// # See also
///
/// Refer to [`flip`] for more detailed documentation.
pub fn into_flip_f<S, D>(
    tensor: TensorBase<S, D>,
    axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
) -> Result<TensorBase<S, D>>
where
    D: DimAPI,
{
    let (storage, mut layout) = tensor.into_raw_parts();
    let axes = axes.try_into().map_err(Into::into)?;
    let axes = match axes {
        AxesIndex::None => (0..layout.ndim() as isize).collect(),
        _ => normalize_axes_index(axes, layout.ndim(), false, true)?,
    };
    for axis in axes {
        layout = layout.dim_narrow(axis, slice!(None, None, -1))?;
    }
    unsafe { Ok(TensorBase::new_unchecked(storage, layout)) }
}

/// Reverses the order of elements in an array along the given axis.
///
/// The shape of the array will be preserved after flipping.
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny)
///
///   - The input tensor.
///   - Note on variant [`into_flip`]: This takes ownership [`Tensor<R, T, B, D>`] of input tensor,
///     and will not perform change to underlying data, only layout changes.
///
/// - `axes`: TryInto [`AxesIndex<isize>`]
///
///   - Axis or axes along which to flip over.
///   - If `axes` is a single integer, flipping is performed along that axis.
///   - If `axes` is a tuple/list of integers, flipping is performed on all specified axes.
///   - If `axes` is `None`, the function will flip over all axes.
///   - If `axes` is an empty tuple `()`, no axes are flipped (returns a view of the original
///     tensor).
///   - Negative values are supported and indicate counting dimensions from the back.
///
/// # Returns
///
/// - [`TensorView<'_, T, B, D>`](TensorView)
///
///   - A view of the input tensor with the entries along the specified axes reversed.
///   - The shape of the array is preserved, but the elements are reordered.
///   - The underlying data is not copied; only the layout of the view is modified.
///   - If you want to convert the tensor itself (taking the ownership instead of returning view),
///     use [`into_flip`] instead.
///
/// # Examples
///
/// ## Flipping along a single axis
///
/// Given a 3D tensor of shape (2, 2, 2):
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((8, &device)).into_shape([2, 2, 2]);
/// println!("{a}");
/// // [[[ 0 1]
/// //   [ 2 3]]
/// //
/// //  [[ 4 5]
/// //   [ 6 7]]]
/// ```
///
/// Flipping the first (0) axis (which is equilvant to slicing with step -1 along that axis):
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// # let a = rt::arange((8, &device)).into_shape([2, 2, 2]);
/// let b = a.flip(0);
/// assert!(rt::allclose(a.i(slice!(None, None, -1)), &b, None));
/// println!("{:}", a.flip(0));
/// // [[[ 4 5]
/// //   [ 6 7]]
/// //
/// //  [[ 0 1]
/// //   [ 2 3]]]
/// ```
///
/// Flipping the second (1) axis:
///
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// # let a = rt::arange((8, &device)).into_shape([2, 2, 2]);
/// let b = a.flip(1);
/// assert!(rt::allclose(a.i((.., slice!(None, None, -1))), &b, None));
/// println!("{:}", a.flip(1));
/// // [[[ 2 3]
/// //   [ 0 1]]
/// //
/// //  [[ 6 7]
/// //   [ 4 5]]]
/// ```
///
/// ## Flipping along multiple axes
///
/// Flipping the first (0) and last (-1 or in this specific case, 2) axes:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// # let a = rt::arange((8, &device)).into_shape([2, 2, 2]);
/// let b = a.flip([0, -1]);
/// println!("{b}");
/// // [[[ 5 4]
/// //   [ 7 6]]
/// //
/// //  [[ 1 0]
/// //   [ 3 2]]]
/// ```
///
/// ## Flipping all axes
///
/// You can specify `None` to flip all axes:
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// # let a = rt::arange((8, &device)).into_shape([2, 2, 2]);
/// let b = a.flip(None);
/// println!("{b}");
/// // [[[ 7 6]
/// //   [ 5 4]]
/// //
/// //  [[ 3 2]
/// //   [ 1 0]]]
/// ```
///
/// ## No flipping (empty axes)
///
/// You can specify an empty tuple `()` to flip no axes (returns a view of the original tensor):
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// # let a = rt::arange((8, &device)).into_shape([2, 2, 2]);
/// let b = a.flip(());
/// println!("{b}");
/// // [[[ 0 1]
/// //   [ 2 3]]
/// //
/// //  [[ 4 5]
/// //   [ 6 7]]]
/// ```
///
/// # Panics
///
/// - If some index in `axes` is greater than the number of axes in the original tensor.
/// - If `axes` has duplicated values.
///
/// # Notes of API accordance
///
/// - Array-API: `flip(x, /, *, axis=None)` ([`flip`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.flip.html))
/// - NumPy: `flip(m, axis=None)` ([`numpy.flip`](https://numpy.org/doc/stable/reference/generated/numpy.flip.html))
/// - RSTSR: `rt::flip(tensor, axes)`
///
/// RSTSR's behavior matches NumPy and Array-API:
/// - `a.flip(None)` flips all axes
/// - `a.flip(())` flips no axes (returns a view of the original tensor)
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`i`](TensorAny::i) or [`slice`](slice()): Basic indexing and slicing of tensors, without
///   modification of the underlying data.
///
/// ## Variants of this function
///
/// - [`flip`]: Borrowing version.
/// - [`flip_f`]: Fallible version.
/// - [`into_flip`]: Consuming version.
/// - [`into_flip_f`]: Consuming and fallible version, actual implementation.
/// - Associated methods on [`TensorAny`]:
///
///   - [`TensorAny::flip`]
///   - [`TensorAny::flip_f`]
///   - [`TensorAny::into_flip`]
///   - [`TensorAny::into_flip_f`]
pub fn flip<R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
) -> TensorView<'_, T, B, D>
where
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_flip_f(tensor.view(), axes).rstsr_unwrap()
}

/// Reverses the order of elements in an array along the given axis.
///
/// # See also
///
/// Refer to [`flip`] for more detailed documentation.
pub fn flip_f<R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
) -> Result<TensorView<'_, T, B, D>>
where
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_flip_f(tensor.view(), axes)
}

/// Reverses the order of elements in an array along the given axis.
///
/// # See also
///
/// Refer to [`flip`] for more detailed documentation.
pub fn into_flip<S, D>(
    tensor: TensorBase<S, D>,
    axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
) -> TensorBase<S, D>
where
    D: DimAPI,
{
    into_flip_f(tensor, axes).rstsr_unwrap()
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
    /// Refer to [`flip`] for more detailed documentation.
    pub fn flip(&self, axis: impl TryInto<AxesIndex<isize>, Error: Into<Error>>) -> TensorView<'_, T, B, D> {
        flip(self, axis)
    }

    pub fn flip_f(&self, axis: impl TryInto<AxesIndex<isize>, Error: Into<Error>>) -> Result<TensorView<'_, T, B, D>> {
        flip_f(self, axis)
    }

    /// Reverses the order of elements in an array along the given axis.
    ///
    /// # See also
    ///
    /// Refer to [`flip`] for more detailed documentation.
    pub fn into_flip(self, axis: impl TryInto<AxesIndex<isize>, Error: Into<Error>>) -> TensorAny<R, T, B, D> {
        into_flip(self, axis)
    }

    /// Reverses the order of elements in an array along the given axis.
    ///
    /// # See also
    ///
    /// Refer to [`flip`] for more detailed documentation.
    pub fn into_flip_f(
        self,
        axis: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    ) -> Result<TensorAny<R, T, B, D>> {
        into_flip_f(self, axis)
    }
}

/* #endregion */
