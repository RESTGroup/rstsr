use crate::prelude_dev::*;

/* #region flip */

/// Reverses the order of elements in an array along the given axis.
///
/// # See also
///
/// Refer to [`flip`] for more detailed documentation.
pub fn into_flip_f<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> Result<TensorBase<S, D>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
{
    let (storage, mut layout) = tensor.into_raw_parts();
    let mut axes = normalize_axes_index(axes.try_into()?, layout.ndim(), false)?;
    if axes.is_empty() {
        axes = (0..layout.ndim() as isize).collect();
    }
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
///   - The input tensor to be flipped.
///
/// - `axes`: TryInto [`AxesIndex<isize>`]
///
///   - Axis or axes along which to flip over.
///   - If `axes` is a single integer, flipping is performed along that axis.
///   - If `axes` is a tuple/list of integers, flipping is performed on all specified axes.
///   - If `axes` is empty, the function will flip over all axes.
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
/// Flipping the first (0) axis:
///
/// ```rust
/// use rstsr::prelude::*;
/// let mut device = DeviceCpu::default();
/// device.set_default_order(RowMajor);
///
/// let a = rt::arange((8, &device)).into_shape([2, 2, 2]);
/// let b = a.flip(0);
/// let b_expected = rt::tensor_from_nested!([[[4, 5], [6, 7]], [[0, 1], [2, 3]]], &device);
/// assert!(rt::allclose(&b, &b_expected, None));
/// # let b_sliced = a.i(slice!(None, None, -1));
/// # assert!(rt::allclose(&b_sliced, &b_expected, None));
/// ```
///
/// The flipping is equivalent to slicing with a step of -1 along the specified axis:
///
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// #
/// # let a = rt::arange((8, &device)).into_shape([2, 2, 2]);
/// # let b = a.flip(0);
/// # let b_expected = rt::tensor_from_nested!([[[4, 5], [6, 7]], [[0, 1], [2, 3]]], &device);
/// # assert!(rt::allclose(&b, &b_expected, None));
/// let b_sliced = a.i(slice!(None, None, -1));
/// assert!(rt::allclose(&b_sliced, &b_expected, None));
/// ```
///
/// Flipping the second (1) axis:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// #
/// # let a = rt::arange((8, &device)).into_shape([2, 2, 2]);
/// let b = a.flip(1);
/// let b_expected = rt::tensor_from_nested!([[[2, 3], [0, 1]], [[6, 7], [4, 5]]], &device);
/// assert!(rt::allclose(&b, &b_expected, None));
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
/// #
/// # let a = rt::arange((8, &device)).into_shape([2, 2, 2]);
/// let b = a.flip([0, -1]);
/// let b_expected = rt::tensor_from_nested!([[[5, 4], [7, 6]], [[1, 0], [3, 2]]], &device);
/// assert!(rt::allclose(&b, &b_expected, None));
/// ```
///
/// ## Flipping all axes
///
/// You can specify `None` or empty tuple `()` to flip all axes:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// #
/// # let a = rt::arange((8, &device)).into_shape([2, 2, 2]);
/// let b = a.flip(None);
/// let b_expected = rt::tensor_from_nested!([[[7, 6], [5, 4]], [[3, 2], [1, 0]]], &device);
/// assert!(rt::allclose(&b, &b_expected, None));
/// ```
///
/// # Panics
///
/// - If some index in `axes` is greater than the number of axes in the original tensor.
/// - If `axes` has duplicated values.
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - Python Array API standard: [`flip`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.flip.html)
/// - NumPy: [`numpy.flip`](https://numpy.org/doc/stable/reference/generated/numpy.flip.html)
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
pub fn flip<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axes: I) -> TensorView<'_, T, B, D>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
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
pub fn flip_f<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axes: I) -> Result<TensorView<'_, T, B, D>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
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
pub fn into_flip<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> TensorBase<S, D>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
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
    /// Refer to [`flip`] for more detailed documentation.
    pub fn into_flip<I>(self, axis: I) -> TensorAny<R, T, B, D>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        into_flip(self, axis)
    }

    /// Reverses the order of elements in an array along the given axis.
    ///
    /// # See also
    ///
    /// Refer to [`flip`] for more detailed documentation.
    pub fn into_flip_f<I>(self, axis: I) -> Result<TensorAny<R, T, B, D>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        into_flip_f(self, axis)
    }
}

/* #endregion */

#[cfg(test)]
mod tests {
    #[test]
    fn doc_flip() {
        use rstsr::prelude::*;

        let mut device = DeviceCpu::default();
        device.set_default_order(RowMajor);

        let a = rt::arange((8, &device)).into_shape([2, 2, 2]);
        let a_expected = rt::tensor_from_nested!([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], &device);
        assert!(rt::allclose(&a, &a_expected, None));

        let b = a.flip(0);
        let b_expected = rt::tensor_from_nested!([[[4, 5], [6, 7]], [[0, 1], [2, 3]]], &device);
        assert!(rt::allclose(&b, &b_expected, None));
        let b_sliced = a.i(slice!(None, None, -1));
        assert!(rt::allclose(&b_sliced, &b_expected, None));

        let b = a.flip(1);
        let b_expected = rt::tensor_from_nested!([[[2, 3], [0, 1]], [[6, 7], [4, 5]]], &device);
        assert!(rt::allclose(&b, &b_expected, None));

        let b = a.flip([0, -1]);
        let b_expected = rt::tensor_from_nested!([[[5, 4], [7, 6]], [[1, 0], [3, 2]]], &device);
        assert!(rt::allclose(&b, &b_expected, None));

        let b = a.flip(None);
        let b_expected = rt::tensor_from_nested!([[[7, 6], [5, 4]], [[3, 2], [1, 0]]], &device);
        assert!(rt::allclose(&b, &b_expected, None));
    }
}
