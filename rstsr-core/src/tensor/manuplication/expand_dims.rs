use crate::prelude_dev::*;

/* #region expand_dims */

/// Expands the shape of an array by inserting a new axis (dimension) of size
/// one at the position specified by `axis`.
///
/// # See also
///
/// Refer to [`expand_dims`] and [`into_expand_dims`] for more detailed documentation.
pub fn into_expand_dims_f<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> Result<TensorBase<S, IxD>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
{
    // convert axis to negative indexes and sort
    let ndim = tensor.ndim();
    let (storage, layout) = tensor.into_raw_parts();
    let mut layout = layout.into_dim::<IxD>()?;
    let axes = axes.try_into()?;
    let len_axes = axes.as_ref().len();
    let axes = normalize_axes_index(axes, ndim + len_axes, false)?;
    for axis in axes {
        layout = layout.dim_insert(axis)?;
    }
    unsafe { Ok(TensorBase::new_unchecked(storage, layout)) }
}

/// Expands the shape of an array by inserting a new axis (dimension) of size
/// one at the position specified by `axis`.
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny)
///
///   - The input tensor.
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
/// # let x = rt::asarray(vec![1, 2]);
/// // [1, 2] -> [[1, 2]]
/// let y = x.expand_dims(0);
/// let y_expected = rt::tensor_from_nested!([[1, 2]]);
/// assert!(rt::allclose(&y, &y_expected, None));
/// assert_eq!(y.shape(), &[1, 2]);
/// assert_eq!(x.i(None).shape(), &[1, 2]);
/// ```
///
/// Expand dims at axis -1 (last axis), which is equilvalent to `x.i((Ellipsis, None))`, or in this
/// 1-dimension specific case, `x.i((.., None))`:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let x = rt::asarray(vec![1, 2]);
/// // [1, 2] -> [[1], [2]]
/// let y = x.expand_dims(-1);
/// let y_expected = rt::tensor_from_nested!([[1], [2]]);
/// assert!(rt::allclose(&y, &y_expected, None));
/// assert_eq!(y.shape(), &[2, 1]);
/// assert_eq!(x.i((Ellipsis, None)).shape(), &[2, 1]);
/// ```
///
/// Expand dims at axes 0 and 1, which is equilvalent to `x.i((None, None))`:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let x = rt::asarray(vec![1, 2]);
/// // Expand dims at axes 0 and 1
/// // [1, 2] -> [[[1, 2]]]
/// let y = x.expand_dims([0, 1]);
/// let y_expected = rt::tensor_from_nested!([[[1, 2]]]);
/// assert!(rt::allclose(&y, &y_expected, None));
/// assert_eq!(y.shape(), &[1, 1, 2]);
/// assert_eq!(x.i((None, None)).shape(), &[1, 1, 2]);
/// ```
///
/// /// Expand dims at axes 0 and 2, which is equilvalent to `x.i((None, Ellipsis, None))`:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let x = rt::asarray(vec![1, 2]);
/// // Expand dims at axes 0 and 2
/// // [1, 2] -> [[[1], [2]]]
/// let y = x.expand_dims([0, 2]);
/// let y_expected = rt::tensor_from_nested!([[[1], [2]]]);
/// assert!(rt::allclose(&y, &y_expected, None));
/// assert_eq!(y.shape(), &[1, 2, 1]);
/// assert_eq!(x.i((None, Ellipsis, None)).shape(), &[1, 2, 1]);
/// ```
///
/// # See also
///
/// ## Similar functions from other crates/libraries
///
/// - Python Array API standard: [`expand_dims`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.expand_dims.html)
/// - NumPy: [`numpy.expand_dims`](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html)
/// - PyTorch: [`torch.unsqueeze`](https://pytorch.org/docs/stable/generated/torch.unsqueeze.html)
///
/// ## Related functions in RSTSR
///
/// - [`i`](TensorAny::i) or [`slice`](slice()): Basic indexing and slicing of tensors, without
///   modification of the underlying data.
/// - [`squeeze`]: Removes singleton dimensions (axes) from `x`.
///
/// ## Variants of this function
///
/// - [`expand_dims_f`]: Failable version.
/// - [`into_expand_dims`]: Consuming version.
/// - [`into_expand_dims_f`]: Failable and consuming version, actual implementation.
pub fn expand_dims<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axes: I) -> TensorView<'_, T, B, IxD>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_expand_dims_f(tensor.view(), axes).rstsr_unwrap()
}

/// Expands the shape of an array by inserting a new axis (dimension) of size
/// one at the position specified by `axis`.
///
/// # See also
///
/// Refer to [`expand_dims`] and [`into_expand_dims`] for more detailed documentation.
pub fn expand_dims_f<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axes: I) -> Result<TensorView<'_, T, B, IxD>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_expand_dims_f(tensor.view(), axes)
}

/// Expands the shape of an array by inserting a new axis (dimension) of size
/// one at the position specified by `axis`.
///
/// # Parameters
///
/// - `tensor`: [`TensorBase<S, D>`]
///
///   - The input tensor.
///   - Please note that this function takes ownership of the input tensor.
///
/// - `axes`: TryInto [`AxesIndex<isize>`]
///
///   - Position in the expanded axes where the new axis (or axes) is placed.
///   - Can be a single integer, or a list/tuple of integers.
///   - Negative values are supported and indicate counting dimensions from the back.
///
/// # Returns
///
/// - [`TensorBase<S, IxD>`]
///
///   - The tensor with the new axis (or axes) inserted.
///   - Ownership of the returned tensor is transferred from the input tensor. Only the layout is
///     modified; the underlying data remains unchanged.
///
/// # See also
///
/// Refer to [`expand_dims`] for more detailed documentation.
pub fn into_expand_dims<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> TensorBase<S, IxD>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
{
    into_expand_dims_f(tensor, axes).rstsr_unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Expands the shape of an array by inserting a new axis (dimension) of size
    /// one at the position specified by `axis`.
    ///
    /// # See also
    ///
    /// Refer to [`expand_dims`] and [`into_expand_dims`] for more detailed documentation.
    pub fn expand_dims<I>(&self, axes: I) -> TensorView<'_, T, B, IxD>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        into_expand_dims(self.view(), axes)
    }

    /// Expands the shape of an array by inserting a new axis (dimension) of size
    /// one at the position specified by `axis`.
    ///
    /// # See also
    ///
    /// Refer to [`expand_dims`] and [`into_expand_dims`] for more detailed documentation.
    pub fn expand_dims_f<I>(&self, axes: I) -> Result<TensorView<'_, T, B, IxD>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        into_expand_dims_f(self.view(), axes)
    }

    /// Expands the shape of an array by inserting a new axis (dimension) of size
    /// one at the position specified by `axis`.
    ///
    /// # See also
    ///
    /// Refer to [`expand_dims`] and [`into_expand_dims`] for more detailed documentation.
    pub fn into_expand_dims<I>(self, axes: I) -> TensorAny<R, T, B, IxD>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        into_expand_dims(self, axes)
    }

    /// Expands the shape of an array by inserting a new axis (dimension) of size
    /// one at the position specified by `axis`.
    ///
    /// # See also
    ///
    /// Refer to [`expand_dims`] and [`into_expand_dims`] for more detailed documentation.
    pub fn into_expand_dims_f<I>(self, axes: I) -> Result<TensorAny<R, T, B, IxD>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        into_expand_dims_f(self, axes)
    }
}

/* #endregion */

#[cfg(test)]
mod tests {
    #[test]
    fn doc_expand_dims() {
        use rstsr::prelude::*;
        let x = rt::asarray(vec![1, 2]);

        // Expand dims at axis 0
        // [1, 2] -> [[1, 2]]
        let y = x.expand_dims(0);
        let y_expected = rt::tensor_from_nested!([[1, 2]]);
        assert!(rt::allclose(&y, &y_expected, None));
        assert_eq!(y.shape(), &[1, 2]);
        assert_eq!(x.i(None).shape(), &[1, 2]);

        // Expand dims at axis -1 (last axis)
        // [1, 2] -> [[1], [2]]
        let y = x.expand_dims(-1);
        let y_expected = rt::tensor_from_nested!([[1], [2]]);
        assert!(rt::allclose(&y, &y_expected, None));
        assert_eq!(y.shape(), &[2, 1]);
        assert_eq!(x.i((Ellipsis, None)).shape(), &[2, 1]);

        // Expand dims at axes 0 and 1
        // [1, 2] -> [[[1, 2]]]
        let y = x.expand_dims([0, 1]);
        let y_expected = rt::tensor_from_nested!([[[1, 2]]]);
        assert!(rt::allclose(&y, &y_expected, None));
        assert_eq!(y.shape(), &[1, 1, 2]);
        assert_eq!(x.i((None, None)).shape(), &[1, 1, 2]);

        // Expand dims at axes 0 and 2
        // [1, 2] -> [[[1], [2]]]
        let y = x.expand_dims([0, 2]);
        let y_expected = rt::tensor_from_nested!([[[1], [2]]]);
        assert!(rt::allclose(&y, &y_expected, None));
        assert_eq!(y.shape(), &[1, 2, 1]);
        assert_eq!(x.i((None, Ellipsis, None)).shape(), &[1, 2, 1]);
    }
}
