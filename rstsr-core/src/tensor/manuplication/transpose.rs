use crate::prelude_dev::*;

/* #region permute_dims */

pub fn into_transpose_f<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> Result<TensorBase<S, D>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
{
    let axes = axes.try_into().map_err(Into::into)?;
    match axes {
        AxesIndex::None => Ok(into_reverse_axes(tensor)),
        _ => {
            let (storage, layout) = tensor.into_raw_parts();
            let layout = layout.transpose(axes.as_ref())?;
            unsafe { Ok(TensorBase::new_unchecked(storage, layout)) }
        },
    }
}

/// Permutes the axes (dimensions) of an array.
///
/// Returns an array with axes transposed.
///
/// - For a 1-D array, this returns an unchanged view of the original array.
/// - For a 2-D array, this is the standard matrix transpose.
/// - For an n-D array, if axes are given, their order indicates how the axes are permuted (see
///   Examples).
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny)
///
///   - The input tensor whose axes are to be permuted.
///
/// - `axes`: TryInto [`AxesIndex<isize>`]
///
///   - The permutation of axes. If `None`, reverses the order of all axes (equivalent to
///     [`reverse_axes`]).
///   - Otherwise, `axes[i]` specifies the new position of axis `i` in the output.
///   - The length of `axes` must match the number of dimensions of the input tensor.
///   - Each axis must appear exactly once in `axes`.
///   - Negative values are supported and indicate counting dimensions from the back.
///
/// # Returns
///
/// - [`TensorView<'_, T, B, D>`](TensorView)
///
///   - A view of the input tensor with permuted axes.
///   - No data is copied; only the shape and strides are modified.
///
/// # Examples
///
/// For a 2-D array, this is the standard matrix transpose:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
/// let result = a.transpose(None);
/// println!("{result}");
/// // [[ 1 3]
/// //  [ 2 4]]
/// ```
///
/// For a 1-D array, this returns an unchanged view:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([1, 2, 3, 4], &device);
/// let result = a.transpose(None);
/// println!("{result}");
/// // [ 1 2 3 4]
/// ```
///
/// For an n-D array, you can specify a custom permutation, or None for reverse order:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // 3-D tensor
/// let a: Tensor<i32, _> = rt::ones(([1, 2, 3], &device));
/// let result = a.transpose(None);
/// println!("{:?}", result.shape());
/// // [3, 2, 1]
/// let result = a.transpose([1, 0, 2]);
/// println!("{:?}", result.shape());
/// // [2, 1, 3]
///
/// // 4-D tensor
/// let a: Tensor<i32, _> = rt::ones(([2, 3, 4, 5], &device));
/// let result = a.transpose(None);
/// println!("{:?}", result.shape());
/// // [5, 4, 3, 2]
/// ```
///
/// Negative indices are also supported:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a: Tensor<i32, _> = rt::arange((3 * 4 * 5, &device)).into_shape([3, 4, 5]);
/// let result = a.transpose([-1, 0, -2]);
/// println!("{:?}", result.shape());
/// // [5, 3, 4]
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `permute_dims(x, /, axes)` ([`permute_dims`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.permute_dims.html))
/// - NumPy: `transpose(a, axes=None)` ([`numpy.transpose`](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html))
/// - RSTSR: `tensor.transpose(axes)` or `rt::transpose(&tensor, axes)`
///
/// Note that `axes=None` in NumPy/RSTSR reverses the order of all axes, which is equivalent to
/// calling [`reverse_axes`] or [`TensorAny::t`] for 2D arrays.
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`permute_dims`] - Alias for this function
/// - [`reverse_axes`] - Reverse all axes order
/// - [`swapaxes`] - Swap two specific axes
///
/// ## Variants of this function
///
/// - [`transpose`] / [`transpose_f`]: Returning a view.
/// - [`into_transpose`] / [`into_transpose_f`]: Consuming version.
///
/// - Associated methods on `TensorAny`:
///
///   - [`TensorAny::transpose`] / [`TensorAny::transpose_f`]
///   - [`TensorAny::into_transpose`] / [`TensorAny::into_transpose_f`]
///   - [`TensorAny::t`] as shorthand for [`reverse_axes`]
pub fn transpose<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axes: I) -> TensorView<'_, T, B, D>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_transpose_f(tensor.view(), axes).rstsr_unwrap()
}

pub fn transpose_f<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axes: I) -> Result<TensorView<'_, T, B, D>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_transpose_f(tensor.view(), axes)
}

pub fn into_transpose<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> TensorBase<S, D>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
{
    into_transpose_f(tensor, axes).rstsr_unwrap()
}

pub use into_transpose as into_permute_dims;
pub use into_transpose_f as into_permute_dims_f;
pub use transpose as permute_dims;
pub use transpose_f as permute_dims_f;

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Permutes the axes (dimensions) of an array `x`.
    ///
    /// # See also
    ///
    /// [`transpose`]
    pub fn transpose<I>(&self, axes: I) -> TensorView<'_, T, B, D>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        transpose(self, axes)
    }

    pub fn transpose_f<I>(&self, axes: I) -> Result<TensorView<'_, T, B, D>>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        transpose_f(self, axes)
    }

    /// Permutes the axes (dimensions) of an array `x`.
    ///
    /// # See also
    ///
    /// [`transpose`]
    pub fn into_transpose<I>(self, axes: I) -> TensorAny<R, T, B, D>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        into_transpose(self, axes)
    }

    pub fn into_transpose_f<I>(self, axes: I) -> Result<TensorAny<R, T, B, D>>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        into_transpose_f(self, axes)
    }

    /// Permutes the axes (dimensions) of an array `x`.
    ///
    /// # See also
    ///
    /// [`transpose`]
    pub fn permute_dims<I>(&self, axes: I) -> TensorView<'_, T, B, D>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        transpose(self, axes)
    }

    pub fn permute_dims_f<I>(&self, axes: I) -> Result<TensorView<'_, T, B, D>>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        transpose_f(self, axes)
    }

    /// Permutes the axes (dimensions) of an array `x`.
    ///
    /// # See also
    ///
    /// [`transpose`]
    pub fn into_permute_dims<I>(self, axes: I) -> TensorAny<R, T, B, D>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        into_transpose(self, axes)
    }

    pub fn into_permute_dims_f<I>(self, axes: I) -> Result<TensorAny<R, T, B, D>>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        into_transpose_f(self, axes)
    }
}

/* #endregion */

/* #region reverse_axes */

/// Reverse the order of the axes (dimensions) of an array.
///
/// # See also
///
/// Refer to [`reverse_axes`] for details and examples.
pub fn into_reverse_axes<S, D>(tensor: TensorBase<S, D>) -> TensorBase<S, D>
where
    D: DimAPI,
{
    let (storage, layout) = tensor.into_raw_parts();
    let layout = layout.reverse_axes();
    unsafe { TensorBase::new_unchecked(storage, layout) }
}

/// Reverse the order of the axes (dimensions) of an array.
///
/// Returns an array with the order of axes reversed.
///
/// For a 2-D array, this is equivalent to a matrix transpose. For
/// higher-dimensional arrays, this reverses the axis order (e.g., for 3D with
/// axes [0, 1, 2], the result has axes [2, 1, 0]).
///
/// This is by definition equivalent to `transpose(None)` or `tensor.t()`.
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny)
///
///   - The input tensor whose axes are to be reversed.
///
/// # Returns
///
/// - [`TensorView<'_, T, B, D>`](TensorView)
///
///   - A view of the input tensor with reversed axis order.
///   - No data is copied; only the shape and strides are modified.
///
/// # Examples
///
/// For a 2-D array, this is equivalent to a matrix transpose:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
/// let result = a.reverse_axes();
/// println!("{result}");
/// // [[ 1 3]
/// //  [ 2 4]]
/// ```
///
/// For a 1-D array, this returns an unchanged view:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([1, 2, 3, 4], &device);
/// let result = a.reverse_axes();
/// println!("{result}");
/// // [ 1 2 3 4]
/// ```
///
/// For higher-dimensional arrays, the axis order is reversed:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // 3-D array: reverse_axes reverses all axis order
/// let a = rt::tensor_from_nested!([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], &device);
/// println!("{:?}", a.shape());
/// // [2, 2, 2]
/// let result = a.reverse_axes();
/// println!("{:?}", result.shape());
/// // [2, 2, 2]
/// // For [2,2,2] shape, reverse doesn't change shape but changes axis order
///
/// // 4-D array: reverse_axes shows clear shape change
/// let a: Tensor<i32, _> = rt::ones(([2, 3, 4, 5], &device));
/// let result = a.reverse_axes();
/// println!("{:?}", result.shape());
/// // [5, 4, 3, 2]
/// ```
///
/// # Notes of API accordance
///
/// - NumPy: `transpose(a)` or `a.T` ([`numpy.transpose`](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html))
/// - RSTSR: `tensor.reverse_axes()` or `tensor.t()`
///
/// Also note for multiple-dimensional arrays, `a.T` (NumPy) is equivalent to `a.reverse_axes()`
/// (RSTSR) (reverse all axes); but the `a.mT` (NumPy) is actually equivalent to `a.swapaxes(-1,
/// -2)` (RSTSR) (only swap the last two axes).
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`transpose`] - General axis permutation
/// - [`swapaxes`] - Swap two specific axes
/// - [`TensorAny::t()`] - Shorthand for reverse axes
///
/// ## Variants of this function
///
/// Note that this function is by definition infallible, so no fallible version is provided.
///
/// - [`reverse_axes`]: Returning a view.
/// - [`into_reverse_axes`]: Consuming version.
///
/// - Associated methods on `TensorAny`:
///
///   - [`TensorAny::reverse_axes`]
///   - [`TensorAny::into_reverse_axes`]
///   - [`TensorAny::t`] as shorthand for reverse axes
pub fn reverse_axes<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> TensorView<'_, T, B, D>
where
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_reverse_axes(tensor.view())
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Reverse the order of elements in an array along the given axis.
    ///
    /// # See also
    ///
    /// Refer to [`reverse_axes`] for details and examples.
    pub fn reverse_axes(&self) -> TensorView<'_, T, B, D> {
        into_reverse_axes(self.view())
    }

    /// Reverse the order of elements in an array along the given axis.
    ///
    /// # See also
    ///
    /// Refer to [`reverse_axes`] for details and examples.
    pub fn into_reverse_axes(self) -> TensorAny<R, T, B, D> {
        into_reverse_axes(self)
    }

    /// Reverse the order of elements in an array along the given axis.
    ///
    /// # See also
    ///
    /// Refer to [`reverse_axes`] for details and examples.
    pub fn t(&self) -> TensorView<'_, T, B, D> {
        into_reverse_axes(self.view())
    }
}

/* #endregion */

/* #region swapaxes */

/// Interchange two axes of an array.
///
/// # See also
///
/// Refer to [`swapaxes`] for details and examples.
pub fn into_swapaxes_f<I, S, D>(tensor: TensorBase<S, D>, axis1: I, axis2: I) -> Result<TensorBase<S, D>>
where
    D: DimAPI,
    I: TryInto<isize>,
{
    let axis1 = axis1.try_into().map_err(|_| rstsr_error!(TryFromIntError))?;
    let axis2 = axis2.try_into().map_err(|_| rstsr_error!(TryFromIntError))?;
    let (storage, layout) = tensor.into_raw_parts();
    let layout = layout.swapaxes(axis1, axis2)?;
    unsafe { Ok(TensorBase::new_unchecked(storage, layout)) }
}

/// Interchange two axes of an array.
///
/// Returns an array with two axes interchanged. No data is copied; only the
/// shape and strides are modified.
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny)
///
///   - The input tensor whose axes are to be swapped.
///
/// - `axis1`: `impl TryInto<isize>`
///
///   - First axis to be swapped.
///   - Negative values are supported and indicate counting dimensions from the back.
///
/// - `axis2`: `impl TryInto<isize>`
///
///   - Second axis to be swapped.
///   - Negative values are supported and indicate counting dimensions from the back.
///
/// # Returns
///
/// - [`TensorView<'_, T, B, D>`](TensorView)
///
///   - A view of the input tensor with the two axes interchanged.
///   - No data is copied; only the shape and strides are modified.
///
/// # Examples
///
/// For a 2-D array, swapping axes 0 and 1 is equivalent to transpose:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let x = rt::tensor_from_nested!([[1, 2, 3]], &device);
/// let result = x.swapaxes(0, 1);
/// println!("{result}");
/// // [[ 1]
/// //  [ 2]
/// //  [ 3]]
/// ```
///
/// For a 3-D array, swapping axes 0 and 2:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let x = rt::tensor_from_nested!([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], &device);
/// let result = x.swapaxes(0, 2);
/// println!("{result}");
/// // [[[ 0 4]
/// //   [ 2 6]]
/// //  [[ 1 5]
/// //   [ 3 7]]]
/// ```
///
/// Using negative indices to swap axes:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let x = rt::tensor_from_nested!([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], &device);
/// let result = x.swapaxes(-1, -3);
/// println!("{:?}", result.shape());
/// // [2, 2, 2]
/// ```
///
/// # Notes of API accordance
///
/// - NumPy: `swapaxes(a, axis1, axis2)` ([`numpy.swapaxes`](https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html))
/// - RSTSR: `tensor.swapaxes(axis1, axis2)` or `rt::swapaxes(&tensor, axis1, axis2)`
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`transpose`] - General axis permutation
/// - [`reverse_axes`] - Reverse all axes order
///
/// ## Variants of this function
///
/// - [`swapaxes`] / [`swapaxes_f`]: Returning a view.
/// - [`into_swapaxes`] / [`into_swapaxes_f`]: Consuming version.
///
/// - Associated methods on `TensorAny`:
///
///   - [`TensorAny::swapaxes`] / [`TensorAny::swapaxes_f`]
///   - [`TensorAny::into_swapaxes`] / [`TensorAny::into_swapaxes_f`]
pub fn swapaxes<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axis1: I, axis2: I) -> TensorView<'_, T, B, D>
where
    D: DimAPI,
    I: TryInto<isize>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_swapaxes_f(tensor.view(), axis1, axis2).rstsr_unwrap()
}

/// Interchange two axes of an array.
///
/// # See also
///
/// Refer to [`swapaxes`] for details and examples.
pub fn swapaxes_f<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axis1: I, axis2: I) -> Result<TensorView<'_, T, B, D>>
where
    D: DimAPI,
    I: TryInto<isize>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_swapaxes_f(tensor.view(), axis1, axis2)
}

/// Interchange two axes of an array.
///
/// # See also
///
/// Refer to [`swapaxes`] for details and examples.
pub fn into_swapaxes<I, S, D>(tensor: TensorBase<S, D>, axis1: I, axis2: I) -> TensorBase<S, D>
where
    D: DimAPI,
    I: TryInto<isize>,
{
    into_swapaxes_f(tensor, axis1, axis2).rstsr_unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Interchange two axes of an array.
    ///
    /// # See also
    ///
    /// Refer to [`swapaxes`] for details and examples.
    pub fn swapaxes<I>(&self, axis1: I, axis2: I) -> TensorView<'_, T, B, D>
    where
        I: TryInto<isize>,
    {
        swapaxes(self, axis1, axis2)
    }

    /// Interchange two axes of an array.
    ///
    /// # See also
    ///
    /// Refer to [`swapaxes`] for details and examples.
    pub fn swapaxes_f<I>(&self, axis1: I, axis2: I) -> Result<TensorView<'_, T, B, D>>
    where
        I: TryInto<isize>,
    {
        swapaxes_f(self, axis1, axis2)
    }

    /// Interchange two axes of an array.
    ///
    /// # See also
    ///
    /// Refer to [`swapaxes`] for details and examples.
    pub fn into_swapaxes<I>(self, axis1: I, axis2: I) -> TensorAny<R, T, B, D>
    where
        I: TryInto<isize>,
    {
        into_swapaxes(self, axis1, axis2)
    }

    /// Interchange two axes of an array.
    ///
    /// # See also
    ///
    /// Refer to [`swapaxes`] for details and examples.
    pub fn into_swapaxes_f<I>(self, axis1: I, axis2: I) -> Result<TensorAny<R, T, B, D>>
    where
        I: TryInto<isize>,
    {
        into_swapaxes_f(self, axis1, axis2)
    }
}

/* #endregion */
