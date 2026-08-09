use crate::prelude_dev::*;

/// Removes singleton dimensions (axes) from a tensor.
///
/// See also [`squeeze`].
pub fn into_squeeze_f<S, D>(
    tensor: TensorBase<S, D>,
    axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
) -> Result<TensorBase<S, IxD>>
where
    D: DimAPI,
{
    // convert axis to positive indexes and (reversed) sort
    let ndim: isize = TryInto::<isize>::try_into(tensor.ndim())?;
    let (storage, layout) = tensor.into_raw_parts();
    let mut layout = layout.into_dim::<IxD>()?;
    let axes = axes.try_into().map_err(Into::into)?;
    let axes = match axes {
        AxesIndex::None => {
            // find all axes with size 1
            let mut axes: Vec<isize> = Vec::new();
            for i in (0..ndim).rev() {
                if layout.shape()[i as usize] == 1 {
                    axes.push(i);
                }
            }
            axes
        },
        _ => {
            let mut axes: Vec<isize> = axes.as_ref().iter().map(|&v| if v >= 0 { v } else { v + ndim }).collect();
            axes.sort_by(|a, b| b.cmp(a));
            if axes.first().is_some_and(|&v| v < 0) {
                return Err(rstsr_error!(InvalidValue, "Some negative index is too small."));
            }
            // check no two axis are the same
            for i in 0..axes.len().saturating_sub(1) {
                rstsr_assert!(axes[i] != axes[i + 1], InvalidValue, "Same axes is not allowed here.")?;
            }
            axes
        },
    };
    // perform squeeze
    for &axis in axes.iter() {
        layout = layout.dim_eliminate(axis)?;
    }
    unsafe { Ok(TensorBase::new_unchecked(storage, layout)) }
}

/// Removes singleton dimensions (axes) from `x`.
///
/// See also [`squeeze`].
pub fn into_squeeze<S, D>(
    tensor: TensorBase<S, D>,
    axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
) -> TensorBase<S, IxD>
where
    D: DimAPI,
{
    into_squeeze_f(tensor, axes).rstsr_unwrap()
}

/// Removes singleton dimensions (axes) from a tensor.
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny)
///
///   - The input tensor.
///   - Note on variant [`into_squeeze`]: This takes ownership [`Tensor<R, T, B, D>`] of input
///     tensor, and will not perform change to underlying data, only layout changes.
///
/// - `axes`: TryInto [`AxesIndex<isize>`]
///
///   - The axis (or axes) to squeeze.
///   - If `axes` is a single integer, squeezing is performed along that axis.
///   - If `axes` is a tuple/list of integers, squeezing is performed on all specified axes.
///   - If `axes` is `None`, the function will squeeze all axes with size 1.
///   - If `axes` is an empty tuple `()`, no axes are squeezed.
///   - Negative values are supported and indicate counting dimensions from the back.
///   - Each axis in `axes` must have size 1; otherwise an error is raised.
///
/// # Returns
///
/// - [`TensorView<'_, T, B, IxD>`](TensorView)
///
///   - A view of the input tensor with the specified singleton dimensions removed.
///   - The underlying data is not copied; only the layout of the view is modified.
///   - If you want to convert the tensor itself (taking the ownership instead of returning view),
///     use [`into_squeeze`] instead.
///
/// # Panics
///
/// - If an axis specified does not have size 1.
/// - If an axis is out of bounds.
/// - If `axes` has duplicated values.
///
/// # Examples
///
/// ## Squeezing a single axis
///
/// Squeeze a tensor along axis 0:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a: Tensor<f64, _> = rt::zeros(([1, 3, 1, 4], &device));
/// let b = a.squeeze(0);
/// assert_eq!(b.shape(), &[3, 1, 4]);
/// ```
///
/// Squeeze a tensor along the axis 2 (third axis with size 1):
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// # let a: Tensor<f64, _> = rt::zeros(([1, 3, 1, 4], &device));
/// let b = a.squeeze(2);
/// assert_eq!(b.shape(), &[1, 3, 4]);
/// ```
///
/// Squeeze using negative index (-2 refers to the axis with size 1):
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// # let a: Tensor<f64, _> = rt::zeros(([1, 3, 1, 4], &device));
/// let b = a.squeeze(-2);
/// assert_eq!(b.shape(), &[1, 3, 4]);
/// ```
///
/// ## Squeezing multiple axes
///
/// Squeeze multiple axes at once:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// # let a: Tensor<f64, _> = rt::zeros(([1, 3, 1, 4, 1], &device));
/// let b = a.squeeze([0, 2]);
/// assert_eq!(b.shape(), &[3, 4, 1]);
/// ```
///
/// Use negative indices to squeeze from the back:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// # let a: Tensor<f64, _> = rt::zeros(([1, 3, 1, 4, 1], &device));
/// let b = a.squeeze([0, -1]);
/// assert_eq!(b.shape(), &[3, 1, 4]);
/// ```
///
/// ## Squeezing all singleton axes
///
/// Use `None` to squeeze all axes with size 1:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a: Tensor<f64, _> = rt::zeros(([1, 3, 1, 4, 1], &device));
/// let b = a.squeeze(None);
/// assert_eq!(b.shape(), &[3, 4]);
/// ```
///
/// ## No squeezing (empty axes)
///
/// Use an empty tuple `()` to squeeze no axes (returns a view of the original tensor):
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// # let a: Tensor<f64, _> = rt::zeros(([1, 3, 1, 4, 1], &device));
/// let b = a.squeeze(());
/// assert_eq!(b.shape(), &[1, 3, 1, 4, 1]);
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `squeeze(x, /, axis)` ([`squeeze`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.squeeze.html))
/// - NumPy: `squeeze(a, axis=None)` ([`numpy.squeeze`](https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html))
/// - RSTSR: `rt::squeeze(tensor, axes)`
///
/// RSTSR's behavior matches NumPy and Array-API:
/// - `a.squeeze(None)` squeezes all axes with size 1
/// - `a.squeeze(())` squeezes no axes (returns a view of the original tensor)
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`expand_dims`]: Adds singleton dimensions (axes) to a tensor.
///
/// ## Variants of this function
///
/// - [`squeeze`] / [`squeeze_f`]: Returning a view.
/// - [`into_squeeze`] / [`into_squeeze_f`]: Consuming version.
/// - Associated methods on [`TensorAny`]:
///
///   - [`TensorAny::squeeze`] / [`TensorAny::squeeze_f`]
///   - [`TensorAny::into_squeeze`] / [`TensorAny::into_squeeze_f`]
pub fn squeeze<R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
) -> TensorView<'_, T, B, IxD>
where
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_squeeze_f(tensor.view(), axes).rstsr_unwrap()
}

/// Removes singleton dimensions (axes) from a tensor.
///
/// See also [`squeeze`].
pub fn squeeze_f<R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
) -> Result<TensorView<'_, T, B, IxD>>
where
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_squeeze_f(tensor.view(), axes)
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Removes singleton dimensions (axes) from a tensor.
    ///
    /// See also [`squeeze`].
    pub fn squeeze(&self, axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>) -> TensorView<'_, T, B, IxD> {
        squeeze(self, axes)
    }

    /// Removes singleton dimensions (axes) from a tensor.
    ///
    /// See also [`squeeze`].
    pub fn squeeze_f(
        &self,
        axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    ) -> Result<TensorView<'_, T, B, IxD>> {
        squeeze_f(self, axes)
    }

    /// Removes singleton dimensions (axes) from a tensor.
    ///
    /// See also [`squeeze`].
    pub fn into_squeeze(self, axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>) -> TensorAny<R, T, B, IxD> {
        into_squeeze(self, axes)
    }

    /// Removes singleton dimensions (axes) from a tensor.
    ///
    /// See also [`squeeze`].
    pub fn into_squeeze_f(
        self,
        axes: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    ) -> Result<TensorAny<R, T, B, IxD>> {
        into_squeeze_f(self, axes)
    }
}
