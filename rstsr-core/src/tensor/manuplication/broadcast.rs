use crate::prelude_dev::*;

/* #region broadcast_shapes */

/// Broadcasts shapes against each other and returns the resulting shape.
///
/// This function computes the shape that results from broadcasting the given
/// shapes against one another according to the standard broadcasting rules.
/// It does not require actual tensor data, only the shapes.
///
/// <div class="warning">
///
/// **Row/Column Major Notice**
///
/// This function behaves differently on default orders ([`RowMajor`] and [`ColMajor`]) of device.
///
/// </div>
///
/// # Parameters
///
/// - `shapes`: A slice of shapes, where each shape is [`IxD`] (`Vec<usize>`).
/// - `order`: The memory order ([`RowMajor`] or [`ColMajor`]) that determines how broadcasting
///   rules are applied.
///
/// # Returns
///
/// - `IxD`: The resulting shape after broadcasting all input shapes together.
///
/// # Examples
///
/// Broadcast two shapes in row-major order:
///
/// ```rust
/// # use rstsr::prelude::*;
///
/// // A      (4d array):  8 x 1 x 6 x 1
/// // B      (3d array):      7 x 1 x 5
/// // ---------------------------------
/// // Result (4d array):  8 x 7 x 6 x 5
/// let shape1 = vec![8, 1, 6, 1];
/// let shape2 = vec![7, 1, 5];
/// let result = rt::broadcast_shapes(&[shape1, shape2], RowMajor);
/// println!("{:?}", result);
/// // [8, 7, 6, 5]
/// assert_eq!(result, vec![8, 7, 6, 5]);
/// ```
///
/// In col-major order, the broadcasting is applied from the left instead of right:
///
/// ```rust
/// # use rstsr::prelude::*;
///
/// // A      (4d array):  1 x 6 x 1 x 8
/// // B      (3d array):  5 x 1 x 7
/// // ---------------------------------
/// // Result (4d array):  5 x 6 x 7 x 8
/// let shape1 = vec![1, 6, 1, 8];
/// let shape2 = vec![5, 1, 7];
/// let result = rt::broadcast_shapes(&[shape1, shape2], ColMajor);
/// println!("{:?}", result);
/// // [5, 6, 7, 8]
/// assert_eq!(result, vec![5, 6, 7, 8]);
/// ```
///
/// Broadcast multiple shapes:
///
/// ```rust
/// # use rstsr::prelude::*;
///
/// // Three shapes: (1,), (3, 1), (3, 4) -> (3, 4)
/// let shapes = vec![vec![1], vec![3, 1], vec![3, 4]];
/// let result = rt::broadcast_shapes(&shapes, RowMajor);
/// println!("{:?}", result);
/// // [3, 4]
/// assert_eq!(result, vec![3, 4]);
/// ```
///
/// # Panics
///
/// - If the shapes cannot be broadcast together (incompatible dimensions).
///
/// # Notes of API accordance
///
/// - Array-API: `broadcast_shapes(*shapes)` ([`broadcast_shapes`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.broadcast_shapes.html))
/// - NumPy: `numpy.broadcast_shapes(*shapes)` ([`numpy.broadcast_shapes`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_shapes.html))
/// - RSTSR: `rt::broadcast_shapes(shapes, order)`
///
/// # See also
///
/// ## Variants of this function
///
/// - [`broadcast_shapes_f`]: Fallible version, actual implementation.
///
/// ## Related functions in RSTSR
///
/// - [`broadcast_arrays`]: Broadcasts any number of arrays against each other.
/// - [`broadcast_to`]: Broadcasts an array to a specified shape.
pub fn broadcast_shapes(shapes: &[IxD], order: FlagOrder) -> IxD {
    broadcast_shapes_f(shapes, order).rstsr_unwrap()
}

/// Broadcasts shapes against each other and returns the resulting shape.
///
/// # See also
///
/// Refer to [`broadcast_shapes`] for detailed documentation.
pub fn broadcast_shapes_f(shapes: &[IxD], order: FlagOrder) -> Result<IxD> {
    // fast return if empty or single shape
    if shapes.is_empty() {
        return Ok(vec![]);
    }
    if shapes.len() == 1 {
        return Ok(shapes[0].clone());
    }

    // iteratively broadcast all shapes
    let mut result = shapes[0].clone();
    for shape in shapes.iter().skip(1) {
        let (new_result, _, _) = broadcast_shape(shape, &result, order)?;
        result = new_result;
    }
    Ok(result)
}

/* #endregion */

/* #region broadcast_arrays */

/// Broadcasts any number of arrays against each other.
///
/// <div class="warning">
///
/// **Row/Column Major Notice**
///
/// This function behaves differently on default orders ([`RowMajor`] and [`ColMajor`]) of device.
///
/// </div>
///
/// # Parameters
///
/// - `tensors`: [`Vec<TensorAny<R, T, B, IxD>>`](TensorAny)
///
///   - The tensors to be broadcasted.
///   - All tensors must be on the same device, and share the same ownerships.
///   - This function takes ownership of the input tensors. If you want to obtain broadcasted views,
///     you need to create a new vector of views first.
///   - This function only accepts dynamic shape tensors ([`IxD`]).
///
/// # Returns
///
/// - [`Vec<TensorAny<R, T, B, IxD>>`](TensorAny)
///
///   - A vector of broadcasted tensors. Each tensor has the same shape after broadcasting.
///   - The ownership of the underlying data is moved from the input tensors to the output tensors.
///   - The tensors are typically not contiguous (with zero strides at the broadcasted axes).
///     Writing values to broadcasted tensors is dangerous, but RSTSR will generally not panic on
///     this behavior. Perform [`to_contig`] afterwards if requires owned contiguous tensors.
///
/// # Examples
///
/// The following example demonstrates how to use `broadcast_arrays` to broadcast two tensors:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
///
/// let a = rt::asarray((vec![1, 2, 3], &device)).into_shape([3]);
/// println!("{a}");
/// // [ 1 2 3]
/// let b = rt::asarray((vec![4, 5], &device)).into_shape([2, 1]);
/// println!("{b}");
/// // [[ 4]
/// //  [ 5]]
///
/// let result = rt::broadcast_arrays(vec![a, b]);
/// println!("broadcasted a:\n{:}", result[0]);
/// // [[ 1 2 3]
/// //  [ 1 2 3]]
/// println!("broadcasted b:\n{:}", result[1]);
/// // [[ 4 4 4]
/// //  [ 5 5 5]]
/// ```
///
/// Please note that the above code only works in [RowMajor].
///
/// For [ColMajor] order, the broadcasting will fail, because the broadcasting rules are applied
/// differently, shapes are incompatible (for col-major, broadcast comparison starts from left
/// instead of row-major's right):
///
/// ```rust,panic
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// device.set_default_order(ColMajor);
/// let a = rt::asarray((vec![1, 2, 3], &device)).into_shape([3]);
/// let b = rt::asarray((vec![4, 5], &device)).into_shape([2, 1]);
/// let result = rt::broadcast_arrays_f(vec![a, b]);
/// ```
///
/// You need to make the following changes to let [ColMajor] case work:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// device.set_default_order(ColMajor);
/// let a = rt::asarray((vec![1, 2, 3], &device)).into_shape([1, 3]);
/// let b = rt::asarray((vec![4, 5], &device)).into_shape([2, 1]);
/// let result = rt::broadcast_arrays_f(vec![a, b]);
/// ```
///
/// # Panics
///
/// - Incompatible shapes to be broadcasted.
/// - Tensors are on different devices.
///
/// # Notes of API accordance
///
/// - Array-API: `broadcast_arrays(*arrays)` ([`broadcast_arrays`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.broadcast_arrays.html))
/// - NumPy: `numpy.broadcast_arrays(*args, subok=False)` ([`numpy.broadcast_arrays`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_arrays.html))
/// - RSTSR: `rt::broadcast_arrays(tensors)`
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`to_broadcast`]: Broadcasts a single array to a specified shape.
///
/// ## Variants of this function
///
/// - [`broadcast_arrays_f`]: Fallible version, actual implementation.
pub fn broadcast_arrays<R, T, B>(tensors: Vec<TensorAny<R, T, B, IxD>>) -> Vec<TensorAny<R, T, B, IxD>>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    broadcast_arrays_f(tensors).rstsr_unwrap()
}

/// Broadcasts any number of arrays against each other.
///
/// # See also
///
/// Refer to [`broadcast_arrays`] for detailed documentation.
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

/// Broadcasts an array to a specified shape.
///
/// # See also
///
/// Refer to [`to_broadcast`] and [`into_broadcast`] for detailed documentation.
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
    // For broadcast_to, the result shape must match the target shape exactly
    // This ensures we cannot broadcast from higher dimensions to lower dimensions
    rstsr_assert_eq!(
        shape.as_ref(),
        shape2.as_ref(),
        InvalidLayout,
        "Cannot broadcast to a shape with fewer dimensions or incompatible size."
    )?;
    let (storage, layout) = tensor.into_raw_parts();
    let layout = update_layout_by_shape(&layout, &shape, &tp1, default_order)?;
    unsafe { Ok(TensorBase::new_unchecked(storage, layout)) }
}

/// Broadcasts an array to a specified shape.
///
/// <div class="warning">
///
/// **Row/Column Major Notice**
///
/// This function behaves differently on default orders ([`RowMajor`] and [`ColMajor`]) of device.
///
/// </div>
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny)
///
///   - The input tensor to be broadcasted.
///   - Note on variant [`into_broadcast`]: This takes ownership [`Tensor<R, T, B, D>`] of input
///     tensor, and will not perform change to underlying data, only layout changes.
///
/// - `shape`: impl [`DimAPI`]
///
///   - The shape of the desired output tensor after broadcasting.
///   - Please note [`IxD`] (`Vec<usize>`) and [`Ix<N>`] (`[usize; N]`) behaves differently here.
///     [`IxD`] will give dynamic shape tensor, while [`Ix<N>`] will give static shape tensor.
///     Unless you are handling static-shape tensors, always use [`IxD`] (`Vec<usize>`) to specify
///     shape.
///
/// # Returns
///
/// - [`TensorView<'_, T, B, D2>`]
///
///   - A readonly view on the original tensor with the given shape. It is typically not contiguous
///     (perform [`to_contig`] afterwards if you require contiguous owned tensors).
///   - Furthermore, more than one element of a broadcasted tensor may refer to a single memory
///     location (zero strides at the broadcasted axes). Writing values to broadcasted tensors is
///     dangerous, but RSTSR will generally not panic on this behavior.
///   - If you want to convert the tensor itself (taking the ownership instead of returning view),
///     use [`into_broadcast`] instead.
///
/// # Examples
///
/// The following example demonstrates how to use `to_broadcast` to broadcast a 1-D tensor
/// (3-element vector) to a 2-D tensor (2x3 matrix) by repeating the original data along a new axis.
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
///
/// // broadcast (3, ) -> (2, 3) in row-major:
/// let a = rt::tensor_from_nested!([1, 2, 3], &device);
/// let result = a.to_broadcast(vec![2, 3]);
/// println!("{result}");
/// // [[ 1 2 3]
/// //  [ 1 2 3]]
/// ```
///
/// Please note the above example is only working in RowMajor order. In ColMajor order, the
/// broadcasting will be done along the other axis:
///
///
///
/// ```rust
/// use rstsr::prelude::*;
/// let mut device = DeviceCpu::default();
/// device.set_default_order(ColMajor);
///
/// let a = rt::tensor_from_nested!([1, 2, 3], &device);
/// // in col-major, broadcast (3, ) -> (2, 3) will fail:
/// let result = a.to_broadcast_f(vec![2, 3]);
/// assert!(result.is_err());
///
/// // broadcast (3, ) -> (3, 2) in col-major:
/// let result = a.to_broadcast(vec![3, 2]);
/// println!("{result}");
/// // [[ 1 1]
/// //  [ 2 2]
/// //  [ 3 3]]
/// ```
///
/// # Panics
///
/// - Incompatible shapes to be broadcasted.
///
/// # Tips on common compilation errors
///
/// You may encounter compilation errors related to shape types like:
///
/// ```compile_fail
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let input_array: Tensor<i32, _> = rt::asarray((0, &device));
/// let result = rt::broadcast_to(&input_array, [3]);
/// ```
///
/// ```text
///  33 |         let result = rt::broadcast_to(&input_array, [3]);
///     |                      ---------------- ^^^^^^^^^^^^ expected `[usize; 1]`, found `Vec<usize>`
///     |                      |
///     |                      required by a bound introduced by this call
///     |
///     = note: expected array `[usize; 1]`
///               found struct `std::vec::Vec<usize>`
/// note: required by a bound in `rstsr_core::prelude_dev::to_broadcast`
///    --> rstsr-core/src/tensor/manuplication/broadcast.rs:427:31
///     |
/// 425 | pub fn to_broadcast<R, T, B, D, D2>(tensor: &TensorAny<R, T, B, D>, shape: D2) -> TensorView<'_, T, B, D2>
///     |        ------------ required by a bound in this function
/// 426 | where
/// 427 |     D: DimAPI + DimMaxAPI<D2, Max = D2>,
///     |                               ^^^^^^^^ required by this bound in `to_broadcast`
/// ```
///
/// To resolve this, you are suggested to use [`IxD`] (`Vec<usize>`) to specify the shape, instead
/// of [`Ix<N>`] (`[usize; N]`), unless you are an advanced API caller to handle static-shape
/// tensors.
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let input_array: Tensor<i32, _> = rt::asarray((0, &device));
/// let result = rt::broadcast_to(&input_array, vec![3]);
/// ```
///
/// # Elaborated examples
///
/// ## Broadcasting behavior (in row-major)
///
/// This example does not directly call this function `to_broadcast`, but demonstrates the
/// broadcasting behavior.
///
/// ```rust
/// use rstsr::prelude::*;
/// let mut device = DeviceCpu::default();
/// device.set_default_order(RowMajor);
///
/// // A      (4d tensor):  8 x 1 x 6 x 1
/// // B      (3d tensor):      7 x 1 x 5
/// // ----------------------------------
/// // Result (4d tensor):  8 x 7 x 6 x 5
/// let a = rt::arange((48, &device)).into_shape([8, 1, 6, 1]);
/// let b = rt::arange((35, &device)).into_shape([7, 1, 5]);
/// let result = &a + &b;
/// assert_eq!(result.shape(), &[8, 7, 6, 5]);
///
/// // A      (2d tensor):  5 x 4
/// // B      (1d tensor):      1
/// // --------------------------
/// // Result (2d tensor):  5 x 4
/// let a = rt::arange((20, &device)).into_shape([5, 4]);
/// let b = rt::arange((1, &device)).into_shape([1]);
/// let result = &a + &b;
/// assert_eq!(result.shape(), &[5, 4]);
///
/// // A      (2d tensor):  5 x 4
/// // B      (1d tensor):      4
/// // --------------------------
/// // Result (2d tensor):  5 x 4
/// let a = rt::arange((20, &device)).into_shape([5, 4]);
/// let b = rt::arange((4, &device)).into_shape([4]);
/// let result = &a + &b;
/// assert_eq!(result.shape(), &[5, 4]);
///
/// // A      (3d tensor):  15 x 3 x 5
/// // B      (3d tensor):  15 x 1 x 5
/// // -------------------------------
/// // Result (3d tensor):  15 x 3 x 5
/// let a = rt::arange((225, &device)).into_shape([15, 3, 5]);
/// let b = rt::arange((75, &device)).into_shape([15, 1, 5]);
/// let result = &a + &b;
/// assert_eq!(result.shape(), &[15, 3, 5]);
///
/// // A      (3d tensor):  15 x 3 x 5
/// // B      (2d tensor):       3 x 5
/// // -------------------------------
/// // Result (3d tensor):  15 x 3 x 5
/// let a = rt::arange((225, &device)).into_shape([15, 3, 5]);
/// let b = rt::arange((15, &device)).into_shape([3, 5]);
/// let result = &a + &b;
/// assert_eq!(result.shape(), &[15, 3, 5]);
///
/// // A      (3d tensor):  15 x 3 x 5
/// // B      (2d tensor):       3 x 1
/// // -------------------------------
/// // Result (3d tensor):  15 x 3 x 5
/// let a = rt::arange((225, &device)).into_shape([15, 3, 5]);
/// let b = rt::arange((3, &device)).into_shape([3, 1]);
/// let result = &a + &b;
/// assert_eq!(result.shape(), &[15, 3, 5]);
/// ```
///
/// ## Broadcasting behavior (in col-major)
///
/// This example does not directly call this function `to_broadcast`, but demonstrates the
/// broadcasting behavior.
///
/// ```rust
/// use rstsr::prelude::*;
/// let mut device = DeviceCpu::default();
/// device.set_default_order(ColMajor);
///
/// // A      (4d tensor):  1 x 6 x 1 x 8
/// // B      (3d tensor):  5 x 1 x 7
/// // ----------------------------------
/// // Result (4d tensor):  5 x 6 x 7 x 8
/// let a = rt::arange((48, &device)).into_shape([1, 6, 1, 8]);
/// let b = rt::arange((35, &device)).into_shape([5, 1, 7]);
/// let result = &a + &b;
/// assert_eq!(result.shape(), &[5, 6, 7, 8]);
///
/// // A      (2d tensor):  4 x 5
/// // B      (1d tensor):  1
/// // --------------------------
/// // Result (2d tensor):  4 x 5
/// let a = rt::arange((20, &device)).into_shape([4, 5]);
/// let b = rt::arange((1, &device)).into_shape([1]);
/// let result = &a + &b;
/// assert_eq!(result.shape(), &[4, 5]);
///
/// // A      (2d tensor):  4 x 5
/// // B      (1d tensor):  4
/// // --------------------------
/// // Result (2d tensor):  4 x 5
/// let a = rt::arange((20, &device)).into_shape([4, 5]);
/// let b = rt::arange((4, &device)).into_shape([4]);
/// let result = &a + &b;
/// assert_eq!(result.shape(), &[4, 5]);
///
/// // A      (3d tensor):  5 x 3 x 15
/// // B      (3d tensor):  5 x 1 x 15
/// // -------------------------------
/// // Result (3d tensor):  5 x 3 x 15
/// let a = rt::arange((225, &device)).into_shape([5, 3, 15]);
/// let b = rt::arange((75, &device)).into_shape([5, 1, 15]);
/// let result = &a + &b;
/// assert_eq!(result.shape(), &[5, 3, 15]);
///
/// // A      (3d tensor):  5 x 3 x 15
/// // B      (2d tensor):  5 x 3
/// // -------------------------------
/// // Result (3d tensor):  5 x 3 x 15
/// let a = rt::arange((225, &device)).into_shape([5, 3, 15]);
/// let b = rt::arange((15, &device)).into_shape([5, 3]);
/// let result = &a + &b;
/// assert_eq!(result.shape(), &[5, 3, 15]);
///
/// // A      (3d tensor):  5 x 3 x 15
/// // B      (2d tensor):  1 x 3
/// // -------------------------------
/// // Result (3d tensor):  5 x 3 x 15
/// let a = rt::arange((225, &device)).into_shape([5, 3, 15]);
/// let b = rt::arange((3, &device)).into_shape([1, 3]);
/// let result = &a + &b;
/// assert_eq!(result.shape(), &[5, 3, 15]);
/// ```
///
/// # See also
///
/// # Notes of API accordance
///
/// - Array-API: `broadcast_to(x, /, shape)` ([`broadcast_to`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.broadcast_to.html))
/// - NumPy: `numpy.broadcast_to(array, shape, subok=False)` ([`numpy.broadcast_to`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html))
/// - ndarray: [`ArrayBase::broadcast`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.broadcast)
/// - RSTSR: `rt::to_broadcast(tensor, shape)` or `rt::broadcast_to(tensor, shape)`
///
/// ## Detailed broadcasting rules
///
/// - Array-API: [Broadcasting rules](https://data-apis.org/array-api/latest/API_specification/broadcasting.html)
/// - NumPy: [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
///
/// ## Related functions in RSTSR
///
/// - [`broadcast_arrays`]: Broadcasts any number of arrays against each other.
///
/// ## Variants of this function
///
/// - [`to_broadcast`]: Standard version.
/// - [`to_broadcast_f`]: Fallible version.
/// - [`into_broadcast`]: Consuming version that takes ownership of the input tensor.
/// - [`into_broadcast_f`]: Consuming and fallible version, actual implementation.
/// - [`broadcast_to`]: Alias for `to_broadcast` (name of Python Array API standard).
/// - Associated methods on [`TensorAny`]:
///
///   - [`TensorAny::to_broadcast`]
///   - [`TensorAny::to_broadcast_f`]
///   - [`TensorAny::into_broadcast`]
///   - [`TensorAny::into_broadcast_f`]
///   - [`TensorAny::broadcast_to`]
pub fn to_broadcast<R, T, B, D, D2>(tensor: &TensorAny<R, T, B, D>, shape: D2) -> TensorView<'_, T, B, D2>
where
    D: DimAPI + DimMaxAPI<D2, Max = D2>,
    D2: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_broadcast_f(tensor.view(), shape).rstsr_unwrap()
}

/// Broadcasts an array to a specified shape.
///
/// # See also
///
/// Refer to [`to_broadcast`] for detailed documentation.
pub fn to_broadcast_f<R, T, B, D, D2>(tensor: &TensorAny<R, T, B, D>, shape: D2) -> Result<TensorView<'_, T, B, D2>>
where
    D: DimAPI + DimMaxAPI<D2, Max = D2>,
    D2: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_broadcast_f(tensor.view(), shape)
}

/// Broadcasts an array to a specified shape.
///
/// # See also
///
/// Refer to [`to_broadcast`] for detailed documentation.
pub fn into_broadcast<R, T, B, D, D2>(tensor: TensorAny<R, T, B, D>, shape: D2) -> TensorAny<R, T, B, D2>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI + DimMaxAPI<D2, Max = D2>,
    D2: DimAPI,
{
    into_broadcast_f(tensor, shape).rstsr_unwrap()
}

pub use into_broadcast as broadcast_into;
pub use into_broadcast_f as broadcast_into_f;
pub use to_broadcast as broadcast_to;
pub use to_broadcast_f as broadcast_to_f;

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
    /// Refer to [`to_broadcast`] for detailed documentation.
    pub fn to_broadcast<D2>(&self, shape: D2) -> TensorView<'_, T, B, D2>
    where
        D2: DimAPI,
        D: DimMaxAPI<D2, Max = D2>,
    {
        to_broadcast(self, shape)
    }

    /// Broadcasts an array to a specified shape.
    ///
    /// # See also
    ///
    /// Refer to [`to_broadcast`] for detailed documentation.
    pub fn broadcast_to<D2>(&self, shape: D2) -> TensorView<'_, T, B, D2>
    where
        D2: DimAPI,
        D: DimMaxAPI<D2, Max = D2>,
    {
        broadcast_to(self, shape)
    }

    /// Broadcasts an array to a specified shape.
    ///
    /// # See also
    ///
    /// Refer to [`to_broadcast`] for detailed documentation.
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
    /// Refer to [`to_broadcast`] for detailed documentation.
    pub fn into_broadcast<D2>(self, shape: D2) -> TensorAny<R, T, B, D2>
    where
        D2: DimAPI,
        D: DimMaxAPI<D2, Max = D2>,
    {
        into_broadcast(self, shape)
    }

    /// Broadcasts an array to a specified shape.
    ///
    /// # See also
    ///
    /// Refer to [`to_broadcast`] for detailed documentation.
    pub fn into_broadcast_f<D2>(self, shape: D2) -> Result<TensorAny<R, T, B, D2>>
    where
        D2: DimAPI,
        D: DimMaxAPI<D2, Max = D2>,
    {
        into_broadcast_f(self, shape)
    }
}

/* #endregion */
