use crate::prelude_dev::*;

/* #region reshape */

/// Reshapes the given tensor to the specified shape.
///
/// # See also
///
/// Refer to [`reshape`], [`into_shape`] and [`change_shape`] for more details and
/// examples.
pub fn change_shape_f<'a, I, R, T, B, D>(tensor: TensorAny<R, T, B, D>, shape: I) -> Result<TensorCow<'a, T, B, IxD>>
where
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    // own shape, this is cheap operation
    let shape_new = reshape_substitute_negatives(shape.try_into()?.as_ref(), tensor.size())?;
    let default_order = tensor.device().default_order();
    if let Some(layout_new) = layout_reshapeable(&tensor.layout().to_dim()?, &shape_new, default_order)? {
        // shape does not need to be changed
        let (storage, _) = tensor.into_raw_parts();
        let layout = layout_new.into_dim::<IxD>()?;
        return unsafe { Ok(TensorBase::new_unchecked(storage, layout).into_cow()) };
    } else {
        // clone underlying data by assign_arbitary
        let (storage, layout) = tensor.into_raw_parts();
        let device = storage.device();
        let layout_new = match default_order {
            RowMajor => shape_new.new_c_contig(None),
            ColMajor => shape_new.new_f_contig(None),
        };
        let mut storage_new = device.uninit_impl(layout_new.size())?;
        device.assign_arbitary_uninit(storage_new.raw_mut(), &layout_new, storage.raw(), &layout)?;
        let storage_new = unsafe { B::assume_init_impl(storage_new)? };
        return unsafe { Ok(TensorBase::new_unchecked(storage_new, layout_new).into_cow()) };
    }
}

/// Reshapes the given tensor to the specified shape.
///
/// This function is not intended to be used by usual users. Please consider using
/// [`reshape`] (take reference of tensor) or [`into_shape`] (take ownership of tensor)
/// instead.
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
/// - `tensor`: [`TensorAny<R, T, B, D>`]
///
///   - The input tensor to be reshaped.
///   - Ownership of input tensor is taken.
///
/// - `axes`: TryInto [`AxesIndex<isize>`]
///
///   - Position in the expanded axes where the new axis (or axes) is placed.
///   - Can be a single integer, or a list/tuple of integers.
///   - Negative values are supported and indicate counting dimensions from the back.
///
/// # Returns
///
/// - [`TensorCow<'a, T, B, IxD>`](TensorCow)
///
///   - The reshaped tensor.
///   - This function will try to avoid data cloning if possible.
///
///     - If layout-compatible, depending on whether the input tensor is owned or other cases,
///       either a view or owned tensor will be returned.
///     - If layout-not-compatible, an owned tensor will be returned, cloning the data.
///     - Cow (Clone-on-Write) semantics is used for representing either view or owned tensor.
///
/// This function is different to [`reshape`], in that it takes ownership of the input
/// tensor.
///
/// This function is also different to [`into_shape`], in that it may return a view, if the input
/// tensor also have the ownership of tensor view, and the layout is compatible.
///
/// # See also
///
/// Refer to [`reshape`] for more details and examples.
pub fn change_shape<'a, I, R, T, B, D>(tensor: TensorAny<R, T, B, D>, shape: I) -> TensorCow<'a, T, B, IxD>
where
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    change_shape_f(tensor, shape).rstsr_unwrap()
}

/// Reshapes the given tensor to the specified shape.
///
/// # See also
///
/// Refer to [`reshape`], [`into_shape`] and [`change_shape`] for more details and
/// examples.
pub fn into_shape_f<'a, I, R, T, B, D>(tensor: TensorAny<R, T, B, D>, shape: I) -> Result<Tensor<T, B, IxD>>
where
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, IxD, D>
        + OpAssignAPI<T, IxD>,
    <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
{
    change_shape_f(tensor, shape).map(|v| v.into_owned())
}

/// Reshapes the given tensor to the specified shape.
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
/// - `tensor`: [`TensorAny<R, T, B, D>`]
///
///   - The input tensor to be reshaped.
///   - Ownership of input tensor is taken.
///
/// - `axes`: TryInto [`AxesIndex<isize>`]
///
///   - Position in the expanded axes where the new axis (or axes) is placed.
///   - Can be a single integer, or a list/tuple of integers.
///   - Negative values are supported and indicate counting dimensions from the back.
///
/// # Returns
///
/// - [`Tensor<T, B, IxD>`]
///
///   - The reshaped tensor.
///   - This function will try to avoid data cloning if possible, but with strict conditions:
///
///     - Layout-compatible after reshaping;
///     - Input tensor owns the underlying data (i.e., not a view);
///     - The input tensor is compact in memory (i.e., the underlying data does not have redundant
///       elements; size of tensor exactly matches the length of underlying data).
///
/// This function is different to [`change_shape`](change_shape()) and [`reshape`], in
/// that it takes ownership of the input tensor, and always returns an owned tensor.
///
/// # Examples
///
/// ```rust
/// use rstsr::prelude::*;
/// let a = rt::arange(6).into_shape([2, 3]);
/// ```
///
/// # Elaborated examples
///
/// Here is some showcases that demonstrate when data cloning happens or not. All examples are
/// row-major.
///
/// A first case is a tensor that is not fully contiguous (containing negative strides), but the
/// tensor is compact (size of tensor is the same to the length of underlying data). In this case,
/// if the new shape is compatible, no data cloning happens:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // shape: (4, 6, 9), stride: (-54, 9, 1), not c-contiguous
/// // contiguous situation: (4, [6, 9]); the first dimension is reversed
/// let a = rt::arange((216, &device)).into_shape([4, 6, 9]).into_flip(0);
/// let a_ptr = a.raw().as_ptr();
/// let b = a.into_shape([4, 54]);
/// let b_ptr = b.raw().as_ptr();
/// assert_eq!(a_ptr, b_ptr); // contiguous dims merged, no data clone happened
/// ```
///
/// However, if the new shape is not compatible, data cloning will happen:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // shape: (4, 6, 9), stride: (-54, 9, 1), not c-contiguous
/// // contiguous situation: (4, [6, 9]); the first dimension is reversed
/// let a = rt::arange((216, &device)).into_shape([4, 6, 9]).into_flip(0);
/// let a_ptr = a.raw().as_ptr();
/// let b = a.into_shape([24, 9]);
/// let b_ptr = b.raw().as_ptr();
/// assert_ne!(a_ptr, b_ptr); // layout not compatible, data clone happened
/// ```
///
/// Another case is a tensor that is not compact (size of tensor is less than the length of
/// underlying data). In this case, even if the new shape is compatible, data cloning will happen:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // shape: (4, 6, 9), stride: (72, 9, 1), not c-contiguous
/// // contiguous situation: (4, [6, 9]), or say the last two dimensions are contiguous
/// let a = rt::arange((288, &device)).into_shape([4, 8, 9]).into_slice((.., 0..6, ..));
/// let a_ptr = a.raw().as_ptr();
/// let b = a.into_shape([4, 54]);
/// let b_ptr = b.raw().as_ptr();
/// assert_ne!(a_ptr, b_ptr); // layout-compatible, but input tensor is not compact (216 < 288)
/// ```
///
/// # See also
///
/// Refer to [`reshape`] for more details and examples.
pub fn into_shape<'a, I, R, T, B, D>(tensor: TensorAny<R, T, B, D>, shape: I) -> Tensor<T, B, IxD>
where
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, IxD, D>
        + OpAssignAPI<T, IxD>,
    <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
{
    into_shape_f(tensor, shape).rstsr_unwrap()
}

/// Reshapes the given tensor to the specified shape.
///
/// # See also
///
/// Refer to [`reshape`], [`into_shape`] and [`change_shape`] for more details and
/// examples.
pub fn to_shape_f<'a, I, R, T, B, D>(tensor: &'a TensorAny<R, T, B, D>, shape: I) -> Result<TensorCow<'a, T, B, IxD>>
where
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    change_shape_f(tensor.view(), shape)
}

/// Reshapes the given tensor to the specified shape.
///
/// # See also
///
/// Refer to [`reshape`], [`into_shape`] and [`change_shape`] for more details and
/// examples.
pub fn to_shape<'a, I, R, T, B, D>(tensor: &'a TensorAny<R, T, B, D>, shape: I) -> TensorCow<'a, T, B, IxD>
where
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    to_shape_f(tensor, shape).rstsr_unwrap()
}

/// Reshapes the given tensor to the specified shape.
///
/// # See also
///
/// Refer to [`reshape`], [`into_shape`] and [`change_shape`] for more details and
/// examples.
pub fn reshape_f<'a, I, R, T, B, D>(tensor: &'a TensorAny<R, T, B, D>, shape: I) -> Result<TensorCow<'a, T, B, IxD>>
where
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    to_shape_f(tensor, shape)
}

/// Reshapes the given tensor to the specified shape.
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
///   - The input tensor to be reshaped.
///
/// - `axes`: TryInto [`AxesIndex<isize>`]
///
///   - Position in the expanded axes where the new axis (or axes) is placed.
///   - Can be a single integer, or a list/tuple of integers.
///   - Negative values are supported and indicate counting dimensions from the back.
///
/// # Returns
///
/// - [`TensorCow<'a, T, B, IxD>`](TensorCow)
///
///   - The reshaped tensor.
///   - This function will try to avoid data cloning if possible.
///
///     - If layout-compatible, a view will be returned.
///     - If shape-not-compatible, an owned tensor will be returned, cloning the data.
///     - Cow (Clone-on-Write) semantics is used for representing either view or owned tensor.
///
/// # Examples
///
/// In row-major order, to reshape a vector of (6, ) to a matrix of (2, 3):
/// ```rust
/// use rstsr::prelude::*;
/// let mut device = DeviceCpu::default();
/// device.set_default_order(RowMajor);
///
/// let a = rt::arange((6, &device));
/// let a_reshaped = a.reshape([2, 3]);
/// let a_expected = rt::tensor_from_nested!(
///     [[0, 1, 2], [3, 4, 5]],
///     &device);
/// assert!(rt::allclose(&a_reshaped, &a_expected, None));
/// ```
///
/// You can also use negative dimension, where -1 means "infer this dimension":
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// #
/// // in this case, unspecified axes length is inferred as 6 / 3 = 2
/// let a = rt::arange((6, &device));
/// let a_reshaped = a.reshape([3, -1]);
/// let a_expected = rt::tensor_from_nested!(
///     [[0, 1], [2, 3], [4, 5]],
///     &device);
/// assert!(rt::allclose(&a_reshaped, &a_expected, None));
/// ```
///
/// # Ownership Semantics between [`reshape`], [`into_shape`] and [`change_shape`]
///
/// [`into_shape`] and [`change_shape`] take ownership of the input tensor. They are important
/// variants to this function [`reshape`].
///
/// | Function | Input Ownership | Output Ownership | Cloning Condition |
/// |--|--|--|--|
/// | [`reshape`] | Borrowed <br> [`&TensorAny`](TensorAny) | View <br> [`TensorCow`] with [`DataCow::Ref`] | not cloned (layout-compatible) |
/// | | | Owned <br> [`TensorCow`] with [`DataCow::Owned`] | cloned (layout-not-compatible) |
/// | [`into_shape`] | Owned <br> [`Tensor`] | Owned <br> [`Tensor`] | not cloned (layout-compatible, input tensor owns data, input tensor is compact) |
/// | | | Owned <br> [`Tensor`] | cloned (otherwise) |
/// | | Otherwise <br> [`TensorAny`] | Owned <br> [`Tensor`] | cloned (always) |
/// | [`change_shape`] | Owned <br> [`Tensor`] | Owned <br> [`TensorCow`] with [`DataCow::Owned`] | not cloned (layout-compatible, input tensor owns data, input tensor is compact) |
/// | | | Owned <br> [`TensorCow`] with [`DataCow::Owned`] | cloned (otherwise) |
/// | | Otherwise <br> [`TensorAny`] | View <br> [`TensorCow`] with [`DataCow::Ref`] | not cloned (layout-compatible) |
/// | | | Owned <br> [`TensorCow`] with [`DataCow::Owned`] | cloned (layout-not-compatible) |
///
/// # Tips on common compilation errors
///
/// You may encounter ownership problem when you try to assign a reshaped tensor like this:
///
/// ```rust,compile_fail
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((6, &device)).reshape([2, 3]);
/// println!("a: {:?}", a);
/// ```
///
/// The compiler may give an error like:
///
/// ```text
/// 704 |    let a = rt::arange((6, &device)).reshape([2, 3]);
///     |            ^^^^^^^^^^^^^^^^^^^^^^^^                - temporary value is freed at the end of this statement
///     |            |
///     |            creates a temporary value which is freed while still in use
/// 705 |    println!("a: {:?}", a);
///     |                        - borrow later used here
///     |
/// help: consider using a `let` binding to create a longer lived value
///     |
/// 704 ~    let binding = rt::arange((6, &device));
/// 705 ~    let a = binding.reshape([2, 3]);
///     |
/// ```
///
/// The suggestion by compiler is correct. However, you have another simpler way to solve this
/// problem by using [`into_shape`] variant that takes ownership:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((6, &device)).into_shape([2, 3]);
/// ```
///
/// # Notes of accordance
///
/// ## To Python Array API Standard
///
/// This function corresponds to Python Array API Standard:
/// [`reshape`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.reshape.html).
///
/// However, please note that this function does not implement the optional keyword `copy` as in
/// the standard. `copy` keyword in the standard specifies whether to return a copy of the array
/// data when the requested shape is not compatible with the original shape.
///
/// This function implements `copy = None` behavior in the standard, which means that it will return
/// a view if possible, and return an owned tensor (cloning the data) if necessary.
///
/// To achieve similar functionality of optional keyword `copy`,
///
/// - For `copy = True` case, you are recommended to
///
///   - use [`into_shape`], which always returns an owned tensor, cloning the data if necessary. But
///     note the necessity of cloning depends on the layout, and RSTSR may still not explicitly
///     perform cloning.
///   - use [`to_contig`], which always returns a contiguous owned tensor, cloning the data if
///     necessary. But note that this function may still not explicitly perform cloning if the
///     tensor is already contiguous.
///   - use [`to_owned`](TensorAny::to_owned) as associated method to give an owned tensor, which
///     always perform cloning.
///
/// - For `copy = False` case, you are recommended to
///
///   - use utility function [`layout_reshapeable`] to check whether the layout is compatible with
///     the new shape.
///
/// ## To NumPy
///
/// This function corresponds to NumPy:
/// [`reshape`](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html).
///
/// However, please note that this function does not implement the optional keyword `order` as in
/// the NumPy version. `order` keyword in NumPy specifies the iteration order to read elements from
/// the tensor to-be-reshaped.
///
/// This function uses the device's default order to determine the layout of the reshaped tensor.
/// You can check the device's current default order by
/// [`device.default_order`](DeviceBaseAPI::default_order). Also see the elaborated examples below.
///
/// To change the device's default order, you can use
///
/// - [`device.set_default_order`](DeviceBaseAPI::set_default_order) to set the default order of a
///   device instance, and then
/// - [`change_device`](TensorDeviceChangeAPI::change_device) or
///   [`into_device`](TensorDeviceChangeAPI::into_device) or
///   [`to_device`](TensorDeviceChangeAPI::to_device) to change the tensor's device to the modified
///   device. Choose the appropriate method depending on the desired ownership semantics.
///
/// # Elaborated examples
///
/// ## Difference between [RowMajor] and [ColMajor]
///
/// Tensor can be uniquely iterated (into a 1-dimension vector), for either row-major or
/// column-major order.
///
/// **Reshape operation does not change the iterated sequence of a tensor**, by definition. In other
/// words, the following code always holds true:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(ColMajor);
/// # let a = rt::tensor_from_nested!([[0, 1, 2], [3, 4, 5]], &device);
/// # let b = a.reshape([3, 2]);
/// // note iteration order of associated method `iter` depends on `device.default_order()`
///
/// // let b = a.reshape(... SOME SHAPE ...);
/// let a_vec = a.iter().collect::<Vec<_>>();
/// let b_vec = b.iter().collect::<Vec<_>>();
/// assert_eq!(a_vec, b_vec); // iterated sequence is the same
/// ```
///
/// For example, in row-major order, reshape a matrix of (2, 3) to (3, 2):
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// // set to row-major order
/// device.set_default_order(RowMajor);
/// // a: [[0, 1, 2], [3, 4, 5]]
/// // b: [[0, 1], [2, 3], [4, 5]]
/// // iterated sequence: [0, 1, 2, 3, 4, 5]
/// let a = rt::tensor_from_nested!([[0, 1, 2], [3, 4, 5]], &device);
/// let b = a.reshape([3, 2]);
/// let b_expected = rt::tensor_from_nested!([[0, 1], [2, 3], [4, 5]], &device);
/// assert!(rt::allclose(&b, &b_expected, None));
/// let a_vec = a.iter().cloned().collect::<Vec<_>>();
/// let b_vec = b.iter().cloned().collect::<Vec<_>>();
/// assert_eq!(a_vec, b_vec); // iterated sequence is the same
/// assert_eq!(a_vec, vec![0, 1, 2, 3, 4, 5]);
/// ```
///
/// In the column-major order, reshape the same matrix of (2, 3) to (3, 2) will yield a different
/// result:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// // set to column-major order
/// device.set_default_order(ColMajor);
/// // a: [[0, 1, 2], [3, 4, 5]]
/// // b: [[0, 4], [3, 2], [1, 5]]
/// // iterated sequence: [0, 3, 1, 4, 2, 5]
/// let a = rt::tensor_from_nested!([[0, 1, 2], [3, 4, 5]], &device);
/// let b = a.reshape([3, 2]);
/// let b_expected = rt::tensor_from_nested!([[0, 4], [3, 2], [1, 5]], &device);
/// assert!(rt::allclose(&b, &b_expected, None));
/// let a_vec = a.iter().cloned().collect::<Vec<_>>();
/// let b_vec = b.iter().cloned().collect::<Vec<_>>();
/// assert_eq!(a_vec, b_vec); // iterated sequence is the same
/// assert_eq!(a_vec, vec![0, 3, 1, 4, 2, 5]);
/// ```
///
/// ## Occasions of data cloning
///
/// The following discussion assumes the tensor is in row-major order. Similar discussion applies to
/// column-major order.
///
/// If the tensor to be reshaped is already in C-contiguous if the device is also row-major, or
/// F-contiguous if the device is column-major, then the reshape operation can be performed without
/// any data cloning.
///
/// Otherwise, whether data cloning is necessary depends. For example, consider a tensor of shape
/// (4, 6, 9) but with non-contiguous strides:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // shape: (4, 6, 9), stride: (72, 9, 1), not c-contiguous
/// // contiguous situation: (4, [6, 9]), or say the last two dimensions are contiguous
/// let a = rt::arange((288, &device)).into_shape([4, 8, 9]).into_slice((.., 0..6, ..));
/// assert_eq!(a.shape(), &[4, 6, 9]);
/// assert_eq!(a.stride(), &[72, 9, 1]);
/// assert!(!a.c_contig());
/// ```
///
/// Those cases will not require data cloning (returns a view, or [`DataCow::Ref`] internally):
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// # let a = rt::arange((288, &device)).into_shape([4, 8, 9]).into_slice((.., 0..6, ..));
/// // split a single dimension into multiple dimensions
/// assert!(!a.reshape([2, 2, 6, 9]).is_owned()); // (4, 6, 9) -> ([2, 2], 6, 9)
/// assert!(!a.reshape([4, 3, 2, 9]).is_owned()); // (4, 6, 9) -> (4, [3, 2], 9)
/// assert!(!a.reshape([4, 2, 3, 3, 3]).is_owned()); // (4, 6, 9) -> (4, [2, 3], [3, 3])
///
/// // merge contiguous dimensions into a single dimension
/// assert!(!a.reshape([4, 54]).is_owned()); // (4, 6, 9) -> (4, 6 * 9)
///
/// // merge contiguous dimensions and then split
/// assert!(!a.reshape([4, 3, 6, 3]).is_owned()); // (4, [6, 9]) -> (4, [3, 6, 3])
/// ```
///
/// However, the following cases will require data cloning (returns an owned tensor, or
/// [`DataCow::Owned`] internally):
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// # let a = rt::arange((288, &device)).into_shape([4, 8, 9]).into_slice((.., 0..6, ..));
/// assert!(a.reshape([24, 9]).is_owned()); // (4, 6, 9) -> (4 * 6, 9)
/// assert!(a.reshape(-1).is_owned()); // (4, 6, 9) -> (4 * 6 * 9)
/// assert!(a.reshape([12, 2, 9]).is_owned()); // (4, 6, 9) -> (4 * [3, 2], 9)
/// ```
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - Python Array API standard: [`reshape`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.reshape.html)
/// - NumPy: [`reshape`](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)
/// - ndarray: [`to_shape`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.to_shape)
///
/// ## Related functions in RSTSR
///
/// - [`reshape_assume_contig`]: Reshape assuming the tensor is contiguous.
/// - [`layout_reshapeable`]: Check whether the layout is compatible with the new shape.
/// - [`to_layout`]: Return a tensor with the specified layout.
/// - [`to_contig`]: Return an owned contiguous tensor.
///
/// ## Variants of this function
///
/// - [`reshape`] / [`reshape_f`]: Taking reference and returning Cow.
/// - [`into_shape`] / [`into_shape_f`]: Taking ownership and returning owned tensor.
/// - [`change_shape`] / [`change_shape_f`]: Taking ownership and returning Cow.
/// - [`to_shape`] / [`to_shape_f`]: Alias to [`reshape`] / [`reshape_f`].
/// - Associated methods on [`TensorAny`]:
///
///   - [`TensorAny::reshape`] / [`TensorAny::reshape_f`]
///   - [`TensorAny::into_shape`] / [`TensorAny::into_shape_f`]
///   - [`TensorAny::change_shape`] / [`TensorAny::change_shape_f`]
///   - [`TensorAny::to_shape`] / [`TensorAny::to_shape_f`]
pub fn reshape<'a, I, R, T, B, D>(tensor: &'a TensorAny<R, T, B, D>, shape: I) -> TensorCow<'a, T, B, IxD>
where
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    to_shape(tensor, shape)
}

/// Reshapes the given tensor to the specified shape.
///
/// # See also
///
/// Refer to [`reshape`], [`into_shape`] and [`change_shape`] for more details and
/// examples.
impl<'a, R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
    T: Clone,
{
    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also
    ///
    /// Refer to [`reshape`], [`into_shape`] and [`change_shape`] for more details and
    /// examples.
    pub fn change_shape_f<I>(self, shape: I) -> Result<TensorCow<'a, T, B, IxD>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        change_shape_f(self, shape)
    }

    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also
    ///
    /// Refer to [`reshape`], [`into_shape`] and [`change_shape`] for more details and
    /// examples.
    pub fn change_shape<I>(self, shape: I) -> TensorCow<'a, T, B, IxD>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        change_shape(self, shape)
    }

    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also
    ///
    /// Refer to [`reshape`], [`into_shape`] and [`change_shape`] for more details and
    /// examples.
    pub fn into_shape_f<I>(self, shape: I) -> Result<Tensor<T, B, IxD>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
        B: OpAssignAPI<T, IxD>,
    {
        into_shape_f(self, shape)
    }

    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also
    ///
    /// Refer to [`reshape`], [`into_shape`] and [`change_shape`] for more details and
    /// examples.
    pub fn into_shape<I>(self, shape: I) -> Tensor<T, B, IxD>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
        B: OpAssignAPI<T, IxD>,
    {
        into_shape(self, shape)
    }

    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also
    ///
    /// Refer to [`reshape`], [`into_shape`] and [`change_shape`] for more details and
    /// examples.
    pub fn to_shape_f<I>(&'a self, shape: I) -> Result<TensorCow<'a, T, B, IxD>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        self.view().change_shape_f(shape)
    }

    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also
    ///
    /// Refer to [`reshape`], [`into_shape`] and [`change_shape`] for more details and
    /// examples.
    pub fn to_shape<I>(&'a self, shape: I) -> TensorCow<'a, T, B, IxD>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        self.view().change_shape(shape)
    }

    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also
    ///
    /// Refer to [`reshape`], [`into_shape`] and [`change_shape`] for more details and
    /// examples.
    pub fn reshape_f<I>(&'a self, shape: I) -> Result<TensorCow<'a, T, B, IxD>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        self.view().change_shape_f(shape)
    }

    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also
    ///
    /// Refer to [`reshape`], [`into_shape`] and [`change_shape`] for more details and
    /// examples.
    pub fn reshape<I>(&'a self, shape: I) -> TensorCow<'a, T, B, IxD>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        self.view().change_shape(shape)
    }
}

/* #endregion */
