use crate::prelude_dev::*;

/* #region to_contig */

/// Convert tensor to contiguous layout, consuming the tensor and returning a
/// copy-on-write tensor.
///
/// This function ensures that the returned tensor has data stored in contiguous
/// memory with the specified order (row-major/C or column-major/F). If the input
/// tensor is already contiguous with the requested order, no data copy occurs and
/// a view is returned. Otherwise, data is copied to a new contiguous layout.
///
/// # Arguments
///
/// - `tensor`: The input tensor to make contiguous. This function takes ownership of the tensor.
/// - `order`: The memory layout order. Use [`RowMajor`] (or [`C`]) for C-contiguous (row-major)
///   layout, or [`ColMajor`] (or [`F`]) for F-contiguous (column-major) layout.
///
/// # Returns
///
/// A [`TensorCow`] containing either a view (if no copy was needed) or an owned
/// tensor (if data was copied).
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // Create a non-contiguous tensor by slicing
/// let a = rt::arange((12, &device)).into_shape([3, 4]);
/// let sliced = a.i((.., ..;2)); // Every other column
/// println!("Original sliced shape: {:?}", sliced.shape());
/// println!("Original sliced stride: {:?}", sliced.stride());
/// // [3, 2]
/// // [4, 2]
///
/// // Convert to C-contiguous
/// let contig = sliced.change_contig(RowMajor);
/// println!("Contiguous shape: {:?}", contig.shape());
/// println!("Contiguous stride: {:?}", contig.stride());
/// // [3, 2]
/// // [2, 1]
/// ```
///
/// # See Also
///
/// - [`to_contig`] - Non-consuming version that takes a reference
/// - [`into_contig`] - Returns an owned tensor directly
/// - [`change_prefer`] - Only converts if not already in preferred layout
/// - [`to_layout`] - Convert to a specific layout
pub fn change_contig_f<'a, R, T, B, D>(
    tensor: TensorAny<R, T, B, D>,
    order: FlagOrder,
) -> Result<TensorCow<'a, T, B, D>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    let shape = tensor.shape();
    let layout_new = match order {
        RowMajor => shape.new_c_contig(None),
        ColMajor => shape.new_f_contig(None),
    };
    change_layout_f(tensor, layout_new)
}

/// Convert tensor to contiguous layout, returning a view or copied tensor.
///
/// This function takes a reference to a tensor and returns a [`TensorCow`] that is
/// either a view (if the tensor is already contiguous with the requested order) or
/// a newly allocated contiguous copy.
///
/// # Arguments
///
/// - `tensor`: A reference to the input tensor.
/// - `order`: The memory layout order ([`RowMajor`] or [`ColMajor`]).
///
/// # Returns
///
/// A [`TensorCow`] containing either a view or an owned tensor with contiguous layout.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // Create a transposed tensor (non-contiguous)
/// let a = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6]], &device);
/// let transposed = a.t(); // transpose creates non-contiguous view
///
/// // Check contiguity
/// assert!(!transposed.c_contig());
///
/// // Convert to C-contiguous
/// let contig = rt::to_contig(&transposed, RowMajor);
/// assert!(contig.c_contig());
/// println!("{}", contig);
/// // [[ 1 4]
/// //  [ 2 5]
/// //  [ 3 6]]
/// ```
///
/// # See Also
///
/// - [`to_contig_f`] - Fallible version
/// - [`change_contig`] - Consuming version that takes ownership
/// - [`into_contig`] - Returns an owned tensor directly
/// - [`to_prefer`] - Only converts if not already in preferred layout
pub fn to_contig<R, T, B, D>(tensor: &TensorAny<R, T, B, D>, order: FlagOrder) -> TensorCow<'_, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    to_contig_f(tensor, order).rstsr_unwrap()
}

/// Fallible version of [`to_contig`].
///
/// # See Also
///
/// - [`to_contig`] - Infallible version
pub fn to_contig_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>, order: FlagOrder) -> Result<TensorCow<'_, T, B, D>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    change_contig_f(tensor.view(), order)
}

/// Convert tensor to contiguous layout, consuming the tensor and returning an owned tensor.
///
/// This is similar to [`change_contig`], but always returns an owned [`Tensor`].
/// If the input tensor is already contiguous with the requested order, the data
/// may be moved without copying. Otherwise, data is copied to a new contiguous layout.
///
/// # Arguments
///
/// - `tensor`: The input tensor to make contiguous. This function takes ownership.
/// - `order`: The memory layout order ([`RowMajor`] or [`ColMajor`]).
///
/// # Returns
///
/// An owned [`Tensor`] with contiguous layout.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // Create a non-contiguous tensor
/// let a = rt::arange((6, &device)).into_shape([2, 3]);
/// let sliced = a.i((.., ..;2)); // shape [2, 2], stride [3, 2]
///
/// // Convert to owned F-contiguous tensor
/// let contig = sliced.into_contig(ColMajor);
/// assert!(contig.f_contig());
/// println!("Shape: {:?}", contig.shape());
/// println!("Stride: {:?}", contig.stride());
/// // Shape: [2, 2]
/// // Stride: [1, 2]
/// ```
///
/// # See Also
///
/// - [`to_contig`] - Non-consuming version returning [`TensorCow`]
/// - [`change_contig`] - Consuming version returning [`TensorCow`]
/// - [`into_prefer`] - Only converts if not already in preferred layout
pub fn into_contig_f<'a, R, T, B, D>(tensor: TensorAny<R, T, B, D>, order: FlagOrder) -> Result<Tensor<T, B, D>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, D, D>
        + OpAssignAPI<T, D>,
    <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
{
    change_contig_f(tensor, order).map(|v| v.into_owned())
}

/// Infallible version of [`change_contig_f`].
///
/// # Panics
///
/// Panics if an error occurs during the conversion.
///
/// # See Also
///
/// - [`change_contig_f`] - Fallible version
/// - [`to_contig`] - Non-consuming version
pub fn change_contig<'a, R, T, B, D>(tensor: TensorAny<R, T, B, D>, order: FlagOrder) -> TensorCow<'a, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    change_contig_f(tensor, order).rstsr_unwrap()
}

/// Infallible version of [`into_contig_f`].
///
/// # Panics
///
/// Panics if an error occurs during the conversion.
///
/// # See Also
///
/// - [`into_contig_f`] - Fallible version
/// - [`to_contig`] - Non-consuming version returning [`TensorCow`]
pub fn into_contig<'a, R, T, B, D>(tensor: TensorAny<R, T, B, D>, order: FlagOrder) -> Tensor<T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, D, D>
        + OpAssignAPI<T, D>,
    <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
{
    into_contig_f(tensor, order).rstsr_unwrap()
}

impl<'a, R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    /// Convert tensor to contiguous layout.
    ///
    /// This method ensures that the tensor's data is stored in contiguous memory
    /// with the specified order. If the tensor is already contiguous with the
    /// requested order, no data copy occurs.
    ///
    /// # Arguments
    ///
    /// - `order`: The memory layout order ([`RowMajor`] or [`ColMajor`]).
    ///
    /// # Returns
    ///
    /// A [`TensorCow`] containing either a view or an owned tensor with contiguous layout.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// // Create a strided (non-contiguous) tensor
    /// let a = rt::arange((12, &device)).into_shape([3, 4]);
    /// let strided = a.i((..;2, ..)); // Every other row
    /// println!("Strided stride: {:?}", strided.stride());
    /// // [8, 1]
    ///
    /// // Convert to contiguous
    /// let contig = strided.to_contig(RowMajor);
    /// println!("Contiguous stride: {:?}", contig.stride());
    /// // [4, 1]
    /// ```
    ///
    /// # See Also
    ///
    /// - [`TensorAny::to_contig_f`] - Fallible version
    /// - [`TensorAny::into_contig`] - Consuming version returning owned tensor
    /// - [`TensorAny::to_prefer`] - Only converts if not already in preferred layout
    /// - [`TensorAny::to_layout`] - Convert to a specific layout
    pub fn to_contig(&self, order: FlagOrder) -> TensorCow<'_, T, B, D>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        to_contig(self, order)
    }

    /// Fallible version of [`to_contig`](TensorAny::to_contig).
    ///
    /// # See Also
    ///
    /// - [`TensorAny::to_contig`] - Infallible version
    pub fn to_contig_f(&self, order: FlagOrder) -> Result<TensorCow<'_, T, B, D>>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        to_contig_f(self, order)
    }

    /// Consuming version of [`to_contig`](TensorAny::to_contig) that returns an owned tensor.
    ///
    /// # See Also
    ///
    /// - [`TensorAny::to_contig`] - Non-consuming version returning [`TensorCow`]
    /// - [`TensorAny::into_contig_f`] - Fallible version
    pub fn into_contig_f(self, order: FlagOrder) -> Result<Tensor<T, B, D>>
    where
        B: DeviceRawAPI<MaybeUninit<T>> + OpAssignArbitaryAPI<T, D, D> + OpAssignAPI<T, D>,
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
    {
        into_contig_f(self, order)
    }

    /// Infallible version of [`into_contig_f`](TensorAny::into_contig_f).
    ///
    /// # See Also
    ///
    /// - [`TensorAny::into_contig_f`] - Fallible version
    /// - [`TensorAny::to_contig`] - Non-consuming version
    pub fn into_contig(self, order: FlagOrder) -> Tensor<T, B, D>
    where
        B: DeviceRawAPI<MaybeUninit<T>> + OpAssignArbitaryAPI<T, D, D> + OpAssignAPI<T, D>,
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
    {
        into_contig(self, order)
    }

    /// Consuming version of [`to_contig`](TensorAny::to_contig) that returns a [`TensorCow`].
    ///
    /// # See Also
    ///
    /// - [`TensorAny::to_contig`] - Non-consuming version
    /// - [`TensorAny::change_contig_f`] - Fallible version
    pub fn change_contig_f(self, order: FlagOrder) -> Result<TensorCow<'a, T, B, D>>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        change_contig_f(self, order)
    }

    /// Infallible version of [`change_contig_f`](TensorAny::change_contig_f).
    ///
    /// # See Also
    ///
    /// - [`TensorAny::change_contig_f`] - Fallible version
    pub fn change_contig(self, order: FlagOrder) -> TensorCow<'a, T, B, D>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        change_contig(self, order)
    }
}

/* #endregion */

/* #region to_prefer */

/// Convert tensor to preferred layout if not already contiguous, consuming the tensor.
///
/// This function checks if the tensor is already contiguous with the specified order.
/// If it is, the tensor is returned as a view without copying data. If not, the data
/// is copied to a new contiguous layout. This is useful for avoiding unnecessary copies
/// when the tensor may or may not already be in the desired layout.
///
/// # Arguments
///
/// - `tensor`: The input tensor. This function takes ownership.
/// - `order`: The memory layout order ([`RowMajor`] or [`ColMajor`]).
///
/// # Returns
///
/// A [`TensorCow`] containing either a view (if already contiguous) or an owned tensor
/// (if data was copied).
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // Already C-contiguous tensor - no copy
/// let a = rt::arange((6, &device)).into_shape([2, 3]);
/// let result = a.change_prefer(RowMajor);
/// assert!(!result.is_owned()); // View returned, no copy
///
/// // Non-contiguous tensor - requires copy
/// let a = rt::arange((6, &device)).into_shape([2, 3]);
/// let transposed = a.t();
/// let result = transposed.change_prefer(RowMajor);
/// assert!(result.is_owned()); // Owned tensor returned, data copied
/// ```
///
/// # See Also
///
/// - [`to_prefer`] - Non-consuming version
/// - [`into_prefer`] - Returns an owned tensor directly
/// - [`change_contig`] - Always converts to contiguous layout
pub fn change_prefer_f<'a, R, T, B, D>(
    tensor: TensorAny<R, T, B, D>,
    order: FlagOrder,
) -> Result<TensorCow<'a, T, B, D>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    if (order == RowMajor && tensor.c_prefer()) || (order == ColMajor && tensor.f_prefer()) {
        Ok(tensor.into_cow())
    } else {
        change_contig_f(tensor, order)
    }
}

/// Convert tensor to preferred layout if not already contiguous.
///
/// This function checks if the tensor is already contiguous with the specified order.
/// If it is, a view is returned without copying data. Otherwise, data is copied to
/// a new contiguous layout.
///
/// # Arguments
///
/// - `tensor`: A reference to the input tensor.
/// - `order`: The memory layout order ([`RowMajor`] or [`ColMajor`]).
///
/// # Returns
///
/// A [`TensorCow`] containing either a view (if already contiguous) or an owned tensor
/// (if data was copied).
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // C-contiguous tensor stays as view
/// let a = rt::tensor_from_nested!([[1, 2], [3, 4], [5, 6]], &device);
/// let result = rt::to_prefer(&a, RowMajor);
/// assert!(!result.is_owned());
///
/// // Transposed (non-contiguous) tensor gets copied
/// let transposed = a.t();
/// let result = rt::to_prefer(&transposed, RowMajor);
/// assert!(result.is_owned());
/// ```
///
/// # See Also
///
/// - [`to_prefer_f`] - Fallible version
/// - [`change_prefer`] - Consuming version
/// - [`into_prefer`] - Returns an owned tensor directly
/// - [`to_contig`] - Always converts to contiguous layout
pub fn to_prefer<R, T, B, D>(tensor: &TensorAny<R, T, B, D>, order: FlagOrder) -> TensorCow<'_, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    to_prefer_f(tensor, order).rstsr_unwrap()
}

/// Fallible version of [`to_prefer`].
///
/// # See Also
///
/// - [`to_prefer`] - Infallible version
pub fn to_prefer_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>, order: FlagOrder) -> Result<TensorCow<'_, T, B, D>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    change_prefer_f(tensor.view(), order)
}

/// Convert tensor to preferred layout if not already contiguous, returning an owned tensor.
///
/// This is similar to [`change_prefer`], but always returns an owned [`Tensor`].
///
/// # Arguments
///
/// - `tensor`: The input tensor. This function takes ownership.
/// - `order`: The memory layout order ([`RowMajor`] or [`ColMajor`]).
///
/// # Returns
///
/// An owned [`Tensor`] with contiguous layout.
///
/// # See Also
///
/// - [`into_prefer`] - Infallible version
/// - [`to_prefer`] - Non-consuming version returning [`TensorCow`]
/// - [`change_prefer`] - Consuming version returning [`TensorCow`]
/// - [`into_contig`] - Always converts to contiguous layout
pub fn into_prefer_f<'a, R, T, B, D>(tensor: TensorAny<R, T, B, D>, order: FlagOrder) -> Result<Tensor<T, B, D>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, D, D>
        + OpAssignAPI<T, D>,
    <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
{
    change_prefer_f(tensor, order).map(|v| v.into_owned())
}

/// Infallible version of [`change_prefer_f`].
///
/// # See Also
///
/// - [`change_prefer_f`] - Fallible version
pub fn change_prefer<'a, R, T, B, D>(tensor: TensorAny<R, T, B, D>, order: FlagOrder) -> TensorCow<'a, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    change_prefer_f(tensor, order).rstsr_unwrap()
}

/// Infallible version of [`into_prefer_f`].
///
/// # See Also
///
/// - [`into_prefer_f`] - Fallible version
pub fn into_prefer<'a, R, T, B, D>(tensor: TensorAny<R, T, B, D>, order: FlagOrder) -> Tensor<T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, D, D>
        + OpAssignAPI<T, D>,
    <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
{
    into_prefer_f(tensor, order).rstsr_unwrap()
}

impl<'a, R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    /// Convert tensor to preferred layout if not already contiguous.
    ///
    /// This method checks if the tensor is already contiguous with the specified order.
    /// If it is, a view is returned without copying data. Otherwise, data is copied to
    /// a new contiguous layout.
    ///
    /// # Arguments
    ///
    /// - `order`: The memory layout order ([`RowMajor`] or [`ColMajor`]).
    ///
    /// # Returns
    ///
    /// A [`TensorCow`] containing either a view (if already contiguous) or an owned tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// // Already contiguous - returns view
    /// let a = rt::arange((6, &device)).into_shape([2, 3]);
    /// let result = a.to_prefer(RowMajor);
    /// assert!(!result.is_owned());
    ///
    /// // Non-contiguous - returns owned
    /// let transposed = a.t();
    /// let result = transposed.to_prefer(RowMajor);
    /// assert!(result.is_owned());
    /// ```
    ///
    /// # See Also
    ///
    /// - [`TensorAny::to_prefer_f`] - Fallible version
    /// - [`TensorAny::into_prefer`] - Consuming version returning owned tensor
    /// - [`TensorAny::to_contig`] - Always converts to contiguous layout
    pub fn to_prefer(&self, order: FlagOrder) -> TensorCow<'_, T, B, D>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        to_prefer(self, order)
    }

    /// Fallible version of [`to_prefer`](TensorAny::to_prefer).
    ///
    /// # See Also
    ///
    /// - [`TensorAny::to_prefer`] - Infallible version
    pub fn to_prefer_f(&self, order: FlagOrder) -> Result<TensorCow<'_, T, B, D>>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        to_prefer_f(self, order)
    }

    /// Consuming version of [`to_prefer`](TensorAny::to_prefer) that returns an owned tensor.
    ///
    /// # See Also
    ///
    /// - [`TensorAny::to_prefer`] - Non-consuming version returning [`TensorCow`]
    /// - [`TensorAny::into_prefer_f`] - Fallible version
    pub fn into_prefer_f(self, order: FlagOrder) -> Result<Tensor<T, B, D>>
    where
        B: DeviceRawAPI<MaybeUninit<T>> + OpAssignArbitaryAPI<T, D, D> + OpAssignAPI<T, D>,
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
    {
        into_prefer_f(self, order)
    }

    /// Infallible version of [`into_prefer_f`](TensorAny::into_prefer_f).
    ///
    /// # See Also
    ///
    /// - [`TensorAny::into_prefer_f`] - Fallible version
    pub fn into_prefer(self, order: FlagOrder) -> Tensor<T, B, D>
    where
        B: DeviceRawAPI<MaybeUninit<T>> + OpAssignArbitaryAPI<T, D, D> + OpAssignAPI<T, D>,
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
    {
        into_prefer(self, order)
    }

    /// Consuming version of [`to_prefer`](TensorAny::to_prefer) that returns a [`TensorCow`].
    ///
    /// # See Also
    ///
    /// - [`TensorAny::to_prefer`] - Non-consuming version
    /// - [`TensorAny::change_prefer_f`] - Fallible version
    pub fn change_prefer_f(self, order: FlagOrder) -> Result<TensorCow<'a, T, B, D>>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        change_prefer_f(self, order)
    }

    /// Infallible version of [`change_prefer_f`](TensorAny::change_prefer_f).
    ///
    /// # See Also
    ///
    /// - [`TensorAny::change_prefer_f`] - Fallible version
    pub fn change_prefer(self, order: FlagOrder) -> TensorCow<'a, T, B, D>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        change_prefer(self, order)
    }
}

/* #endregion */
