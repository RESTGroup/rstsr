use crate::prelude_dev::*;

/* #region to_contig */

/// Convert tensor to contiguous layout.
///
/// See also [`to_contig`].
pub fn change_contig_f<'a, R, T, B, D>(
    tensor: TensorAny<R, T, B, D>,
    order: FlagOrder,
) -> Result<TensorCow<'a, T, B, D>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    // view decision follows the NumPy-style contiguity flags (singleton-axis
    // strides are irrelevant), not exact layout equality
    let is_contig = match order {
        RowMajor => tensor.layout().c_contig(),
        ColMajor => tensor.layout().f_contig(),
    };
    if is_contig {
        // no copy: return a view with singleton-axis strides reset, so the
        // layout becomes the usual contiguous one over the same elements
        let layout_old = tensor.layout().clone();
        let shape = layout_old.shape().clone();
        let layout_new = match order {
            RowMajor => shape.new_c_contig(Some(layout_old.offset())),
            ColMajor => shape.new_f_contig(Some(layout_old.offset())),
        };
        let (storage, _) = tensor.into_raw_parts();
        // safety: `is_contig` ensures the normalized layout references exactly
        // the same elements of `storage` as the original layout
        let tensor = unsafe { TensorBase::new_unchecked(storage, layout_new) };
        return Ok(tensor.into_cow());
    }
    // layout is not contiguous in the requested order; copy data by assign
    let shape = tensor.shape().clone();
    let layout_new = match order {
        RowMajor => shape.new_c_contig(None),
        ColMajor => shape.new_f_contig(None),
    };
    change_layout_f(tensor, layout_new)
}

/// Convert tensor to contiguous layout.
///
/// This function takes a reference to a tensor and returns a [`TensorCow`] that is
/// either a view (if the tensor is already contiguous with the requested order) or
/// a newly allocated contiguous copy. Contiguity is decided by the NumPy-style
/// flags ([`TensorBase::c_contig`] / [`TensorBase::f_contig`]), so a
/// padded-singleton tensor is returned as a view; the singleton-axis strides of
/// a viewed result are reset, making the layout the usual contiguous one.
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
///
/// # Parameters
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
/// // Create a non-contiguous tensor to contiguous
/// let a = rt::arange((12, &device)).into_shape([3, 4]);
/// let sliced = a.i((.., slice!(None, None, 2))); // Every other column
/// println!("layout of sliced tensor: {:?}", sliced.layout());
/// // 2-Dim (dyn), contiguous: Custom
/// // shape: [3, 2], stride: [4, 2], offset: 0
///
/// // Convert to C-contiguous
/// let contig = sliced.to_contig(RowMajor);
/// println!("Contiguous layout: {:?}", contig.layout());
/// // 2-Dim (dyn), contiguous: Cc
/// // shape: [3, 2], stride: [2, 1], offset: 0
/// ```
///
/// A padded-singleton tensor that is already contiguous by the NumPy-style
/// flags is returned as a view, with the singleton-axis stride reset:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let reshaped = rt::arange((15, &device)).into_shape([3, 5]);
/// let parent = reshaped.to_contig(ColMajor);
/// let a = parent.i((.., 0..1)); // shape [3, 1], stride [1, 3]
/// println!("a layout: {:?}", a.layout());
/// // 2-Dim (dyn), contiguous: CcFf
/// // shape: [3, 1], stride: [1, 3], offset: 0
///
/// let b = a.to_contig(RowMajor); // no copy; stride reset to [1, 1]
/// println!("b layout: {:?}", b.layout());
/// // 2-Dim (dyn), contiguous: CcFf
/// // shape: [3, 1], stride: [1, 1], offset: 0
/// # assert!(!b.is_owned());
/// # assert_eq!(b.stride(), &[1, 1]);
/// ```
///
/// # Panics
///
/// - Panics if the internal copy path fails (e.g. an overflowing element count or a device
///   allocation error).
///
/// For a fallible version, use [`to_contig_f`].
///
/// # See also
///
/// ## Similar functions in RSTSR
///
/// - [`reshape`]: Change the shape of a tensor without changing its data layout (returns
///   copy-on-write tensor, view or necessarily clone). Reshape function inputs shape instead of
///   layout.
/// - [`to_prefer`]: Only converts if not already in preferred layout.
///
/// ## Variants of this function
///
/// - [`to_contig`] / [`to_contig_f`]: Non-consuming version that takes a reference and returns a
///   view or owned tensor.
/// - [`into_contig`] / [`into_contig_f`]: Consuming version that returns an owned tensor directly.
/// - [`change_contig`] / [`change_contig_f`]: Consuming version that returns a view or owned
///   tensor.
/// - Associated methods on [`TensorAny`]:
///
///   - [`TensorAny::to_contig`] / [`TensorAny::to_contig_f`]
///   - [`TensorAny::into_contig`] / [`TensorAny::into_contig_f`]
///   - [`TensorAny::change_contig`] / [`TensorAny::change_contig_f`]
pub fn to_contig<R, T, B, D>(tensor: &TensorAny<R, T, B, D>, order: FlagOrder) -> TensorCow<'_, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    to_contig_f(tensor, order).rstsr_unwrap()
}

/// Convert tensor to contiguous layout.
///
/// See also [`to_contig`].
pub fn to_contig_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>, order: FlagOrder) -> Result<TensorCow<'_, T, B, D>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    change_contig_f(tensor.view(), order)
}

/// Convert tensor to contiguous layout.
///
/// See also [`to_contig`].
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

/// Convert tensor to contiguous layout.
///
/// See also [`to_contig`].
pub fn change_contig<'a, R, T, B, D>(tensor: TensorAny<R, T, B, D>, order: FlagOrder) -> TensorCow<'a, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    change_contig_f(tensor, order).rstsr_unwrap()
}

/// Convert tensor to contiguous layout.
///
/// See also [`to_contig`].
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
    /// See also [`to_contig`].
    pub fn to_contig(&self, order: FlagOrder) -> TensorCow<'_, T, B, D>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        to_contig(self, order)
    }

    /// Convert tensor to contiguous layout.
    ///
    /// See also [`to_contig`].
    pub fn to_contig_f(&self, order: FlagOrder) -> Result<TensorCow<'_, T, B, D>>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        to_contig_f(self, order)
    }

    /// Convert tensor to contiguous layout.
    ///
    /// See also [`to_contig`].
    pub fn into_contig_f(self, order: FlagOrder) -> Result<Tensor<T, B, D>>
    where
        B: DeviceRawAPI<MaybeUninit<T>> + OpAssignArbitaryAPI<T, D, D> + OpAssignAPI<T, D>,
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
    {
        into_contig_f(self, order)
    }

    /// Convert tensor to contiguous layout.
    ///
    /// # See also
    ///
    /// Refer to [`to_contig`] for more detailed documentation.
    pub fn into_contig(self, order: FlagOrder) -> Tensor<T, B, D>
    where
        B: DeviceRawAPI<MaybeUninit<T>> + OpAssignArbitaryAPI<T, D, D> + OpAssignAPI<T, D>,
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
    {
        into_contig(self, order)
    }

    /// Convert tensor to contiguous layout.
    ///
    /// See also [`to_contig`].
    pub fn change_contig_f(self, order: FlagOrder) -> Result<TensorCow<'a, T, B, D>>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        change_contig_f(self, order)
    }

    /// Convert tensor to contiguous layout.
    ///
    /// See also [`to_contig`].
    pub fn change_contig(self, order: FlagOrder) -> TensorCow<'a, T, B, D>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        change_contig(self, order)
    }
}

/* #endregion */

/* #region to_prefer */

/// Convert tensor to preferred layout if not already contiguous.
///
/// See also [`to_prefer`].
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
/// # Parameters
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
/// # See also
///
/// ## Similar functions in RSTSR
///
/// - [`reshape`]: Change the shape of a tensor without changing its data layout (returns
///   copy-on-write tensor, view or necessarily clone). Reshape function inputs shape instead of
///   layout.
/// - [`to_contig`]: Always converts to contiguous layout regardless of current layout.
///
/// ## Variants of this function
///
/// - [`to_prefer`] / [`to_prefer_f`]: Non-consuming version that takes a reference and returns a
///   view or owned tensor.
/// - [`into_prefer`] / [`into_prefer_f`]: Consuming version that returns an owned tensor directly.
/// - [`change_prefer`] / [`change_prefer_f`]: Consuming version that returns a view or owned
///   tensor.
/// - Associated methods on [`TensorAny`]:
///
///   - [`TensorAny::to_prefer`] / [`TensorAny::to_prefer_f`]
///   - [`TensorAny::into_prefer`] / [`TensorAny::into_prefer_f`]
///   - [`TensorAny::change_prefer`] / [`TensorAny::change_prefer_f`]
pub fn to_prefer<R, T, B, D>(tensor: &TensorAny<R, T, B, D>, order: FlagOrder) -> TensorCow<'_, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    to_prefer_f(tensor, order).rstsr_unwrap()
}

/// Convert tensor to preferred layout if not already contiguous.
///
/// See also [`to_prefer`].
pub fn to_prefer_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>, order: FlagOrder) -> Result<TensorCow<'_, T, B, D>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    change_prefer_f(tensor.view(), order)
}

/// Convert tensor to preferred layout if not already contiguous.
///
/// See also [`to_prefer`].
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

/// Convert tensor to preferred layout if not already contiguous.
///
/// See also [`to_prefer`].
pub fn change_prefer<'a, R, T, B, D>(tensor: TensorAny<R, T, B, D>, order: FlagOrder) -> TensorCow<'a, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    change_prefer_f(tensor, order).rstsr_unwrap()
}

/// Convert tensor to preferred layout if not already contiguous.
///
/// See also [`to_prefer`].
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
    /// See also [`to_prefer`].
    pub fn to_prefer(&self, order: FlagOrder) -> TensorCow<'_, T, B, D>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        to_prefer(self, order)
    }

    /// Convert tensor to preferred layout if not already contiguous.
    ///
    /// See also [`to_prefer`].
    pub fn to_prefer_f(&self, order: FlagOrder) -> Result<TensorCow<'_, T, B, D>>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        to_prefer_f(self, order)
    }

    /// Convert tensor to preferred layout if not already contiguous.
    ///
    /// See also [`to_prefer`].
    pub fn into_prefer_f(self, order: FlagOrder) -> Result<Tensor<T, B, D>>
    where
        B: DeviceRawAPI<MaybeUninit<T>> + OpAssignArbitaryAPI<T, D, D> + OpAssignAPI<T, D>,
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
    {
        into_prefer_f(self, order)
    }

    /// Convert tensor to preferred layout if not already contiguous.
    ///
    /// See also [`to_prefer`].
    pub fn into_prefer(self, order: FlagOrder) -> Tensor<T, B, D>
    where
        B: DeviceRawAPI<MaybeUninit<T>> + OpAssignArbitaryAPI<T, D, D> + OpAssignAPI<T, D>,
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
    {
        into_prefer(self, order)
    }

    /// Convert tensor to preferred layout if not already contiguous.
    ///
    /// See also [`to_prefer`].
    pub fn change_prefer_f(self, order: FlagOrder) -> Result<TensorCow<'a, T, B, D>>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        change_prefer_f(self, order)
    }

    /// Convert tensor to preferred layout if not already contiguous.
    ///
    /// See also [`to_prefer`].
    pub fn change_prefer(self, order: FlagOrder) -> TensorCow<'a, T, B, D>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        change_prefer(self, order)
    }
}

/* #endregion */
