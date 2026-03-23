use crate::prelude_dev::*;

/* #region to_layout */

/// Convert tensor to a specified layout.
///
/// See also [`to_layout`].
pub fn change_layout_f<'a, R, T, B, D, D2>(
    tensor: TensorAny<R, T, B, D>,
    layout: Layout<D2>,
) -> Result<TensorCow<'a, T, B, D2>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    D2: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D2, D>,
{
    let shape = layout.shape();
    rstsr_assert_eq!(tensor.size(), shape.shape_size(), InvalidLayout)?;
    let same_layout = tensor.layout().to_dim::<IxD>()? == layout.to_dim::<IxD>()?;
    if same_layout {
        // no data cloned
        let (storage, _) = tensor.into_raw_parts();
        let tensor = unsafe { TensorBase::new_unchecked(storage, layout) };
        return Ok(tensor.into_cow());
    } else {
        // layout changed, or not c and f contiguous with same layout
        // clone data by assign
        let (storage_old, layout_old) = tensor.into_raw_parts();
        let device = storage_old.device();
        let (_, idx_max) = layout.bounds_index()?;
        let mut storage_new = device.uninit_impl(idx_max)?;
        device.assign_arbitary_uninit(storage_new.raw_mut(), &layout, storage_old.raw(), &layout_old)?;
        let storage_new = unsafe { B::assume_init_impl(storage_new)? };
        let tensor = unsafe { TensorBase::new_unchecked(storage_new, layout) };
        return Ok(tensor.into_cow());
    }
}

/// Convert tensor to a specified layout.
///
/// This function takes a reference to a tensor and a target layout, returning a [`TensorCow`]
/// that is either a view (if the layout matches or both are contiguous) or a newly allocated
/// copy with the requested layout.
///
/// The layout can differ from the original in shape, strides, or even dimensionality,
/// as long as the total number of elements remains the same.
///
/// # Arguments
///
/// - `tensor`: A reference to the input tensor.
/// - `layout`: The target [`Layout`] for the output tensor.
///
/// # Returns
///
/// A [`TensorCow`] containing either a view (if no copy needed) or an owned tensor with the
/// specified layout.
///
/// # Errors
///
/// Returns an error if the layout size doesn't match the tensor size.
/// Use [`to_layout_f`] for the fallible version.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // Convert tensor to a different layout
/// let a = rt::arange((12, &device)).into_shape([3, 4]);
/// println!("a layout: {:?}", a.layout());
/// // 2-Dim (dyn), contiguous: Cc
/// // shape: [3, 4], stride: [4, 1], offset: 0
///
/// // Convert to F-contiguous layout
/// let layout_f = [3, 4].f();
/// let b = a.to_layout(layout_f);
/// println!("b layout: {:?}", b.layout());
/// // 2-Dim (dyn), contiguous: Fc
/// // shape: [3, 4], stride: [1, 3], offset: 0
/// assert!(b.f_contig());
/// ```
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // Using to_layout to reshape tensor
/// let a = rt::arange((12, &device)).into_shape([3, 4]);
///
/// // Flatten to 1D
/// let layout_1d = [12].c();
/// let b = a.to_layout(layout_1d);
/// assert_eq!(b.shape(), &[12]);
///
/// // Reshape to different 2D
/// let layout_2d = [2, 6].c();
/// let c = b.to_layout(layout_2d);
/// assert_eq!(c.shape(), &[2, 6]);
/// ```
///
/// # See also
///
/// ## Similar functions in RSTSR
///
/// - [`reshape`]: Change the shape of a tensor (inputs shape instead of layout).
/// - [`to_contig`]: Convert tensor to C or F contiguous layout.
/// - [`transpose`]: Permute dimensions of a tensor (returns a view).
///
/// ## Variants of this function
///
/// - [`to_layout`] / [`to_layout_f`]: Non-consuming version that takes a reference and returns a
///   view or owned tensor.
/// - [`into_layout`] / [`into_layout_f`]: Consuming version that returns an owned tensor directly.
/// - [`change_layout`] / [`change_layout_f`]: Consuming version that returns a view or owned
///   tensor.
/// - Associated methods on [`TensorAny`]:
///
///   - [`TensorAny::to_layout`] / [`TensorAny::to_layout_f`]
///   - [`TensorAny::into_layout`] / [`TensorAny::into_layout_f`]
///   - [`TensorAny::change_layout`] / [`TensorAny::change_layout_f`]
pub fn to_layout<R, T, D, B, D2>(tensor: &TensorAny<R, T, B, D>, layout: Layout<D2>) -> TensorCow<'_, T, B, D2>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    D2: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D2, D>,
{
    change_layout_f(tensor.view(), layout).rstsr_unwrap()
}

/// Convert tensor to a specified layout.
///
/// See also [`to_layout`].
pub fn to_layout_f<R, T, D, B, D2>(
    tensor: &TensorAny<R, T, B, D>,
    layout: Layout<D2>,
) -> Result<TensorCow<'_, T, B, D2>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    D2: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D2, D>,
{
    change_layout_f(tensor.view(), layout)
}

/// Convert tensor to a specified layout.
///
/// See also [`to_layout`].
pub fn into_layout_f<'a, R, T, B, D, D2>(tensor: TensorAny<R, T, B, D>, layout: Layout<D2>) -> Result<Tensor<T, B, D2>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    D2: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, D2, D>
        + OpAssignAPI<T, D2>,
    <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
{
    change_layout_f(tensor, layout).map(|v| v.into_owned())
}

/// Convert tensor to a specified layout.
///
/// See also [`to_layout`].
pub fn into_layout<'a, R, T, B, D, D2>(tensor: TensorAny<R, T, B, D>, layout: Layout<D2>) -> Tensor<T, B, D2>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    D2: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, D2, D>
        + OpAssignAPI<T, D2>,
    <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
{
    into_layout_f(tensor, layout).rstsr_unwrap()
}

/// Convert tensor to a specified layout.
///
/// See also [`to_layout`].
pub fn change_layout<'a, R, T, B, D, D2>(tensor: TensorAny<R, T, B, D>, layout: Layout<D2>) -> TensorCow<'a, T, B, D2>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    D2: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D2, D>,
{
    change_layout_f(tensor, layout).rstsr_unwrap()
}

impl<'a, R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    /// Convert tensor to a specified layout.
    ///
    /// See also [`to_layout`].
    pub fn to_layout<D2>(&self, layout: Layout<D2>) -> TensorCow<'_, T, B, D2>
    where
        D2: DimAPI,
        B: OpAssignArbitaryAPI<T, D2, D>,
    {
        to_layout(self, layout)
    }

    /// Convert tensor to a specified layout.
    ///
    /// See also [`to_layout`].
    pub fn to_layout_f<D2>(&self, layout: Layout<D2>) -> Result<TensorCow<'_, T, B, D2>>
    where
        D2: DimAPI,
        B: OpAssignArbitaryAPI<T, D2, D>,
    {
        to_layout_f(self, layout)
    }

    /// Convert tensor to a specified layout.
    ///
    /// See also [`to_layout`].
    pub fn into_layout_f<D2>(self, layout: Layout<D2>) -> Result<Tensor<T, B, D2>>
    where
        D2: DimAPI,
        B: DeviceRawAPI<MaybeUninit<T>> + OpAssignArbitaryAPI<T, D2, D> + OpAssignAPI<T, D2>,
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
    {
        into_layout_f(self, layout)
    }

    /// Convert tensor to a specified layout.
    ///
    /// See also [`to_layout`].
    pub fn into_layout<D2>(self, layout: Layout<D2>) -> Tensor<T, B, D2>
    where
        D2: DimAPI,
        B: DeviceRawAPI<MaybeUninit<T>> + OpAssignArbitaryAPI<T, D2, D> + OpAssignAPI<T, D2>,
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
    {
        into_layout(self, layout)
    }

    /// Convert tensor to a specified layout.
    ///
    /// See also [`to_layout`].
    pub fn change_layout_f<D2>(self, layout: Layout<D2>) -> Result<TensorCow<'a, T, B, D2>>
    where
        D2: DimAPI,
        B: OpAssignArbitaryAPI<T, D2, D>,
    {
        change_layout_f(self, layout)
    }

    /// Convert tensor to a specified layout.
    ///
    /// See also [`to_layout`].
    pub fn change_layout<D2>(self, layout: Layout<D2>) -> TensorCow<'a, T, B, D2>
    where
        D2: DimAPI,
        B: OpAssignArbitaryAPI<T, D2, D>,
    {
        change_layout(self, layout)
    }
}

/* #endregion */
