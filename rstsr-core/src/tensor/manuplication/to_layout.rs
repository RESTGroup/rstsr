use crate::prelude_dev::*;

/* #region to_layout */

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
    let contig_c = tensor.c_contig() && layout.c_contig() && tensor.layout().offset() == layout.offset();
    let contig_f = tensor.f_contig() && layout.f_contig() && tensor.layout().offset() == layout.offset();
    let default_order = tensor.device().default_order();
    let contig = match default_order {
        RowMajor => contig_c,
        ColMajor => contig_f,
    };
    if same_layout || contig {
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

/// Convert tensor to the other layout.
pub fn to_layout<R, T, D, B, D2>(tensor: &TensorAny<R, T, B, D>, layout: Layout<D2>) -> TensorCow<'_, T, B, D2>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    D2: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D2, D>,
{
    change_layout_f(tensor.view(), layout).rstsr_unwrap()
}

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
    /// Convert tensor to the other layout.
    ///
    /// # See also
    ///
    /// [`to_layout`]
    pub fn to_layout<D2>(&self, layout: Layout<D2>) -> TensorCow<'_, T, B, D2>
    where
        D2: DimAPI,
        B: OpAssignArbitaryAPI<T, D2, D>,
    {
        to_layout(self, layout)
    }

    pub fn to_layout_f<D2>(&self, layout: Layout<D2>) -> Result<TensorCow<'_, T, B, D2>>
    where
        D2: DimAPI,
        B: OpAssignArbitaryAPI<T, D2, D>,
    {
        to_layout_f(self, layout)
    }

    pub fn into_layout_f<D2>(self, layout: Layout<D2>) -> Result<Tensor<T, B, D2>>
    where
        D2: DimAPI,
        B: DeviceRawAPI<MaybeUninit<T>> + OpAssignArbitaryAPI<T, D2, D> + OpAssignAPI<T, D2>,
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
    {
        into_layout_f(self, layout)
    }

    pub fn into_layout<D2>(self, layout: Layout<D2>) -> Tensor<T, B, D2>
    where
        D2: DimAPI,
        B: DeviceRawAPI<MaybeUninit<T>> + OpAssignArbitaryAPI<T, D2, D> + OpAssignAPI<T, D2>,
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
    {
        into_layout(self, layout)
    }

    pub fn change_layout_f<D2>(self, layout: Layout<D2>) -> Result<TensorCow<'a, T, B, D2>>
    where
        D2: DimAPI,
        B: OpAssignArbitaryAPI<T, D2, D>,
    {
        change_layout_f(self, layout)
    }

    pub fn change_layout<D2>(self, layout: Layout<D2>) -> TensorCow<'a, T, B, D2>
    where
        D2: DimAPI,
        B: OpAssignArbitaryAPI<T, D2, D>,
    {
        change_layout(self, layout)
    }
}

/* #endregion */
