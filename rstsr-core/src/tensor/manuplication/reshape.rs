use crate::prelude_dev::*;

/* #region reshape */

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

pub fn change_shape<'a, I, R, T, B, D>(tensor: TensorAny<R, T, B, D>, shape: I) -> TensorCow<'a, T, B, IxD>
where
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    change_shape_f(tensor, shape).rstsr_unwrap()
}

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

pub fn to_shape_f<'a, I, R, T, B, D>(tensor: &'a TensorAny<R, T, B, D>, shape: I) -> Result<TensorCow<'a, T, B, IxD>>
where
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    change_shape_f(tensor.view(), shape)
}

pub fn to_shape<'a, I, R, T, B, D>(tensor: &'a TensorAny<R, T, B, D>, shape: I) -> TensorCow<'a, T, B, IxD>
where
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    to_shape_f(tensor, shape).rstsr_unwrap()
}

pub fn reshape_f<'a, I, R, T, B, D>(tensor: &'a TensorAny<R, T, B, D>, shape: I) -> Result<TensorCow<'a, T, B, IxD>>
where
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    to_shape_f(tensor, shape)
}

pub fn reshape<'a, I, R, T, B, D>(tensor: &'a TensorAny<R, T, B, D>, shape: I) -> TensorCow<'a, T, B, IxD>
where
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    to_shape(tensor, shape)
}

impl<'a, R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
    T: Clone,
{
    pub fn change_shape_f<I>(self, shape: I) -> Result<TensorCow<'a, T, B, IxD>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        change_shape_f(self, shape)
    }

    pub fn change_shape<I>(self, shape: I) -> TensorCow<'a, T, B, IxD>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        change_shape(self, shape)
    }

    pub fn into_shape_f<I>(self, shape: I) -> Result<Tensor<T, B, IxD>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
        B: OpAssignAPI<T, IxD>,
    {
        into_shape_f(self, shape)
    }

    pub fn into_shape<I>(self, shape: I) -> Tensor<T, B, IxD>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
        B: OpAssignAPI<T, IxD>,
    {
        into_shape(self, shape)
    }

    pub fn to_shape_f<I>(&'a self, shape: I) -> Result<TensorCow<'a, T, B, IxD>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        self.view().change_shape_f(shape)
    }

    pub fn to_shape<I>(&'a self, shape: I) -> TensorCow<'a, T, B, IxD>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        self.view().change_shape(shape)
    }

    pub fn reshape_f<I>(&'a self, shape: I) -> Result<TensorCow<'a, T, B, IxD>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        self.view().change_shape_f(shape)
    }

    pub fn reshape<I>(&'a self, shape: I) -> TensorCow<'a, T, B, IxD>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        self.view().change_shape(shape)
    }
}

/* #endregion */
