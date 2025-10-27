use crate::prelude_dev::*;

/* #region to_contig */

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

pub fn to_contig_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>, order: FlagOrder) -> Result<TensorCow<'_, T, B, D>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    change_contig_f(tensor.view(), order)
}

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

pub fn change_contig<'a, R, T, B, D>(tensor: TensorAny<R, T, B, D>, order: FlagOrder) -> TensorCow<'a, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    change_contig_f(tensor, order).unwrap()
}

pub fn to_contig<R, T, B, D>(tensor: &TensorAny<R, T, B, D>, order: FlagOrder) -> TensorCow<'_, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    to_contig_f(tensor, order).unwrap()
}

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
    into_contig_f(tensor, order).unwrap()
}

impl<'a, R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    /// Convert tensor to contiguous, with specified layout.
    pub fn to_contig(&self, order: FlagOrder) -> TensorCow<'_, T, B, D>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        to_contig(self, order)
    }

    pub fn to_contig_f(&self, order: FlagOrder) -> Result<TensorCow<'_, T, B, D>>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        to_contig_f(self, order)
    }

    pub fn into_contig_f(self, order: FlagOrder) -> Result<Tensor<T, B, D>>
    where
        B: DeviceRawAPI<MaybeUninit<T>> + OpAssignArbitaryAPI<T, D, D> + OpAssignAPI<T, D>,
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
    {
        into_contig_f(self, order)
    }

    pub fn into_contig(self, order: FlagOrder) -> Tensor<T, B, D>
    where
        B: DeviceRawAPI<MaybeUninit<T>> + OpAssignArbitaryAPI<T, D, D> + OpAssignAPI<T, D>,
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
    {
        into_contig(self, order)
    }

    pub fn change_contig_f(self, order: FlagOrder) -> Result<TensorCow<'a, T, B, D>>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        change_contig_f(self, order)
    }

    pub fn change_contig(self, order: FlagOrder) -> TensorCow<'a, T, B, D>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        change_contig(self, order)
    }
}

/* #endregion */

/* #region to_prefer */

pub fn change_prefer_f<'a, R, T, B, D>(
    tensor: TensorAny<R, T, B, D>,
    order: FlagOrder,
) -> Result<TensorCow<'a, T, B, D>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    if (order == FlagOrder::C && tensor.c_prefer()) || (order == FlagOrder::F && tensor.f_prefer()) {
        Ok(tensor.into_cow())
    } else {
        change_contig_f(tensor, order)
    }
}

pub fn to_prefer_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>, order: FlagOrder) -> Result<TensorCow<'_, T, B, D>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    change_prefer_f(tensor.view(), order)
}

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

pub fn change_prefer<'a, R, T, B, D>(tensor: TensorAny<R, T, B, D>, order: FlagOrder) -> TensorCow<'a, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    change_prefer_f(tensor, order).unwrap()
}

pub fn to_prefer<R, T, B, D>(tensor: &TensorAny<R, T, B, D>, order: FlagOrder) -> TensorCow<'_, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    to_prefer_f(tensor, order).unwrap()
}

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
    into_prefer_f(tensor, order).unwrap()
}

impl<'a, R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    /// Convert tensor to contiguous, with specified layout.
    pub fn to_prefer(&self, order: FlagOrder) -> TensorCow<'_, T, B, D>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        to_prefer(self, order)
    }

    pub fn to_prefer_f(&self, order: FlagOrder) -> Result<TensorCow<'_, T, B, D>>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        to_prefer_f(self, order)
    }

    pub fn into_prefer_f(self, order: FlagOrder) -> Result<Tensor<T, B, D>>
    where
        B: DeviceRawAPI<MaybeUninit<T>> + OpAssignArbitaryAPI<T, D, D> + OpAssignAPI<T, D>,
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
    {
        into_prefer_f(self, order)
    }

    pub fn into_prefer(self, order: FlagOrder) -> Tensor<T, B, D>
    where
        B: DeviceRawAPI<MaybeUninit<T>> + OpAssignArbitaryAPI<T, D, D> + OpAssignAPI<T, D>,
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
    {
        into_prefer(self, order)
    }

    pub fn change_prefer_f(self, order: FlagOrder) -> Result<TensorCow<'a, T, B, D>>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        change_prefer_f(self, order)
    }

    pub fn change_prefer(self, order: FlagOrder) -> TensorCow<'a, T, B, D>
    where
        B: OpAssignArbitaryAPI<T, D, D>,
    {
        change_prefer(self, order)
    }
}

/* #endregion */
