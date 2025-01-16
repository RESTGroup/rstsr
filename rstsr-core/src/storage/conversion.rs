use crate::prelude_dev::*;

pub trait DeviceChangeAPI<'l, BOut, R, T, D>
where
    Self: DeviceRawAPI<T>,
    BOut: DeviceRawAPI<T>,
    D: DimAPI,
    R: DataAPI<Data = Self::Raw>,
{
    type Repr: DataAPI<Data = BOut::Raw>;
    type ReprTo: DataAPI<Data = BOut::Raw>;

    fn change_device(
        tensor: TensorAny<R, T, Self, D>,
        device: &BOut,
    ) -> Result<TensorAny<Self::Repr, T, BOut, D>>;

    fn into_device(
        tensor: TensorAny<R, T, Self, D>,
        device: &BOut,
    ) -> Result<TensorAny<DataOwned<BOut::Raw>, T, BOut, D>>;

    fn to_device(
        tensor: &'l TensorAny<R, T, Self, D>,
        device: &BOut,
    ) -> Result<TensorAny<Self::ReprTo, T, BOut, D>>;
}
