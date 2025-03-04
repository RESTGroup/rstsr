use crate::prelude_dev::*;

#[allow(clippy::type_complexity)]
pub trait TensorDeviceChangeAPI<'l, BOut>
where
    BOut: DeviceRawAPI<Self::Type>,
{
    type Repr;
    type ReprTo;
    type Type;
    type Dim: DimAPI;

    fn change_device_f(
        self,
        device: &BOut,
    ) -> Result<TensorAny<Self::Repr, Self::Type, BOut, Self::Dim>>;
    fn into_device_f(
        self,
        device: &BOut,
    ) -> Result<TensorAny<DataOwned<BOut::Raw>, Self::Type, BOut, Self::Dim>>;
    fn to_device_f(
        &'l self,
        device: &BOut,
    ) -> Result<TensorAny<Self::ReprTo, Self::Type, BOut, Self::Dim>>;

    fn change_device(self, device: &BOut) -> TensorAny<Self::Repr, Self::Type, BOut, Self::Dim>
    where
        Self: Sized,
    {
        self.change_device_f(device).unwrap()
    }

    fn into_device(
        self,
        device: &BOut,
    ) -> TensorAny<DataOwned<BOut::Raw>, Self::Type, BOut, Self::Dim>
    where
        Self: Sized,
    {
        self.into_device_f(device).unwrap()
    }

    fn to_device(&'l self, device: &BOut) -> TensorAny<Self::ReprTo, Self::Type, BOut, Self::Dim> {
        self.to_device_f(device).unwrap()
    }
}

impl<'a, R, T, B, D, BOut> TensorDeviceChangeAPI<'a, BOut> for TensorAny<R, T, B, D>
where
    B: DeviceRawAPI<T> + DeviceChangeAPI<'a, BOut, R, T, D>,
    BOut: DeviceRawAPI<T>,
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
{
    type Repr = B::Repr;
    type ReprTo = B::ReprTo;
    type Type = T;
    type Dim = D;

    fn change_device_f(self, device: &BOut) -> Result<TensorAny<B::Repr, T, BOut, D>> {
        B::change_device(self, device)
    }

    fn into_device_f(self, device: &BOut) -> Result<Tensor<T, BOut, D>> {
        B::into_device(self, device)
    }

    fn to_device_f(
        &'a self,
        device: &BOut,
    ) -> Result<TensorAny<B::ReprTo, Self::Type, BOut, Self::Dim>> {
        B::to_device(self, device)
    }
}

#[allow(clippy::type_complexity)]
pub trait TensorChangeFromDevice<'l, BOut>
where
    BOut: DeviceRawAPI<Self::Type>,
{
    type Repr;
    type ReprTo;
    type Type;
    type Dim: DimAPI;

    fn change_device_f(
        self,
        device: &BOut,
    ) -> Result<TensorAny<Self::Repr, Self::Type, BOut, Self::Dim>>;
    fn into_device_f(
        self,
        device: &BOut,
    ) -> Result<TensorAny<DataOwned<BOut::Raw>, Self::Type, BOut, Self::Dim>>;
    fn to_device_f(
        &'l self,
        device: &BOut,
    ) -> Result<TensorAny<Self::ReprTo, Self::Type, BOut, Self::Dim>>;

    fn change_device(self, device: &BOut) -> TensorAny<Self::Repr, Self::Type, BOut, Self::Dim>
    where
        Self: Sized,
    {
        self.change_device_f(device).unwrap()
    }

    fn into_device(
        self,
        device: &BOut,
    ) -> TensorAny<DataOwned<BOut::Raw>, Self::Type, BOut, Self::Dim>
    where
        Self: Sized,
    {
        self.into_device_f(device).unwrap()
    }

    fn to_device(&'l self, device: &BOut) -> TensorAny<Self::ReprTo, Self::Type, BOut, Self::Dim> {
        self.to_device_f(device).unwrap()
    }
}
