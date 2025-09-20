use crate::prelude_dev::*;

pub trait DeviceOpPackTriAPI<T>
where
    Self: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>>,
{
    fn pack_tri(
        &self,
        a: &mut <Self as DeviceRawAPI<MaybeUninit<T>>>::Raw,
        la: &Layout<IxD>,
        b: &<Self as DeviceRawAPI<T>>::Raw,
        lb: &Layout<IxD>,
        uplo: FlagUpLo,
    ) -> Result<()>;
}

pub trait DeviceOpUnpackTriAPI<T>
where
    Self: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>>,
{
    fn unpack_tri(
        &self,
        a: &mut <Self as DeviceRawAPI<MaybeUninit<T>>>::Raw,
        la: &Layout<IxD>,
        b: &<Self as DeviceRawAPI<T>>::Raw,
        lb: &Layout<IxD>,
        uplo: FlagUpLo,
        symm: FlagSymm,
    ) -> Result<()>;
}
