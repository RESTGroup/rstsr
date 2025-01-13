use crate::prelude_dev::*;

pub trait DeviceOpPackTriAPI<T>
where
    Self: DeviceAPI<T>,
{
    fn pack_tri(
        &self,
        a: &mut <Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<IxD>,
        b: &<Self as DeviceRawAPI<T>>::Raw,
        lb: &Layout<IxD>,
        uplo: TensorUpLo,
    ) -> Result<()>;
}

pub trait DeviceOpUnpackTriAPI<T>
where
    Self: DeviceAPI<T>,
{
    fn unpack_tri(
        &self,
        a: &mut <Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<IxD>,
        b: &<Self as DeviceRawAPI<T>>::Raw,
        lb: &Layout<IxD>,
        uplo: TensorUpLo,
        symm: TensorSymm,
    ) -> Result<()>;
}
