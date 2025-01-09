use crate::prelude_dev::*;

pub trait DeviceOpPackTriAPI<T>
where
    Self: DeviceAPI<T>,
{
    fn pack_tri(
        &self,
        a: &mut Storage<T, Self>,
        la: &Layout<IxD>,
        b: &Storage<T, Self>,
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
        a: &mut Storage<T, Self>,
        la: &Layout<IxD>,
        b: &Storage<T, Self>,
        lb: &Layout<IxD>,
        uplo: TensorUpLo,
        symm: TensorSymm,
    ) -> Result<()>;
}
