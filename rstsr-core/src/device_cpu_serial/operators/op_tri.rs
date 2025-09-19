use crate::prelude_dev::*;
use num::complex::ComplexFloat;

impl<T> DeviceOpPackTriAPI<T> for DeviceCpuSerial
where
    T: Clone,
{
    fn pack_tri(
        &self,
        a: &mut Vec<MaybeUninit<T>>,
        la: &Layout<IxD>,
        b: &Vec<T>,
        lb: &Layout<IxD>,
        uplo: FlagUpLo,
    ) -> Result<()> {
        let default_order = self.default_order();
        match default_order {
            RowMajor => pack_tri_cpu_serial(a, la, b, lb, uplo),
            ColMajor => {
                let la = la.reverse_axes();
                let lb = lb.reverse_axes();
                let uplo = uplo.flip()?;
                pack_tri_cpu_serial(a, &la, b, &lb, uplo)
            },
        }
    }
}

impl<T> DeviceOpUnpackTriAPI<T> for DeviceCpuSerial
where
    T: ComplexFloat,
{
    fn unpack_tri(
        &self,
        a: &mut Vec<MaybeUninit<T>>,
        la: &Layout<IxD>,
        b: &Vec<T>,
        lb: &Layout<IxD>,
        uplo: FlagUpLo,
        symm: FlagSymm,
    ) -> Result<()> {
        let default_order = self.default_order();
        match default_order {
            RowMajor => unpack_tri_cpu_serial(a, la, b, lb, uplo, symm),
            ColMajor => {
                let la = la.reverse_axes();
                let lb = lb.reverse_axes();
                let uplo = uplo.flip()?;
                unpack_tri_cpu_serial(a, &la, b, &lb, uplo, symm)
            },
        }
    }
}
