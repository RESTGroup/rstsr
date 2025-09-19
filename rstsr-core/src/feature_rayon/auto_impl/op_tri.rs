use crate::prelude_dev::*;
use num::complex::ComplexFloat;

impl<T> DeviceOpPackTriAPI<T> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync,
{
    fn pack_tri(
        &self,
        a: &mut Vec<MaybeUninit<T>>,
        la: &Layout<IxD>,
        b: &Vec<T>,
        lb: &Layout<IxD>,
        uplo: FlagUpLo,
    ) -> Result<()> {
        let pool = self.get_current_pool();
        let default_order = self.default_order();
        match default_order {
            RowMajor => pack_tri_cpu_rayon(a, la, b, lb, uplo, pool),
            ColMajor => {
                let la = la.reverse_axes();
                let lb = lb.reverse_axes();
                let uplo = uplo.flip()?;
                pack_tri_cpu_rayon(a, &la, b, &lb, uplo, pool)
            },
        }
    }
}

impl<T> DeviceOpUnpackTriAPI<T> for DeviceRayonAutoImpl
where
    T: ComplexFloat + Send + Sync,
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
        let pool = self.get_current_pool();
        let default_order = self.default_order();
        match default_order {
            RowMajor => unpack_tri_cpu_rayon(a, la, b, lb, uplo, symm, pool),
            ColMajor => {
                let la = la.reverse_axes();
                let lb = lb.reverse_axes();
                let uplo = uplo.flip()?;
                unpack_tri_cpu_rayon(a, &la, b, &lb, uplo, symm, pool)
            },
        }
    }
}
