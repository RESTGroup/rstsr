use crate::feature_rayon::*;
use crate::prelude_dev::*;
use num::complex::ComplexFloat;

impl<T> DeviceOpPackTriAPI<T> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync,
{
    fn pack_tri(
        &self,
        a: &mut Vec<T>,
        la: &Layout<IxD>,
        b: &Vec<T>,
        lb: &Layout<IxD>,
        uplo: FlagUpLo,
    ) -> Result<()> {
        let pool = self.get_current_pool();
        pack_tri_cpu_rayon(a, la, b, lb, uplo, pool)
    }
}

impl<T> DeviceOpUnpackTriAPI<T> for DeviceRayonAutoImpl
where
    T: ComplexFloat + Send + Sync,
{
    fn unpack_tri(
        &self,
        a: &mut Vec<T>,
        la: &Layout<IxD>,
        b: &Vec<T>,
        lb: &Layout<IxD>,
        uplo: FlagUpLo,
        symm: FlagSymm,
    ) -> Result<()> {
        let pool = self.get_current_pool();
        unpack_tri_cpu_rayon(a, la, b, lb, uplo, symm, pool)
    }
}
