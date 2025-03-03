
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
        let nthreads = self.get_num_threads();
        pack_tri_cpu_rayon(a, la, b, lb, uplo, nthreads)
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
        let nthreads = self.get_num_threads();
        unpack_tri_cpu_rayon(a, la, b, lb, uplo, symm, nthreads)
    }
}
