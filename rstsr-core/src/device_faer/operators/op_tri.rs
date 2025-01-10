use crate::feature_rayon::*;
use crate::prelude_dev::*;
use num::complex::ComplexFloat;

impl<T> DeviceOpPackTriAPI<T> for DeviceFaer
where
    T: Clone + Send + Sync,
{
    fn pack_tri(
        &self,
        a: &mut Storage<T, Self>,
        la: &Layout<IxD>,
        b: &Storage<T, Self>,
        lb: &Layout<IxD>,
        uplo: TensorUpLo,
    ) -> Result<()> {
        let a = a.rawvec_mut();
        let b = b.rawvec();
        let nthreads = self.get_num_threads();
        pack_tri_cpu_rayon(a, la, b, lb, uplo, nthreads)
    }
}

impl<T> DeviceOpUnpackTriAPI<T> for DeviceFaer
where
    T: ComplexFloat + Send + Sync,
{
    fn unpack_tri(
        &self,
        a: &mut Storage<T, Self>,
        la: &Layout<IxD>,
        b: &Storage<T, Self>,
        lb: &Layout<IxD>,
        uplo: TensorUpLo,
        symm: TensorSymm,
    ) -> Result<()> {
        let a = a.rawvec_mut();
        let b = b.rawvec();
        let nthreads = self.get_num_threads();
        unpack_tri_cpu_rayon(a, la, b, lb, uplo, symm, nthreads)
    }
}
