#[macro_export]
macro_rules! macro_impl_rayon_op_tri {
    ($Device: ident) => {
        use num::complex::ComplexFloat;
        use $crate::feature_rayon::*;
        use $crate::prelude_dev::*;

        impl<T> DeviceOpPackTriAPI<T> for $Device
        where
            T: Clone + Send + Sync,
        {
            fn pack_tri(
                &self,
                a: &mut Vec<T>,
                la: &Layout<IxD>,
                b: &Vec<T>,
                lb: &Layout<IxD>,
                uplo: TensorUpLo,
            ) -> Result<()> {
                let nthreads = self.get_num_threads();
                pack_tri_cpu_rayon(a, la, b, lb, uplo, nthreads)
            }
        }

        impl<T> DeviceOpUnpackTriAPI<T> for $Device
        where
            T: ComplexFloat + Send + Sync,
        {
            fn unpack_tri(
                &self,
                a: &mut Vec<T>,
                la: &Layout<IxD>,
                b: &Vec<T>,
                lb: &Layout<IxD>,
                uplo: TensorUpLo,
                symm: TensorSymm,
            ) -> Result<()> {
                let nthreads = self.get_num_threads();
                unpack_tri_cpu_rayon(a, la, b, lb, uplo, symm, nthreads)
            }
        }
    };
}
