use crate::prelude_dev::*;
use num::ToPrimitive;

/* #region pack_tri */

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: DeviceAPI<T> + DeviceOpPackTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    pub fn pack_tri_f(&self, uplo: FlagUpLo) -> Result<Tensor<T, B, D::SmallerOne>> {
        // layouts manuplication
        let lb = self.layout().to_dim::<IxD>()?;

        // generate layout for output
        let default_order = self.device().default_order();
        let la_shape = match default_order {
            RowMajor => {
                // check last two dimensions are equal
                let (lb_rest, lb_inner) = lb.dim_split_at(-2)?;
                rstsr_assert_eq!(
                    lb_inner.shape()[0],
                    lb_inner.shape()[1],
                    InvalidLayout,
                    "Last two dimensions should be the same for pack_tri."
                )?;
                let n: usize = lb_inner.shape()[0];
                let n_tp = n * (n + 1) / 2;

                // layouts for output
                let mut la_shape = lb_rest.shape().to_vec();
                la_shape.push(n_tp);
                la_shape
            },
            ColMajor => {
                // check first two dimensions are equal
                let (lb_inner, lb_rest) = lb.dim_split_at(2)?;
                rstsr_assert_eq!(
                    lb_inner.shape()[0],
                    lb_inner.shape()[1],
                    InvalidLayout,
                    "First two dimensions should be the same for pack_tri."
                )?;
                let n: usize = lb_inner.shape()[0];
                let n_tp = n * (n + 1) / 2;

                // layouts for output
                let mut la_shape = vec![n_tp];
                la_shape.append(&mut lb_rest.shape().to_vec());
                la_shape
            },
        };

        let la = match (lb.c_prefer(), lb.f_prefer()) {
            (true, false) => la_shape.c(),
            (false, true) => la_shape.f(),
            _ => match self.device().default_order() {
                RowMajor => la_shape.c(),
                ColMajor => la_shape.f(),
            },
        };

        // device alloc and compute
        let device = self.device();
        let mut storage_a = unsafe { device.empty_impl(la.bounds_index()?.1)? };
        device.pack_tri(storage_a.raw_mut(), &la, self.raw(), &lb, uplo)?;
        Tensor::new_f(storage_a, la.into_dim()?)
    }

    pub fn pack_tri(&self, uplo: FlagUpLo) -> Tensor<T, B, D::SmallerOne> {
        self.pack_tri_f(uplo).unwrap()
    }

    pub fn pack_tril_f(&self) -> Result<Tensor<T, B, D::SmallerOne>> {
        self.pack_tri_f(FlagUpLo::L)
    }

    pub fn pack_tril(&self) -> Tensor<T, B, D::SmallerOne> {
        self.pack_tril_f().unwrap()
    }

    pub fn pack_triu_f(&self) -> Result<Tensor<T, B, D::SmallerOne>> {
        self.pack_tri_f(FlagUpLo::U)
    }

    pub fn pack_triu(&self) -> Tensor<T, B, D::SmallerOne> {
        self.pack_triu_f().unwrap()
    }
}

/* #endregion */

/* #region unpack_tri */

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI + DimLargerOneAPI,
    D::LargerOne: DimAPI,
    B: DeviceAPI<T> + DeviceOpUnpackTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    pub fn unpack_tri(&self, uplo: FlagUpLo, symm: FlagSymm) -> Result<Tensor<T, B, D::LargerOne>> {
        // layouts manuplication
        let lb = self.layout().to_dim::<IxD>()?;

        let default_order = self.device().default_order();
        let la_shape = match default_order {
            RowMajor => {
                // check last two dimensions are equal
                let (lb_rest, lb_inner) = lb.dim_split_at(-1)?;
                let n_tp: usize = lb_inner.shape()[0];
                let n: usize = (2 * n_tp).to_f64().unwrap().sqrt().floor().to_usize().unwrap();
                rstsr_assert_eq!(
                    n * (n + 1) / 2,
                    n_tp,
                    InvalidLayout,
                    "Last dimension should be triangular number for unpack_tri."
                )?;

                // layouts for output
                let mut la_shape = lb_rest.shape().to_vec();
                la_shape.append(&mut vec![n, n]);
                la_shape
            },
            ColMajor => {
                // check first two dimensions are equal
                let (lb_inner, lb_rest) = lb.dim_split_at(1)?;
                let n_tp: usize = lb_inner.shape()[0];
                let n: usize = (2 * n_tp).to_f64().unwrap().sqrt().floor().to_usize().unwrap();
                rstsr_assert_eq!(
                    n * (n + 1) / 2,
                    n_tp,
                    InvalidLayout,
                    "First dimension should be triangular number for unpack_tri."
                )?;

                // layouts for output
                let mut la_shape = vec![n, n];
                la_shape.append(&mut lb_rest.shape().to_vec());
                la_shape
            },
        };

        let la = match (lb.c_prefer(), lb.f_prefer()) {
            (true, false) => la_shape.c(),
            (false, true) => la_shape.f(),
            _ => match self.device().default_order() {
                RowMajor => la_shape.c(),
                ColMajor => la_shape.f(),
            },
        };

        // device alloc and compute
        let device = self.device();
        let mut storage_a = unsafe { device.empty_impl(la.bounds_index()?.1)? };
        device.unpack_tri(storage_a.raw_mut(), &la, self.raw(), &lb, uplo, symm)?;
        Tensor::new_f(storage_a, la.into_dim()?)
    }

    pub fn unpack_tril(&self, symm: FlagSymm) -> Tensor<T, B, D::LargerOne> {
        self.unpack_tri(FlagUpLo::L, symm).unwrap()
    }

    pub fn unpack_triu(&self, symm: FlagSymm) -> Tensor<T, B, D::LargerOne> {
        self.unpack_tri(FlagUpLo::U, symm).unwrap()
    }

    pub fn unpack_tri_f(
        &self,
        uplo: FlagUpLo,
        symm: FlagSymm,
    ) -> Result<Tensor<T, B, D::LargerOne>> {
        self.unpack_tri(uplo, symm)
    }

    pub fn unpack_tril_f(&self, symm: FlagSymm) -> Result<Tensor<T, B, D::LargerOne>> {
        self.unpack_tri(FlagUpLo::L, symm)
    }
}

/* #endregion */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_tri() {
        #[cfg(not(feature = "col_major"))]
        {
            let a = {
                let a = arange((48., &DeviceCpuSerial::default()));
                let storage_a = a.into_raw_parts().0;
                Tensor::new(storage_a, [3, 4, 4].f())
            };
            let a_triu = a.pack_tril();
            println!("{:?}", a_triu);
            println!("{:?}", a.slice(0));
            println!("{:?}", a_triu.slice(0).to_vec());
            assert_eq!(a_triu.slice(1).to_vec(), [1., 4., 16., 7., 19., 31., 10., 22., 34., 46.]);

            let b = a_triu.unpack_tril(FlagSymm::Sy);
            println!("{:?}", b);
            assert_eq!(b.slice((0, 1)).to_vec(), [3., 15., 18., 21.]);
        }
        #[cfg(feature = "col_major")]
        {
            let a = {
                let a = arange((48., &DeviceCpuSerial::default()));
                let storage_a = a.into_raw_parts().0;
                Tensor::new(storage_a, [4, 4, 3].c())
            };
            let a_triu = a.pack_triu();
            println!("{:?}", a_triu);
            println!("{:?}", a.slice((.., 0)));
            println!("{:?}", a_triu.slice((.., 0)).to_vec());
            assert_eq!(a_triu.slice((.., 1)).to_vec(), [
                1., 4., 16., 7., 19., 31., 10., 22., 34., 46.
            ]);

            let b = a_triu.unpack_triu(FlagSymm::Sy);
            println!("{:?}", b);
            assert_eq!(b.slice((.., 1, 0)).to_vec(), [3., 15., 18., 21.]);
        }
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_par_pack_tril_compiles() {
        #[cfg(not(feature = "col_major"))]
        {
            use num::complex::c64;
            let a = linspace((c64(-2.0, 1.5), c64(1.7, -2.3), 256 * 256 * 256))
                .into_layout([4, 64, 256, 256].f());
            let a_tril = a.pack_tril();
            println!("{:20.5}", a_tril);
            let b = a_tril.unpack_tril(FlagSymm::Ah);
            println!("{:20.5}", b);
        }
        #[cfg(feature = "col_major")]
        {
            use num::complex::c64;
            let a = linspace((c64(-2.0, 1.5), c64(1.7, -2.3), 256 * 256 * 256))
                .into_layout([256, 256, 64, 4].c());
            let a_tril = a.pack_tril();
            println!("{:20.5}", a_tril);
            let b = a_tril.unpack_tril(FlagSymm::Ah);
            println!("{:20.5}", b);
        }
    }

    #[test]
    fn test_correctness() {
        let a = {
            let a = arange((16., &DeviceCpuSerial::default()));
            let storage_a = a.into_raw_parts().0;
            Tensor::new(storage_a, [4, 4].c())
        };
        println!("{:}", a);

        let a_tril1 = a.pack_tril();
        println!("{:}", a_tril1);

        let a = a.to_contig(FlagOrder::F);
        let a_tril2 = a.pack_tril();
        println!("{:}", a_tril2);

        assert!((&a_tril1 - &a_tril2).l2_norm_all() < 1e-6);
    }
}
