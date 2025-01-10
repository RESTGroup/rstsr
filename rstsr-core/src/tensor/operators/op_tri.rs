use crate::prelude_dev::*;
use num::ToPrimitive;

/* #region pack_tri */

impl<R, T, D, B> TensorBase<R, D>
where
    R: DataMutAPI<Data = Storage<T, B>>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: DeviceAPI<T> + DeviceOpPackTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    pub fn pack_tri_f(&self, uplo: TensorUpLo) -> Result<Tensor<T, D::SmallerOne, B>> {
        // layouts manuplication
        let lb = self.layout().to_dim::<IxD>()?;
        let (lb_rest, lb_inner) = lb.dim_split_at(-2)?;

        // check last two dimensions are equal
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
        let la = match (lb.c_prefer(), lb.f_prefer()) {
            (true, false) => la_shape.c(),
            (false, true) => la_shape.f(),
            _ => match TensorOrder::default() {
                TensorOrder::C => la_shape.c(),
                TensorOrder::F => la_shape.f(),
            },
        };

        // device alloc and compute
        let device = self.device();
        let mut storage_a = unsafe { device.empty_impl(la.bounds_index()?.1)? };
        let storage_b = self.storage();
        device.pack_tri(&mut storage_a, &la, storage_b, &lb, uplo)?;
        Tensor::new_f(DataOwned::from(storage_a), la.into_dim()?)
    }

    pub fn pack_tri(&self, uplo: TensorUpLo) -> Tensor<T, D::SmallerOne, B> {
        self.pack_tri_f(uplo).unwrap()
    }

    pub fn pack_tril_f(&self) -> Result<Tensor<T, D::SmallerOne, B>> {
        self.pack_tri_f(TensorUpLo::L)
    }

    pub fn pack_tril(&self) -> Tensor<T, D::SmallerOne, B> {
        self.pack_tril_f().unwrap()
    }

    pub fn pack_triu_f(&self) -> Result<Tensor<T, D::SmallerOne, B>> {
        self.pack_tri_f(TensorUpLo::U)
    }

    pub fn pack_triu(&self) -> Tensor<T, D::SmallerOne, B> {
        self.pack_triu_f().unwrap()
    }
}

/* #endregion */

/* #region unpack_tri */

impl<R, T, D, B> TensorBase<R, D>
where
    R: DataMutAPI<Data = Storage<T, B>>,
    D: DimAPI + DimLargerOneAPI,
    D::LargerOne: DimAPI,
    B: DeviceAPI<T> + DeviceOpUnpackTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    pub fn unpack_tri(
        &self,
        uplo: TensorUpLo,
        symm: TensorSymm,
    ) -> Result<Tensor<T, D::LargerOne, B>> {
        // layouts manuplication
        let lb = self.layout().to_dim::<IxD>()?;
        let (lb_rest, lb_inner) = lb.dim_split_at(-1)?;

        // check last two dimensions are equal
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
        la_shape.push(n);
        la_shape.push(n);
        let la = match (lb.c_prefer(), lb.f_prefer()) {
            (true, false) => la_shape.c(),
            (false, true) => la_shape.f(),
            _ => match TensorOrder::default() {
                TensorOrder::C => la_shape.c(),
                TensorOrder::F => la_shape.f(),
            },
        };

        // device alloc and compute
        let device = self.device();
        let mut storage_a = unsafe { device.empty_impl(la.bounds_index()?.1)? };
        let storage_b = self.storage();
        device.unpack_tri(&mut storage_a, &la, storage_b, &lb, uplo, symm)?;
        Tensor::new_f(DataOwned::from(storage_a), la.into_dim()?)
    }

    pub fn unpack_tril(&self, symm: TensorSymm) -> Tensor<T, D::LargerOne, B> {
        self.unpack_tri(TensorUpLo::L, symm).unwrap()
    }

    pub fn unpack_triu(&self, symm: TensorSymm) -> Tensor<T, D::LargerOne, B> {
        self.unpack_tri(TensorUpLo::U, symm).unwrap()
    }

    pub fn unpack_tri_f(
        &self,
        uplo: TensorUpLo,
        symm: TensorSymm,
    ) -> Result<Tensor<T, D::LargerOne, B>> {
        self.unpack_tri(uplo, symm)
    }

    pub fn unpack_tril_f(&self, symm: TensorSymm) -> Result<Tensor<T, D::LargerOne, B>> {
        self.unpack_tri(TensorUpLo::L, symm)
    }
}

/* #endregion */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_tri() {
        let a = arange((48., &DeviceCpuSerial)).into_layout([3, 4, 4].f());
        let a_triu = a.pack_tril();
        println!("{:?}", a_triu);
        println!("{:?}", a.slice(0));
        println!("{:?}", a_triu.slice(0).to_vec());
        assert_eq!(a_triu.slice(1).to_vec(), [1., 4., 16., 7., 19., 31., 10., 22., 34., 46.]);

        let b = a_triu.unpack_tril(TensorSymm::Sy);
        println!("{:?}", b);
        assert_eq!(b.slice((0, 1)).to_vec(), [3., 15., 18., 21.]);
    }

    #[test]
    fn test_par_pack_tril_compiles() {
        use num::complex::c64;
        let a = linspace((c64(-2.0, 1.5), c64(1.7, -2.3), 256 * 256 * 256))
            .into_layout([4, 64, 256, 256].f());
        let a_tril = a.pack_tril();
        println!("{:20.5}", a_tril);
        let b = a_tril.unpack_tril(TensorSymm::Ah);
        println!("{:20.5}", b);
    }
}
