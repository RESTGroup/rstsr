use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

impl<R, T, D> EighAPI<DeviceBLAS> for &TensorAny<R, T, DeviceBLAS, D>
where
    T: BlasFloat + Send + Sync,
    R: DataCloneAPI<Data = Vec<T>>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: BlasThreadAPI
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<T::Real, Raw = Vec<T::Real>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceComplexFloatAPI<T::Real, Ix2>
        + SYGVDriverAPI<T>
        + SYGVDDriverAPI<T>
        + SYEVDriverAPI<T>
        + SYEVDDriverAPI<T>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tensor<T, DeviceBLAS, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let a = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let eigh_args = EighArgs::default().a(a_view).build()?;
        let (vals, vecs) = blas_eigh_simple_f(eigh_args)?;
        let vals = vals.into_dim::<IxD>().into_dim::<D::SmallerOne>();
        let vecs = vecs.unwrap().into_owned().into_dim::<IxD>().into_dim::<D>();
        return Ok(EighResult { eigenvalues: vals, eigenvectors: vecs });
    }
}

impl<T, D> EighAPI<DeviceBLAS> for TensorView<'_, T, DeviceBLAS, D>
where
    T: BlasFloat + Send + Sync,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: BlasThreadAPI
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<T::Real, Raw = Vec<T::Real>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceComplexFloatAPI<T::Real, Ix2>
        + SYGVDriverAPI<T>
        + SYGVDDriverAPI<T>
        + SYEVDriverAPI<T>
        + SYEVDDriverAPI<T>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tensor<T, DeviceBLAS, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        EighAPI::<DeviceBLAS>::eigh_f(&self)
    }
}

impl<Ra, Rb, T, D> EighAPI<DeviceBLAS>
    for (&TensorAny<Ra, T, DeviceBLAS, D>, &TensorAny<Rb, T, DeviceBLAS, D>)
where
    T: BlasFloat + Send + Sync,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    Ra: DataCloneAPI<Data = Vec<T>>,
    Rb: DataCloneAPI<Data = Vec<T>>,
    DeviceBLAS: BlasThreadAPI
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<T::Real, Raw = Vec<T::Real>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceComplexFloatAPI<T::Real, Ix2>
        + SYGVDriverAPI<T>
        + SYGVDDriverAPI<T>
        + SYEVDriverAPI<T>
        + SYEVDDriverAPI<T>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tensor<T, DeviceBLAS, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, b) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = b.view().into_dim::<Ix2>();
        let eigh_args = EighArgs::default().a(a_view).b(b_view).build()?;
        let (vals, vecs) = blas_eigh_simple_f(eigh_args)?;
        let vals = vals.into_dim::<IxD>().into_dim::<D::SmallerOne>();
        let vecs = vecs.unwrap().into_owned().into_dim::<IxD>().into_dim::<D>();
        return Ok(EighResult { eigenvalues: vals, eigenvectors: vecs });
    }
}

impl<T, D> EighAPI<DeviceBLAS>
    for (TensorView<'_, T, DeviceBLAS, D>, TensorView<'_, T, DeviceBLAS, D>)
where
    T: BlasFloat + Send + Sync,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: BlasThreadAPI
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<T::Real, Raw = Vec<T::Real>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceComplexFloatAPI<T::Real, Ix2>
        + SYGVDriverAPI<T>
        + SYGVDDriverAPI<T>
        + SYEVDriverAPI<T>
        + SYEVDDriverAPI<T>,
{
    type Out = EighResult<Tensor<T::Real, DeviceBLAS, D::SmallerOne>, Tensor<T, DeviceBLAS, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, b) = self;
        EighAPI::<DeviceBLAS>::eigh_f((&a, &b))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::DeviceBLAS;
    use rstsr_test_manifest::get_vec;

    #[test]
    fn playground() {
        let device = DeviceBLAS::default();
        let la = [2048, 2048].c();
        let a = Tensor::new(Storage::new(get_vec::<f64>('a').into(), device.clone()), la);
        let (w, v) = eigh(&a).into();
        println!("{:8.3?}", v);
        println!("{:8.3?}", w);

        let device = DeviceBLAS::default();
        let vec_a = [0, 1, 9, 1, 5, 3, 9, 3, 6].iter().map(|&x| x as f64).collect::<Vec<_>>();
        let a = asarray((vec_a, [3, 3].c(), &device)).into_dim::<Ix2>();
        let vec_b = [1, 1, 2, 1, 3, 1, 2, 1, 8].iter().map(|&x| x as f64).collect::<Vec<_>>();
        let b = asarray((vec_b, [3, 3].c(), &device)).into_dim::<Ix2>();
        let (w, v) = eigh((&a, &b)).into();
        println!("{:?}", v);
        println!("{:?}", w);
    }
}
