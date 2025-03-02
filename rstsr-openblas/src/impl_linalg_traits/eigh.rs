use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

impl<R, T> LinalgEighAPI<DeviceBLAS> for &TensorAny<R, T, DeviceBLAS, Ix2>
where
    T: BlasFloat + Send + Sync,
    R: DataAPI<Data = Vec<T>>,
    DeviceBLAS: BlasThreadAPI
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<T::Real, Raw = Vec<T::Real>>
        + DeviceAPI<blasint, Raw = Vec<blasint>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceComplexFloatAPI<T::Real, Ix2>
        + DeviceNumAPI<blasint, Ix1>
        + SYGVDriverAPI<T>
        + SYGVDDriverAPI<T>
        + SYEVDriverAPI<T>
        + SYEVDDriverAPI<T>,
{
    type Out = (Tensor<T::Real, DeviceBLAS, Ix1>, Tensor<T, DeviceBLAS, Ix2>);
    fn eigh_f(args: Self) -> Result<Self::Out> {
        let a = args;
        let eigh_args = EighArgs::default().a(a.view()).build()?;
        let (v, w) = blas_eigh_simple_f(eigh_args)?;
        return Ok((v, w.unwrap().into_owned()));
    }
}

impl<T> LinalgEighAPI<DeviceBLAS> for TensorView<'_, T, DeviceBLAS, Ix2>
where
    T: BlasFloat + Send + Sync,
    DeviceBLAS: BlasThreadAPI
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<T::Real, Raw = Vec<T::Real>>
        + DeviceAPI<blasint, Raw = Vec<blasint>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceComplexFloatAPI<T::Real, Ix2>
        + DeviceNumAPI<blasint, Ix1>
        + SYGVDriverAPI<T>
        + SYGVDDriverAPI<T>
        + SYEVDriverAPI<T>
        + SYEVDDriverAPI<T>,
{
    type Out = (Tensor<T::Real, DeviceBLAS, Ix1>, Tensor<T, DeviceBLAS, Ix2>);
    fn eigh_f(args: Self) -> Result<Self::Out> {
        LinalgEighAPI::<DeviceBLAS>::eigh_f(&args)
    }
}

impl<Ra, Rb, T> LinalgEighAPI<DeviceBLAS>
    for (&TensorAny<Ra, T, DeviceBLAS, Ix2>, &TensorAny<Rb, T, DeviceBLAS, Ix2>)
where
    T: BlasFloat + Send + Sync,
    Ra: DataAPI<Data = Vec<T>>,
    Rb: DataAPI<Data = Vec<T>>,
    DeviceBLAS: BlasThreadAPI
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<T::Real, Raw = Vec<T::Real>>
        + DeviceAPI<blasint, Raw = Vec<blasint>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceComplexFloatAPI<T::Real, Ix2>
        + DeviceNumAPI<blasint, Ix1>
        + SYGVDriverAPI<T>
        + SYGVDDriverAPI<T>
        + SYEVDriverAPI<T>
        + SYEVDDriverAPI<T>,
{
    type Out = (Tensor<T::Real, DeviceBLAS, Ix1>, Tensor<T, DeviceBLAS, Ix2>);
    fn eigh_f(args: Self) -> Result<Self::Out> {
        let (a, b) = args;
        let eigh_args = EighArgs::default().a(a.view()).b(b.view()).build()?;
        let (v, w) = blas_eigh_simple_f(eigh_args)?;
        return Ok((v, w.unwrap().into_owned()));
    }
}

impl<T> LinalgEighAPI<DeviceBLAS>
    for (TensorView<'_, T, DeviceBLAS, Ix2>, TensorView<'_, T, DeviceBLAS, Ix2>)
where
    T: BlasFloat + Send + Sync,
    DeviceBLAS: BlasThreadAPI
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<T::Real, Raw = Vec<T::Real>>
        + DeviceAPI<blasint, Raw = Vec<blasint>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceComplexFloatAPI<T::Real, Ix2>
        + DeviceNumAPI<blasint, Ix1>
        + SYGVDriverAPI<T>
        + SYGVDDriverAPI<T>
        + SYEVDriverAPI<T>
        + SYEVDDriverAPI<T>,
{
    type Out = (Tensor<T::Real, DeviceBLAS, Ix1>, Tensor<T, DeviceBLAS, Ix2>);
    fn eigh_f(args: Self) -> Result<Self::Out> {
        let (a, b) = args;
        LinalgEighAPI::<DeviceBLAS>::eigh_f((&a, &b))
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
        let (v, w) = eigh(&a);
        println!("{:8.3?}", v);
        println!("{:8.3?}", w);

        let device = DeviceBLAS::default();
        let vec_a = [0, 1, 9, 1, 5, 3, 9, 3, 6].iter().map(|&x| x as f64).collect::<Vec<_>>();
        let a = asarray((vec_a, [3, 3].c(), &device)).into_dim::<Ix2>();
        let vec_b = [1, 1, 2, 1, 3, 1, 2, 1, 8].iter().map(|&x| x as f64).collect::<Vec<_>>();
        let b = asarray((vec_b, [3, 3].c(), &device)).into_dim::<Ix2>();
        let (v, w) = eigh((&a, &b));
        println!("{:?}", v);
        println!("{:?}", w);
    }
}
