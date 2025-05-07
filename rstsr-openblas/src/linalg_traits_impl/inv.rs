use crate::DeviceBLAS;
use duplicate::duplicate_item;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

impl<R, T, D> InvAPI<DeviceBLAS> for &TensorAny<R, T, DeviceBLAS, D>
where
    T: BlasFloat,
    R: DataAPI<Data = Vec<T>>,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn inv_f(self) -> Result<Self::Out> {
        rstsr_assert_eq!(
            self.ndim(),
            2,
            InvalidLayout,
            "Currently we can only handle 2-D matrix."
        )?;
        let a = self.view().into_dim::<Ix2>();
        let result = ref_impl_inv_f(a.into())?.into_owned();
        Ok(result.into_dim::<IxD>().into_dim::<D>())
    }
}

#[duplicate_item(
    TSR;
    [TensorView<'_, T, DeviceBLAS, D>];
    [TensorCow<'_, T, DeviceBLAS, D>]
)]
impl<T, D> InvAPI<DeviceBLAS> for TSR
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn inv_f(self) -> Result<Self::Out> {
        InvAPI::<DeviceBLAS>::inv_f(&self)
    }
}

impl<'a, T, D> InvAPI<DeviceBLAS> for TensorMut<'a, T, DeviceBLAS, D>
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = TensorMutable<'a, T, DeviceBLAS, D>;
    fn inv_f(self) -> Result<Self::Out> {
        rstsr_assert_eq!(
            self.ndim(),
            2,
            InvalidLayout,
            "Currently we can only handle 2-D matrix."
        )?;
        let a = self.into_dim::<Ix2>();
        let result = ref_impl_inv_f(a.into())?;
        Ok(result.into_dim::<IxD>().into_dim::<D>())
    }
}

impl<T, D> InvAPI<DeviceBLAS> for Tensor<T, DeviceBLAS, D>
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn inv_f(mut self) -> Result<Self::Out> {
        InvAPI::<DeviceBLAS>::inv_f(self.view_mut())?;
        Ok(self)
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
        let a_inv = inv(&a);
        println!("{:?}", a_inv.into_owned());
    }
}
