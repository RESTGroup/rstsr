use crate::prelude_dev::*;

use num::Float;
use rstsr_sci_traits::distance::metric::{MetricDistAPI, MetricDistWeightedAPI, MetricEuclidean};
use rstsr_sci_traits::distance::native_impl::{cdist_rayon, cdist_weighted_rayon};
use rstsr_sci_traits::distance::traits::CDistAPI;

impl<T, D, M, TW, DW> CDistAPI<DeviceRayonAutoImpl>
    for (
        TensorView<'_, T, DeviceRayonAutoImpl, D>,
        TensorView<'_, T, DeviceRayonAutoImpl, D>,
        M,
        TensorView<'_, TW, DeviceRayonAutoImpl, DW>,
    )
where
    M: MetricDistWeightedAPI<Vec<T>, Weight = Vec<TW>, Out = TW> + Send + Sync,
    T: Send + Sync,
    TW: Float + Send + Sync,
    M::Out: Send + Sync,
    DeviceRayonAutoImpl: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<TW, Raw = M::Weight>
        + DeviceAPI<M::Out, Raw = Vec<M::Out>>
        + DeviceCreationAnyAPI<M::Out>
        + DeviceCreationAnyAPI<TW>
        + OpAssignArbitaryAPI<TW, DW, DW>
        + OpAssignAPI<TW, DW>,
    D: DimAPI + DimIntoAPI<Ix2>,
    DW: DimAPI + DimIntoAPI<Ix1>,
{
    type Out = Tensor<M::Out, DeviceRayonAutoImpl, D>;

    fn cdist_f(self) -> Result<Self::Out> {
        let (xa, xb, kernel, weight) = self;
        rstsr_assert_eq!(xa.ndim(), 2, InvalidLayout, "xa must be a 2D tensor")?;
        rstsr_assert_eq!(xb.ndim(), 2, InvalidLayout, "xb must be a 2D tensor")?;
        rstsr_assert_eq!(weight.ndim(), 1, InvalidLayout, "weight must be a 1D tensor")?;
        rstsr_assert!(xa.device().same_device(xb.device()), DeviceMismatch)?;
        rstsr_assert!(xa.device().same_device(weight.device()), DeviceMismatch)?;
        let la = xa.layout().to_dim::<Ix2>()?;
        let lb = xb.layout().to_dim::<Ix2>()?;
        let device = xa.device().clone();
        let order = device.default_order();
        let weight = weight.into_contig_f(RowMajor)?;
        let pool = device.get_current_pool();
        let dist = cdist_weighted_rayon(xa.raw(), xb.raw(), &la, &lb, weight.raw(), kernel, order, pool)?;

        let m = la.shape()[0];
        let n = lb.shape()[0];
        asarray_f((dist, [m, n], &device))?.into_dim_f::<D>()
    }
}

impl<T, D, M> CDistAPI<DeviceRayonAutoImpl>
    for (TensorView<'_, T, DeviceRayonAutoImpl, D>, TensorView<'_, T, DeviceRayonAutoImpl, D>, M)
where
    M: MetricDistAPI<Vec<T>> + Send + Sync,
    T: Send + Sync,
    M::Out: Send + Sync,
    DeviceRayonAutoImpl:
        DeviceAPI<T, Raw = Vec<T>> + DeviceAPI<M::Out, Raw = Vec<M::Out>> + DeviceCreationAnyAPI<M::Out>,
    D: DimAPI + DimIntoAPI<Ix2>,
{
    type Out = Tensor<M::Out, DeviceRayonAutoImpl, D>;

    fn cdist_f(self) -> Result<Self::Out> {
        let (xa, xb, kernel) = self;
        rstsr_assert_eq!(xa.ndim(), 2, InvalidLayout, "xa must be a 2D tensor")?;
        rstsr_assert_eq!(xb.ndim(), 2, InvalidLayout, "xb must be a 2D tensor")?;
        rstsr_assert!(xa.device().same_device(xb.device()), DeviceMismatch)?;
        let la = xa.layout().to_dim::<Ix2>()?;
        let lb = xb.layout().to_dim::<Ix2>()?;
        let device = xa.device().clone();
        let order = device.default_order();
        let pool = device.get_current_pool();
        let dist = cdist_rayon(xa.raw(), xb.raw(), &la, &lb, kernel, order, pool)?;

        let m = la.shape()[0];
        let n = lb.shape()[0];
        asarray_f((dist, [m, n], &device))?.into_dim_f::<D>()
    }
}

impl<T, D> CDistAPI<DeviceRayonAutoImpl>
    for (TensorView<'_, T, DeviceRayonAutoImpl, D>, TensorView<'_, T, DeviceRayonAutoImpl, D>)
where
    T: Float + Send + Sync,
    DeviceRayonAutoImpl: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T>,
    D: DimAPI + DimIntoAPI<Ix2>,
{
    type Out = Tensor<T, DeviceRayonAutoImpl, D>;

    fn cdist_f(self) -> Result<Self::Out> {
        let (xa, xb) = self;
        CDistAPI::<DeviceRayonAutoImpl>::cdist_f((xa, xb, MetricEuclidean))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rstsr_sci_traits::distance::metric::MetricEuclidean;
    use rstsr_sci_traits::distance::traits::cdist;

    #[test]
    fn playground() {
        let device = DeviceRayonAutoImpl::default();
        let a = linspace((0., 1., 6400, &device)).into_shape((1600, 4));
        let b = linspace((0., 1., 8000, &device)).into_shape((2000, 4)).into_flip(-1);

        let d = cdist((a.view(), b.view(), MetricEuclidean));
        println!("{d:16.8?}");

        let d = cdist((a.view(), b.view()));
        println!("{d:16.8?}");

        let w = asarray((vec![1.5, 1.2, 0.7, 1.3], &device));
        let d_w = cdist((a.view(), b.view(), MetricEuclidean, w.view()));
        println!("{d_w:16.8?}");
    }
}
