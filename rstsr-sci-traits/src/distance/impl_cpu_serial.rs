use crate::distance::metric::{MetricDistAPI, MetricDistWeightedAPI, MetricEuclidean};
use crate::distance::native_impl::{cdist_serial, cdist_weighted_serial};
use crate::distance::traits::CDistAPI;
use num::Float;
use rstsr_core::prelude_dev::*;

impl<T, D, M, TW, DW> CDistAPI<DeviceCpuSerial>
    for (
        TensorView<'_, T, DeviceCpuSerial, D>,
        TensorView<'_, T, DeviceCpuSerial, D>,
        M,
        TensorView<'_, TW, DeviceCpuSerial, DW>,
    )
where
    M: MetricDistWeightedAPI<Vec<T>, Weight = Vec<TW>, Out = TW>,
    TW: Float + Clone,
    DeviceCpuSerial: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<TW, Raw = M::Weight>
        + DeviceAPI<M::Out, Raw = Vec<M::Out>>
        + DeviceCreationAnyAPI<M::Out>
        + DeviceCreationAnyAPI<TW>
        + OpAssignArbitaryAPI<TW, DW, DW>
        + OpAssignAPI<TW, DW>,
    D: DimAPI + DimIntoAPI<Ix2>,
    DW: DimAPI + DimIntoAPI<Ix1>,
{
    type Out = Tensor<M::Out, DeviceCpuSerial, D>;

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
        let dist = cdist_weighted_serial(xa.raw(), xb.raw(), &la, &lb, weight.raw(), kernel, order)?;

        let m = la.shape()[0];
        let n = lb.shape()[0];
        asarray_f((dist, [m, n], &device))?.into_dim_f::<D>()
    }
}

impl<T, D, M> CDistAPI<DeviceCpuSerial>
    for (TensorView<'_, T, DeviceCpuSerial, D>, TensorView<'_, T, DeviceCpuSerial, D>, M)
where
    DeviceCpuSerial: DeviceAPI<T, Raw = Vec<T>> + DeviceAPI<M::Out, Raw = Vec<M::Out>> + DeviceCreationAnyAPI<M::Out>,
    M: MetricDistAPI<Vec<T>>,
    D: DimAPI + DimIntoAPI<Ix2>,
{
    type Out = Tensor<M::Out, DeviceCpuSerial, D>;

    fn cdist_f(self) -> Result<Self::Out> {
        let (xa, xb, kernel) = self;
        rstsr_assert_eq!(xa.ndim(), 2, InvalidLayout, "xa must be a 2D tensor")?;
        rstsr_assert_eq!(xb.ndim(), 2, InvalidLayout, "xb must be a 2D tensor")?;
        rstsr_assert!(xa.device().same_device(xb.device()), DeviceMismatch)?;
        let la = xa.layout().to_dim::<Ix2>()?;
        let lb = xb.layout().to_dim::<Ix2>()?;
        let device = xa.device().clone();
        let order = device.default_order();
        let dist = cdist_serial(xa.raw(), xb.raw(), &la, &lb, kernel, order)?;

        let m = la.shape()[0];
        let n = lb.shape()[0];
        asarray_f((dist, [m, n], &device))?.into_dim_f::<D>()
    }
}

impl<T, D> CDistAPI<DeviceCpuSerial> for (TensorView<'_, T, DeviceCpuSerial, D>, TensorView<'_, T, DeviceCpuSerial, D>)
where
    T: Float,
    DeviceCpuSerial: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T>,
    D: DimAPI + DimIntoAPI<Ix2>,
{
    type Out = Tensor<T, DeviceCpuSerial, D>;

    fn cdist_f(self) -> Result<Self::Out> {
        let (xa, xb) = self;
        CDistAPI::<DeviceCpuSerial>::cdist_f((xa, xb, MetricEuclidean))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::distance::metric::*;
    use crate::distance::traits::cdist;

    #[test]
    fn playground() {
        let device = DeviceCpuSerial::default();
        let a = linspace((0., 1., 64, &device)).into_shape((16, 4));
        let b = linspace((0., 1., 80, &device)).into_shape((20, 4)).into_flip(-1);

        let d = cdist((a.view(), b.view(), MetricEuclidean));
        println!("{d:16.8?}");

        let d = cdist((a.view(), b.view()));
        println!("{d:16.8?}");

        let w = asarray((vec![1.5, 1.2, 0.7, 1.3], &device));
        let d_w = cdist((a.view(), b.view(), MetricEuclidean, w.view()));
        println!("{d_w:16.8?}");

        let d = cdist((a.view(), b.view(), MetricMinkowski::new(3.0)));
        println!("{d:16.8?}");

        let d = cdist((a.view(), b.view(), MetricCityBlock));
        println!("{d:16.8?}");

        let d = cdist((a.view(), b.view(), MetricSqEuclidean));
        println!("{d:16.8?}");

        let a = linspace((0., 1., 64, &device)).into_shape((16, 4));
        let b = linspace((0., 1., 80, &device)).into_shape((20, 4)).into_flip(-1);
        let d = cdist((a.view(), b.view(), MetricHamming));
        println!("{d:16.8?}");

        let d = cdist((a.view(), b.view(), MetricHamming, w.view()));
        println!("{d:16.8?}");

        let d = cdist((a.view(), b.view(), MetricBrayCurtis, w.view()));
        println!("{d:16.8?}");

        let d = cdist((a.view(), b.view(), MetricCosine::new(), w.view()));
        println!("{d:16.8?}");

        let d = cdist((a.view(), b.view(), MetricJensenShannon::new()));
        println!("{d:16.8?}");

        let vec_a =
            vec![true, false, false, false, false, false, true, true, true, true, true, false, true, true, true, true];
        let vec_b = vec![
            false, false, true, false, false, true, true, false, false, true, true, false, false, false, false, false,
        ];
        let a = asarray((vec_a, &device)).into_shape((4, 4));
        let b = asarray((vec_b, &device)).into_shape((4, 4));

        let d = cdist((a.view(), b.view(), MetricSokalSneath, w.view()));
        println!("{d:16.8?}");

        let d = cdist((a.view(), b.view(), MetricYule, w.view()));
        println!("{d:16.8?}");

        let a = asarray((vec![-3., 6., 3., 6., -3., 3., 6., 1., 0., -8., 1., -2.], &device)).into_shape((3, 4));
        let b = asarray((vec![3., 1., -1., 1., -6., 0., 3., -1., -5., -4., 4., -2.], &device)).into_shape((3, 4));

        let d = cdist((a.view(), b.view(), MetricCorrelation::default(), w.view()));
        println!("{d:16.8?}");

        let d = cdist((a.abs().view(), b.abs().view(), MetricJensenShannon::default()));
        println!("{d:16.8?}");
    }
}
