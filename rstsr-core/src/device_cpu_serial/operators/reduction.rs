use crate::device_cpu_serial::reduction::*;
use crate::prelude_dev::*;
use core::ops::{Add, Mul};
use num::complex::ComplexFloat;
use num::{Bounded, FromPrimitive, One, Zero};
use rstsr_dtype_traits::MinMaxAPI;

impl<T, D> OpSumAPI<T, D> for DeviceCpuSerial
where
    T: Zero + Add<Output = T> + Clone,
    D: DimAPI,
{
    fn sum_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T> {
        let f_init = T::zero;
        let f = |acc, x| acc + x;
        let f_sum = |acc1, acc2| acc1 + acc2;
        let f_out = |acc| acc;

        reduce_all_cpu_serial(a, la, f_init, f, f_sum, f_out)
    }

    fn sum(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
        let f_init = T::zero;
        let f = |acc, x| acc + x;
        let f_sum = |acc1, acc2| acc1 + acc2;
        let f_out = |acc| acc;

        let (out, layout_out) =
            reduce_axes_cpu_serial(a, &la.to_dim()?, axes, f_init, f, f_sum, f_out)?;
        Ok((Storage::new(out.into(), self.clone()), layout_out))
    }
}

impl<T, D> OpMinAPI<T, D> for DeviceCpuSerial
where
    T: Clone + MinMaxAPI + Bounded,
    D: DimAPI,
{
    fn min_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T> {
        if la.size() == 0 {
            rstsr_raise!(InvalidValue, "zero-size array is not supported for min")?;
        }

        let f_init = T::max_value;
        let f = |acc: T, x: T| acc.min(x);
        let f_sum = |acc1: T, acc2: T| acc1.min(acc2);
        let f_out = |acc| acc;
        reduce_all_cpu_serial(a, la, f_init, f, f_sum, f_out)
    }

    fn min(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
        if la.size() == 0 {
            rstsr_raise!(InvalidValue, "zero-size array is not supported for min")?;
        }

        let f_init = T::max_value;
        let f = |acc: T, x: T| acc.min(x);
        let f_sum = |acc1: T, acc2: T| acc1.min(acc2);
        let f_out = |acc| acc;

        let (out, layout_out) =
            reduce_axes_cpu_serial(a, &la.to_dim()?, axes, f_init, f, f_sum, f_out)?;
        Ok((Storage::new(out.into(), self.clone()), layout_out))
    }
}

impl<T, D> OpMaxAPI<T, D> for DeviceCpuSerial
where
    T: Clone + MinMaxAPI + Bounded,
    D: DimAPI,
{
    fn max_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T> {
        if la.size() == 0 {
            rstsr_raise!(InvalidValue, "zero-size array is not supported for max")?;
        }

        let f_init = T::min_value;
        let f = |acc: T, x: T| acc.max(x);
        let f_sum = |acc1: T, acc2: T| acc1.max(acc2);
        let f_out = |acc| acc;

        reduce_all_cpu_serial(a, la, f_init, f, f_sum, f_out)
    }

    fn max(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
        if la.size() == 0 {
            rstsr_raise!(InvalidValue, "zero-size array is not supported for max")?;
        }

        let f_init = T::min_value;
        let f = |acc: T, x: T| acc.max(x);
        let f_sum = |acc1: T, acc2: T| acc1.max(acc2);
        let f_out = |acc| acc;

        let (out, layout_out) =
            reduce_axes_cpu_serial(a, &la.to_dim()?, axes, f_init, f, f_sum, f_out)?;
        Ok((Storage::new(out.into(), self.clone()), layout_out))
    }
}

impl<T, D> OpProdAPI<T, D> for DeviceCpuSerial
where
    T: Clone + One + Mul<Output = T>,
    D: DimAPI,
{
    fn prod_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T> {
        let f_init = T::one;
        let f = |acc, x| acc * x;
        let f_sum = |acc1, acc2| acc1 * acc2;
        let f_out = |acc| acc;

        reduce_all_cpu_serial(a, la, f_init, f, f_sum, f_out)
    }

    fn prod(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
        let f_init = T::one;
        let f = |acc, x| acc * x;
        let f_sum = |acc1, acc2| acc1 * acc2;
        let f_out = |acc| acc;

        let (out, layout_out) =
            reduce_axes_cpu_serial(a, &la.to_dim()?, axes, f_init, f, f_sum, f_out)?;
        Ok((Storage::new(out.into(), self.clone()), layout_out))
    }
}

impl<T, D> OpMeanAPI<T, D> for DeviceCpuSerial
where
    T: Clone + ComplexFloat + FromPrimitive,
    D: DimAPI,
{
    fn mean_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T> {
        let size = la.size();
        let f_init = T::zero;
        let f = |acc, x| acc + x;
        let f_sum = |acc, x| acc + x;
        let f_out = |acc| acc / T::from_usize(size).unwrap();

        let sum = reduce_all_cpu_serial(a, la, f_init, f, f_sum, f_out)?;
        Ok(sum)
    }

    fn mean(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
        let (layout_axes, _) = la.dim_split_axes(axes)?;
        let size = layout_axes.size();
        let f_init = T::zero;
        let f = |acc, x| acc + x;
        let f_sum = |acc, x| acc + x;
        let f_out = |acc| acc / T::from_usize(size).unwrap();

        let (out, layout_out) =
            reduce_axes_cpu_serial(a, &la.to_dim()?, axes, f_init, f, f_sum, f_out)?;
        Ok((Storage::new(out.into(), self.clone()), layout_out))
    }
}

impl<T, D> OpVarAPI<T, D> for DeviceCpuSerial
where
    T: Clone + ComplexFloat + FromPrimitive,
    D: DimAPI,
{
    fn var_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T> {
        let size = la.size();

        let f_init = || (T::zero(), T::zero());
        let f = |(acc_1, acc_2), x| (acc_1 + x, acc_2 + x * x);
        let f_sum = |(acc_1, acc_2), (x_1, x_2)| (acc_1 + x_1, acc_2 + x_2);
        let f_out = |(acc_1, acc_2)| {
            let size = T::from_usize(size).unwrap();
            let mean = acc_1 / size;
            acc_2 / size - mean * mean
        };

        let result = reduce_all_cpu_serial(a, la, f_init, f, f_sum, f_out)?;
        Ok(result)
    }

    fn var(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
        let (layout_axes, _) = la.dim_split_axes(axes)?;
        let size = layout_axes.size();

        let f_init = || (T::zero(), T::zero());
        let f = |(acc_1, acc_2), x| (acc_1 + x, acc_2 + x * x);
        let f_sum = |(acc_1, acc_2), (x_1, x_2)| (acc_1 + x_1, acc_2 + x_2);
        let f_out = |(acc_1, acc_2)| {
            let size = T::from_usize(size).unwrap();
            let mean = acc_1 / size;
            acc_2 / size - mean * mean
        };

        let (out, layout_out) =
            reduce_axes_difftype_cpu_serial(a, &la.to_dim()?, axes, f_init, f, f_sum, f_out)?;

        Ok((Storage::new(out.into(), self.clone()), layout_out))
    }
}

impl<T, D> OpStdAPI<T, D> for DeviceCpuSerial
where
    T: Clone + ComplexFloat + FromPrimitive,
    D: DimAPI,
{
    fn std_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T> {
        let size = la.size();

        let f_init = || (T::zero(), T::zero());
        let f = |(acc_1, acc_2), x| (acc_1 + x, acc_2 + x * x);
        let f_sum = |(acc_1, acc_2), (x_1, x_2)| (acc_1 + x_1, acc_2 + x_2);
        let f_out = |(acc_1, acc_2)| {
            let size = T::from_usize(size).unwrap();
            let mean = acc_1 / size;
            let var: T = acc_2 / size - mean * mean;
            var.sqrt()
        };

        let result = reduce_all_cpu_serial(a, la, f_init, f, f_sum, f_out)?;
        Ok(result)
    }

    fn std(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
        let (layout_axes, _) = la.dim_split_axes(axes)?;
        let size = layout_axes.size();

        let f_init = || (T::zero(), T::zero());
        let f = |(acc_1, acc_2), x| (acc_1 + x, acc_2 + x * x);
        let f_sum = |(acc_1, acc_2), (x_1, x_2)| (acc_1 + x_1, acc_2 + x_2);
        let f_out = |(acc_1, acc_2)| {
            let size = T::from_usize(size).unwrap();
            let mean = acc_1 / size;
            let var: T = acc_2 / size - mean * mean;
            var.sqrt()
        };

        let (out, layout_out) =
            reduce_axes_difftype_cpu_serial(a, &la.to_dim()?, axes, f_init, f, f_sum, f_out)?;

        Ok((Storage::new(out.into(), self.clone()), layout_out))
    }
}
