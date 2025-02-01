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
        reduce_all_cpu_serial(a, la, T::zero, |acc, x| acc + x, |acc| acc)
    }

    fn sum(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
        let (out, layout_out) =
            reduce_axes_cpu_serial(a, &la.to_dim()?, axes, T::zero, |acc, x| acc + x, |acc| acc)?;
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
        reduce_all_cpu_serial(a, la, || T::max_value(), |acc, x| acc.min(x), |acc| acc)
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
        let (out, layout_out) = reduce_axes_cpu_serial(
            a,
            &la.to_dim()?,
            axes,
            || T::max_value(),
            |acc, x| acc.min(x),
            |acc| acc,
        )?;
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
        reduce_all_cpu_serial(a, la, || T::min_value(), |acc, x| acc.max(x), |acc| acc)
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
        let (out, layout_out) = reduce_axes_cpu_serial(
            a,
            &la.to_dim()?,
            axes,
            || T::min_value(),
            |acc, x| acc.max(x),
            |acc| acc,
        )?;
        Ok((Storage::new(out.into(), self.clone()), layout_out))
    }
}

impl<T, D> OpProdAPI<T, D> for DeviceCpuSerial
where
    T: Clone + One + Mul<Output = T>,
    D: DimAPI,
{
    fn prod_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T> {
        reduce_all_cpu_serial(a, la, T::one, |acc, x| acc * x, |acc| acc)
    }

    fn prod(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
        let (out, layout_out) =
            reduce_axes_cpu_serial(a, &la.to_dim()?, axes, T::one, |acc, x| acc * x, |acc| acc)?;
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
        let sum = reduce_all_cpu_serial(
            a,
            la,
            T::zero,
            |acc, x| acc + x,
            |acc| acc / T::from_usize(size).unwrap(),
        )?;
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
        let (out, layout_out) = reduce_axes_cpu_serial(
            a,
            &la.to_dim()?,
            axes,
            T::zero,
            |acc, x| acc + x,
            |acc| acc / T::from_usize(size).unwrap(),
        )?;
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
        let e_x1 = reduce_all_cpu_serial(
            a,
            la,
            T::zero,
            |acc, x| acc + x,
            |acc| acc / T::from_usize(size).unwrap(),
        )?;
        let e_x2 = reduce_all_cpu_serial(
            a,
            la,
            T::zero,
            |acc, x| acc + x.clone() * x.clone(),
            |acc| acc / T::from_usize(size).unwrap(),
        )?;
        Ok(e_x2 - e_x1.clone() * e_x1.clone())
    }

    fn var(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
        let (layout_axes, _) = la.dim_split_axes(axes)?;
        let size = layout_axes.size();
        let (mut e_x1, layout_out) = reduce_axes_cpu_serial(
            a,
            &la.to_dim()?,
            axes,
            T::zero,
            |acc, x| acc + x,
            |acc| acc / T::from_usize(size).unwrap(),
        )?;
        let (e_x2, _) = reduce_axes_cpu_serial(
            a,
            &la.to_dim()?,
            axes,
            T::zero,
            |acc, x| acc + x.clone() * x.clone(),
            |acc| acc / T::from_usize(size).unwrap(),
        )?;
        e_x1.iter_mut()
            .zip(e_x2.iter())
            .for_each(|(x1, x2)| *x1 = x2.clone() - x1.clone() * x1.clone());
        Ok((Storage::new(e_x1.into(), self.clone()), layout_out))
    }
}

impl<T, D> OpStdAPI<T, D> for DeviceCpuSerial
where
    T: Clone + ComplexFloat + FromPrimitive,
    D: DimAPI,
{
    fn std_all(&self, a: &Vec<T>, la: &Layout<D>) -> Result<T> {
        let size = la.size();
        let e_x1 = reduce_all_cpu_serial(
            a,
            la,
            T::zero,
            |acc, x| acc + x,
            |acc| acc / T::from_usize(size).unwrap(),
        )?;
        let e_x2 = reduce_all_cpu_serial(
            a,
            la,
            T::zero,
            |acc, x| acc + x.clone() * x.clone(),
            |acc| acc / T::from_usize(size).unwrap(),
        )?;
        Ok((e_x2 - e_x1.clone() * e_x1.clone()).sqrt())
    }

    fn std(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
        let (layout_axes, _) = la.dim_split_axes(axes)?;
        let size = layout_axes.size();
        let (mut e_x1, layout_out) = reduce_axes_cpu_serial(
            a,
            &la.to_dim()?,
            axes,
            T::zero,
            |acc, x| acc + x,
            |acc| acc / T::from_usize(size).unwrap(),
        )?;
        let (e_x2, _) = reduce_axes_cpu_serial(
            a,
            &la.to_dim()?,
            axes,
            T::zero,
            |acc, x| acc + x.clone() * x.clone(),
            |acc| acc / T::from_usize(size).unwrap(),
        )?;
        e_x1.iter_mut()
            .zip(e_x2.iter())
            .for_each(|(x1, x2)| *x1 = (x2.clone() - x1.clone() * x1.clone()).sqrt());
        Ok((Storage::new(e_x1.into(), self.clone()), layout_out))
    }
}
