use crate::prelude_dev::*;

/* #region sum */

pub fn sum_all_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> Result<T>
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: OpSumAPI<T, D>,
{
    tensor.device().sum_all(tensor.raw(), tensor.layout())
}

pub fn sum_f<R, T, B, D, I>(tensor: &TensorAny<R, T, B, D>, axes: I) -> Result<Tensor<T, B, IxD>>
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: OpSumAPI<T, D> + DeviceAPI<T> + DeviceCreationAnyAPI<T>,
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
{
    let axes = axes.try_into()?;

    // special case for summing all axes
    if axes.as_ref().is_empty() {
        let sum = tensor.device().sum_all(tensor.raw(), tensor.layout())?;
        let storage = tensor.device().outof_cpu_vec(vec![sum])?;
        let layout = Layout::new(vec![], vec![], 0)?;
        return Tensor::new_f(storage, layout);
    }

    let (storage, layout) = tensor.device().sum(tensor.raw(), tensor.layout(), axes.as_ref())?;
    Tensor::new_f(storage, layout)
}

pub fn sum_all<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> T
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: OpSumAPI<T, D>,
{
    sum_all_f(tensor).unwrap()
}

pub fn sum<R, T, B, D, I>(tensor: &TensorAny<R, T, B, D>, axes: I) -> Tensor<T, B, IxD>
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: OpSumAPI<T, D> + DeviceCreationAnyAPI<T>,
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
{
    sum_f(tensor, axes).unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: OpSumAPI<T, D>,
{
    pub fn sum_all_f(&self) -> Result<T> {
        sum_all_f(self)
    }

    pub fn sum_all(&self) -> T {
        sum_all(self)
    }

    pub fn sum_f<I>(&self, axes: I) -> Result<Tensor<T, B, IxD>>
    where
        B: DeviceCreationAnyAPI<T>,
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        sum_f(self, axes)
    }

    pub fn sum<I>(&self, axes: I) -> Tensor<T, B, IxD>
    where
        B: DeviceCreationAnyAPI<T>,
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        sum(self, axes)
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_sum_all() {
        let a = arange((24, &DeviceCpuSerial));
        let s = sum_all(&a);
        assert_eq!(s, 276);

        // np.arange(3240).reshape(12, 15, 18)
        //   .swapaxes(-1, -2)[2:-3, 1:-4:2, -1:3:-2].sum()
        let a_owned =
            arange((3240, &DeviceCpuSerial)).into_shape([12, 15, 18]).into_swapaxes(-1, -2);
        let a = a_owned.i((slice!(2, -3), slice!(1, -4, 2), slice!(-1, 3, -2)));
        let s = a.sum_all();
        assert_eq!(s, 446586);

        let s = a.sum(());
        println!("{:?}", s);
        assert_eq!(s.to_scalar(), 446586);
    }

    #[test]
    fn test_sum_axes() {
        let s = arange((3240, &DeviceCpuSerial))
            .into_shape([4, 6, 15, 9])
            .transpose([2, 0, 3, 1])
            .sum([0, -2]);
        println!("{:?}", s);
        assert_eq!(s[[0, 1]], 27270);
        assert_eq!(s[[1, 2]], 154845);
        assert_eq!(s[[3, 5]], 428220);
    }
}
