use crate::prelude_dev::*;

/* #region sum */

pub fn sum_all_f<R, T, B, D>(tensor: &TensorBase<R, D>) -> Result<T>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: OpSumAPI<T, D>,
{
    let storage = tensor.storage();
    let layout = tensor.layout();
    tensor.device().sum_all(storage, layout)
}

pub fn sum_f<R, T, B, D, I>(tensor: &TensorBase<R, D>, axes: I) -> Result<Tensor<T, IxD, B>>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: OpSumAPI<T, D> + DeviceAPI<T> + DeviceCreationAnyAPI<T>,
    I: TryInto<AxesIndex<isize>, Error = Error>,
{
    let axes = axes.try_into()?;
    let storage = tensor.storage();
    let layout = tensor.layout();

    // special case for summing all axes
    if axes.as_ref().is_empty() {
        let sum = tensor.device().sum_all(storage, layout)?;
        let storage = tensor.device().outof_cpu_vec(vec![sum])?;
        let data = DataOwned::from(storage);
        let layout = Layout::new(vec![], vec![], 0)?;
        return Tensor::new_f(data, layout);
    }

    let (storage, layout) = tensor.device().sum(storage, layout, axes.as_ref())?;
    let data = DataOwned::from(storage);
    Tensor::new_f(data, layout)
}

pub fn sum_all<R, T, B, D>(tensor: &TensorBase<R, D>) -> T
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: OpSumAPI<T, D>,
{
    sum_all_f(tensor).unwrap()
}

pub fn sum<R, T, B, D, I>(tensor: &TensorBase<R, D>, axes: I) -> Tensor<T, IxD, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: OpSumAPI<T, D> + DeviceCreationAnyAPI<T>,
    I: TryInto<AxesIndex<isize>, Error = Error>,
{
    sum_f(tensor, axes).unwrap()
}

impl<R, T, B, D> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: OpSumAPI<T, D>,
{
    pub fn sum_all_f(&self) -> Result<T> {
        sum_all_f(self)
    }

    pub fn sum_all(&self) -> T {
        sum_all(self)
    }

    pub fn sum_f<I>(&self, axes: I) -> Result<Tensor<T, IxD, B>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
        B: DeviceCreationAnyAPI<T>,
    {
        sum_f(self, axes)
    }

    pub fn sum<I>(&self, axes: I) -> Tensor<T, IxD, B>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
        B: DeviceCreationAnyAPI<T>,
    {
        sum(self, axes)
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use crate::prelude::TensorSliceAPI;

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

        let empty: [isize; 0] = [];
        let s = a.sum(empty);
        println!("{:?}", s);
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
