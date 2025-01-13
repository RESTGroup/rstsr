use crate::prelude_dev::*;

pub trait OpSumAPI<T, D>
where
    D: DimAPI,
    Self: DeviceAPI<T>,
{
    fn sum_all(&self, a: &Self::Raw, la: &Layout<D>) -> Result<T>;
    fn sum(
        &self,
        a: &Self::Raw,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<Self::Raw>, T, Self>, Layout<IxD>)>;
}
