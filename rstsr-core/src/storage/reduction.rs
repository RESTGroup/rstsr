use crate::prelude_dev::*;

pub trait OpSumAPI<T, D>
where
    D: DimAPI,
    Self: DeviceAPI<T>,
{
    fn sum_all(&self, a: &Storage<T, Self>, la: &Layout<D>) -> Result<T>;
    fn sum(
        &self,
        a: &Storage<T, Self>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<T, Self>, Layout<IxD>)>;
}
