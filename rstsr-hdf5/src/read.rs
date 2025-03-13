use crate::prelude_dev::*;

pub fn read_slice_to_cpu<T, D>(_dataset: &Dataset, _layout: &D) -> Result<(Vec<T>, Layout<D>)>
where
    T: H5Type,
    D: DimAPI,
{
    todo!()
}
