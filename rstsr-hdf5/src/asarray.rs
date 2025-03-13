use rstsr_core::storage;

use crate::{device::DeviceHDF5, prelude_dev::*};

/// A dummy struct that involves `DeviceHDF5` and `T`.
///
/// This struct is used to implement `AsArrayAPI` for `Dataset` to avoid orphan
/// rule restriction.
pub struct DeviceHDF5WithDType<T> {
    _phantom_hdf5: std::marker::PhantomData<DeviceHDF5>,
    _phantom_t: std::marker::PhantomData<T>,
}

impl<T> AsArrayAPI<DeviceHDF5WithDType<T>> for Dataset
where
    T: H5Type,
{
    type Out = Tensor<T, DeviceHDF5, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let layout = self.shape().c();
        let storage = storage::Storage::new(self.into(), DeviceHDF5);
        Ok(Tensor::new(storage, layout))
    }
}
