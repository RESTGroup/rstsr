pub(crate) use crate as rstsr_sci_traits;
pub(crate) use rstsr_core::prelude_dev::*;

#[cfg(feature = "faer")]
pub(crate) type DeviceRayonAutoImpl = rstsr_core::prelude_dev::DeviceFaer;
