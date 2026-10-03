//! Device mapping between rstsr devices and DLPack device descriptors.

use dlpack_ffi::{DLDevice, DLDeviceType};
use rstsr_common::error::Result;
use rstsr_core::prelude::*;

/// Mapping seam: the DLPack device descriptor of an rstsr device.
///
/// The blanket implementation covers every device whose raw buffer is
/// `Vec<T>` — `DeviceCpuSerial`, `DeviceFaer` and all `crates-device`
/// backends. An accelerator backend (whose `Raw` is a device pointer type)
/// would provide its own implementations in its own bridge crate.
pub trait DeviceDlpackAPI<T> {
    /// The DLPack device descriptor (v1: `{kDLCPU, 0}` for every CPU device).
    fn to_dlpack_device(&self) -> DLDevice;
}

impl<T, B> DeviceDlpackAPI<T> for B
where
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    fn to_dlpack_device(&self) -> DLDevice {
        DLDevice { device_type: DLDeviceType::kDLCPU, device_id: 0 }
    }
}

/// Check that a DLPack device descriptor denotes plain CPU memory.
pub(crate) fn check_cpu_device(device: DLDevice) -> Result<()> {
    if device.device_type != DLDeviceType::kDLCPU {
        return rstsr_raise!(
            UnImplemented,
            "DLPack device type {} is not supported by rstsr-cpu-dlpack (only kDLCPU)",
            device.device_type.0
        );
    }
    Ok(())
}
