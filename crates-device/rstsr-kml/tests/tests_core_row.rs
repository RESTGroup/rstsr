mod core_func;
mod test_utils;

pub use rstsr::prelude::*;
pub use std::sync::LazyLock;
pub use test_utils::TestCfg;

pub use DeviceKML as DeviceType;

pub static TESTCFG: LazyLock<TestCfg<DeviceType>> = LazyLock::new(|| {
    let mut device = DeviceType::default();
    device.set_default_order(RowMajor);
    TestCfg::init(device, vec![], None)
});
