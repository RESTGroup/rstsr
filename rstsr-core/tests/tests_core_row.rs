mod tests_core;
mod tests_utils;

use rstsr::prelude::*;
use std::sync::LazyLock;
use tests_utils::TestCfg;

pub static TESTCFG: LazyLock<TestCfg<DeviceCpuSerial>> = LazyLock::new(|| {
    let mut device = DeviceCpuSerial::default();
    device.set_default_order(RowMajor);
    TestCfg::init(device, vec![], None)
});
