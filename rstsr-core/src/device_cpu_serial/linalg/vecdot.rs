use crate::prelude_dev::*;
use num::Zero;

impl<TA, TB, TC, DA, DB, DC> DeviceVecdotAPI<TA, TB, TC, DA, DB, DC> for DeviceCpuSerial
where
    TA: Clone,
    TB: Clone,
    TC: Clone + Zero,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: Mul<TB, Output = TC>,
    TA: ExtNum,
    Self: DeviceAPI<TA, Raw = Vec<TA>> + DeviceAPI<TB, Raw = Vec<TB>> + DeviceAPI<TC, Raw = Vec<TC>>,
{
    fn vecdot(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<DC>,
        a: &Vec<TA>,
        la: &Layout<DA>,
        b: &Vec<TB>,
        lb: &Layout<DB>,
        axes_a: &[isize],
        axes_b: &[isize],
    ) -> Result<()> {
        vecdot_naive_cpu_serial(c, lc, a, la, b, lb, axes_a, axes_b, self.default_order())
    }
}
