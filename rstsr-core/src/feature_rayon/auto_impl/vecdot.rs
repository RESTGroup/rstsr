use crate::prelude_dev::*;
use num::Zero;

impl<TA, TB, TC, DA, DB, DC> DeviceVecdotAPI<TA, TB, TC, DA, DB, DC> for DeviceRayonAutoImpl
where
    TA: Clone + Send + Sync,
    TB: Clone + Send + Sync,
    TC: Clone + Send + Sync + Zero,
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
        axis: isize,
    ) -> Result<()> {
        let pool = self.get_current_pool();
        vecdot_naive_cpu_rayon(c, lc, a, la, b, lb, axis, self.default_order(), pool)
    }
}
