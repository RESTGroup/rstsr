use crate::prelude_dev::*;

impl<TC, DC, DA, TA> OpAssignArbitaryAPI<TC, DC, DA, TA> for DeviceCpuSerial
where
    TC: Clone,
    TA: Clone + DTypePromotionAPI<TC>,
    DC: DimAPI,
    DA: DimAPI,
{
    fn assign_arbitary(&self, c: &mut Vec<TC>, lc: &Layout<DC>, a: &Vec<TA>, la: &Layout<DA>) -> Result<()> {
        let default_order = self.default_order();
        return assign_arbitary_promote_cpu_serial(c, lc, a, la, default_order);
    }

    fn assign_arbitary_uninit(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<DC>,
        a: &Vec<TA>,
        la: &Layout<DA>,
    ) -> Result<()> {
        let default_order = self.default_order();
        return assign_arbitary_uninit_promote_cpu_serial(c, lc, a, la, default_order);
    }
}

impl<TC, D, TA> OpAssignAPI<TC, D, TA> for DeviceCpuSerial
where
    TC: Clone,
    TA: Clone + DTypePromotionAPI<TC>,
    D: DimAPI,
{
    fn assign(&self, c: &mut Vec<TC>, lc: &Layout<D>, a: &Vec<TA>, la: &Layout<D>) -> Result<()> {
        return assign_promote_cpu_serial(c, lc, a, la);
    }

    fn assign_uninit(&self, c: &mut Vec<MaybeUninit<TC>>, lc: &Layout<D>, a: &Vec<TA>, la: &Layout<D>) -> Result<()> {
        return assign_uninit_promote_cpu_serial(c, lc, a, la);
    }

    fn fill(&self, c: &mut Vec<TC>, lc: &Layout<D>, fill: TA) -> Result<()> {
        return fill_promote_cpu_serial(c, lc, fill);
    }
}
