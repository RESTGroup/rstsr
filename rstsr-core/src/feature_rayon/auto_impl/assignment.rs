use crate::prelude_dev::*;

impl<TC, TA, DC, DA> OpAssignArbitaryAPI<TC, DC, DA, TA> for DeviceRayonAutoImpl
where
    TC: Clone + Send + Sync,
    TA: Clone + Send + Sync + PromotionAPI<TC>,
    DC: DimAPI,
    DA: DimAPI,
{
    fn assign_arbitary(&self, c: &mut Vec<TC>, lc: &Layout<DC>, a: &Vec<TA>, la: &Layout<DA>) -> Result<()> {
        let pool = self.get_current_pool();
        let default_order = self.default_order();
        assign_arbitary_promote_cpu_rayon(c, lc, a, la, default_order, pool)
    }

    fn assign_arbitary_uninit(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<DC>,
        a: &Vec<TA>,
        la: &Layout<DA>,
    ) -> Result<()> {
        let pool = self.get_current_pool();
        let default_order = self.default_order();
        return assign_arbitary_uninit_promote_cpu_rayon(c, lc, a, la, default_order, pool);
    }
}

impl<TC, TA, D> OpAssignAPI<TC, D, TA> for DeviceRayonAutoImpl
where
    TC: Clone + Send + Sync,
    TA: Clone + Send + Sync + PromotionAPI<TC>,
    D: DimAPI,
{
    fn assign(&self, c: &mut Vec<TC>, lc: &Layout<D>, a: &Vec<TA>, la: &Layout<D>) -> Result<()> {
        let pool = self.get_current_pool();
        assign_promote_cpu_rayon(c, lc, a, la, pool)
    }

    fn assign_uninit(&self, c: &mut Vec<MaybeUninit<TC>>, lc: &Layout<D>, a: &Vec<TA>, la: &Layout<D>) -> Result<()> {
        let pool = self.get_current_pool();
        return assign_uninit_promote_cpu_rayon(c, lc, a, la, pool);
    }

    fn fill(&self, c: &mut Vec<TC>, lc: &Layout<D>, fill: TA) -> Result<()> {
        let pool = self.get_current_pool();
        fill_promote_cpu_rayon(c, lc, fill, pool)
    }
}
