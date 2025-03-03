use crate::prelude_dev::*;

impl<T, DC, DA> OpAssignArbitaryAPI<T, DC, DA> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync,
    DC: DimAPI,
    DA: DimAPI,
{
    fn assign_arbitary(
        &self,
        c: &mut Vec<T>,
        lc: &Layout<DC>,
        a: &Vec<T>,
        la: &Layout<DA>,
    ) -> Result<()> {
        let pool = self.get_pool();
        assign_arbitary_cpu_rayon(c, lc, a, la, pool)
    }
}

impl<T, D> OpAssignAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync,
    D: DimAPI,
{
    fn assign(&self, c: &mut Vec<T>, lc: &Layout<D>, a: &Vec<T>, la: &Layout<D>) -> Result<()> {
        let pool = self.get_pool();
        assign_cpu_rayon(c, lc, a, la, pool)
    }

    fn fill(&self, c: &mut Vec<T>, lc: &Layout<D>, fill: T) -> Result<()> {
        let pool = self.get_pool();
        fill_cpu_rayon(c, lc, fill, pool)
    }
}
