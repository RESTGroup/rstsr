use crate::prelude_dev::*;

impl<T, DC, DA> OpAssignArbitaryAPI<T, DC, DA> for DeviceCpuSerial
where
    T: Clone,
    DC: DimAPI,
    DA: DimAPI,
{
    fn assign_arbitary(&self, c: &mut Vec<T>, lc: &Layout<DC>, a: &Vec<T>, la: &Layout<DA>) -> Result<()> {
        let default_order = self.default_order();
        return assign_arbitary_cpu_serial(c, lc, a, la, default_order);
    }
}

impl<T, D> OpAssignAPI<T, D> for DeviceCpuSerial
where
    T: Clone,
    D: DimAPI,
{
    fn assign(&self, c: &mut Vec<T>, lc: &Layout<D>, a: &Vec<T>, la: &Layout<D>) -> Result<()> {
        return assign_cpu_serial(c, lc, a, la);
    }

    fn fill(&self, c: &mut Vec<T>, lc: &Layout<D>, fill: T) -> Result<()> {
        return fill_cpu_serial(c, lc, fill);
    }
}
