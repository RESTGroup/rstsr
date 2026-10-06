use crate::prelude_dev::*;

/* #region impl op_func for DeviceRayonAutoImpl */

impl<TA, TB, TC, D, F> Op_MutC_RefA_RefB_API<TA, TB, TC, D, F> for DeviceRayonAutoImpl
where
    TA: Clone + Send + Sync,
    TB: Clone + Send + Sync,
    TC: Clone + Send + Sync,
    D: DimAPI,
    F: Fn(&mut MaybeUninit<TC>, &TA, &TB) + ?Sized + Send + Sync,
{
    fn op_mutc_refa_refb_func(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: &Vec<TB>,
        lb: &Layout<D>,
        f: &mut F,
    ) -> Result<()> {
        let pool = self.get_current_pool();
        op_mutc_refa_refb_func_cpu_rayon(c, lc, a, la, b, lb, f, pool)
    }
}

impl<TA, TB, TC, D, F> Op_MutC_RefA_NumB_API<TA, TB, TC, D, F> for DeviceRayonAutoImpl
where
    TA: Clone + Send + Sync,
    TB: Clone + Send + Sync,
    TC: Clone + Send + Sync,
    D: DimAPI,
    F: Fn(&mut MaybeUninit<TC>, &TA, &TB) + ?Sized + Send + Sync,
{
    fn op_mutc_refa_numb_func(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: TB,
        f: &mut F,
    ) -> Result<()> {
        let pool = self.get_current_pool();
        op_mutc_refa_numb_func_cpu_rayon(c, lc, a, la, b, f, pool)
    }
}

impl<TA, TB, TC, D, F> Op_MutC_NumA_RefB_API<TA, TB, TC, D, F> for DeviceRayonAutoImpl
where
    TA: Clone + Send + Sync,
    TB: Clone + Send + Sync,
    TC: Clone + Send + Sync,
    D: DimAPI,
    F: Fn(&mut MaybeUninit<TC>, &TA, &TB) + ?Sized + Send + Sync,
{
    fn op_mutc_numa_refb_func(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: TA,
        b: &Vec<TB>,
        lb: &Layout<D>,
        f: &mut F,
    ) -> Result<()> {
        let pool = self.get_current_pool();
        op_mutc_numa_refb_func_cpu_rayon(c, lc, a, b, lb, f, pool)
    }
}

impl<TA, TB, D, F> Op_MutA_RefB_API<TA, TB, D, F> for DeviceRayonAutoImpl
where
    TA: Clone + Send + Sync,
    TB: Clone + Send + Sync,
    D: DimAPI,
    F: Fn(&mut MaybeUninit<TA>, &TB) + ?Sized + Send + Sync,
{
    fn op_muta_refb_func(
        &self,
        a: &mut Vec<MaybeUninit<TA>>,
        la: &Layout<D>,
        b: &Vec<TB>,
        lb: &Layout<D>,
        f: &mut F,
    ) -> Result<()> {
        let pool = self.get_current_pool();
        op_muta_refb_func_cpu_rayon(a, la, b, lb, f, pool)
    }
}

impl<TA, TB, D, F> Op_MutA_NumB_API<TA, TB, D, F> for DeviceRayonAutoImpl
where
    TA: Clone + Send + Sync,
    TB: Clone + Send + Sync,
    D: DimAPI,
    F: Fn(&mut MaybeUninit<TA>, &TB) + ?Sized + Send + Sync,
{
    fn op_muta_numb_func(&self, a: &mut Vec<MaybeUninit<TA>>, la: &Layout<D>, b: TB, f: &mut F) -> Result<()> {
        let pool = self.get_current_pool();
        op_muta_numb_func_cpu_rayon(a, la, b, f, pool)
    }
}

impl<T, D, F> Op_MutA_API<T, D, F> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync,
    D: DimAPI,
    F: Fn(&mut MaybeUninit<T>) + ?Sized + Send + Sync,
{
    fn op_muta_func(&self, a: &mut Vec<MaybeUninit<T>>, la: &Layout<D>, f: &mut F) -> Result<()> {
        let pool = self.get_current_pool();
        op_muta_func_cpu_rayon(a, la, f, pool)
    }
}

impl<TA, TB, TC, TD, D, F> Op_MutD_RefA_RefB_RefC_API<TA, TB, TC, TD, D, F> for DeviceRayonAutoImpl
where
    TA: Clone + Send + Sync,
    TB: Clone + Send + Sync,
    TC: Clone + Send + Sync,
    TD: Clone + Send + Sync,
    D: DimAPI,
    F: Fn(&mut MaybeUninit<TD>, &TA, &TB, &TC) + ?Sized + Send + Sync,
{
    fn op_mutd_refa_refb_refc_func(
        &self,
        d: &mut Vec<MaybeUninit<TD>>,
        ld: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: &Vec<TB>,
        lb: &Layout<D>,
        c: &Vec<TC>,
        lc: &Layout<D>,
        f: &mut F,
    ) -> Result<()> {
        let pool = self.get_current_pool();
        op_mutd_refa_refb_refc_func_cpu_rayon(d, ld, a, la, b, lb, c, lc, f, pool)
    }
}

/* #endregion */
