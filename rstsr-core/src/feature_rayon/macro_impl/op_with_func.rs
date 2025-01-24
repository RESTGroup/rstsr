#[macro_export]
macro_rules! macro_impl_rayon_op_with_func {
    ($Device: ident) => {
        use $crate::feature_rayon::*;
        use $crate::prelude_dev::*;

        /* #region impl op_func for $Device */

        impl<TA, TB, TC, D, F> DeviceOp_MutC_RefA_RefB_API<TA, TB, TC, D, F> for $Device
        where
            TA: Clone + Send + Sync,
            TB: Clone + Send + Sync,
            TC: Clone + Send + Sync,
            D: DimAPI,
            F: Fn(&mut TC, &TA, &TB) + ?Sized + Send + Sync,
        {
            fn op_mutc_refa_refb_func(
                &self,
                c: &mut Vec<TC>,
                lc: &Layout<D>,
                a: &Vec<TA>,
                la: &Layout<D>,
                b: &Vec<TB>,
                lb: &Layout<D>,
                f: &mut F,
            ) -> Result<()> {
                let pool = self.get_pool();
                op_mutc_refa_refb_func_cpu_rayon(c, lc, a, la, b, lb, f, pool)
            }
        }

        impl<TA, TB, TC, D, F> DeviceOp_MutC_RefA_NumB_API<TA, TB, TC, D, F> for $Device
        where
            TA: Clone + Send + Sync,
            TB: Clone + Send + Sync,
            TC: Clone + Send + Sync,
            D: DimAPI,
            F: Fn(&mut TC, &TA, &TB) + ?Sized + Send + Sync,
        {
            fn op_mutc_refa_numb_func(
                &self,
                c: &mut Vec<TC>,
                lc: &Layout<D>,
                a: &Vec<TA>,
                la: &Layout<D>,
                b: TB,
                f: &mut F,
            ) -> Result<()> {
                let pool = self.get_pool();
                op_mutc_refa_numb_func_cpu_rayon(c, lc, a, la, b, f, pool)
            }
        }

        impl<TA, TB, TC, D, F> DeviceOp_MutC_NumA_RefB_API<TA, TB, TC, D, F> for $Device
        where
            TA: Clone + Send + Sync,
            TB: Clone + Send + Sync,
            TC: Clone + Send + Sync,
            D: DimAPI,
            F: Fn(&mut TC, &TA, &TB) + ?Sized + Send + Sync,
        {
            fn op_mutc_numa_refb_func(
                &self,
                c: &mut Vec<TC>,
                lc: &Layout<D>,
                a: TA,
                b: &Vec<TB>,
                lb: &Layout<D>,
                f: &mut F,
            ) -> Result<()> {
                let pool = self.get_pool();
                op_mutc_numa_refb_func_cpu_rayon(c, lc, a, b, lb, f, pool)
            }
        }

        impl<TA, TB, D, F> DeviceOp_MutA_RefB_API<TA, TB, D, F> for $Device
        where
            TA: Clone + Send + Sync,
            TB: Clone + Send + Sync,
            D: DimAPI,
            F: Fn(&mut TA, &TB) + ?Sized + Send + Sync,
        {
            fn op_muta_refb_func(
                &self,
                a: &mut Vec<TA>,
                la: &Layout<D>,
                b: &Vec<TB>,
                lb: &Layout<D>,
                f: &mut F,
            ) -> Result<()> {
                let pool = self.get_pool();
                op_muta_refb_func_cpu_rayon(a, la, b, lb, f, pool)
            }
        }

        impl<TA, TB, D, F> DeviceOp_MutA_NumB_API<TA, TB, D, F> for $Device
        where
            TA: Clone + Send + Sync,
            TB: Clone + Send + Sync,
            D: DimAPI,
            F: Fn(&mut TA, &TB) + ?Sized + Send + Sync,
        {
            fn op_muta_numb_func(
                &self,
                a: &mut Vec<TA>,
                la: &Layout<D>,
                b: TB,
                f: &mut F,
            ) -> Result<()> {
                let pool = self.get_pool();
                op_muta_numb_func_cpu_rayon(a, la, b, f, pool)
            }
        }

        impl<T, D, F> DeviceOp_MutA_API<T, D, F> for $Device
        where
            T: Clone + Send + Sync,
            D: DimAPI,
            F: Fn(&mut T) + ?Sized + Send + Sync,
        {
            fn op_muta_func(&self, a: &mut Vec<T>, la: &Layout<D>, f: &mut F) -> Result<()> {
                let pool = self.get_pool();
                op_muta_func_cpu_rayon(a, la, f, pool)
            }
        }

        /* #endregion */
    };
}
