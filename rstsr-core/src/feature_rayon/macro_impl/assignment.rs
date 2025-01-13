#[macro_export]
macro_rules! macro_impl_rayon_assignment {
    ($Device: ident) => {
        use $crate::prelude_dev::*;

        impl<T, DC, DA> OpAssignArbitaryAPI<T, DC, DA> for $Device
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
                let nthreads = self.get_num_threads();
                assign_arbitary_cpu_rayon(c, lc, a, la, nthreads)
            }
        }

        impl<T, D> OpAssignAPI<T, D> for $Device
        where
            T: Clone + Send + Sync,
            D: DimAPI,
        {
            fn assign(
                &self,
                c: &mut Vec<T>,
                lc: &Layout<D>,
                a: &Vec<T>,
                la: &Layout<D>,
            ) -> Result<()> {
                let nthreads = self.get_num_threads();
                assign_cpu_rayon(c, lc, a, la, nthreads)
            }

            fn fill(&self, c: &mut Vec<T>, lc: &Layout<D>, fill: T) -> Result<()> {
                let nthreads = self.get_num_threads();
                fill_cpu_rayon(c, lc, fill, nthreads)
            }
        }
    };
}
