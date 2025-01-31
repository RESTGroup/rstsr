use crate::prelude_dev::*;

macro_rules! trait_reduction {
    ($OpReduceAPI: ident, $func: ident, $func_all: ident) => {
        pub trait $OpReduceAPI<T, D>
        where
            D: DimAPI,
            Self: DeviceAPI<T>,
        {
            fn $func_all(&self, a: &Self::Raw, la: &Layout<D>) -> Result<T>;
            fn $func(
                &self,
                a: &Self::Raw,
                la: &Layout<D>,
                axes: &[isize],
            ) -> Result<(Storage<DataOwned<Self::Raw>, T, Self>, Layout<IxD>)>;
        }
    };
}

trait_reduction!(OpSumAPI, sum, sum_all);
trait_reduction!(OpMinAPI, min, min_all);
trait_reduction!(OpMaxAPI, max, max_all);
trait_reduction!(OpProdAPI, prod, prod_all);
trait_reduction!(OpMeanAPI, mean, mean_all);
trait_reduction!(OpVarAPI, var, var_all);
trait_reduction!(OpStdAPI, std, std_all);
