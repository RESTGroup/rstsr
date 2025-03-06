use crate::prelude_dev::*;

macro_rules! trait_reduction {
    ($OpReduceAPI: ident, $func: ident, $func_all: ident) => {
        pub trait $OpReduceAPI<T, D>
        where
            D: DimAPI,
            Self: DeviceAPI<T> + DeviceAPI<Self::TOut>,
        {
            type TOut;
            fn $func_all(
                &self,
                a: &<Self as DeviceRawAPI<T>>::Raw,
                la: &Layout<D>,
            ) -> Result<Self::TOut>;
            fn $func(
                &self,
                a: &<Self as DeviceRawAPI<T>>::Raw,
                la: &Layout<D>,
                axes: &[isize],
            ) -> Result<(
                Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>,
                Layout<IxD>,
            )>;
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
trait_reduction!(OpL2NormAPI, l2_norm, l2_norm_all);

macro_rules! trait_reduction_arg {
    ($OpReduceAPI: ident, $func: ident, $func_all: ident) => {
        pub trait $OpReduceAPI<T, D>
        where
            D: DimAPI,
            Self: DeviceAPI<T>,
        {
            fn $func_all(&self, a: &<Self as DeviceRawAPI<T>>::Raw, la: &Layout<D>) -> Result<D>;
            fn $func(
                &self,
                a: &<Self as DeviceRawAPI<T>>::Raw,
                la: &Layout<D>,
                axes: &[isize],
            ) -> Result<(
                Storage<DataOwned<<Self as DeviceRawAPI<IxD>>::Raw>, IxD, Self>,
                Layout<IxD>,
            )>
            where
                Self: DeviceAPI<IxD>;
        }
    };
}

trait_reduction_arg!(OpArgMinAPI, argmin, argmin_all);
trait_reduction_arg!(OpArgMaxAPI, argmax, argmax_all);
