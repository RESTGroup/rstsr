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

trait_reduction!(OpSumAPI, sum_axes, sum_all);
trait_reduction!(OpMinAPI, min_axes, min_all);
trait_reduction!(OpMaxAPI, max_axes, max_all);
trait_reduction!(OpProdAPI, prod_axes, prod_all);
trait_reduction!(OpMeanAPI, mean_axes, mean_all);
trait_reduction!(OpVarAPI, var_axes, var_all);
trait_reduction!(OpStdAPI, std_axes, std_all);
trait_reduction!(OpL2NormAPI, l2_norm_axes, l2_norm_all);

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

trait_reduction_arg!(OpArgMinAPI, argmin_axes, argmin_all);
trait_reduction_arg!(OpArgMaxAPI, argmax_axes, argmax_all);
