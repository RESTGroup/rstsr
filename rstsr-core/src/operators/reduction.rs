use crate::prelude_dev::*;

#[allow(clippy::type_complexity)]
#[duplicate_item(
    OpReduceAPI   func           func_all    ;
   [OpSumAPI   ] [sum_axes    ] [sum_all    ];
   [OpMinAPI   ] [min_axes    ] [min_all    ];
   [OpMaxAPI   ] [max_axes    ] [max_all    ];
   [OpProdAPI  ] [prod_axes   ] [prod_all   ];
   [OpMeanAPI  ] [mean_axes   ] [mean_all   ];
   [OpVarAPI   ] [var_axes    ] [var_all    ];
   [OpStdAPI   ] [std_axes    ] [std_all    ];
   [OpL2NormAPI] [l2_norm_axes] [l2_norm_all];
   [OpArgMinAPI] [argmin_axes ] [argmin_all ];
   [OpArgMaxAPI] [argmax_axes ] [argmax_all ];
   [OpAllAPI   ] [all_axes    ] [all_all    ];
   [OpAnyAPI   ] [any_axes    ] [any_all    ];
   [OpCountNonZeroAPI] [count_nonzero_axes] [count_nonzero_all];
)]
pub trait OpReduceAPI<T, D>
where
    D: DimAPI,
    Self: DeviceAPI<T> + DeviceAPI<Self::TOut>,
{
    type TOut;
    fn func_all(&self, a: &<Self as DeviceRawAPI<T>>::Raw, la: &Layout<D>) -> Result<Self::TOut>;
    fn func(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>)>;
}

#[allow(clippy::type_complexity)]
#[duplicate_item(
    OpReduceAPI            func                    func_all             ;
   [OpUnraveledArgMinAPI] [unraveled_argmin_axes] [unraveled_argmin_all];
   [OpUnraveledArgMaxAPI] [unraveled_argmax_axes] [unraveled_argmax_all];
)]
pub trait OpReduceAPI<T, D>
where
    D: DimAPI,
    Self: DeviceAPI<T>,
{
    fn func_all(&self, a: &<Self as DeviceRawAPI<T>>::Raw, la: &Layout<D>) -> Result<D>;
    fn func(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<IxD>>::Raw>, IxD, Self>, Layout<IxD>)>
    where
        Self: DeviceAPI<IxD>;
}

#[allow(clippy::type_complexity)]
pub trait OpSumBoolAPI<D>
where
    D: DimAPI,
    Self: DeviceAPI<bool> + DeviceAPI<usize>,
{
    fn sum_all(&self, a: &<Self as DeviceRawAPI<bool>>::Raw, la: &Layout<D>) -> Result<usize>;
    fn sum_axes(
        &self,
        a: &<Self as DeviceRawAPI<bool>>::Raw,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<usize>>::Raw>, usize, Self>, Layout<IxD>)>;
}

#[allow(clippy::type_complexity)]
pub trait OpAllCloseAPI<TA, TB, TE, D>
where
    D: DimAPI,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<bool>,
{
    fn allclose_all(
        &self,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<D>,
        isclose_args: &IsCloseArgs<TE>,
    ) -> Result<bool>;
    fn allclose_axes(
        &self,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<D>,
        axes: &[isize],
        isclose_args: &IsCloseArgs<TE>,
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<bool>>::Raw>, bool, Self>, Layout<IxD>)>;
}
