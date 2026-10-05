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
   [OpNanArgMinAPI] [nanargmin_axes ] [nanargmin_all ];
   [OpNanArgMaxAPI] [nanargmax_axes ] [nanargmax_all ];
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

/* #region dtype / correction variants */

/// Reductions with an explicit accumulator (and output) dtype; the input
/// elements are cast to `TOut` inside the fold, so no intermediate tensor is
/// created.
#[allow(clippy::type_complexity)]
#[duplicate_item(
    OpReduceDtypeAPI   func_dtype         func_all_dtype     ;
   [OpSumDtypeAPI   ] [sum_axes_dtype   ] [sum_all_dtype    ];
   [OpProdDtypeAPI  ] [prod_axes_dtype  ] [prod_all_dtype   ];
   [OpMeanDtypeAPI  ] [mean_axes_dtype  ] [mean_all_dtype   ];
)]
pub trait OpReduceDtypeAPI<T, D, TOut>
where
    D: DimAPI,
    Self: DeviceAPI<T> + DeviceAPI<TOut>,
{
    fn func_all_dtype(&self, a: &<Self as DeviceRawAPI<T>>::Raw, la: &Layout<D>) -> Result<TOut>;
    fn func_dtype(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<TOut>>::Raw>, TOut, Self>, Layout<IxD>)>;
}

/// `var`/`std` with an explicit accumulator dtype and a Delta degrees of
/// freedom (`correction`); the divisor is `M - correction`.
#[allow(clippy::type_complexity)]
#[duplicate_item(
    OpVarStdDtypeAPI   func_dtype         func_all_dtype     ;
   [OpVarDtypeAPI  ] [var_axes_dtype   ] [var_all_dtype    ];
   [OpStdDtypeAPI  ] [std_axes_dtype   ] [std_all_dtype    ];
)]
pub trait OpVarStdDtypeAPI<T, D, TOut>
where
    D: DimAPI,
    Self: DeviceAPI<T> + DeviceAPI<TOut>,
{
    fn func_all_dtype(&self, a: &<Self as DeviceRawAPI<T>>::Raw, la: &Layout<D>, correction: f64) -> Result<TOut>;
    fn func_dtype(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        axes: &[isize],
        correction: f64,
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<TOut>>::Raw>, TOut, Self>, Layout<IxD>)>;
}

/// `var`/`std` with a Delta degrees of freedom (`correction`) and the
/// input's natural real accumulator dtype (`ComplexFloat::Real`, i.e. `T::Real`).
#[allow(clippy::type_complexity)]
#[duplicate_item(
    OpReduceCorrAPI   func_corr         func_all_corr     ;
   [OpVarCorrAPI   ] [var_axes_corr   ] [var_all_corr    ];
   [OpStdCorrAPI   ] [std_axes_corr   ] [std_all_corr    ];
)]
pub trait OpReduceCorrAPI<T, D>
where
    D: DimAPI,
    Self: DeviceAPI<T> + DeviceAPI<Self::TOut>,
{
    type TOut;
    fn func_all_corr(&self, a: &<Self as DeviceRawAPI<T>>::Raw, la: &Layout<D>, correction: f64) -> Result<Self::TOut>;
    fn func_corr(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        axes: &[isize],
        correction: f64,
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>)>;
}

/* #endregion */

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
