use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/// n-dim `solve_symmetric`: solve `a x = b` for a stack `a` of `(M, M)`
/// symmetric/Hermitian matrices and right-hand side `b` of `(M, K)` matrices
/// or `(M)` vectors (the two matrix axes are the last two shape axes under a
/// row-major device, the first two under a column-major one); the batch dims
/// of `a` and `b` broadcast against each other.
///
/// `a` is never written through: an immutable `a` is materialized into a
/// freshly allocated broadcast buffer; a mutable `a` is factored from a
/// per-slice copy and must already fit the broadcast batch (it cannot grow).
/// `b` follows the shared
/// solve ownership rule: an immutable `b` is solved into a freshly allocated
/// broadcast solution; a mutable `b` is solved in place when its batch fits
/// the broadcast batch, or in a contiguous work buffer written back by
/// `clone_to_mut` at the call site.
#[allow(clippy::type_complexity)]
pub fn blas_impl_solve_symmetric_f<'b, T>(
    a: TensorReference<'b, T, DeviceBLAS, IxD>,
    b: TensorReference<'b, T, DeviceBLAS, IxD>,
    hermi: bool,
    uplo: Option<FlagUpLo>,
) -> Result<TensorMutable<'b, T, DeviceBLAS, IxD>>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationNumAPI<T>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
{
    let order = a.device().default_order();
    let plan = solve_plan(a.shape(), b.shape(), order, "solve_symmetric")?;

    let a_buffer: Option<Tensor<T, DeviceBLAS, IxD>>;
    let a_converted: Option<TensorMutable<'b, T, DeviceBLAS, IxD>>;
    let a_view = if a.is_ref() {
        let mut full: Tensor<T, DeviceBLAS, IxD> = zeros_f((plan.a_shape(order), a.device()))?;
        full.assign_f(a.broadcast_to(plan.a_shape(order)))?;
        a_buffer = Some(full);
        a_buffer.as_ref().unwrap().view()
    } else {
        let converted = overwritable_convert_with_order(a, order)?;
        let (batch_a, _) = stack_split_shape(converted.view().shape(), 2, order)?;
        rstsr_assert_eq!(
            batch_a.as_slice(),
            plan.batch_out.as_slice(),
            InvalidLayout,
            "solve_symmetric: mutable a cannot grow to the broadcast batch; pass an immutable a"
        )?;
        a_converted = Some(converted);
        a_converted.as_ref().unwrap().view()
    };

    solve_stack_f(a_view, b, &plan, order, |a_slice, b_slice| {
        let result = ref_impl_solve_symmetric_f(a_slice.into(), b_slice.into(), hermi, uplo)?;
        result.clone_to_mut();
        Ok(())
    })
}

/* #region full-args */

#[duplicate_item(
    ImplType                                                            TrA                                 TrB                               ;
   [T, DA, DB, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceBLAS, DA>] [&TensorAny<Rb, T, DeviceBLAS, DB>];
   [T, DA, DB, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceBLAS, DA> ] [TensorView<'_, T, DeviceBLAS, DB>];
   [T, DA, DB, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceBLAS, DA>] [&TensorAny<R, T, DeviceBLAS, DB> ];
   [T, DA, DB,                                                       ] [TensorView<'_, T, DeviceBLAS, DA>] [TensorView<'_, T, DeviceBLAS, DB>];
)]
impl<ImplType> SolveSymmetricAPI<DeviceBLAS> for (TrA, TrB, bool, Option<FlagUpLo>)
where
    T: BlasFloat,
    DA: DimAPI,
    DB: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationNumAPI<T>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
{
    type Out = Tensor<T, DeviceBLAS, IxD>;
    fn solve_symmetric_f(self) -> Result<Self::Out> {
        let (a, b, hermi, uplo) = self;
        let a_view = a.view();
        let a_dyn = a_view.to_dyn();
        let b_view = b.view();
        let b_dyn = b_view.to_dyn();
        Ok(blas_impl_solve_symmetric_f(a_dyn.into(), b_dyn.into(), hermi, uplo)?.into_owned())
    }
}

#[duplicate_item(
    ImplType                                   TrA                                 TrB                              ;
   ['b, T, DA, DB, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, DA> ] [TensorMut<'b, T, DeviceBLAS, DB>];
   ['b, T, DA, DB,                          ] [TensorView<'_, T, DeviceBLAS, DA>] [TensorMut<'b, T, DeviceBLAS, DB>];
   [    T, DA, DB, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, DA> ] [Tensor<T, DeviceBLAS, DB>       ];
   [    T, DA, DB,                          ] [TensorView<'_, T, DeviceBLAS, DA>] [Tensor<T, DeviceBLAS, DB>       ];
)]
impl<ImplType> SolveSymmetricAPI<DeviceBLAS> for (TrA, TrB, bool, Option<FlagUpLo>)
where
    T: BlasFloat,
    DA: DimAPI,
    DB: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationNumAPI<T>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
{
    type Out = TrB;
    fn solve_symmetric_f(self) -> Result<Self::Out> {
        let (a, mut b, hermi, uplo) = self;
        let a_view = a.view();
        let a_dyn = a_view.to_dyn();
        let b_ref = b.view_mut().into_dim::<IxD>();
        let result = blas_impl_solve_symmetric_f(a_dyn.into(), b_ref.into(), hermi, uplo)?;
        result.clone_to_mut();
        Ok(b)
    }
}

#[duplicate_item(
    ImplType                               TrA                                TrB                               ;
   [T, DA, DB, R: DataAPI<Data = Vec<T>>] [TensorMut<'_, T, DeviceBLAS, DA>] [&TensorAny<R, T, DeviceBLAS, DB> ];
   [T, DA, DB,                          ] [TensorMut<'_, T, DeviceBLAS, DA>] [TensorView<'_, T, DeviceBLAS, DB>];
   [T, DA, DB, R: DataAPI<Data = Vec<T>>] [Tensor<T, DeviceBLAS, DA>       ] [&TensorAny<R, T, DeviceBLAS, DB> ];
   [T, DA, DB,                          ] [Tensor<T, DeviceBLAS, DA>       ] [TensorView<'_, T, DeviceBLAS, DB>];
)]
impl<ImplType> SolveSymmetricAPI<DeviceBLAS> for (TrA, TrB, bool, Option<FlagUpLo>)
where
    T: BlasFloat,
    DA: DimAPI,
    DB: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationNumAPI<T>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
{
    type Out = Tensor<T, DeviceBLAS, IxD>;
    fn solve_symmetric_f(self) -> Result<Self::Out> {
        let (mut a, b, hermi, uplo) = self;
        let a_ref = a.view_mut().into_dim::<IxD>();
        let b_view = b.view();
        let b_dyn = b_view.to_dyn();
        Ok(blas_impl_solve_symmetric_f(a_ref.into(), b_dyn.into(), hermi, uplo)?.into_owned())
    }
}

#[duplicate_item(
    ImplType        TrA                                TrB                              ;
   ['b, T, DA, DB] [TensorMut<'_, T, DeviceBLAS, DA>] [TensorMut<'b, T, DeviceBLAS, DB>];
   [    T, DA, DB] [TensorMut<'_, T, DeviceBLAS, DA>] [Tensor<T, DeviceBLAS, DB>       ];
   ['b, T, DA, DB] [Tensor<T, DeviceBLAS, DA>       ] [TensorMut<'b, T, DeviceBLAS, DB>];
   [    T, DA, DB] [Tensor<T, DeviceBLAS, DA>       ] [Tensor<T, DeviceBLAS, DB>       ];
)]
impl<ImplType> SolveSymmetricAPI<DeviceBLAS> for (TrA, TrB, bool, Option<FlagUpLo>)
where
    T: BlasFloat,
    DA: DimAPI,
    DB: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationNumAPI<T>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
{
    type Out = TrB;
    fn solve_symmetric_f(self) -> Result<Self::Out> {
        let (mut a, mut b, hermi, uplo) = self;
        let a_ref = a.view_mut().into_dim::<IxD>();
        let b_ref = b.view_mut().into_dim::<IxD>();
        let result = blas_impl_solve_symmetric_f(a_ref.into(), b_ref.into(), hermi, uplo)?;
        result.clone_to_mut();
        Ok(b)
    }
}

/* #endregion */

/* #region sub-args */

#[duplicate_item(
    ImplStruct                   args_tuple            internal_tuple            ;
   [(TrA, TrB, bool, FlagUpLo)] [(a, b, hermi, uplo)] [(a, b, hermi, Some(uplo))];
   [(TrA, TrB, bool,         )] [(a, b, hermi,     )] [(a, b, hermi, None      )];
   [(TrA, TrB,       FlagUpLo)] [(a, b,        uplo)] [(a, b, true , Some(uplo))];
   [(TrA, TrB,               )] [(a, b,            )] [(a, b, true , None      )];
)]
impl<TrA, TrB> SolveSymmetricAPI<DeviceBLAS> for ImplStruct
where
    (TrA, TrB, bool, Option<FlagUpLo>): SolveSymmetricAPI<DeviceBLAS>,
{
    type Out = <(TrA, TrB, bool, Option<FlagUpLo>) as SolveSymmetricAPI<DeviceBLAS>>::Out;
    fn solve_symmetric_f(self) -> Result<Self::Out> {
        let args_tuple = self;
        SolveSymmetricAPI::<DeviceBLAS>::solve_symmetric_f(internal_tuple)
    }
}

/* #endregion */
