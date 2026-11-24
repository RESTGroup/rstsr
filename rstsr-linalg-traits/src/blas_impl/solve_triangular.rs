use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/// n-dim `solve_triangular`: solve `a x = b` for a triangular stack `a` of
/// `(M, M)` matrices and right-hand side `b` of `(M, K)` matrices or `(M)`
/// vectors (the two matrix axes are the last two shape axes under a row-major
/// device, the first two under a column-major one); the batch dims of `a` and
/// `b` broadcast against each other.
///
/// An immutable `b` is solved into a freshly allocated broadcast solution; a
/// mutable `b` is solved in place when its batch fits the broadcast batch
/// (the in-place solve cannot grow), or in a contiguous work buffer written
/// back by `clone_to_mut` at the call site.
pub fn blas_impl_solve_triangular_f<'b, T>(
    a: TensorView<'_, T, DeviceBLAS, IxD>,
    b: TensorReference<'b, T, DeviceBLAS, IxD>,
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
    let plan = solve_plan(a.shape(), b.shape(), order, "solve_triangular")?;
    solve_stack_f(a, b, &plan, order, |a_slice, b_slice| {
        let result = ref_impl_solve_triangular_f(a_slice.into(), b_slice.into(), uplo)?;
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
impl<ImplType> SolveTriangularAPI<DeviceBLAS> for (TrA, TrB, Option<FlagUpLo>)
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
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let (a, b, uplo) = self;
        let a_view = a.view();
        let a_dyn = a_view.to_dyn();
        let b_view = b.view();
        let b_dyn = b_view.to_dyn();
        Ok(blas_impl_solve_triangular_f(a_dyn, b_dyn.into(), uplo)?.into_owned())
    }
}

#[duplicate_item(
    ImplType                                   TrA                                 TrB                              ;
   ['b, T, DA, DB, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, DA> ] [TensorMut<'b, T, DeviceBLAS, DB>];
   ['b, T, DA, DB,                          ] [TensorView<'_, T, DeviceBLAS, DA>] [TensorMut<'b, T, DeviceBLAS, DB>];
   [    T, DA, DB, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, DA> ] [Tensor<T, DeviceBLAS, DB>       ];
   [    T, DA, DB,                          ] [TensorView<'_, T, DeviceBLAS, DA>] [Tensor<T, DeviceBLAS, DB>       ];
)]
impl<ImplType> SolveTriangularAPI<DeviceBLAS> for (TrA, TrB, Option<FlagUpLo>)
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
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let (a, mut b, uplo) = self;
        let a_view = a.view();
        let a_dyn = a_view.to_dyn();
        let b_ref = b.view_mut().into_dim::<IxD>();
        let result = blas_impl_solve_triangular_f(a_dyn, b_ref.into(), uplo)?;
        result.clone_to_mut();
        Ok(b)
    }
}

/* #endregion */

/* #region sub-args */

#[duplicate_item(
    ImplStruct             args_tuple     internal_tuple     ;
   [(TrA, TrB, FlagUpLo)] [(a, b, uplo)] [(a, b, Some(uplo))];
   [(TrA, TrB,         )] [(a, b,     )] [(a, b, None      )];
)]
impl<TrA, TrB> SolveTriangularAPI<DeviceBLAS> for ImplStruct
where
    (TrA, TrB, Option<FlagUpLo>): SolveTriangularAPI<DeviceBLAS>,
{
    type Out = <(TrA, TrB, Option<FlagUpLo>) as SolveTriangularAPI<DeviceBLAS>>::Out;
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let args_tuple = self;
        SolveTriangularAPI::<DeviceBLAS>::solve_triangular_f(internal_tuple)
    }
}

/* #endregion */

#[duplicate_item(
    ImplType        TrA                                TrB                              ;
   ['b, T, DA, DB] [TensorMut<'_, T, DeviceBLAS, DA>] [TensorMut<'b, T, DeviceBLAS, DB>];
   [    T, DA, DB] [TensorMut<'_, T, DeviceBLAS, DA>] [Tensor<T, DeviceBLAS, DB>       ];
   ['b, T, DA, DB] [Tensor<T, DeviceBLAS, DA>       ] [TensorMut<'b, T, DeviceBLAS, DB>];
   [    T, DA, DB] [Tensor<T, DeviceBLAS, DA>       ] [Tensor<T, DeviceBLAS, DB>       ];
)]
impl<ImplType> SolveTriangularAPI<DeviceBLAS> for (TrA, TrB, Option<FlagUpLo>)
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
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let (a, mut b, uplo) = self;
        let a_view = a.view();
        let a_dyn = a_view.to_dyn();
        let b_ref = b.view_mut().into_dim::<IxD>();
        let result = blas_impl_solve_triangular_f(a_dyn, b_ref.into(), uplo)?;
        result.clone_to_mut();
        Ok(b)
    }
}
