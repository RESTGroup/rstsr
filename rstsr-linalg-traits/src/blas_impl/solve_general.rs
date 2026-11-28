use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/// n-dim `solve_general`: solve `a x = b` for a stack `a` of `(M, M)` matrices
/// and right-hand side `b` of `(M, K)` matrices or `(M)` vectors (the two
/// matrix axes are the last two shape axes under a row-major device, the
/// first two under a column-major one); the batch dims of `a` and `b`
/// broadcast against each other. A rectangular `a` is rejected rather than
/// solved in the least-squares sense (the array-API contract).
///
/// An immutable `b` is solved into a freshly allocated broadcast solution; a
/// mutable `b` is solved in place when its batch fits the broadcast batch
/// (the in-place solve cannot grow), or in a contiguous work buffer written
/// back by `clone_to_mut` at the call site.
pub fn blas_impl_solve_general_f<'b, T>(
    a: TensorView<'_, T, DeviceBLAS, IxD>,
    b: TensorReference<'b, T, DeviceBLAS, IxD>,
) -> Result<TensorMutable<'b, T, DeviceBLAS, IxD>>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>,
{
    let order = a.device().default_order();
    let plan = solve_plan(a.shape(), b.shape(), order, "solve_general")?;
    solve_stack_f(a, b, &plan, order, |a_slice, b_slice| {
        let result = ref_impl_solve_general_f(a_slice.into(), b_slice.into())?;
        result.clone_to_mut();
        Ok(())
    })
}

#[duplicate_item(
    ImplType                                                            TrA                                 TrB                               ;
   [T, DA, DB, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceBLAS, DA>] [&TensorAny<Rb, T, DeviceBLAS, DB>];
   [T, DA, DB, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceBLAS, DA> ] [TensorView<'_, T, DeviceBLAS, DB>];
   [T, DA, DB, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceBLAS, DA>] [&TensorAny<R, T, DeviceBLAS, DB> ];
   [T, DA, DB,                                                       ] [TensorView<'_, T, DeviceBLAS, DA>] [TensorView<'_, T, DeviceBLAS, DB>];
)]
impl<ImplType> SolveGeneralAPI<DeviceBLAS> for (TrA, TrB)
where
    T: BlasFloat,
    DA: DimAPI,
    DB: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, IxD>;
    fn solve_general_f(self) -> Result<Self::Out> {
        let (a, b) = self;
        let a_view = a.view();
        let a_dyn = a_view.to_dyn();
        let b_view = b.view();
        let b_dyn = b_view.to_dyn();
        Ok(blas_impl_solve_general_f(a_dyn, b_dyn.into())?.into_owned())
    }
}

#[duplicate_item(
    ImplType                                   TrA                                 TrB                              ;
   ['b, T, DA, DB, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, DA> ] [TensorMut<'b, T, DeviceBLAS, DB>];
   ['b, T, DA, DB,                          ] [TensorView<'_, T, DeviceBLAS, DA>] [TensorMut<'b, T, DeviceBLAS, DB>];
   [    T, DA, DB, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, DA> ] [Tensor<T, DeviceBLAS, DB>       ];
   [    T, DA, DB,                          ] [TensorView<'_, T, DeviceBLAS, DA>] [Tensor<T, DeviceBLAS, DB>       ];
)]
impl<ImplType> SolveGeneralAPI<DeviceBLAS> for (TrA, TrB)
where
    T: BlasFloat,
    DA: DimAPI,
    DB: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = TrB;
    fn solve_general_f(self) -> Result<Self::Out> {
        let (a, mut b) = self;
        let a_view = a.view();
        let a_dyn = a_view.to_dyn();
        let b_ref = b.view_mut().into_dim::<IxD>();
        let result = blas_impl_solve_general_f(a_dyn, b_ref.into())?;
        result.clone_to_mut();
        Ok(b)
    }
}

#[duplicate_item(
    ImplType        TrA                                TrB                              ;
   ['b, T, DA, DB] [TensorMut<'_, T, DeviceBLAS, DA>] [TensorMut<'b, T, DeviceBLAS, DB>];
   [    T, DA, DB] [TensorMut<'_, T, DeviceBLAS, DA>] [Tensor<T, DeviceBLAS, DB>       ];
   ['b, T, DA, DB] [Tensor<T, DeviceBLAS, DA>       ] [TensorMut<'b, T, DeviceBLAS, DB>];
   [    T, DA, DB] [Tensor<T, DeviceBLAS, DA>       ] [Tensor<T, DeviceBLAS, DB>       ];
)]
impl<ImplType> SolveGeneralAPI<DeviceBLAS> for (TrA, TrB)
where
    T: BlasFloat,
    DA: DimAPI,
    DB: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = TrB;
    fn solve_general_f(self) -> Result<Self::Out> {
        let (a, mut b) = self;
        let a_view = a.view();
        let a_dyn = a_view.to_dyn();
        let b_ref = b.view_mut().into_dim::<IxD>();
        let result = blas_impl_solve_general_f(a_dyn, b_ref.into())?;
        result.clone_to_mut();
        Ok(b)
    }
}
