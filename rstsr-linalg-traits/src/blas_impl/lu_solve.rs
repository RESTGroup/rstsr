//! n-dim `lu_solve` on the BLAS device: solve `A x = b` (or `Aᵀ x = b`,
//! `Aᴴ x = b`) from a packed LU factorization of every square matrix slice
//! of the stack.
//!
//! Matrix slices: the two matrix axes are the last two shape axes under a
//! row-major device and the first two under a column-major one; the rest is
//! the batch. The batch dims of the factors and `b` broadcast against each
//! other.
//!
//! # Shapes
//!
//! - Packed form `(lu, piv)`: `lu` is `(batch, N, N)` and `piv` `(batch, N)` with the same batch (a
//!   [`LuFactorResult`]); `b` is `(batch_b, N, K)` or the vector form `(batch_b, N)`; the solution
//!   `x` has the broadcast shape of `b`, `(N, K)` or `(N,)` per slice.
//! - The form of `b` is relative to the rank of the factors: `b` with one fewer axis than `lu` is
//!   the vector form `(batch_b, N)`; with as many axes it is `(batch_b, N, K)`. Next to a stacked
//!   factor a 2-D `b` is therefore a batch of vectors, not one matrix shared by every factor as
//!   SciPy reads it; to share one matrix, give it a size-1 batch axis, `(1, N, K)`.
//! - Factor form `(&LuResult, b)`: `p` `(batch, N)`, `l` and `u` `(batch, N, N)` with the same
//!   batch. The factors must be unpermuted (`p` present): a result of `lu` with `permute_l` is an
//!   `InvalidValue` error.
//!
//! # Overloads
//!
//! - Packed factors, flat `lu_solve((lu, piv, b))`; nested as in SciPy `lu_solve(((lu, piv), b))`;
//!   or as the result of [`lu_factor`](crate::linalg_auto_impl::lu_factor):
//!   `lu_solve((lu_factor(a), b))` and `lu_solve((&factors, b))`.
//! - Explicit factors of [`lu`](crate::linalg_auto_impl::lu): `lu_solve((&factors, b))`.
//! - Each form takes an optional last `FlagTrans`.
//!
//! `lu` and `piv` are `&TensorAny`, `TensorView` or an owned `Tensor` with a read-only `b`, and
//! any operand kind with a consumed `b`. `b` is read-only (`&TensorAny`, `TensorView`) for a fresh
//! solution, or consumed (`TensorMut`, `Tensor`) to be solved in place.
//!
//! # Policy
//!
//! - `trans` selects the system (`FlagTrans::N`, `T` or `C`; the bare forms solve `A x = b`).
//! - An immutable `b` is solved into a freshly allocated solution of the broadcast shape; an owned
//!   or mutable `b` is solved in place when its batch is the broadcast batch (the in-place solve
//!   cannot grow), or in a contiguous work buffer written back into `b`.
//! - The factors are only read; `lu` and `piv` must share their batch, and `piv` must hold exactly
//!   `N` pivots per slice.
//! - A singular `U` gives `inf` / `NaN` entries (no error). Zero-size slices are empty without a
//!   driver call.
//!
//! See also [`lu`](crate::linalg_auto_impl::lu) and
//! [`lu_factor`](crate::linalg_auto_impl::lu_factor).

use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/// The explicit factor form of `lu`, as [`blas_impl_lu_solve_plu_f`] and the
/// `(&LuResult, b)` overloads take it.
type LuResultRef<'x, T, DP, DA> =
    &'x LuResult<Option<Tensor<usize, DeviceBLAS, DP>>, Tensor<T, DeviceBLAS, DA>, Tensor<T, DeviceBLAS, DA>>;

/// Device-facing entry of the n-dim `lu_solve` on the packed factor form;
/// see the module documentation for the shapes and the input policy.
pub fn blas_impl_lu_solve_f<'b, T>(
    lu: TensorView<'_, T, DeviceBLAS, IxD>,
    piv: TensorView<'_, usize, DeviceBLAS, IxD>,
    b: TensorReference<'b, T, DeviceBLAS, IxD>,
    trans: FlagTrans,
) -> Result<TensorMutable<'b, T, DeviceBLAS, IxD>>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>,
{
    lu_solve_stack_f(lu, None, piv, b, trans, true)
}

/// Device-facing entry of the n-dim `lu_solve` on the factor form of `lu`
/// (the factors must be unpermuted: `p` present).
pub fn blas_impl_lu_solve_plu_f<'b, DP, DA, T>(
    lu: LuResultRef<'_, T, DP, DA>,
    b: TensorReference<'b, T, DeviceBLAS, IxD>,
    trans: FlagTrans,
) -> Result<TensorMutable<'b, T, DeviceBLAS, IxD>>
where
    T: BlasFloat,
    DP: DimAPI,
    DA: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    let p = lu.p.as_ref().ok_or_else(|| {
        rstsr_error!(
            InvalidValue,
            "lu_solve: the factors are permuted (lu with permute_l gives A = L @ U without p); this form needs p"
        )
    })?;
    lu_solve_stack_f(lu.l.view().to_dyn(), Some(lu.u.view().to_dyn()), p.view().to_dyn(), b, trans, false)
}

/// The stack walk shared by the two forms: `l` unit lower triangular and `u`
/// upper triangular squares (`u` is `None` for the packed form, where `l`
/// carries both triangles), `piv` either the `?GETRF` pivot sequence or the
/// row indices of `A = L[P] @ U` (`piv_are_swaps` selects).
fn lu_solve_stack_f<'b, T>(
    l: TensorView<'_, T, DeviceBLAS, IxD>,
    u: Option<TensorView<'_, T, DeviceBLAS, IxD>>,
    piv: TensorView<'_, usize, DeviceBLAS, IxD>,
    b: TensorReference<'b, T, DeviceBLAS, IxD>,
    trans: FlagTrans,
    piv_are_swaps: bool,
) -> Result<TensorMutable<'b, T, DeviceBLAS, IxD>>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>,
{
    let device = b.device().clone();
    rstsr_assert!(
        device.same_device(l.device())
            && u.as_ref().is_none_or(|u| device.same_device(u.device()))
            && device.same_device(piv.device()),
        DeviceMismatch,
        "lu_solve: the factors and b are on different devices"
    )?;
    let order = device.default_order();
    rstsr_assert!(
        l.shape().len() >= 2,
        InvalidLayout,
        "lu_solve: the factorization must be at least 2-D, got {}-D",
        l.shape().len()
    )?;
    let (batch_a, inner_a) = stack_split_shape(l.shape(), 2, order)?;
    rstsr_assert_eq!(
        inner_a[0],
        inner_a[1],
        InvalidLayout,
        "lu_solve: the factorization must be square, got {}x{}",
        inner_a[0],
        inner_a[1]
    )?;
    let n = inner_a[0];
    let plan = solve_plan(l.shape(), b.shape(), order, "lu_solve")?;
    if let Some(u) = &u {
        rstsr_assert_eq!(
            u.shape(),
            l.shape(),
            InvalidLayout,
            "lu_solve: l and u must have the same shape, got {:?} vs {:?}",
            u.shape(),
            l.shape()
        )?;
    }
    let piv_shape = stack_shape(&batch_a, &[n], order);
    rstsr_assert_eq!(
        piv.shape(),
        piv_shape.as_slice(),
        InvalidLayout,
        "lu_solve: piv must have the batch of the factors and {} entries, i.e. shape {:?}, got {:?}",
        n,
        piv_shape,
        piv.shape()
    )?;

    // the factors follow the broadcast batch of the plan
    let u_full = u.as_ref().map(|u| u.broadcast_to(plan.a_shape(order)));
    let u_stack = u_full.as_ref().map(|u| StackLayout::new(u.layout(), 2, order)).transpose()?;
    let piv_full = piv.broadcast_to(stack_shape(&plan.batch_out, &[n], order));
    let piv_stack = StackLayout::new(piv_full.layout(), 1, order)?;

    solve_stack_enumerated_f(l, b, &plan, order, |i, l_slice, mut b_slice| {
        let u_slice = match (&u_full, &u_stack) {
            (Some(u_full), Some(u_stack)) => stack_slice_at(u_full.view(), u_stack, i)?.into_dim::<Ix2>(),
            // the packed form: `l_slice` carries both triangles
            _ => l_slice.view(),
        };
        let piv_slice = stack_slice_at(piv_full.view(), &piv_stack, i)?.into_dim::<Ix1>();
        let ipiv = match piv_are_swaps {
            true => piv_slice.to_vec(),
            false => lu_perm_to_ipiv(&piv_slice.to_vec()),
        };
        ref_impl_lu_solve_f(l_slice.view(), u_slice, &ipiv, b_slice.view_mut().into(), trans)?;
        Ok(())
    })
}

/* #region packed factors, read-only b */

#[duplicate_item(
    ImplTypes                                                                                                TrLu                                 TrPiv                                    TrB                                ;
   [T, DA, DP, DB, Ra: DataAPI<Data = Vec<T>>, Rp: DataAPI<Data = Vec<usize>>, Rb: DataAPI<Data = Vec<T>>]  [&TensorAny<Ra, T, DeviceBLAS, DA>]  [&TensorAny<Rp, usize, DeviceBLAS, DP>]  [&TensorAny<Rb, T, DeviceBLAS, DB>];
   [T, DA, DP, DB, Ra: DataAPI<Data = Vec<T>>, Rp: DataAPI<Data = Vec<usize>>]                              [&TensorAny<Ra, T, DeviceBLAS, DA>]  [&TensorAny<Rp, usize, DeviceBLAS, DP>]  [TensorView<'_, T, DeviceBLAS, DB>];
   [T, DA, DP, DB, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>]                                  [&TensorAny<Ra, T, DeviceBLAS, DA>]  [TensorView<'_, usize, DeviceBLAS, DP>]  [&TensorAny<Rb, T, DeviceBLAS, DB>];
   [T, DA, DP, DB, Ra: DataAPI<Data = Vec<T>>]                                                              [&TensorAny<Ra, T, DeviceBLAS, DA>]  [TensorView<'_, usize, DeviceBLAS, DP>]  [TensorView<'_, T, DeviceBLAS, DB>];
   [T, DA, DP, DB, Rp: DataAPI<Data = Vec<usize>>, Rb: DataAPI<Data = Vec<T>>]                              [TensorView<'_, T, DeviceBLAS, DA>]  [&TensorAny<Rp, usize, DeviceBLAS, DP>]  [&TensorAny<Rb, T, DeviceBLAS, DB>];
   [T, DA, DP, DB, Rp: DataAPI<Data = Vec<usize>>]                                                          [TensorView<'_, T, DeviceBLAS, DA>]  [&TensorAny<Rp, usize, DeviceBLAS, DP>]  [TensorView<'_, T, DeviceBLAS, DB>];
   [T, DA, DP, DB, Rb: DataAPI<Data = Vec<T>>]                                                              [TensorView<'_, T, DeviceBLAS, DA>]  [TensorView<'_, usize, DeviceBLAS, DP>]  [&TensorAny<Rb, T, DeviceBLAS, DB>];
   [T, DA, DP, DB]                                                                                          [TensorView<'_, T, DeviceBLAS, DA>]  [TensorView<'_, usize, DeviceBLAS, DP>]  [TensorView<'_, T, DeviceBLAS, DB>];
   [T, DA, DP, DB, Rb: DataAPI<Data = Vec<T>>]                                                              [Tensor<T, DeviceBLAS, DA>]          [Tensor<usize, DeviceBLAS, DP>]          [&TensorAny<Rb, T, DeviceBLAS, DB>];
   [T, DA, DP, DB]                                                                                          [Tensor<T, DeviceBLAS, DA>]          [Tensor<usize, DeviceBLAS, DP>]          [TensorView<'_, T, DeviceBLAS, DB>];
   [T, DA, DP, DB, Rb: DataAPI<Data = Vec<T>>]                                                              [Tensor<T, DeviceBLAS, DA>]          [TensorView<'_, usize, DeviceBLAS, DP>]  [&TensorAny<Rb, T, DeviceBLAS, DB>];
   [T, DA, DP, DB]                                                                                          [Tensor<T, DeviceBLAS, DA>]          [TensorView<'_, usize, DeviceBLAS, DP>]  [TensorView<'_, T, DeviceBLAS, DB>];
   [T, DA, DP, DB, Rb: DataAPI<Data = Vec<T>>]                                                              [TensorView<'_, T, DeviceBLAS, DA>]  [Tensor<usize, DeviceBLAS, DP>]          [&TensorAny<Rb, T, DeviceBLAS, DB>];
   [T, DA, DP, DB]                                                                                          [TensorView<'_, T, DeviceBLAS, DA>]  [Tensor<usize, DeviceBLAS, DP>]          [TensorView<'_, T, DeviceBLAS, DB>];
)]
impl<ImplTypes> LuSolveAPI<DeviceBLAS> for (TrLu, TrPiv, TrB, FlagTrans)
where
    T: BlasFloat,
    DA: DimAPI,
    DP: DimAPI,
    DB: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, IxD>;
    fn lu_solve_f(self) -> Result<Self::Out> {
        let (lu, piv, b, trans) = self;
        Ok(blas_impl_lu_solve_f(lu.view().to_dyn(), piv.view().to_dyn(), b.view().to_dyn().into(), trans)?.into_owned())
    }
}

/* #endregion */

/* #region packed factors, consumed b */

#[duplicate_item(
    ImplTypes                                                                        TrLu                               TrPiv                                  TrB;
   ['b, T, DA, DP, DB, Ra: DataAPI<Data = Vec<T>>, Rp: DataAPI<Data = Vec<usize>>]  [&TensorAny<Ra, T, DeviceBLAS, DA> ] [&TensorAny<Rp, usize, DeviceBLAS, DP> ] [TensorMut<'b, T, DeviceBLAS, DB>];
   ['b, T, DA, DP, DB, Ra: DataAPI<Data = Vec<T>>]                                  [&TensorAny<Ra, T, DeviceBLAS, DA> ] [TensorView<'_, usize, DeviceBLAS, DP> ] [TensorMut<'b, T, DeviceBLAS, DB>];
   ['b, T, DA, DP, DB, Ra: DataAPI<Data = Vec<T>>]                                  [&TensorAny<Ra, T, DeviceBLAS, DA> ] [TensorMut<'_, usize, DeviceBLAS, DP>  ] [TensorMut<'b, T, DeviceBLAS, DB>];
   ['b, T, DA, DP, DB, Ra: DataAPI<Data = Vec<T>>]                                  [&TensorAny<Ra, T, DeviceBLAS, DA> ] [Tensor<usize, DeviceBLAS, DP>         ] [TensorMut<'b, T, DeviceBLAS, DB>];
   ['b, T, DA, DP, DB, Rp: DataAPI<Data = Vec<usize>>]                              [TensorView<'_, T, DeviceBLAS, DA> ] [&TensorAny<Rp, usize, DeviceBLAS, DP> ] [TensorMut<'b, T, DeviceBLAS, DB>];
   ['b, T, DA, DP, DB]                                                              [TensorView<'_, T, DeviceBLAS, DA> ] [TensorView<'_, usize, DeviceBLAS, DP> ] [TensorMut<'b, T, DeviceBLAS, DB>];
   ['b, T, DA, DP, DB]                                                              [TensorView<'_, T, DeviceBLAS, DA> ] [TensorMut<'_, usize, DeviceBLAS, DP>  ] [TensorMut<'b, T, DeviceBLAS, DB>];
   ['b, T, DA, DP, DB]                                                              [TensorView<'_, T, DeviceBLAS, DA> ] [Tensor<usize, DeviceBLAS, DP>         ] [TensorMut<'b, T, DeviceBLAS, DB>];
   ['b, T, DA, DP, DB, Rp: DataAPI<Data = Vec<usize>>]                              [TensorMut<'_, T, DeviceBLAS, DA>  ] [&TensorAny<Rp, usize, DeviceBLAS, DP> ] [TensorMut<'b, T, DeviceBLAS, DB>];
   ['b, T, DA, DP, DB]                                                              [TensorMut<'_, T, DeviceBLAS, DA>  ] [TensorView<'_, usize, DeviceBLAS, DP> ] [TensorMut<'b, T, DeviceBLAS, DB>];
   ['b, T, DA, DP, DB]                                                              [TensorMut<'_, T, DeviceBLAS, DA>  ] [TensorMut<'_, usize, DeviceBLAS, DP>  ] [TensorMut<'b, T, DeviceBLAS, DB>];
   ['b, T, DA, DP, DB]                                                              [TensorMut<'_, T, DeviceBLAS, DA>  ] [Tensor<usize, DeviceBLAS, DP>         ] [TensorMut<'b, T, DeviceBLAS, DB>];
   ['b, T, DA, DP, DB, Rp: DataAPI<Data = Vec<usize>>]                              [Tensor<T, DeviceBLAS, DA>         ] [&TensorAny<Rp, usize, DeviceBLAS, DP> ] [TensorMut<'b, T, DeviceBLAS, DB>];
   ['b, T, DA, DP, DB]                                                              [Tensor<T, DeviceBLAS, DA>         ] [TensorView<'_, usize, DeviceBLAS, DP> ] [TensorMut<'b, T, DeviceBLAS, DB>];
   ['b, T, DA, DP, DB]                                                              [Tensor<T, DeviceBLAS, DA>         ] [TensorMut<'_, usize, DeviceBLAS, DP>  ] [TensorMut<'b, T, DeviceBLAS, DB>];
   ['b, T, DA, DP, DB]                                                              [Tensor<T, DeviceBLAS, DA>         ] [Tensor<usize, DeviceBLAS, DP>         ] [TensorMut<'b, T, DeviceBLAS, DB>];
   [T, DA, DP, DB, Ra: DataAPI<Data = Vec<T>>, Rp: DataAPI<Data = Vec<usize>>]      [&TensorAny<Ra, T, DeviceBLAS, DA> ] [&TensorAny<Rp, usize, DeviceBLAS, DP> ] [Tensor<T, DeviceBLAS, DB>];
   [T, DA, DP, DB, Ra: DataAPI<Data = Vec<T>>]                                      [&TensorAny<Ra, T, DeviceBLAS, DA> ] [TensorView<'_, usize, DeviceBLAS, DP> ] [Tensor<T, DeviceBLAS, DB>];
   [T, DA, DP, DB, Ra: DataAPI<Data = Vec<T>>]                                      [&TensorAny<Ra, T, DeviceBLAS, DA> ] [TensorMut<'_, usize, DeviceBLAS, DP>  ] [Tensor<T, DeviceBLAS, DB>];
   [T, DA, DP, DB, Ra: DataAPI<Data = Vec<T>>]                                      [&TensorAny<Ra, T, DeviceBLAS, DA> ] [Tensor<usize, DeviceBLAS, DP>         ] [Tensor<T, DeviceBLAS, DB>];
   [T, DA, DP, DB, Rp: DataAPI<Data = Vec<usize>>]                                  [TensorView<'_, T, DeviceBLAS, DA> ] [&TensorAny<Rp, usize, DeviceBLAS, DP> ] [Tensor<T, DeviceBLAS, DB>];
   [T, DA, DP, DB]                                                                  [TensorView<'_, T, DeviceBLAS, DA> ] [TensorView<'_, usize, DeviceBLAS, DP> ] [Tensor<T, DeviceBLAS, DB>];
   [T, DA, DP, DB]                                                                  [TensorView<'_, T, DeviceBLAS, DA> ] [TensorMut<'_, usize, DeviceBLAS, DP>  ] [Tensor<T, DeviceBLAS, DB>];
   [T, DA, DP, DB]                                                                  [TensorView<'_, T, DeviceBLAS, DA> ] [Tensor<usize, DeviceBLAS, DP>         ] [Tensor<T, DeviceBLAS, DB>];
   [T, DA, DP, DB, Rp: DataAPI<Data = Vec<usize>>]                                  [TensorMut<'_, T, DeviceBLAS, DA>  ] [&TensorAny<Rp, usize, DeviceBLAS, DP> ] [Tensor<T, DeviceBLAS, DB>];
   [T, DA, DP, DB]                                                                  [TensorMut<'_, T, DeviceBLAS, DA>  ] [TensorView<'_, usize, DeviceBLAS, DP> ] [Tensor<T, DeviceBLAS, DB>];
   [T, DA, DP, DB]                                                                  [TensorMut<'_, T, DeviceBLAS, DA>  ] [TensorMut<'_, usize, DeviceBLAS, DP>  ] [Tensor<T, DeviceBLAS, DB>];
   [T, DA, DP, DB]                                                                  [TensorMut<'_, T, DeviceBLAS, DA>  ] [Tensor<usize, DeviceBLAS, DP>         ] [Tensor<T, DeviceBLAS, DB>];
   [T, DA, DP, DB, Rp: DataAPI<Data = Vec<usize>>]                                  [Tensor<T, DeviceBLAS, DA>         ] [&TensorAny<Rp, usize, DeviceBLAS, DP> ] [Tensor<T, DeviceBLAS, DB>];
   [T, DA, DP, DB]                                                                  [Tensor<T, DeviceBLAS, DA>         ] [TensorView<'_, usize, DeviceBLAS, DP> ] [Tensor<T, DeviceBLAS, DB>];
   [T, DA, DP, DB]                                                                  [Tensor<T, DeviceBLAS, DA>         ] [TensorMut<'_, usize, DeviceBLAS, DP>  ] [Tensor<T, DeviceBLAS, DB>];
   [T, DA, DP, DB]                                                                  [Tensor<T, DeviceBLAS, DA>         ] [Tensor<usize, DeviceBLAS, DP>         ] [Tensor<T, DeviceBLAS, DB>];
)]
impl<ImplTypes> LuSolveAPI<DeviceBLAS> for (TrLu, TrPiv, TrB, FlagTrans)
where
    T: BlasFloat,
    DA: DimAPI,
    DP: DimAPI,
    DB: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = TrB;
    fn lu_solve_f(self) -> Result<Self::Out> {
        let (lu, piv, mut b, trans) = self;
        let result = blas_impl_lu_solve_f(
            lu.view().to_dyn(),
            piv.view().to_dyn(),
            b.view_mut().into_dim::<IxD>().into(),
            trans,
        )?;
        result.clone_to_mut();
        Ok(b)
    }
}

/* #endregion */

/* #region packed factors, default trans */

impl<TrLu, TrPiv, TrB> LuSolveAPI<DeviceBLAS> for (TrLu, TrPiv, TrB)
where
    (TrLu, TrPiv, TrB, FlagTrans): LuSolveAPI<DeviceBLAS>,
{
    type Out = <(TrLu, TrPiv, TrB, FlagTrans) as LuSolveAPI<DeviceBLAS>>::Out;
    fn lu_solve_f(self) -> Result<Self::Out> {
        let (lu, piv, b) = self;
        LuSolveAPI::<DeviceBLAS>::lu_solve_f((lu, piv, b, NoTrans))
    }
}

/* #endregion */

/* #region factor form of lu */

#[duplicate_item(
    ImplTypes                                   TrB;
   [T, DP, DA, DB, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Rb, T, DeviceBLAS, DB>];
   [T, DP, DA, DB]                             [TensorView<'_, T, DeviceBLAS, DB>];
)]
impl<ImplTypes> LuSolveAPI<DeviceBLAS> for (LuResultRef<'_, T, DP, DA>, TrB, FlagTrans)
where
    T: BlasFloat,
    DP: DimAPI,
    DA: DimAPI,
    DB: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, IxD>;
    fn lu_solve_f(self) -> Result<Self::Out> {
        let (lu, b, trans) = self;
        Ok(blas_impl_lu_solve_plu_f(lu, b.view().to_dyn().into(), trans)?.into_owned())
    }
}

#[duplicate_item(
    ImplTypes       TrB;
   ['b, T, DP, DA, DB] [TensorMut<'b, T, DeviceBLAS, DB>];
   [    T, DP, DA, DB] [Tensor<T, DeviceBLAS, DB>       ];
)]
impl<ImplTypes> LuSolveAPI<DeviceBLAS> for (LuResultRef<'_, T, DP, DA>, TrB, FlagTrans)
where
    T: BlasFloat,
    DP: DimAPI,
    DA: DimAPI,
    DB: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = TrB;
    fn lu_solve_f(self) -> Result<Self::Out> {
        let (lu, mut b, trans) = self;
        let result = blas_impl_lu_solve_plu_f(lu, b.view_mut().into_dim::<IxD>().into(), trans)?;
        result.clone_to_mut();
        Ok(b)
    }
}

impl<TrF, TrB> LuSolveAPI<DeviceBLAS> for (TrF, TrB)
where
    (TrF, TrB, FlagTrans): LuSolveAPI<DeviceBLAS>,
{
    type Out = <(TrF, TrB, FlagTrans) as LuSolveAPI<DeviceBLAS>>::Out;
    fn lu_solve_f(self) -> Result<Self::Out> {
        let (factors, b) = self;
        LuSolveAPI::<DeviceBLAS>::lu_solve_f((factors, b, NoTrans))
    }
}

/* #endregion */

/* #region SciPy form ((lu, piv), b) and the lu_factor result */

impl<TrLu, TrPiv, TrB> LuSolveAPI<DeviceBLAS> for ((TrLu, TrPiv), TrB, FlagTrans)
where
    (TrLu, TrPiv, TrB, FlagTrans): LuSolveAPI<DeviceBLAS>,
{
    type Out = <(TrLu, TrPiv, TrB, FlagTrans) as LuSolveAPI<DeviceBLAS>>::Out;
    fn lu_solve_f(self) -> Result<Self::Out> {
        let ((lu, piv), b, trans) = self;
        LuSolveAPI::<DeviceBLAS>::lu_solve_f((lu, piv, b, trans))
    }
}

impl<LU, Piv, TrB> LuSolveAPI<DeviceBLAS> for (LuFactorResult<LU, Piv>, TrB, FlagTrans)
where
    ((LU, Piv), TrB, FlagTrans): LuSolveAPI<DeviceBLAS>,
{
    type Out = <((LU, Piv), TrB, FlagTrans) as LuSolveAPI<DeviceBLAS>>::Out;
    fn lu_solve_f(self) -> Result<Self::Out> {
        let (LuFactorResult { lu, piv }, b, trans) = self;
        LuSolveAPI::<DeviceBLAS>::lu_solve_f(((lu, piv), b, trans))
    }
}

impl<'x, LU, Piv, TrB> LuSolveAPI<DeviceBLAS> for (&'x LuFactorResult<LU, Piv>, TrB, FlagTrans)
where
    (&'x LU, &'x Piv, TrB, FlagTrans): LuSolveAPI<DeviceBLAS>,
{
    type Out = <(&'x LU, &'x Piv, TrB, FlagTrans) as LuSolveAPI<DeviceBLAS>>::Out;
    fn lu_solve_f(self) -> Result<Self::Out> {
        let (factors, b, trans) = self;
        LuSolveAPI::<DeviceBLAS>::lu_solve_f((&factors.lu, &factors.piv, b, trans))
    }
}

/* #endregion */
