//! n-dim `cho_solve` on the BLAS device: solve `A x = b` from the Cholesky
//! factorization of every square matrix slice of the stack (two triangular
//! solves per slice, composed like `?POTRS`).
//!
//! Matrix slices: the two matrix axes are the last two shape axes under a
//! row-major device and the first two under a column-major one; the rest is
//! the batch. The batch dims of the factor and `b` broadcast against each
//! other.
//!
//! # Shapes
//!
//! - Factor `c` is `(batch, N, N)` and `b` is `(batch_b, N, K)` or the vector form `(batch_b, N)`;
//!   the solution `x` has the broadcast shape of `b`, `(N, K)` or `(N,)` per slice.
//! - The form of `b` is relative to the rank of the factor: `b` with one fewer axis than `c` is the
//!   vector form. Next to a stacked factor a 2-D `b` is therefore a batch of vectors, not one
//!   matrix shared by every factor; to share one matrix, give it a size-1 batch axis, `(1, N, K)`.
//!
//! # Overloads
//!
//! - Flat `cho_solve((c, uplo, b))`; nested as in SciPy `cho_solve(((c, uplo), b))`.
//! - The result of [`cho_factor`](crate::linalg_auto_impl::cho_factor), which carries the triangle:
//!   `cho_solve((cho_factor(a), b))` and `cho_solve((&factors, b))`.
//! - Any triangular matrix with an explicit flag, e.g. the result of
//!   [`cholesky`](crate::linalg_auto_impl::cholesky): `cho_solve(((&l, Lower), &b))`.
//!
//! `c` is `&TensorAny`, `TensorView` or an owned `Tensor` with a read-only
//! `b`, and any operand kind with a consumed `b`. `b` is read-only
//! (`&TensorAny`, `TensorView`) for a fresh solution, or consumed
//! (`TensorMut`, `Tensor`) to be solved in place.
//!
//! # Policy
//!
//! - Only the `uplo` triangle of `c` is read: a factor of `cholesky` or `cho_factor` (cleaned or
//!   not), or one with arbitrary values elsewhere, serves; the factor is only read. `Lower` reads
//!   `L z = b`, `Lᴴ x = z`; `Upper` reads `Uᴴ z = b`, `U x = z`.
//! - A factor in either storage order is solved without copying it; an off-order factor with the
//!   conjugated step runs as the conjugated transposed solve.
//! - An immutable `b` is solved into a freshly allocated broadcast solution; an owned or mutable
//!   `b` is solved in place when its batch is the broadcast batch (the in-place solve cannot grow),
//!   or in a contiguous work buffer written back into `b`.
//! - Zero-size (`N == 0` or `K == 0`) is empty without a driver call. Devices that differ are a
//!   `DeviceMismatch` error; a non-square factor or a `b` breaking the shape or batch rules is an
//!   `InvalidLayout` error.
//!
//! See also [`cho_factor`](crate::linalg_auto_impl::cho_factor) and
//! [`solve_triangular`](crate::linalg_auto_impl::solve_triangular).

use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/// Device-facing entry of the n-dim `cho_solve`; see the module
/// documentation for the shapes and the input policy.
pub fn blas_impl_cho_solve_f<'b, T>(
    c: TensorView<'_, T, DeviceBLAS, IxD>,
    uplo: FlagUpLo,
    b: TensorReference<'b, T, DeviceBLAS, IxD>,
) -> Result<TensorMutable<'b, T, DeviceBLAS, IxD>>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>,
{
    let order = c.device().default_order();
    rstsr_assert!(
        c.shape().len() >= 2,
        InvalidLayout,
        "cho_solve: the factor must be at least 2-D, got {}-D",
        c.shape().len()
    )?;
    let (_batch, inner) = stack_split_shape(c.shape(), 2, order)?;
    rstsr_assert_eq!(
        inner[0],
        inner[1],
        InvalidLayout,
        "cho_solve: the factor must be square, got {}x{}",
        inner[0],
        inner[1]
    )?;
    let plan = solve_plan(c.shape(), b.shape(), order, "cho_solve")?;
    solve_stack_f(c, b, &plan, order, |c_slice, b_slice| {
        ref_impl_cho_solve_f(c_slice, uplo, b_slice.into())?;
        Ok(())
    })
}

/* #region core flat form, read-only b */

#[duplicate_item(
    ImplTypes                                                           TrC                                 TrB                               ;
   [T, DA, DB, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceBLAS, DA>] [&TensorAny<Rb, T, DeviceBLAS, DB>];
   [T, DA, DB, R: DataAPI<Data = Vec<T>>                            ] [&TensorAny<R, T, DeviceBLAS, DA> ] [TensorView<'_, T, DeviceBLAS, DB>];
   [T, DA, DB, R: DataAPI<Data = Vec<T>>                            ] [TensorView<'_, T, DeviceBLAS, DA>] [&TensorAny<R, T, DeviceBLAS, DB> ];
   [T, DA, DB,                                                      ] [TensorView<'_, T, DeviceBLAS, DA>] [TensorView<'_, T, DeviceBLAS, DB>];
   [T, DA, DB, R: DataAPI<Data = Vec<T>>                            ] [Tensor<T, DeviceBLAS, DA>        ] [&TensorAny<R, T, DeviceBLAS, DB> ];
   [T, DA, DB,                                                      ] [Tensor<T, DeviceBLAS, DA>        ] [TensorView<'_, T, DeviceBLAS, DB>];
)]
impl<ImplTypes> ChoSolveAPI<DeviceBLAS> for (TrC, FlagUpLo, TrB)
where
    T: BlasFloat,
    DA: DimAPI,
    DB: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, IxD>;
    fn cho_solve_f(self) -> Result<Self::Out> {
        let (c, uplo, b) = self;
        Ok(blas_impl_cho_solve_f(c.view().to_dyn(), uplo, b.view().to_dyn().into())?.into_owned())
    }
}

/* #endregion */

/* #region core flat form, consumed b */

#[duplicate_item(
    ImplTypes                                  TrC                                 TrB                              ;
   ['b, T, DA, DB, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, DA> ] [TensorMut<'b, T, DeviceBLAS, DB>];
   ['b, T, DA, DB,                          ] [TensorView<'_, T, DeviceBLAS, DA>] [TensorMut<'b, T, DeviceBLAS, DB>];
   ['b, T, DA, DB,                          ] [TensorMut<'_, T, DeviceBLAS, DA> ] [TensorMut<'b, T, DeviceBLAS, DB>];
   ['b, T, DA, DB,                          ] [Tensor<T, DeviceBLAS, DA>        ] [TensorMut<'b, T, DeviceBLAS, DB>];
   [    T, DA, DB, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, DA> ] [Tensor<T, DeviceBLAS, DB>       ];
   [    T, DA, DB,                          ] [TensorView<'_, T, DeviceBLAS, DA>] [Tensor<T, DeviceBLAS, DB>       ];
   [    T, DA, DB,                          ] [TensorMut<'_, T, DeviceBLAS, DA> ] [Tensor<T, DeviceBLAS, DB>       ];
   [    T, DA, DB,                          ] [Tensor<T, DeviceBLAS, DA>        ] [Tensor<T, DeviceBLAS, DB>       ];
)]
impl<ImplTypes> ChoSolveAPI<DeviceBLAS> for (TrC, FlagUpLo, TrB)
where
    T: BlasFloat,
    DA: DimAPI,
    DB: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = TrB;
    fn cho_solve_f(self) -> Result<Self::Out> {
        let (c, uplo, mut b) = self;
        let result = blas_impl_cho_solve_f(c.view().to_dyn(), uplo, b.view_mut().into_dim::<IxD>().into())?;
        result.clone_to_mut();
        Ok(b)
    }
}

/* #endregion */

/* #region SciPy nested form and the cho_factor result */

#[duplicate_item(
    ImplStruct                 args_tuple       internal_tuple    ;
   [((TrC, FlagUpLo), TrB)]  [((c, uplo), b)] [(c, uplo, b)    ];
)]
impl<TrC, TrB> ChoSolveAPI<DeviceBLAS> for ImplStruct
where
    (TrC, FlagUpLo, TrB): ChoSolveAPI<DeviceBLAS>,
{
    type Out = <(TrC, FlagUpLo, TrB) as ChoSolveAPI<DeviceBLAS>>::Out;
    fn cho_solve_f(self) -> Result<Self::Out> {
        let args_tuple = self;
        ChoSolveAPI::<DeviceBLAS>::cho_solve_f(internal_tuple)
    }
}

impl<TrC, TrB> ChoSolveAPI<DeviceBLAS> for (ChoFactorResult<TrC>, TrB)
where
    (TrC, FlagUpLo, TrB): ChoSolveAPI<DeviceBLAS>,
{
    type Out = <(TrC, FlagUpLo, TrB) as ChoSolveAPI<DeviceBLAS>>::Out;
    fn cho_solve_f(self) -> Result<Self::Out> {
        let (ChoFactorResult { c, uplo }, b) = self;
        ChoSolveAPI::<DeviceBLAS>::cho_solve_f((c, uplo, b))
    }
}

impl<'x, TrC, TrB> ChoSolveAPI<DeviceBLAS> for (&'x ChoFactorResult<TrC>, TrB)
where
    (&'x TrC, FlagUpLo, TrB): ChoSolveAPI<DeviceBLAS>,
{
    type Out = <(&'x TrC, FlagUpLo, TrB) as ChoSolveAPI<DeviceBLAS>>::Out;
    fn cho_solve_f(self) -> Result<Self::Out> {
        let (factors, b) = self;
        ChoSolveAPI::<DeviceBLAS>::cho_solve_f((&factors.c, factors.uplo, b))
    }
}

/* #endregion */
