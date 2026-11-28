//! Matrix and vector norms on the BLAS devices (the array-API `linalg` norms).
//!
//! A direct port of the faer entries: `matrix_norm` follows NumPy with the
//! matrix axes as the device-order pair (the last two under `RowMajor`, the
//! first two under `ColMajor`); `'fro'` is the L2 norm over both axes,
//! `1`/`-1`/`inf`/`-inf` are abs -> sum -> max/min, and `2`/`-2`/`'nuc'` are
//! the max/min/sum of the singular values (via [`super::svdvals::
//! blas_impl_svdvals_f`]). `vector_norm` reduces `|x|` over the requested axes
//! with the L2 reduction for `p = 2`, a plain sum for `p = 1`, a count of
//! nonzeros for `p = 0`, max/min for `p = ±inf`, and the general
//! `(sum |x|^p)^(1/p)` otherwise.
//!
//! A normed region with a zero-size dimension yields `0` under **every** order.
//! That is a deliberate divergence from NumPy, which is shape-dependent on the
//! min-based orders of an empty matrix: for a `(0, N)` matrix it returns `0`
//! for `-1` and raises for `-2` and `-inf`, for an `(M, 0)` matrix it returns
//! `0` for `-inf` and raises for `-1` and `-2`; the other matrix orders return
//! `0`. For an empty vector it returns `0` except for `-inf` (raises) and the
//! other negative orders (`inf`).

use crate::DeviceBLAS;
use num::FromPrimitive;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

use super::svdvals::blas_impl_svdvals_f;

/// Output shape of a matrix norm: the batch shape, with the two matrix axes
/// re-inserted as singletons under `keepdims` (trailing under `RowMajor`,
/// leading under `ColMajor`).
fn matrix_out_shape(batch_shape: &[usize], order: FlagOrder, keepdims: bool) -> Vec<usize> {
    if keepdims {
        stack_shape(batch_shape, &[1, 1], order)
    } else {
        batch_shape.to_vec()
    }
}

/// n-dim `matrix_norm` on the BLAS devices: one norm per matrix slice of the
/// stack, real-typed and batch-shaped.
///
/// The matrix axes are the last two shape axes under a row-major device and
/// the first two under a column-major one; the rest is the batch. The order
/// resolution (`None` -> `'fro'`, only `±1`/`±2` numerically) happens before
/// the empty-input short-circuit, so an invalid order errors on an empty
/// input too. The SVD-based orders run GESDD without `U`/`Vᵀ`.
pub fn blas_impl_matrix_norm_f<T>(
    a: TensorView<'_, T, DeviceBLAS, IxD>,
    ord: NormOrd,
    keepdims: bool,
) -> Result<Tensor<T::Real, DeviceBLAS, IxD>>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>,
{
    let ord = ord.resolve_matrix()?;
    let device = a.device().clone();
    let order = device.default_order();
    let (batch_shape, inner) = stack_split_shape(a.shape(), 2, order)?;
    let [m, n] = [inner[0], inner[1]];

    // an empty input yields an empty (or, when only a matrix axis vanished, all
    // zero) result: NumPy's `initial=0`, and `max`/`min` reject a zero-size input
    if m == 0 || n == 0 || batch_shape.contains(&0) {
        let out: Tensor<T::Real, DeviceBLAS, IxD> =
            zeros_f((matrix_out_shape(&batch_shape, order, keepdims), &device))?;
        return Ok(out);
    }

    // matrix axis indices, and the surviving axis after the first reduction
    let (row_ax, col_ax, after_row, after_col, k_ax): (isize, isize, isize, isize, isize) = match order {
        RowMajor => (-2, -1, -1, -1, -1),
        ColMajor => (0, 1, 0, 0, 0),
    };

    let reduced: Tensor<T::Real, DeviceBLAS, IxD> = match ord {
        // `l2_norm` on the (possibly complex) matrix already computes
        // `sqrt(sum |x|^2)`, so no `abs` copy is needed
        NormOrd::Fro => l2_norm_with_args_f(&a, ([row_ax, col_ax], false))?,
        NormOrd::Nuc => sum_with_args_f(blas_impl_svdvals_f(a.into())?, (k_ax, false))?,
        NormOrd::Inf => {
            let t = sum_with_args_f(abs_f(&a)?, (col_ax, false))?;
            max_with_args_f(&t, (after_col, false))?
        },
        NormOrd::NegInf => {
            let t = sum_with_args_f(abs_f(&a)?, (col_ax, false))?;
            min_with_args_f(&t, (after_col, false))?
        },
        // the numeric orders a matrix has a norm for: ±1 (abs -> sum -> max/min)
        // and ±2 (largest/smallest singular value); `resolve_matrix` fixed the set
        NormOrd::P(p) => {
            if p == 1.0 {
                let t = sum_with_args_f(abs_f(&a)?, (row_ax, false))?;
                max_with_args_f(&t, (after_row, false))?
            } else if p == -1.0 {
                let t = sum_with_args_f(abs_f(&a)?, (row_ax, false))?;
                min_with_args_f(&t, (after_row, false))?
            } else if p == 2.0 {
                max_with_args_f(blas_impl_svdvals_f(a.into())?, (k_ax, false))?
            } else if p == -2.0 {
                min_with_args_f(blas_impl_svdvals_f(a.into())?, (k_ax, false))?
            } else {
                return Err(rstsr_error!(InvalidValue, "matrix_norm: invalid norm order"));
            }
        },
        _ => return Err(rstsr_error!(InvalidValue, "matrix_norm: invalid norm order")),
    };

    if keepdims {
        let shape = matrix_out_shape(&batch_shape, order, true).into_iter().map(|v| v as isize).collect::<Vec<_>>();
        Ok(reduced.into_shape_f(shape)?)
    } else {
        Ok(reduced)
    }
}

/// n-dim `vector_norm` on the BLAS devices: `|x|` reduced over the requested
/// axes, real-typed.
///
/// `None` resolves to the `2`-norm; `'fro'`/`'nuc'` are rejected. The axes
/// are validated before the empty-input short-circuit, so a bad axis errors
/// on an empty input too.
pub fn blas_impl_vector_norm_f<T>(
    a: TensorView<'_, T, DeviceBLAS, IxD>,
    ord: NormOrd,
    axes: AxesIndex<isize>,
    keepdims: bool,
) -> Result<Tensor<T::Real, DeviceBLAS, IxD>>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>,
{
    let ord = ord.resolve_vector()?;
    let device = a.device().clone();

    // validate the axes up front, so a bad axis errors even when the empty-input
    // short-circuit below would otherwise swallow it
    let ndim = a.ndim();
    let axes_norm = match &axes {
        AxesIndex::None => (0..ndim as isize).collect::<Vec<_>>(),
        _ => normalize_axes_index(axes.clone(), ndim, false, true)?,
    };

    // an empty input yields an empty (or, when only a reduced axis vanished, all
    // zero) result: NumPy's `initial=0`, and `max`/`min` reject a zero-size input
    if a.shape().contains(&0) {
        let mut is_reduced = vec![false; ndim];
        for &ax in &axes_norm {
            is_reduced[ax as usize] = true;
        }
        let out_shape = is_reduced
            .iter()
            .enumerate()
            .filter_map(|(i, &r)| (!r).then(|| a.shape()[i]).or_else(|| keepdims.then_some(1)))
            .collect::<Vec<_>>();
        let out: Tensor<T::Real, DeviceBLAS, IxD> = zeros_f((out_shape, &device))?;
        return Ok(out);
    }

    match ord {
        NormOrd::Inf => max_with_args_f(abs_f(&a)?, (axes.clone(), keepdims)),
        NormOrd::NegInf => min_with_args_f(abs_f(&a)?, (axes.clone(), keepdims)),
        NormOrd::P(p) => {
            // p = 2 (the default) is the L2 reduction on the input directly; p = 1
            // a plain sum of |x|; p = 0 a count of nonzeros (`sign(|x|)` is 1
            // exactly on nonzero entries); anything else the general (sum |x|^p)^(1/p)
            if p == 2.0 {
                l2_norm_with_args_f(&a, (axes.clone(), keepdims))
            } else if p == 1.0 {
                sum_with_args_f(abs_f(&a)?, (axes.clone(), keepdims))
            } else if p == 0.0 {
                sum_with_args_f(sign_f(abs_f(&a)?)?, (axes.clone(), keepdims))
            } else {
                let abs_a = abs_f(&a)?;
                let out_of_range = || rstsr_error!(InvalidValue, "vector_norm: order {p} out of range");
                let exp = T::Real::from_f64(p).ok_or_else(out_of_range)?;
                let inv_exp = T::Real::from_f64(1.0 / p).ok_or_else(out_of_range)?;
                let t = pow_f(&abs_a, exp)?;
                let s = sum_with_args_f(&t, (axes.clone(), keepdims))?;
                pow_f(&s, inv_exp)
            }
        },
        _ => Err(rstsr_error!(InvalidValue, "vector_norm: invalid norm order")),
    }
}

#[duplicate_item(
    ImplType                          Tr                                  ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D                           ] [Tensor<T, DeviceBLAS, D>        ];
)]
impl<ImplType> MatrixNormAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T::Real, DeviceBLAS, IxD>;
    fn matrix_norm_f(self) -> Result<Self::Out> {
        blas_impl_matrix_norm_f(self.view().to_dyn(), NormOrd::None, false)
    }
}

#[duplicate_item(
    ImplType                          Tr                                  ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D                           ] [Tensor<T, DeviceBLAS, D>        ];
)]
impl<ImplType, A> MatrixNormAPI<DeviceBLAS> for (Tr, A)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
    A: TryInto<MatrixNormArgs>,
    A::Error: Into<Error>,
{
    type Out = Tensor<T::Real, DeviceBLAS, IxD>;
    fn matrix_norm_f(self) -> Result<Self::Out> {
        let (a, args) = self;
        let MatrixNormArgs { ord, keepdims } = args.try_into().map_err(Into::into)?;
        blas_impl_matrix_norm_f(a.view().to_dyn(), ord, keepdims)
    }
}

#[duplicate_item(
    ImplType                          Tr                                  ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D                           ] [Tensor<T, DeviceBLAS, D>        ];
)]
impl<ImplType> VectorNormAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T::Real, DeviceBLAS, IxD>;
    fn vector_norm_f(self) -> Result<Self::Out> {
        blas_impl_vector_norm_f(self.view().to_dyn(), NormOrd::None, AxesIndex::None, false)
    }
}

#[duplicate_item(
    ImplType                          Tr                                  ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D                           ] [Tensor<T, DeviceBLAS, D>        ];
)]
impl<ImplType, A> VectorNormAPI<DeviceBLAS> for (Tr, A)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
    A: TryInto<VectorNormArgs>,
    A::Error: Into<Error>,
{
    type Out = Tensor<T::Real, DeviceBLAS, IxD>;
    fn vector_norm_f(self) -> Result<Self::Out> {
        let (a, args) = self;
        let VectorNormArgs { ord, axes, keepdims } = args.try_into().map_err(Into::into)?;
        blas_impl_vector_norm_f(a.view().to_dyn(), ord, axes, keepdims)
    }
}
