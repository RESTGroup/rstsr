//! Matrix power on a stack of square matrices (BLAS devices).
//!
//! NumPy's `matrix_power` with the faer entry's exponentiation: `n = 0` yields
//! the identity, `n < 0` inverts the base once, then the base is raised to
//! `|n|` by exponentiation-by-squaring over the bits of `|n|` (LSB to MSB).
//!
//! An owned or mutable input is raised **in place** (the `inv` rule): the
//! power is written into the input's own storage when its layout matches the
//! device order, and into a contiguous work buffer written back by
//! `clone_to_mut` at the call site otherwise; pass `a.view()` to keep the
//! input. A singular base with `n < 0` **errors** (`InvalidValue` from the
//! `GETRF` factorization behind `inv`) — the faer device returns `inf`/`NaN`
//! entries there instead, and NumPy raises. That error leaves an in-place
//! input **unspecified**: the slices before the singular one are already
//! inverted and the singular one holds its `LU` factors (a work-buffer input
//! is not written back); pass a view when the input must survive.

use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

use super::inv::blas_impl_inv_f;

/// Raise one stack to a nonnegative power `e >= 2` by exponentiation by
/// squaring (`e` = 0 and 1 are handled by the caller, `e < 0` via `inv`).
/// The base is read through a view; it is copied only when the result starts
/// from it (odd `e >= 5`).
fn power_of<T>(base: TensorView<'_, T, DeviceBLAS, IxD>, e: usize) -> Result<Tensor<T, DeviceBLAS, IxD>>
where
    T: BlasFloat,
    DeviceBLAS: DeviceCreationAnyAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceMatMulAPI<T, T, T, IxD, IxD, IxD>,
{
    match e {
        2 => matmul_f(&base, &base),
        3 => {
            let base2 = matmul_f(&base, &base)?;
            matmul_f(&base2, &base)
        },
        _ => {
            // binary exponentiation over the bits of `e`, LSB to MSB: the first
            // set bit makes the result start from the base, the later ones
            // accumulate the running square `z`
            let mut z = matmul_f(&base, &base)?;
            let mut result = if e & 1 == 1 { Some(base.into_owned()) } else { None };
            let mut e = e >> 1;
            while e > 0 {
                if e & 1 == 1 {
                    result = Some(match result {
                        Some(acc) => matmul_f(&acc, &z)?,
                        None => z.clone(),
                    });
                }
                e >>= 1;
                if e > 0 {
                    z = matmul_f(&z, &z)?;
                }
            }
            match result {
                Some(result) => Ok(result),
                // `e >= 4` keeps at least one bit after the first shift
                None => unreachable!("matrix_power: exponentiation loop set no result"),
            }
        },
    }
}

/// n-dim `matrix_power` on the BLAS devices: raise every square matrix slice
/// of the stack to the integer power `n`.
///
/// The matrix axes are the last two shape axes under a row-major device and
/// the first two under a column-major one; the rest is the batch. An owned or
/// mutable input is consumed and returned raised in place (the `inv` rule);
/// a read-only input yields a fresh result. See the module doc for the
/// singular-base behavior.
pub fn blas_impl_matrix_power_f<T>(
    a: TensorReference<'_, T, DeviceBLAS, IxD>,
    n: isize,
) -> Result<TensorMutable<'_, T, DeviceBLAS, IxD>>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationNumAPI<T>
        + DeviceCreationTriAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceAPI<T, Raw = Vec<T>>
        + OpAssignArbitaryAPI<T, Ix2, Ix2>
        + OpAssignAPI<T, Ix2>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>
        + OpAssignAPI<T, Ix1>
        + DeviceMatMulAPI<T, T, T, IxD, IxD, IxD>,
{
    let device = a.device().clone();
    let order = device.default_order();
    let stack = StackLayout::new(a.layout(), 2, order)?;
    let inner_shape = stack.inner_shape();
    rstsr_assert_eq!(
        inner_shape[0],
        inner_shape[1],
        InvalidLayout,
        "matrix_power: the matrix must be square, got {}x{}",
        inner_shape[0],
        inner_shape[1]
    )?;

    let mut converted = overwritable_convert_with_order(a, order)?;
    let converted_stack = StackLayout::new(converted.view().layout(), 2, order)?;

    if n == 0 {
        // the identity, written per slice into the converted buffer
        let eye: Tensor<T, DeviceBLAS, Ix2> = eye_f((inner_shape[0], &device))?.into_dim::<Ix2>();
        for i in 0..stack.len() {
            let mut slice = stack_slice_at_mut(converted.view_mut(), &converted_stack, i)?.into_dim::<Ix2>();
            slice.assign_f(eye.view())?;
        }
        return Ok(converted);
    }

    // negative exponent: invert once in place, then raise the inverse to |n|
    if n < 0 {
        let inverted = blas_impl_inv_f(converted.view_mut().into())?;
        inverted.clone_to_mut();
    }

    let e = n.unsigned_abs();
    if e == 1 {
        return Ok(converted);
    }

    // the squaring body reads the converted buffer through a view; the result
    // is written back into it, so every input kind ends up in the same storage
    let result = power_of(converted.view(), e)?;
    converted.view_mut().assign_f(result.view())?;
    Ok(converted)
}

#[duplicate_item(
    ImplType                          Tr                                  ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> MatrixPowerAPI<DeviceBLAS> for (Tr, isize)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationNumAPI<T>
        + DeviceCreationTriAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceAPI<T, Raw = Vec<T>>
        + OpAssignArbitaryAPI<T, Ix2, Ix2>
        + OpAssignAPI<T, Ix2>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>
        + OpAssignAPI<T, Ix1>
        + DeviceMatMulAPI<T, T, T, IxD, IxD, IxD>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn matrix_power_f(self) -> Result<Self::Out> {
        let (a, n) = self;
        let v = a.view();
        let view = v.to_dyn();
        let result = blas_impl_matrix_power_f(view.into(), n)?;
        Ok(result.into_owned().into_dim::<D>())
    }
}

#[duplicate_item(
    ImplType   Tr                              ;
   ['a, T, D] [TensorMut<'a, T, DeviceBLAS, D>];
   [    T, D] [Tensor<T, DeviceBLAS, D>       ];
)]
impl<ImplType> MatrixPowerAPI<DeviceBLAS> for (Tr, isize)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationNumAPI<T>
        + DeviceCreationTriAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceAPI<T, Raw = Vec<T>>
        + OpAssignArbitaryAPI<T, Ix2, Ix2>
        + OpAssignAPI<T, Ix2>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>
        + OpAssignAPI<T, Ix1>
        + DeviceMatMulAPI<T, T, T, IxD, IxD, IxD>,
{
    type Out = Tr;
    fn matrix_power_f(self) -> Result<Self::Out> {
        let (a, n) = self;
        let mut a = a;
        let result = blas_impl_matrix_power_f(a.view_mut().into_dim::<IxD>().into(), n)?;
        result.clone_to_mut();
        Ok(a)
    }
}
