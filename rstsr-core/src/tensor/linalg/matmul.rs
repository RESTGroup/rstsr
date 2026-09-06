//! Matrix-multiplication for tensor.

use crate::prelude_dev::*;
use core::ops::{Mul, Rem};
use num::{One, Zero};

/* #region matmul by function */

/// Matrix multiplication of two tensors, following the Array API `matmul`
/// semantics.
///
/// <div class="warning">
///
/// **Row/Column Major Notice**
///
/// This function behaves differently on default orders ([`RowMajor`] and [`ColMajor`]) of device.
///
/// </div>
///
/// The last two axes of each operand are the matrix dimensions, and any
/// leading axes broadcast against each other; one-dimensional operands are
/// folded into the matrix dimensions (see the rule table below). The result
/// is an owned tensor, contiguous in the device default order. Under
/// [`ColMajor`], the same rules apply with all axes reversed: the matrix
/// dimensions are the *first* two axes, and trailing axes broadcast. See
/// [`order_semantics`](crate::order_semantics) for the two orders.
///
/// The supported shape combinations (written for [`RowMajor`]; `M`, `K`, `N`
/// are matrix dimensions and `...` denotes broadcast batch axes):
///
/// | A | B | C |
/// |----|---|---|
/// | `N` | `N` | scalar |
/// | `M, K` | `K, N` | `M, N` |
/// | `K` | `..., K, N` | `..., N` |
/// | `..., M, K` | `K` | `..., M` |
/// | `M, K` | `..., K, N` | `..., M, N` |
/// | `..., M, K` | `K, N` | `..., M, N` |
/// | `..., M, K` | `..., K, N` | `..., M, N` |
///
/// # Parameters
///
/// - `a`: the left operand (views and owned tensors both accepted).
/// - `b`: the right operand (views and owned tensors both accepted).
///
/// # Returns
///
/// - [`Tensor<TC, B, DC>`][`Tensor`]: the matrix product, owning its data.
///
/// # Examples
///
/// Matrix multiplication, and matrix-vector products:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
/// let b = rt::tensor_from_nested!([[1, 0], [1, 1]], &device);
/// println!("{}", rt::matmul(&a, &b));
/// // [[ 3 2]
/// //  [ 7 4]]
/// let v = rt::tensor_from_nested!([1, 2], &device);
/// println!("{}", rt::matmul(&a, &v));
/// // [ 5 11]
/// # assert_eq!(format!("{}", rt::matmul(&a, &v)), "[ 5 11]");
/// ```
///
/// One-dimensional operands form an inner product (scalar tensor):
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let x = rt::tensor_from_nested!([1, 2, 3], &device);
/// let y = rt::tensor_from_nested!([4, 5, 6], &device);
/// println!("{}", rt::matmul(&x, &y));
/// // 32
/// # assert_eq!(format!("{}", rt::matmul(&x, &y)), "32");
/// ```
///
/// Batched matrices broadcast their leading axes:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((12, &device)).into_shape([2, 2, 3]);
/// let b = rt::arange((6, &device)).into_shape([3, 2]);
/// let c = rt::matmul(&a, &b);
/// println!("{c}");
/// // [[[ 10 13]
/// //   [ 28 40]]
/// //
/// //  [[ 46 67]
/// //   [ 64 94]]]
/// # assert_eq!(format!("{c}"), "[[[ 10 13]\n  [ 28 40]]\n\n [[ 46 67]\n  [ 64 94]]]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `matmul(x1, x2, /)` ([`matmul`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.matmul.html))
/// - NumPy: `numpy.matmul(x1, x2)` ([`numpy.matmul`](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html))
/// - RSTSR: `rt::matmul(&a, &b)`, method `a.matmul(&b)`, or operator `a % b`.
///
/// # Panics
///
/// - Panics if the operand shapes do not fit the rule table (for example, mismatching matrix
///   dimensions, or a 0-dimensional operand).
///
/// For a fallible version, use [`matmul_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - Python Array API standard: [`matmul`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.matmul.html)
/// - NumPy: [`numpy.matmul`](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)
///
/// ## Related functions in RSTSR
///
/// - [`vecdot`]: dot product along specified axes.
/// - [`matmul_from`]: GEMM-style `c = beta * c + alpha * (a @ b)`.
/// - [`matmul_with_output`]: write the plain product into a provided output.
/// - [`rem`](crate::tensor::operators::exports::rem()): element-wise remainder
///   - this is *not* `%` between two tensors.
///
/// ## Variants of this function
///
/// - [`matmul_f`]: fallible version.
/// - Associated methods on [`TensorAny`]: [`TensorAny::matmul`] / [`TensorAny::matmul_f`].
/// - Operator [`Rem`]: `a % b` calls this function.
pub fn matmul<TA, TB, TC, DA, DB, DC, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
) -> Tensor<TC, B, DC>
where
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: Mul<TB, Output = TC>,
    TC: Zero + One,
    B: DeviceCreationAnyAPI<TC>,
    LayoutMatMulConfig<DA, DB>: LayoutMatMulAPI<DA, DB, DC = DC>,
    B: DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    op_refa_refb_matmul(a, b, TC::one()).rstsr_unwrap()
}

/// GEMM-style matrix multiplication with output scaling: writes
/// `c = beta * c + alpha * (a @ b)`.
///
/// The shapes follow the same rules as [`matmul`]; the output `c` must have
/// the resulting shape (its batch axes may already be broadcast-shaped). This
/// is the direct analogue of BLAS `GEMM` with arbitrary strides on the
/// matrix dimensions.
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders for the values it writes (the operands' existing layouts are
/// used as given).
///
/// # Parameters
///
/// - `c`: the output tensor (mutable view or owned tensor).
/// - `a` / `b`: the operands.
/// - `alpha`: scaling factor of the matrix product.
/// - `beta`: scaling factor of the existing `c`.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);
/// let b = rt::tensor_from_nested!([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], &device);
/// let mut c: Tensor<f64, _> = rt::ones(([2, 2], &device));
/// rt::matmul_from(&mut c, &a, &b, 2.0, 1.5);
/// println!("{c}");
/// // [[ 21.5 27.5]
/// //  [ 57.5 81.5]]
/// # assert_eq!(format!("{c}"), "[[ 21.5 27.5]\n [ 57.5 81.5]]");
/// ```
///
/// # Notes of API accordance
///
/// - BLAS: `GEMM` / `GEMV` family (`C := alpha * A @ B + beta * C`)
/// - RSTSR: `rt::matmul_from(&mut c, &a, &b, alpha, beta)`; method form `c.matmul_from(&a, &b,
///   alpha, beta)`.
///
/// # Panics
///
/// - Panics if the operands' shapes do not follow the [`matmul`] rules, if `c` mismatches the
///   resulting shape, or if the devices differ.
///
/// For a fallible version, use [`matmul_from_f`].
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`matmul`]: allocate the output internally.
/// - [`matmul_with_output`]: write the plain product (`alpha = 1`, `beta = 0`).
///
/// ## Variants of this function
///
/// - [`matmul_from_f`]: fallible version.
/// - Associated methods on [`TensorAny`]: [`TensorAny::matmul_from`] /
///   [`TensorAny::matmul_from_f`].
pub fn matmul_from<TA, TB, TC, DA, DB, DC, B>(
    c: impl TensorViewMutAPI<Type = TC, Backend = B, Dim = DC>,
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    alpha: TC,
    beta: TC,
) where
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    B: DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    op_mutc_refa_refb_matmul(c, a, b, alpha, beta).rstsr_unwrap()
}

/// Device-level driver of matmul with output: writes `alpha * (a @ b) + beta * c`.
///
/// See also [`matmul_from`].
pub fn op_mutc_refa_refb_matmul<TA, TB, TC, DA, DB, DC, B>(
    mut c: impl TensorViewMutAPI<Type = TC, Backend = B, Dim = DC>,
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    alpha: TC,
    beta: TC,
) -> Result<()>
where
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    B: DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    let (a, b, mut c) = (a.view(), b.view(), c.view_mut());
    rstsr_assert!(c.device().same_device(a.device()), DeviceMismatch)?;
    rstsr_assert!(c.device().same_device(b.device()), DeviceMismatch)?;
    let device = c.device().clone();
    let la = a.layout();
    let lb = b.layout();
    let lc = c.layout().clone();
    let sa = a.raw();
    let sb = b.raw();
    let sc = c.raw_mut();
    device.matmul(sc, &lc, sa, la, sb, lb, alpha, beta)
}

/// Device-level driver of matmul, allocating the output; see also [`matmul`].
pub fn op_refa_refb_matmul<TA, TB, TC, DA, DB, DC, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    alpha: TC,
) -> Result<Tensor<TC, B, DC>>
where
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TC: Zero,
    B: DeviceCreationAnyAPI<TC>,
    LayoutMatMulConfig<DA, DB>: LayoutMatMulAPI<DA, DB, DC = DC>,
    B: DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    let (a, b) = (a.view(), b.view());
    rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
    let default_order = a.device().default_order();
    let cfg = LayoutMatMulConfig::<DA, DB>::layout_matmul(a.layout(), b.layout(), default_order)?;
    let lc = cfg.lc;
    let mut c: Tensor<TC, B, _> = unsafe { empty((lc, a.device())) }.into_dim_f()?;
    op_mutc_refa_refb_matmul(&mut c, &a, &b, alpha, TC::zero())?;
    return Ok(c);
}

/// Matrix multiplication, writing the plain product into a provided output.
///
/// See also [`matmul_with_output`].
pub fn matmul_with_output_f<TA, TB, TC, DA, DB, DC, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    c: impl TensorViewMutAPI<Type = TC, Backend = B, Dim = DC>,
) -> Result<()>
where
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TC: Zero + One,
    B: DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    op_mutc_refa_refb_matmul(c, a, b, TC::one(), TC::zero())
}

/// Matrix multiplication, writing the plain product into a provided output.
///
/// The same operation as [`matmul`] (`a @ b`), but the result is written into
/// `c` instead of being allocated; this is [`matmul_from`] with `alpha = 1`
/// and `beta = 0` (the previous contents of `c` are overwritten).
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders for the values it writes (the operands' existing layouts are
/// used as given).
///
/// # Parameters
///
/// - `a` / `b`: the operands.
/// - `c`: the output tensor (mutable view or owned tensor), filled with `a @ b`.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);
/// let b = rt::tensor_from_nested!([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], &device);
/// let mut d: Tensor<f64, _> = rt::zeros(([2, 2], &device));
/// rt::matmul_with_output(&a, &b, &mut d);
/// println!("{d}");
/// // [[ 10 13]
/// //  [ 28 40]]
/// # assert_eq!(format!("{d}"), "[[ 10 13]\n [ 28 40]]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `matmul(x1, x2, /)` with an explicit `out` ([`matmul`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.matmul.html))
/// - RSTSR: `rt::matmul_with_output(&a, &b, &mut c)`; method form `a.matmul_with_output(&b, &mut
///   c)`.
///
/// # Panics
///
/// - Panics if the operands' shapes do not follow the [`matmul`] rules, if `c` mismatches the
///   resulting shape, or if the devices differ.
///
/// For a fallible version, use [`matmul_with_output_f`].
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`matmul`]: allocate the output internally.
/// - [`matmul_from`]: GEMM-style scaling of operands and output.
///
/// ## Variants of this function
///
/// - [`matmul_with_output_f`]: fallible version.
/// - Associated methods on [`TensorAny`]: [`TensorAny::matmul_with_output`] /
///   [`TensorAny::matmul_with_output_f`].
pub fn matmul_with_output<TA, TB, TC, DA, DB, DC, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    c: impl TensorViewMutAPI<Type = TC, Backend = B, Dim = DC>,
) where
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TC: Zero + One,
    B: DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    op_mutc_refa_refb_matmul(c, a, b, TC::one(), TC::zero()).rstsr_unwrap()
}

/// GEMM-style matrix multiplication with output scaling: writes `c = beta * c + alpha * (a @ b)`.
///
/// See also [`matmul_from`].
pub fn matmul_from_f<TA, TB, TC, DA, DB, DC, B>(
    c: impl TensorViewMutAPI<Type = TC, Backend = B, Dim = DC>,
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    alpha: TC,
    beta: TC,
) -> Result<()>
where
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    B: DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    op_mutc_refa_refb_matmul(c, a, b, alpha, beta)
}

/// Matrix multiplication of two tensors, following the Array API `matmul` semantics.
///
/// See also [`matmul`].
pub fn matmul_f<TA, TB, TC, DA, DB, DC, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
) -> Result<Tensor<TC, B, DC>>
where
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: Mul<TB, Output = TC>,
    TC: Zero + One,
    B: DeviceCreationAnyAPI<TC>,
    LayoutMatMulConfig<DA, DB>: LayoutMatMulAPI<DA, DB, DC = DC>,
    B: DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    op_refa_refb_matmul(a, b, TC::one())
}

/* #endregion */

/* #region matmul implementation to core ops */

#[duplicate_item(
     TrA                         TrB                       ;
    [ TensorAny<RA, TA, B, DA>] [ TensorAny<RB, TB, B, DB>];
    [&TensorAny<RA, TA, B, DA>] [ TensorAny<RB, TB, B, DB>];
    [ TensorAny<RA, TA, B, DA>] [&TensorAny<RB, TB, B, DB>];
    [&TensorAny<RA, TA, B, DA>] [&TensorAny<RB, TB, B, DB>];
)]
/// Matrix multiplication by the `%` operator; see [`matmul`].
impl<RA, RB, TA, TB, TC, DA, DB, DC, B> Rem<TrB> for TrA
where
    // storage
    RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: Mul<TB, Output = TC>,
    TC: Zero + One,
    B: DeviceCreationAnyAPI<TC>,
    LayoutMatMulConfig<DA, DB>: LayoutMatMulAPI<DA, DB, DC = DC>,
    B: DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    type Output = Tensor<TC, B, DC>;
    fn rem(self, rhs: TrB) -> Self::Output {
        op_refa_refb_matmul(self, rhs, TC::one()).rstsr_unwrap()
    }
}

/* #endregion */

/* #region matmul tensor trait */

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Matrix multiplication of two tensors.
    ///
    /// See also [`matmul`].
    pub fn matmul_f<TB, TC, DB, DC>(
        &self,
        rhs: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    ) -> Result<Tensor<TC, B, DC>>
    where
        // dimension
        DB: DimAPI,
        DC: DimAPI,
        // operation specific
        T: Mul<TB, Output = TC>,
        TC: Zero + One,
        B: DeviceCreationAnyAPI<TC>,
        LayoutMatMulConfig<D, DB>: LayoutMatMulAPI<D, DB, DC = DC>,
        B: DeviceMatMulAPI<T, TB, TC, D, DB, DC>,
    {
        op_refa_refb_matmul(self.view(), rhs, TC::one())
    }

    /// Matrix multiplication of two tensors.
    ///
    /// See also [`matmul`].
    pub fn matmul<TB, TC, DB, DC>(&self, rhs: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>) -> Tensor<TC, B, DC>
    where
        // dimension
        DB: DimAPI,
        DC: DimAPI,
        // operation specific
        T: Mul<TB, Output = TC>,
        TC: Zero + One,
        B: DeviceCreationAnyAPI<TC>,
        LayoutMatMulConfig<D, DB>: LayoutMatMulAPI<D, DB, DC = DC>,
        B: DeviceMatMulAPI<T, TB, TC, D, DB, DC>,
    {
        op_refa_refb_matmul(self.view(), rhs, TC::one()).rstsr_unwrap()
    }

    /// Matrix multiplication, writing the plain product into a provided output.
    ///
    /// See also [`matmul_with_output`].
    pub fn matmul_with_output_f<TB, TC, DB, DC>(
        &self,
        rhs: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
        c: impl TensorViewMutAPI<Type = TC, Backend = B, Dim = DC>,
    ) -> Result<()>
    where
        // dimension
        DB: DimAPI,
        DC: DimAPI,
        // operation specific
        TC: Zero + One,
        B: DeviceMatMulAPI<T, TB, TC, D, DB, DC>,
    {
        op_mutc_refa_refb_matmul(c, self.view(), rhs, TC::one(), TC::zero())
    }

    /// Matrix multiplication, writing the plain product into a provided output.
    ///
    /// See also [`matmul_with_output`].
    pub fn matmul_with_output<TB, TC, DB, DC>(
        &self,
        rhs: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
        c: impl TensorViewMutAPI<Type = TC, Backend = B, Dim = DC>,
    ) where
        // dimension
        DB: DimAPI,
        DC: DimAPI,
        // operation specific
        TC: Zero + One,
        B: DeviceMatMulAPI<T, TB, TC, D, DB, DC>,
    {
        op_mutc_refa_refb_matmul(c, self.view(), rhs, TC::one(), TC::zero()).rstsr_unwrap()
    }

    /// GEMM-style matrix multiplication with output scaling.
    ///
    /// See also [`matmul_from`].
    pub fn matmul_from_f<TA, TB, DA, DB>(
        &mut self,
        a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
        b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
        alpha: T,
        beta: T,
    ) -> Result<()>
    where
        // storage
        R: DataMutAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
        // dimension
        DA: DimAPI,
        DB: DimAPI,
        // operation specific
        B: DeviceMatMulAPI<TA, TB, T, DA, DB, D>,
    {
        op_mutc_refa_refb_matmul(self.view_mut(), a, b, alpha, beta)
    }

    /// GEMM-style matrix multiplication with output scaling.
    ///
    /// See also [`matmul_from`].
    pub fn matmul_from<TA, TB, DA, DB>(
        &mut self,
        a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
        b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
        alpha: T,
        beta: T,
    ) where
        // storage
        R: DataMutAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
        // dimension
        DA: DimAPI,
        DB: DimAPI,
        // operation specific
        B: DeviceMatMulAPI<TA, TB, T, DA, DB, D>,
    {
        op_mutc_refa_refb_matmul(self.view_mut(), a, b, alpha, beta).rstsr_unwrap()
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_matmul() {
        let a = linspace((0.0, 14.0, 15)).into_shape([3, 5]);
        let b = linspace((0.0, 14.0, 15)).into_shape([5, 3]);
        let mut c: Tensor<f64> = zeros([3, 3]);

        op_mutc_refa_refb_matmul(&mut c, &a, &b, 1.0, 0.0).unwrap();
        println!("{c}");

        let d = &a % &b;
        println!("{d}");

        let a = linspace((0.0, 14.0, 15));
        let b = linspace((0.0, 14.0, 15));
        println!("{:}", &a % &b);

        #[cfg(not(feature = "col_major"))]
        {
            let a = linspace((0.0, 2.0, 3));
            let b = linspace((0.0, 29.0, 30)).into_shape([2, 3, 5]);
            println!("{:}", &a % &b);

            let a = linspace((0.0, 29.0, 30)).into_shape([2, 3, 5]);
            let b = linspace((0.0, 4.0, 5));
            println!("{:}", &a % &b);

            let a = linspace((0.0, 14.0, 15)).into_shape([5, 3]);
            let b = linspace((0.0, 29.0, 30)).into_shape([2, 3, 5]);
            println!("{:}", &a % &b);

            let a = linspace((0.0, 29.0, 30)).into_shape([2, 3, 5]);
            let b = linspace((0.0, 14.0, 15)).into_shape([5, 3]);
            println!("{:}", &a % &b);
        }
    }

    #[test]
    fn test_matmul_from() {
        #[cfg(not(feature = "col_major"))]
        {
            let a = linspace((0.0, 14.0, 15)).into_shape([3, 5]);
            let b = linspace((0.0, 19.0, 20)).into_shape([5, 4]);
            let mut c = linspace((0.0, 11.0, 12)).into_shape([3, 4]);
            c.matmul_from(&a, &b, 2.0, 1.5);
            println!("{c}");

            let c_ref = vec![240., 261.5, 283., 304.5, 646., 717.5, 789., 860.5, 1052., 1173.5, 1295., 1416.5];
            assert!(allclose_f64(&c.raw().into(), &c_ref.into()));
        }
        #[cfg(feature = "col_major")]
        {
            let a = linspace((0.0, 14.0, 15)).into_shape([3, 5]);
            let b = linspace((0.0, 19.0, 20)).into_shape([5, 4]);
            let mut c = linspace((0.0, 11.0, 12)).into_shape([3, 4]);
            c.matmul_from(&a, &b, 2.0, 1.5);
            println!("{c}");

            let c_ref = vec![180.0, 201.5, 223.0, 484.5, 556.0, 627.5, 789.0, 910.5, 1032.0, 1093.5, 1265.0, 1436.5];
            assert!(allclose_f64(&c.raw().into(), &c_ref.into()));
        }
    }
}
