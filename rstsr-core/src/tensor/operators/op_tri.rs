//! Triangular-matrix pack/unpack operations (`trf`-style, used by
//! LAPACK-backed solvers): [`pack_tri`] / [`unpack_tri`] and their
//! lower-/upper-triangle variants, available as free functions and
//! [`TensorAny`] methods.

use crate::prelude_dev::*;
use core::mem::transmute;
use num::{Float as NumFloat, ToPrimitive};

/* #region helpers */

/// Shape of the packed output for `pack_tri`, from the input layout.
fn pack_tri_output_shape(lb: &Layout<IxD>, default_order: FlagOrder) -> Result<Vec<usize>> {
    match default_order {
        RowMajor => {
            // the square block is the trailing two dimensions
            let (lb_rest, lb_inner) = lb.dim_split_at(-2)?;
            rstsr_assert_eq!(
                lb_inner.shape()[0],
                lb_inner.shape()[1],
                InvalidLayout,
                "Last two dimensions should be the same for pack_tri."
            )?;
            let n: usize = lb_inner.shape()[0];
            let n_tp = n * (n + 1) / 2;

            let mut shape = lb_rest.shape().to_vec();
            shape.push(n_tp);
            Ok(shape)
        },
        ColMajor => {
            // the square block is the leading two dimensions
            let (lb_inner, lb_rest) = lb.dim_split_at(2)?;
            rstsr_assert_eq!(
                lb_inner.shape()[0],
                lb_inner.shape()[1],
                InvalidLayout,
                "First two dimensions should be the same for pack_tri."
            )?;
            let n: usize = lb_inner.shape()[0];
            let n_tp = n * (n + 1) / 2;

            let mut shape = vec![n_tp];
            shape.append(&mut lb_rest.shape().to_vec());
            Ok(shape)
        },
    }
}

/// Side length of the square matrix described by a packed length `n_tp`.
fn unpack_tri_size(n_tp: usize, axis: &str) -> Result<usize> {
    let n: usize = NumFloat::floor(NumFloat::sqrt((2 * n_tp).to_f64().unwrap())).to_usize().unwrap();
    rstsr_assert_eq!(
        n * (n + 1) / 2,
        n_tp,
        InvalidLayout,
        "{} dimension should be triangular number for unpack_tri.",
        axis
    )?;
    Ok(n)
}

/// Shape of the unpacked output for `unpack_tri`, from the input layout.
fn unpack_tri_output_shape(lb: &Layout<IxD>, default_order: FlagOrder) -> Result<Vec<usize>> {
    match default_order {
        RowMajor => {
            // the packed vector is the last dimension
            let (lb_rest, lb_inner) = lb.dim_split_at(-1)?;
            let n = unpack_tri_size(lb_inner.shape()[0], "Last")?;

            let mut shape = lb_rest.shape().to_vec();
            shape.push(n);
            shape.push(n);
            Ok(shape)
        },
        ColMajor => {
            // the packed vector is the first dimension
            let (lb_inner, lb_rest) = lb.dim_split_at(1)?;
            let n = unpack_tri_size(lb_inner.shape()[0], "First")?;

            let mut shape = vec![n, n];
            shape.append(&mut lb_rest.shape().to_vec());
            Ok(shape)
        },
    }
}

/* #endregion */

/* #region pack_tri */

/// Packs a triangle of the trailing square matrices into vectors (fallible).
///
/// See also [`pack_tri`].
pub fn pack_tri_f<T, B, D>(
    a: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    uplo: FlagUpLo,
) -> Result<Tensor<T, B, D::SmallerOne>>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: DeviceAPI<T> + OpPackTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    let a = a.view();
    let lb = a.layout().to_dim::<IxD>()?;
    let default_order = a.device().default_order();
    let la_shape = pack_tri_output_shape(&lb, default_order)?;

    let la = match (lb.c_prefer(), lb.f_prefer()) {
        (true, false) => la_shape.c(),
        (false, true) => la_shape.f(),
        _ => match default_order {
            RowMajor => la_shape.c(),
            ColMajor => la_shape.f(),
        },
    };

    let device = a.device();
    let mut storage_a = device.uninit_impl(la.bounds_index()?.1)?;
    device.pack_tri(storage_a.raw_mut(), &la, a.raw(), &lb, uplo)?;
    // SAFETY: `pack_tri` above wrote every element of the fresh storage.
    let storage_a = unsafe { B::assume_init_impl(storage_a)? };
    Tensor::new_f(storage_a, la.into_dim()?)
}

/// Packs a triangle of the trailing square matrices into vectors.
///
/// The square block is `n x n`; the triangle selected by `uplo` is written into a
/// length-`n * (n + 1) / 2` vector, its entries visited in the device default order.
///
/// <div class="warning">
///
/// **Row/Column Major Notice**
///
/// This function behaves differently on default orders ([`RowMajor`] and [`ColMajor`]) of device.
///
/// </div>
///
/// The square block is the **trailing** two dimensions under [`RowMajor`] and the **leading**
/// two under [`ColMajor`], and the packed vector is appended as a new **last** axis under
/// [`RowMajor`] but prepended as a new **first** axis under [`ColMajor`]. For a 2-D input both
/// orders give the same length-`n(n+1)/2` vector. See
/// [`order_semantics`](crate::order_semantics).
///
/// # Parameters
///
/// - `a`: [`impl TensorViewAPI<Type = T, Backend = B, Dim = D>`][TensorViewAPI] — the input tensor
///   whose triangular block is packed.
/// - `uplo`: [`FlagUpLo`] — [`FlagUpLo::L`] packs the lower triangle, [`FlagUpLo::U`] the upper.
///
/// # Returns
///
/// - [`Tensor<T, B, D::SmallerOne>`][Tensor] — a new owned tensor: the input shape with its square
///   block replaced by the packed vector.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], &device);
/// println!("{}", rt::pack_tri(&a, FlagUpLo::L));
/// // [ 1 4 5 7 8 9]
/// # assert_eq!(format!("{}", rt::pack_tri(&a, FlagUpLo::L)), "[ 1 4 5 7 8 9]");
/// ```
///
/// # Elaborated examples
///
/// ## Difference between [`RowMajor`] and [`ColMajor`]
///
/// For an input of rank at least three the packed axis is appended (row-major) or prepended
/// (column-major):
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], &device);
/// println!("{:?}", rt::pack_tril(&a).shape());
/// // [2, 3]
///
/// device.set_default_order(ColMajor);
/// let a = rt::tensor_from_nested!([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], &device);
/// println!("{:?}", rt::pack_tril(&a).shape());
/// // [3, 2]
/// # assert_eq!(rt::tensor_from_nested!([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], &device).pack_tril().shape(), &[3, 2]);
/// ```
///
/// # Panics
///
/// - Panics if the last two dimensions (row-major) or first two dimensions (column-major) are not
///   equal.
///
/// For a fallible version, use [`pack_tri_f`].
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`unpack_tri`]: the inverse operation.
///
/// ## Variants of this function
///
/// - [`pack_tri_f`]: fallible version.
/// - [`pack_tri_with_output`]: write into a provided output tensor.
/// - [`pack_tril`] / [`pack_triu`]: the lower- and upper-triangle convenience forms.
/// - Associated methods on [`TensorAny`]: [`TensorAny::pack_tri`] / [`TensorAny::pack_tri_f`].
pub fn pack_tri<T, B, D>(
    a: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    uplo: FlagUpLo,
) -> Tensor<T, B, D::SmallerOne>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: DeviceAPI<T> + OpPackTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    pack_tri_f(a, uplo).rstsr_unwrap()
}

/// Packs the lower triangle of the trailing square matrices into vectors (fallible).
///
/// See also [`pack_tri`].
pub fn pack_tril_f<T, B, D>(
    a: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
) -> Result<Tensor<T, B, D::SmallerOne>>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: DeviceAPI<T> + OpPackTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    pack_tri_f(a, FlagUpLo::L)
}

/// Packs the lower triangle of the trailing square matrices into vectors.
///
/// See also [`pack_tri`].
pub fn pack_tril<T, B, D>(a: impl TensorViewAPI<Type = T, Backend = B, Dim = D>) -> Tensor<T, B, D::SmallerOne>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: DeviceAPI<T> + OpPackTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    pack_tril_f(a).rstsr_unwrap()
}

/// Packs the upper triangle of the trailing square matrices into vectors (fallible).
///
/// See also [`pack_tri`].
pub fn pack_triu_f<T, B, D>(
    a: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
) -> Result<Tensor<T, B, D::SmallerOne>>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: DeviceAPI<T> + OpPackTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    pack_tri_f(a, FlagUpLo::U)
}

/// Packs the upper triangle of the trailing square matrices into vectors.
///
/// See also [`pack_tri`].
pub fn pack_triu<T, B, D>(a: impl TensorViewAPI<Type = T, Backend = B, Dim = D>) -> Tensor<T, B, D::SmallerOne>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: DeviceAPI<T> + OpPackTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    pack_triu_f(a).rstsr_unwrap()
}

/// Packs a triangle of the trailing square matrices into a provided output tensor (fallible).
///
/// See also [`pack_tri_with_output`].
pub fn pack_tri_with_output_f<T, B, D, DC>(
    a: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    uplo: FlagUpLo,
    output: impl TensorViewMutAPI<Type = T, Backend = B, Dim = DC>,
) -> Result<()>
where
    D: DimAPI,
    DC: DimAPI,
    B: DeviceAPI<T> + OpPackTriAPI<T>,
{
    let a = a.view();
    let mut output = output;
    let mut output = output.view_mut();
    rstsr_assert!(output.device().same_device(a.device()), DeviceMismatch)?;
    // writing through a broadcast layout would alias elements
    rstsr_assert!(!output.layout().is_broadcasted(), InvalidLayout, "cannot write into broadcasted tensor")?;

    let lb = a.layout().to_dim::<IxD>()?;
    let lc = output.layout().to_dim::<IxD>()?;
    let expected = pack_tri_output_shape(&lb, a.device().default_order())?;
    rstsr_assert_eq!(
        lc.shape().as_slice(),
        expected.as_slice(),
        InvalidLayout,
        "Output shape should match the packed shape for pack_tri."
    )?;

    let device = output.device().clone();
    // SAFETY: `Raw<T>` and `Raw<MaybeUninit<T>>` have the same layout; `pack_tri`
    // writes every output element, so the buffer stays initialized.
    let output_raw = unsafe {
        transmute::<&mut <B as DeviceRawAPI<T>>::Raw, &mut <B as DeviceRawAPI<MaybeUninit<T>>>::Raw>(output.raw_mut())
    };
    device.pack_tri(output_raw, &lc, a.raw(), &lb, uplo)
}

/// Packs a triangle of the trailing square matrices into a provided output tensor.
///
/// This is [`pack_tri`] with the result written into `output` instead of a new tensor; the
/// previous contents of `output` are overwritten. The output shape must be exactly the shape
/// [`pack_tri`] would return.
///
/// <div class="warning">
///
/// **Row/Column Major Notice**
///
/// This function behaves differently on default orders ([`RowMajor`] and [`ColMajor`]) of device.
///
/// </div>
///
/// The square block is the **trailing** two dimensions under [`RowMajor`] and the **leading**
/// two under [`ColMajor`], and the packed vector is the new **last** axis under [`RowMajor`] but
/// the new **first** axis under [`ColMajor`]; see [`pack_tri`] and
/// [`order_semantics`](crate::order_semantics).
///
/// # Parameters
///
/// - `a`: [`impl TensorViewAPI<Type = T, Backend = B, Dim = D>`][TensorViewAPI] — the input tensor
///   whose triangular block is packed.
/// - `uplo`: [`FlagUpLo`] — [`FlagUpLo::L`] packs the lower triangle, [`FlagUpLo::U`] the upper.
/// - `output`: [`impl TensorViewMutAPI`][TensorViewMutAPI] — the destination tensor; every element
///   is overwritten.
///
/// # Returns
///
/// - `()`: the result is written into `output`.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], &device);
/// let mut packed: Tensor<f64, _> = rt::zeros(([6], &device));
/// rt::pack_tri_with_output(&a, FlagUpLo::L, &mut packed);
/// println!("{packed}");
/// // [ 1 4 5 7 8 9]
/// # assert_eq!(format!("{packed}"), "[ 1 4 5 7 8 9]");
/// ```
///
/// # Panics
///
/// - Panics if the input's triangular block is not square, if `output` does not have exactly the
///   packed shape, if `output` is broadcasted, or if the devices differ.
///
/// For a fallible version, use [`pack_tri_with_output_f`].
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`pack_tri`]: allocate the output internally.
/// - [`unpack_tri_with_output`]: the inverse operation.
///
/// ## Variants of this function
///
/// - [`pack_tri_with_output_f`]: fallible version.
/// - Associated methods on [`TensorAny`]: [`TensorAny::pack_tri_with_output`] /
///   [`TensorAny::pack_tri_with_output_f`].
pub fn pack_tri_with_output<T, B, D, DC>(
    a: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    uplo: FlagUpLo,
    output: impl TensorViewMutAPI<Type = T, Backend = B, Dim = DC>,
) where
    D: DimAPI,
    DC: DimAPI,
    B: DeviceAPI<T> + OpPackTriAPI<T>,
{
    pack_tri_with_output_f(a, uplo, output).rstsr_unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: DeviceAPI<T> + OpPackTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    /// Packs a triangle of the trailing square matrices into vectors (fallible).
    ///
    /// See also [`pack_tri`].
    pub fn pack_tri_f(&self, uplo: FlagUpLo) -> Result<Tensor<T, B, D::SmallerOne>> {
        pack_tri_f(self, uplo)
    }

    /// Packs a triangle of the trailing square matrices into vectors.
    ///
    /// See also [`pack_tri`].
    pub fn pack_tri(&self, uplo: FlagUpLo) -> Tensor<T, B, D::SmallerOne> {
        pack_tri(self, uplo)
    }

    /// Packs the lower triangle of the trailing square matrices into vectors (fallible).
    ///
    /// See also [`pack_tri`].
    pub fn pack_tril_f(&self) -> Result<Tensor<T, B, D::SmallerOne>> {
        pack_tril_f(self)
    }

    /// Packs the lower triangle of the trailing square matrices into vectors.
    ///
    /// See also [`pack_tri`].
    pub fn pack_tril(&self) -> Tensor<T, B, D::SmallerOne> {
        pack_tril(self)
    }

    /// Packs the upper triangle of the trailing square matrices into vectors (fallible).
    ///
    /// See also [`pack_tri`].
    pub fn pack_triu_f(&self) -> Result<Tensor<T, B, D::SmallerOne>> {
        pack_triu_f(self)
    }

    /// Packs the upper triangle of the trailing square matrices into vectors.
    ///
    /// See also [`pack_tri`].
    pub fn pack_triu(&self) -> Tensor<T, B, D::SmallerOne> {
        pack_triu(self)
    }
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + OpPackTriAPI<T>,
{
    /// Packs a triangle of the trailing square matrices into a provided output tensor (fallible).
    ///
    /// See also [`pack_tri_with_output`].
    pub fn pack_tri_with_output_f<DC>(
        &self,
        uplo: FlagUpLo,
        output: impl TensorViewMutAPI<Type = T, Backend = B, Dim = DC>,
    ) -> Result<()>
    where
        DC: DimAPI,
    {
        pack_tri_with_output_f(self, uplo, output)
    }

    /// Packs a triangle of the trailing square matrices into a provided output tensor.
    ///
    /// See also [`pack_tri_with_output`].
    pub fn pack_tri_with_output<DC>(
        &self,
        uplo: FlagUpLo,
        output: impl TensorViewMutAPI<Type = T, Backend = B, Dim = DC>,
    ) where
        DC: DimAPI,
    {
        pack_tri_with_output(self, uplo, output)
    }
}

/* #endregion */

/* #region unpack_tri */

/// Unpacks packed triangular vectors back into square matrices (fallible).
///
/// See also [`unpack_tri`].
pub fn unpack_tri_f<T, B, D>(
    a: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    uplo: FlagUpLo,
    symm: FlagSymm,
) -> Result<Tensor<T, B, D::LargerOne>>
where
    D: DimAPI + DimLargerOneAPI,
    D::LargerOne: DimAPI,
    B: DeviceAPI<T> + OpUnpackTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    let a = a.view();
    let lb = a.layout().to_dim::<IxD>()?;
    let default_order = a.device().default_order();
    let la_shape = unpack_tri_output_shape(&lb, default_order)?;

    let la = match (lb.c_prefer(), lb.f_prefer()) {
        (true, false) => la_shape.c(),
        (false, true) => la_shape.f(),
        _ => match default_order {
            RowMajor => la_shape.c(),
            ColMajor => la_shape.f(),
        },
    };

    let device = a.device();
    let mut storage_a = device.uninit_impl(la.bounds_index()?.1)?;
    device.unpack_tri(storage_a.raw_mut(), &la, a.raw(), &lb, uplo, symm)?;
    // SAFETY: `unpack_tri` above wrote every element of the fresh storage.
    let storage_a = unsafe { B::assume_init_impl(storage_a)? };
    Tensor::new_f(storage_a, la.into_dim()?)
}

/// Unpacks packed triangular vectors back into square matrices.
///
/// The input carries vectors of length `n * (n + 1) / 2`; each is expanded back into an
/// `n x n` matrix by writing the vector into the triangle selected by `uplo` and filling the
/// other triangle according to `symm`.
///
/// <div class="warning">
///
/// **Row/Column Major Notice**
///
/// This function behaves differently on default orders ([`RowMajor`] and [`ColMajor`]) of device.
///
/// </div>
///
/// The packed vector is the **last** dimension under [`RowMajor`] and the **first** under
/// [`ColMajor`], and the restored `n x n` block is inserted at the back (row-major) or at the
/// front (column-major). For a 1-D input both orders give the same `n x n` matrix. See
/// [`order_semantics`](crate::order_semantics).
///
/// # Parameters
///
/// - `a`: [`impl TensorViewAPI<Type = T, Backend = B, Dim = D>`][TensorViewAPI] — the input tensor
///   holding the packed vector(s).
/// - `uplo`: [`FlagUpLo`] — the triangle the vector is written into ([`FlagUpLo::L`] or
///   [`FlagUpLo::U`]).
/// - `symm`: [`FlagSymm`] — how the opposite triangle is derived from the packed one: `Sy` mirrors
///   it (symmetric), `He` mirrors its complex conjugate (Hermitian), `Ay` negates it and zeroes the
///   diagonal (anti-symmetric), `Ah` negates the conjugate and zeroes the diagonal
///   (anti-Hermitian); `N` zero-fills the opposite triangle (no symmetry implied by the packed
///   data).
///
/// # Returns
///
/// - [`Tensor<T, B, D::LargerOne>`][Tensor] — a new owned tensor: the input shape with its packed
///   vector replaced by the `n x n` matrix.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let packed = rt::tensor_from_nested!([1.0, 4.0, 5.0, 7.0, 8.0, 9.0], &device);
/// println!("{}", rt::unpack_tri(&packed, FlagUpLo::L, FlagSymm::Sy));
/// // [[ 1 4 7]
/// //  [ 4 5 8]
/// //  [ 7 8 9]]
/// # assert_eq!(format!("{}", rt::unpack_tri(&packed, FlagUpLo::L, FlagSymm::Sy)), "[[ 1 4 7]\n [ 4 5 8]\n [ 7 8 9]]");
/// ```
///
/// With `N`, no symmetry is assumed: the packed triangle is written as-is and the opposite
/// triangle is zero-filled.
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let packed = rt::tensor_from_nested!([1.0, 4.0, 5.0, 7.0, 8.0, 9.0], &device);
/// println!("{}", rt::unpack_tri(&packed, FlagUpLo::L, FlagSymm::N));
/// // [[ 1 0 0]
/// //  [ 4 5 0]
/// //  [ 7 8 9]]
/// # assert_eq!(format!("{}", rt::unpack_tri(&packed, FlagUpLo::L, FlagSymm::N)), "[[ 1 0 0]\n [ 4 5 0]\n [ 7 8 9]]");
/// ```
///
/// # Panics
///
/// - Panics if the last dimension (row-major) or first dimension (column-major) is not a triangular
///   number of the form `n * (n + 1) / 2`.
///
/// For a fallible version, use [`unpack_tri_f`].
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`pack_tri`]: the inverse operation.
///
/// ## Variants of this function
///
/// - [`unpack_tri_f`]: fallible version.
/// - [`unpack_tri_with_output`]: write into a provided output tensor.
/// - [`unpack_tril`] / [`unpack_triu`]: the lower- and upper-triangle convenience forms.
/// - Associated methods on [`TensorAny`]: [`TensorAny::unpack_tri`] / [`TensorAny::unpack_tri_f`].
pub fn unpack_tri<T, B, D>(
    a: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    uplo: FlagUpLo,
    symm: FlagSymm,
) -> Tensor<T, B, D::LargerOne>
where
    D: DimAPI + DimLargerOneAPI,
    D::LargerOne: DimAPI,
    B: DeviceAPI<T> + OpUnpackTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    unpack_tri_f(a, uplo, symm).rstsr_unwrap()
}

/// Unpacks packed lower-triangular vectors back into square matrices (fallible).
///
/// See also [`unpack_tri`].
pub fn unpack_tril_f<T, B, D>(
    a: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    symm: FlagSymm,
) -> Result<Tensor<T, B, D::LargerOne>>
where
    D: DimAPI + DimLargerOneAPI,
    D::LargerOne: DimAPI,
    B: DeviceAPI<T> + OpUnpackTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    unpack_tri_f(a, FlagUpLo::L, symm)
}

/// Unpacks packed lower-triangular vectors back into square matrices.
///
/// See also [`unpack_tri`].
pub fn unpack_tril<T, B, D>(
    a: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    symm: FlagSymm,
) -> Tensor<T, B, D::LargerOne>
where
    D: DimAPI + DimLargerOneAPI,
    D::LargerOne: DimAPI,
    B: DeviceAPI<T> + OpUnpackTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    unpack_tril_f(a, symm).rstsr_unwrap()
}

/// Unpacks packed upper-triangular vectors back into square matrices (fallible).
///
/// See also [`unpack_tri`].
pub fn unpack_triu_f<T, B, D>(
    a: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    symm: FlagSymm,
) -> Result<Tensor<T, B, D::LargerOne>>
where
    D: DimAPI + DimLargerOneAPI,
    D::LargerOne: DimAPI,
    B: DeviceAPI<T> + OpUnpackTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    unpack_tri_f(a, FlagUpLo::U, symm)
}

/// Unpacks packed upper-triangular vectors back into square matrices.
///
/// See also [`unpack_tri`].
pub fn unpack_triu<T, B, D>(
    a: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    symm: FlagSymm,
) -> Tensor<T, B, D::LargerOne>
where
    D: DimAPI + DimLargerOneAPI,
    D::LargerOne: DimAPI,
    B: DeviceAPI<T> + OpUnpackTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    unpack_triu_f(a, symm).rstsr_unwrap()
}

/// Unpacks packed triangular vectors into a provided output tensor (fallible).
///
/// See also [`unpack_tri_with_output`].
pub fn unpack_tri_with_output_f<T, B, D, DC>(
    a: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    uplo: FlagUpLo,
    symm: FlagSymm,
    output: impl TensorViewMutAPI<Type = T, Backend = B, Dim = DC>,
) -> Result<()>
where
    D: DimAPI,
    DC: DimAPI,
    B: DeviceAPI<T> + OpUnpackTriAPI<T>,
{
    let a = a.view();
    let mut output = output;
    let mut output = output.view_mut();
    rstsr_assert!(output.device().same_device(a.device()), DeviceMismatch)?;
    // writing through a broadcast layout would alias elements
    rstsr_assert!(!output.layout().is_broadcasted(), InvalidLayout, "cannot write into broadcasted tensor")?;

    let lb = a.layout().to_dim::<IxD>()?;
    let la = output.layout().to_dim::<IxD>()?;
    let expected = unpack_tri_output_shape(&lb, a.device().default_order())?;
    rstsr_assert_eq!(
        la.shape().as_slice(),
        expected.as_slice(),
        InvalidLayout,
        "Output shape should match the unpacked shape for unpack_tri."
    )?;

    let device = output.device().clone();
    // SAFETY: `Raw<T>` and `Raw<MaybeUninit<T>>` have the same layout; `unpack_tri`
    // writes every output element, so the buffer stays initialized.
    let output_raw = unsafe {
        transmute::<&mut <B as DeviceRawAPI<T>>::Raw, &mut <B as DeviceRawAPI<MaybeUninit<T>>>::Raw>(output.raw_mut())
    };
    device.unpack_tri(output_raw, &la, a.raw(), &lb, uplo, symm)
}

/// Unpacks packed triangular vectors into a provided output tensor.
///
/// This is [`unpack_tri`] with the result written into `output` instead of a new tensor; the
/// previous contents of `output` are overwritten. The output shape must be exactly the shape
/// [`unpack_tri`] would return.
///
/// <div class="warning">
///
/// **Row/Column Major Notice**
///
/// This function behaves differently on default orders ([`RowMajor`] and [`ColMajor`]) of device.
///
/// </div>
///
/// The packed vector is the **last** dimension under [`RowMajor`] and the **first** under
/// [`ColMajor`], and the restored `n x n` block is inserted at the back (row-major) or at the
/// front (column-major); see [`unpack_tri`] and
/// [`order_semantics`](crate::order_semantics).
///
/// # Parameters
///
/// - `a`: [`impl TensorViewAPI<Type = T, Backend = B, Dim = D>`][TensorViewAPI] — the input tensor
///   holding the packed vector(s).
/// - `uplo`: [`FlagUpLo`] — the triangle the vector is written into.
/// - `symm`: [`FlagSymm`] — how the opposite triangle is derived from the packed one; see
///   [`unpack_tri`].
/// - `output`: [`impl TensorViewMutAPI`][TensorViewMutAPI] — the destination tensor; every element
///   is overwritten.
///
/// # Returns
///
/// - `()`: the result is written into `output`.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let packed = rt::tensor_from_nested!([1.0, 4.0, 5.0, 7.0, 8.0, 9.0], &device);
/// let mut a: Tensor<f64, _> = rt::zeros(([3, 3], &device));
/// rt::unpack_tri_with_output(&packed, FlagUpLo::L, FlagSymm::Sy, &mut a);
/// println!("{a}");
/// // [[ 1 4 7]
/// //  [ 4 5 8]
/// //  [ 7 8 9]]
/// # assert_eq!(format!("{a}"), "[[ 1 4 7]\n [ 4 5 8]\n [ 7 8 9]]");
/// ```
///
/// # Panics
///
/// - Panics if the input does not hold a triangular number of packed entries, if `output` does not
///   have exactly the unpacked shape, if `output` is broadcasted, or if the devices differ.
///
/// For a fallible version, use [`unpack_tri_with_output_f`].
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`unpack_tri`]: allocate the output internally.
/// - [`pack_tri_with_output`]: the inverse operation.
///
/// ## Variants of this function
///
/// - [`unpack_tri_with_output_f`]: fallible version.
/// - Associated methods on [`TensorAny`]: [`TensorAny::unpack_tri_with_output`] /
///   [`TensorAny::unpack_tri_with_output_f`].
pub fn unpack_tri_with_output<T, B, D, DC>(
    a: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    uplo: FlagUpLo,
    symm: FlagSymm,
    output: impl TensorViewMutAPI<Type = T, Backend = B, Dim = DC>,
) where
    D: DimAPI,
    DC: DimAPI,
    B: DeviceAPI<T> + OpUnpackTriAPI<T>,
{
    unpack_tri_with_output_f(a, uplo, symm, output).rstsr_unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI + DimLargerOneAPI,
    D::LargerOne: DimAPI,
    B: DeviceAPI<T> + OpUnpackTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    /// Unpacks packed triangular vectors back into square matrices (fallible).
    ///
    /// See also [`unpack_tri`].
    pub fn unpack_tri_f(&self, uplo: FlagUpLo, symm: FlagSymm) -> Result<Tensor<T, B, D::LargerOne>> {
        unpack_tri_f(self, uplo, symm)
    }

    /// Unpacks packed triangular vectors back into square matrices.
    ///
    /// See also [`unpack_tri`].
    pub fn unpack_tri(&self, uplo: FlagUpLo, symm: FlagSymm) -> Tensor<T, B, D::LargerOne> {
        unpack_tri(self, uplo, symm)
    }

    /// Unpacks packed lower-triangular vectors back into square matrices (fallible).
    ///
    /// See also [`unpack_tri`].
    pub fn unpack_tril_f(&self, symm: FlagSymm) -> Result<Tensor<T, B, D::LargerOne>> {
        unpack_tril_f(self, symm)
    }

    /// Unpacks packed lower-triangular vectors back into square matrices.
    ///
    /// See also [`unpack_tri`].
    pub fn unpack_tril(&self, symm: FlagSymm) -> Tensor<T, B, D::LargerOne> {
        unpack_tril(self, symm)
    }

    /// Unpacks packed upper-triangular vectors back into square matrices (fallible).
    ///
    /// See also [`unpack_tri`].
    pub fn unpack_triu_f(&self, symm: FlagSymm) -> Result<Tensor<T, B, D::LargerOne>> {
        unpack_triu_f(self, symm)
    }

    /// Unpacks packed upper-triangular vectors back into square matrices.
    ///
    /// See also [`unpack_tri`].
    pub fn unpack_triu(&self, symm: FlagSymm) -> Tensor<T, B, D::LargerOne> {
        unpack_triu(self, symm)
    }
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + OpUnpackTriAPI<T>,
{
    /// Unpacks packed triangular vectors into a provided output tensor (fallible).
    ///
    /// See also [`unpack_tri_with_output`].
    pub fn unpack_tri_with_output_f<DC>(
        &self,
        uplo: FlagUpLo,
        symm: FlagSymm,
        output: impl TensorViewMutAPI<Type = T, Backend = B, Dim = DC>,
    ) -> Result<()>
    where
        DC: DimAPI,
    {
        unpack_tri_with_output_f(self, uplo, symm, output)
    }

    /// Unpacks packed triangular vectors into a provided output tensor.
    ///
    /// See also [`unpack_tri_with_output`].
    pub fn unpack_tri_with_output<DC>(
        &self,
        uplo: FlagUpLo,
        symm: FlagSymm,
        output: impl TensorViewMutAPI<Type = T, Backend = B, Dim = DC>,
    ) where
        DC: DimAPI,
    {
        unpack_tri_with_output(self, uplo, symm, output)
    }
}

/* #endregion */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_tri() {
        #[cfg(not(feature = "col_major"))]
        {
            let a = {
                let a = arange((48., &DeviceCpuSerial::default()));
                let storage_a = a.into_raw_parts().0;
                Tensor::new(storage_a, [3, 4, 4].f())
            };
            let a_triu = a.pack_tril();
            println!("{a_triu:?}");
            println!("{:?}", a.slice(0));
            println!("{:?}", a_triu.slice(0).to_vec());
            assert_eq!(a_triu.slice(1).to_vec(), [1., 4., 16., 7., 19., 31., 10., 22., 34., 46.]);

            let b = a_triu.unpack_tril(FlagSymm::Sy);
            println!("{b:?}");
            assert_eq!(b.slice((0, 1)).to_vec(), [3., 15., 18., 21.]);
        }
        #[cfg(feature = "col_major")]
        {
            let a = {
                let a = arange((48., &DeviceCpuSerial::default()));
                let storage_a = a.into_raw_parts().0;
                Tensor::new(storage_a, [4, 4, 3].c())
            };
            let a_triu = a.pack_triu();
            println!("{a_triu:?}");
            println!("{:?}", a.slice((.., 0)));
            println!("{:?}", a_triu.slice((.., 0)).to_vec());
            assert_eq!(a_triu.slice((.., 1)).to_vec(), [1., 4., 16., 7., 19., 31., 10., 22., 34., 46.]);

            let b = a_triu.unpack_triu(FlagSymm::Sy);
            println!("{b:?}");
            assert_eq!(b.slice((.., 1, 0)).to_vec(), [3., 15., 18., 21.]);
        }
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_par_pack_tril_compiles() {
        #[cfg(not(feature = "col_major"))]
        {
            use num::complex::c64;
            let a = linspace((c64(-2.0, 1.5), c64(1.7, -2.3), 256 * 256 * 256)).into_layout([4, 64, 256, 256].f());
            let a_tril = a.pack_tril();
            println!("{a_tril:20.5}");
            let b = a_tril.unpack_tril(FlagSymm::Ah);
            println!("{b:20.5}");
        }
        #[cfg(feature = "col_major")]
        {
            use num::complex::c64;
            let a = linspace((c64(-2.0, 1.5), c64(1.7, -2.3), 256 * 256 * 256)).into_layout([256, 256, 64, 4].c());
            let a_tril = a.pack_tril();
            println!("{a_tril:20.5}");
            let b = a_tril.unpack_tril(FlagSymm::Ah);
            println!("{b:20.5}");
        }
    }

    #[test]
    fn test_correctness() {
        let a = {
            let a = arange((16., &DeviceCpuSerial::default()));
            let storage_a = a.into_raw_parts().0;
            Tensor::new(storage_a, [4, 4].c())
        };
        println!("{a:}");

        let a_tril1 = a.pack_tril();
        println!("{a_tril1:}");

        let a = a.to_contig(FlagOrder::F);
        let a_tril2 = a.pack_tril();
        println!("{a_tril2:}");

        assert!((&a_tril1 - &a_tril2).l2_norm_all() < 1e-6);
    }
}
