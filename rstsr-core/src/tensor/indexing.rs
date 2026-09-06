//! Basic indexing of tensors: slicing ([`slice`](slice()),
//! [`slice_mut`](slice_mut()), operator `[]` for scalar access) and diagonal
//! extraction ([`diagonal`]).
//!
//! For advanced indexing (selecting by boolean masks or integer indices), see
//! [`adv_indexing`](crate::tensor::adv_indexing).

use core::ops::{Index, IndexMut};

use crate::prelude_dev::*;

/* #region slice */

/// Basic slicing of a tensor.
///
/// See also [`into_slice`].
pub fn into_slice_f<S, D, I>(tensor: TensorBase<S, D>, index: I) -> Result<TensorBase<S, IxD>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<Indexer>, Error: Into<Error>>,
{
    let (data, layout) = tensor.into_raw_parts();
    let index = index.try_into().map_err(Into::into)?;
    let layout = layout.dim_slice(index.as_ref())?;
    return unsafe { Ok(TensorBase::new_unchecked(data, layout)) };
}

/// Basic slicing of a tensor.
///
/// See also [`slice`](slice()).
pub fn into_slice<S, D, I>(tensor: TensorBase<S, D>, index: I) -> TensorBase<S, IxD>
where
    D: DimAPI,
    I: TryInto<AxesIndex<Indexer>, Error: Into<Error>>,
{
    into_slice_f(tensor, index).rstsr_unwrap()
}

/// Basic slicing of a tensor, returning a view.
///
/// See also [`slice`](slice()).
pub fn slice_f<R, T, B, D, I>(tensor: &TensorAny<R, T, B, D>, index: I) -> Result<TensorView<'_, T, B, IxD>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<Indexer>, Error: Into<Error>>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_slice_f(tensor.view(), index)
}

/// Basic slicing of a tensor, returning a view.
///
/// Each dimension can be indexed by one of the following indexers, typically
/// written with the [`s!`] macro:
///
/// - a range (`1..4`), optionally with step (via the [`slice!`] macro: `slice!(2, 7, 2)`) - keeps
///   the axis, narrowing it;
/// - an integer (`2`) - selects one position and drops the axis (like NumPy's `x[2]`); negative
///   values count from the back;
/// - `None` - inserts a new axis of size one (like NumPy's `x[:, None]`);
/// - [`Indexer::Ellipsis`] - expands to as many full ranges as needed.
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny): the input tensor.
/// - `index`: the per-axis indexers, anything that converts into [`AxesIndex<Indexer>`][AxesIndex].
///   - Overloads:
///     - the result of the [`s!`] macro: a list of indexers, one per axis (shorter lists index the
///       leading axes);
///     - a single integer: select the first axis and drop it;
///     - a single range: slice the first axis.
///
/// # Returns
///
/// - [`TensorView<'_, T, B, IxD>`][`TensorView`]: a view of the sliced tensor (no data is copied).
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((10, &device));
/// println!("{}", a.slice(s![1..4]));
/// // [ 1 2 3]
/// println!("{}", a.slice(slice!(2, 7, 2)));
/// // [ 2 4 6]
/// println!("{}", a.slice(s![-3..]));
/// // [ 7 8 9]
/// # assert_eq!(format!("{}", a.slice(s![-3..])), "[ 7 8 9]");
/// ```
///
/// For n-dimensional tensors, indexers apply per axis; an integer drops the
/// axis, `None` inserts one:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let b = rt::arange((24, &device)).into_shape([2, 3, 4]);
/// println!("{}", b.slice(s![1..2, 1..3, 1..4]));
/// // [[[ 17 18 19]
/// //   [ 21 22 23]]]
/// println!("{}", b.slice(s![1]));
/// // [[ 12 13 14 15]
/// //  [ 16 17 18 19]
/// //  [ 20 21 22 23]]
/// ```
///
/// Mutable slicing writes through to the original tensor:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let mut a: Tensor<i32, _> = rt::arange((5, &device));
/// let mut v = a.slice_mut(s![1..4]);
/// v += 10;
/// println!("{a}");
/// // [ 0 11 12 13 4]
/// # assert_eq!(format!("{a}"), "[ 0 11 12 13 4]");
/// ```
///
/// # Notes of API accordance
///
/// - NumPy: basic indexing / slicing, `x[1:4]`, `x[2]`, `x[:, None]` ([`numpy.indexing`](https://numpy.org/doc/stable/user/basics.indexing.html#basic-slicing-and-indexing))
/// - RSTSR: `rt::slice(&tensor, index)` or `tensor.slice(index)`; indexers are built by the [`s!`]
///   macro instead of Python's bracket syntax.
///
/// # Panics
///
/// - Panics if any integer indexer is out of bound; range ends are clamped to the axis bounds (as
///   in Python), which yields an empty slice rather than an error.
///
/// For a fallible version, use [`slice_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - NumPy: [basic indexing](https://numpy.org/doc/stable/user/basics.indexing.html#basic-slicing-and-indexing)
/// - rust `ndarray`: [`ArrayBase::slice`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.slice)
///
/// ## Related functions in RSTSR
///
/// - [`diagonal`]: diagonal views.
/// - [`index_select`] / [`take`] / [`bool_select`]: advanced indexing (copies).
/// - Operator `[]` ([`Index`]) / `[]=` ([`IndexMut`]): scalar element access, e.g. `a[[1, 2]]`.
///
/// ## Variants of this function
///
/// - [`slice_f`]: fallible version.
/// - [`slice_mut`] / [`slice_mut_f`]: mutable views.
/// - [`into_slice`] / [`into_slice_f`]: ownership-consuming forms.
/// - [`TensorAny::i`] / [`TensorAny::i_mut`]: aliases of [`slice`](slice()) /
///   [`slice_mut`](slice_mut()).
/// - Associated methods on [`TensorAny`]: [`TensorAny::slice`] / [`TensorAny::slice_f`] /
///   [`TensorAny::slice_mut`] / [`TensorAny::slice_mut_f`] / [`TensorAny::into_slice`] /
///   [`TensorAny::into_slice_f`] / [`TensorAny::i`] / [`TensorAny::i_mut`].
pub fn slice<R, T, B, D, I>(tensor: &TensorAny<R, T, B, D>, index: I) -> TensorView<'_, T, B, IxD>
where
    D: DimAPI,
    I: TryInto<AxesIndex<Indexer>, Error: Into<Error>>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    slice_f(tensor, index).rstsr_unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    pub fn into_slice_f<I>(self, index: I) -> Result<TensorAny<R, T, B, IxD>>
    where
        I: TryInto<AxesIndex<Indexer>, Error: Into<Error>>,
    {
        into_slice_f(self, index)
    }

    pub fn into_slice<I>(self, index: I) -> TensorAny<R, T, B, IxD>
    where
        I: TryInto<AxesIndex<Indexer>, Error: Into<Error>>,
    {
        into_slice(self, index)
    }

    pub fn slice_f<I>(&self, index: I) -> Result<TensorView<'_, T, B, IxD>>
    where
        I: TryInto<AxesIndex<Indexer>, Error: Into<Error>>,
    {
        slice_f(self, index)
    }

    pub fn slice<I>(&self, index: I) -> TensorView<'_, T, B, IxD>
    where
        I: TryInto<AxesIndex<Indexer>, Error: Into<Error>>,
    {
        slice(self, index)
    }

    /// Basic slicing (alias of [`TensorAny::slice`]).
    ///
    /// See also [`slice`](slice()).
    pub fn i_f<I>(&self, index: I) -> Result<TensorView<'_, T, B, IxD>>
    where
        I: TryInto<AxesIndex<Indexer>, Error: Into<Error>>,
    {
        slice_f(self, index)
    }

    /// Basic slicing (alias of [`TensorAny::slice`]).
    ///
    /// See also [`slice`](slice()).
    pub fn i<I>(&self, index: I) -> TensorView<'_, T, B, IxD>
    where
        I: TryInto<AxesIndex<Indexer>, Error: Into<Error>>,
    {
        slice(self, index)
    }
}

/* #endregion */

/* #region slice mut */

/// Mutable basic slicing of a tensor, returning a mutable view.
///
/// See also [`slice`](slice()).
pub fn slice_mut_f<R, T, B, D, I>(tensor: &mut TensorAny<R, T, B, D>, index: I) -> Result<TensorMut<'_, T, B, IxD>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<Indexer>, Error: Into<Error>>,
    R: DataMutAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_slice_f(tensor.view_mut(), index)
}

/// Mutable basic slicing of a tensor, returning a mutable view.
///
/// See also [`slice`](slice()).
pub fn slice_mut<R, T, B, D, I>(tensor: &mut TensorAny<R, T, B, D>, index: I) -> TensorMut<'_, T, B, IxD>
where
    D: DimAPI,
    I: TryInto<AxesIndex<Indexer>, Error: Into<Error>>,
    R: DataMutAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    slice_mut_f(tensor, index).rstsr_unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataMutAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    pub fn slice_mut_f<I>(&mut self, index: I) -> Result<TensorMut<'_, T, B, IxD>>
    where
        I: TryInto<AxesIndex<Indexer>, Error: Into<Error>>,
    {
        slice_mut_f(self, index)
    }

    pub fn slice_mut<I>(&mut self, index: I) -> TensorMut<'_, T, B, IxD>
    where
        I: TryInto<AxesIndex<Indexer>, Error: Into<Error>>,
    {
        slice_mut(self, index)
    }

    /// Mutable basic slicing (alias of [`TensorAny::slice_mut`]).
    ///
    /// See also [`slice_mut`](slice_mut()).
    pub fn i_mut_f<I>(&mut self, index: I) -> Result<TensorMut<'_, T, B, IxD>>
    where
        I: TryInto<AxesIndex<Indexer>, Error: Into<Error>>,
    {
        slice_mut_f(self, index)
    }

    /// Mutable basic slicing (alias of [`TensorAny::slice_mut`]).
    ///
    /// See also [`slice_mut`](slice_mut()).
    pub fn i_mut<I>(&mut self, index: I) -> TensorMut<'_, T, B, IxD>
    where
        I: TryInto<AxesIndex<Indexer>, Error: Into<Error>>,
    {
        slice_mut(self, index)
    }
}

/* #endregion */

/* #region diagonal */

/// Arguments of the diagonal operation ([`diagonal`], [`diagonal_mut`], and
/// their variants).
///
/// - `offset`: diagonal index: `0` the main diagonal, positive above it, negative below it.
///   Defaults to `0` if absent.
/// - `axis1` / `axis2`: the two axes defining the diagonals. Default to the first two axes (`0`,
///   `1`) if absent.
///
/// Constructed by conversion from `()`, a single integer (offset only), or a
/// tuple `(offset, axis1, axis2)` of integers (any mix of `isize`/`usize`,
/// `i32`, or `i64`).
pub struct DiagonalArgs {
    pub offset: Option<isize>,
    pub axis1: Option<isize>,
    pub axis2: Option<isize>,
}

#[duplicate_item(
    S0      S1      S2;
   [isize] [isize] [isize];
   [usize] [isize] [isize];
   [usize] [usize] [usize];
   [i32  ] [i32  ] [i32  ];
   [i64  ] [i64  ] [i64  ];
)]
#[allow(clippy::unnecessary_cast)]
impl From<(S0, S1, S2)> for DiagonalArgs {
    fn from(args: (S0, S1, S2)) -> Self {
        let (offset, axis1, axis2) = args;
        Self { offset: Some(offset as isize), axis1: Some(axis1 as isize), axis2: Some(axis2 as isize) }
    }
}

#[duplicate_item(S; [isize]; [usize]; [i32]; [i64];)]
#[allow(clippy::unnecessary_cast)]
impl From<S> for DiagonalArgs {
    fn from(offset: S) -> Self {
        Self { offset: Some(offset as isize), axis1: None, axis2: None }
    }
}

impl From<()> for DiagonalArgs {
    fn from(_: ()) -> Self {
        Self { offset: None, axis1: None, axis2: None }
    }
}

impl From<Option<isize>> for DiagonalArgs {
    fn from(offset: Option<isize>) -> Self {
        Self { offset, axis1: None, axis2: None }
    }
}

/// Return specified diagonal views of a tensor.
///
/// See also [`diagonal`].
pub fn into_diagonal_f<S, D>(
    tensor: TensorBase<S, D>,
    diagonal_args: impl Into<DiagonalArgs>,
) -> Result<TensorBase<S, D::SmallerOne>>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    let (data, layout) = tensor.into_raw_parts();
    let DiagonalArgs { offset, axis1, axis2 } = diagonal_args.into();
    let layout = layout.diagonal(offset, axis1, axis2)?;
    return unsafe { Ok(TensorBase::new_unchecked(data, layout)) };
}

/// Return specified diagonal views of a tensor.
///
/// See also [`diagonal`].
pub fn into_diagonal<S, D>(
    tensor: TensorBase<S, D>,
    diagonal_args: impl Into<DiagonalArgs>,
) -> TensorBase<S, D::SmallerOne>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    into_diagonal_f(tensor, diagonal_args).rstsr_unwrap()
}

/// Return specified diagonal views of a tensor.
///
/// See also [`diagonal`].
pub fn diagonal_f<R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    diagonal_args: impl Into<DiagonalArgs>,
) -> Result<TensorView<'_, T, B, D::SmallerOne>>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_diagonal_f(tensor.view(), diagonal_args)
}

/// Return specified diagonal views of a tensor.
///
/// The two axes `axis1` and `axis2` of the input are collapsed into the
/// diagonal: the returned tensor has these two axes removed, and the diagonal
/// appended as the **last** axis (NumPy `np.diagonal` convention). For a 2-D
/// input the result is one-dimensional.
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny): the input tensor.
/// - `diagonal_args`: [`DiagonalArgs`]: offset and axes; accepts `()` (main diagonal), a single
///   integer (offset, default axes), or a tuple `(offset, axis1, axis2)`.
///
/// # Returns
///
/// - [`TensorView<'_, T, B, D::SmallerOne>`][`TensorView`]: a view of the diagonals (no data is
///   copied); dimensionality is one less than the input.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((9, &device)).into_shape([3, 3]);
/// println!("{}", a.diagonal(()));
/// // [ 0 4 8]
/// println!("{}", a.diagonal(1));
/// // [ 1 5]
/// # assert_eq!(format!("{}", a.diagonal(1)), "[ 1 5]");
/// ```
///
/// For a 3-D tensor, the diagonal over axes `(0, 1)` is appended as the last
/// axis:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let b = rt::arange((24, &device)).into_shape([2, 3, 4]);
/// let d = b.diagonal((0, 0, 1));
/// println!("{d}");
/// // [[ 0 16]
/// //  [ 1 17]
/// //  [ 2 18]
/// //  [ 3 19]]
/// # assert_eq!(d.shape(), &[4, 2]);
/// ```
///
/// The mutable form writes through to the original tensor:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let mut c: Tensor<i32, _> = rt::zeros(([3, 3], &device));
/// let mut d = c.diagonal_mut(());
/// d += 2;
/// println!("{c}");
/// // [[ 2 0 0]
/// //  [ 0 2 0]
/// //  [ 0 0 2]]
/// # assert_eq!(format!("{c}"), "[[ 2 0 0]\n [ 0 2 0]\n [ 0 0 2]]");
/// ```
///
/// # Notes of API accordance
///
/// - NumPy: `numpy.diagonal(a, offset=0, axis1=0, axis2=1)` ([`numpy.diagonal`](https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html))
/// - RSTSR: `rt::diagonal(&tensor, args)` or `tensor.diagonal(args)`; the diagonal axis is appended
///   as the last axis, as in NumPy.
///
/// # Panics
///
/// - Panics if `axis1` / `axis2` are out of range or equal. An `offset` that selects no element
///   yields a zero-length diagonal (as in NumPy).
///
/// For a fallible version, use [`diagonal_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - NumPy: [`numpy.diagonal`](https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html)
/// - Python Array API standard: `diag` (extract, via `__getitem__`)
///
/// ## Related functions in RSTSR
///
/// - [`diag`]: extract a diagonal as an owned tensor, or construct a diagonal matrix from a 1-D
///   tensor.
/// - [`slice`](slice()): basic slicing views.
///
/// ## Variants of this function
///
/// - [`diagonal_f`]: fallible version.
/// - [`diagonal_mut`] / [`diagonal_mut_f`]: mutable views.
/// - [`into_diagonal`] / [`into_diagonal_f`]: ownership-consuming forms.
/// - [`into_diagonal_mut`] / [`into_diagonal_mut_f`]: ownership-consuming mutable forms.
/// - Associated methods on [`TensorAny`]: [`TensorAny::diagonal`] / [`TensorAny::diagonal_f`] /
///   [`TensorAny::diagonal_mut`] / [`TensorAny::diagonal_mut_f`] / [`TensorAny::into_diagonal`] /
///   [`TensorAny::into_diagonal_f`] / [`TensorAny::into_diagonal_mut`] /
///   [`TensorAny::into_diagonal_mut_f`].
pub fn diagonal<R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    diagonal_args: impl Into<DiagonalArgs>,
) -> TensorView<'_, T, B, D::SmallerOne>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    diagonal_f(tensor, diagonal_args).rstsr_unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    pub fn into_diagonal_f(self, diagonal_args: impl Into<DiagonalArgs>) -> Result<TensorAny<R, T, B, D::SmallerOne>> {
        into_diagonal_f(self, diagonal_args)
    }

    pub fn into_diagonal(self, diagonal_args: impl Into<DiagonalArgs>) -> TensorAny<R, T, B, D::SmallerOne> {
        into_diagonal(self, diagonal_args)
    }

    pub fn diagonal_f(&self, diagonal_args: impl Into<DiagonalArgs>) -> Result<TensorView<'_, T, B, D::SmallerOne>> {
        diagonal_f(self, diagonal_args)
    }

    pub fn diagonal(&self, diagonal_args: impl Into<DiagonalArgs>) -> TensorView<'_, T, B, D::SmallerOne> {
        diagonal(self, diagonal_args)
    }
}

/* #endregion */

/* #region diagonal_mut */

/// Mutable diagonal views of a tensor.
///
/// See also [`diagonal`].
pub fn into_diagonal_mut_f<S, D>(
    tensor: TensorBase<S, D>,
    diagonal_args: impl Into<DiagonalArgs>,
) -> Result<TensorBase<S, D::SmallerOne>>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    let (data, layout) = tensor.into_raw_parts();
    let DiagonalArgs { offset, axis1, axis2 } = diagonal_args.into();
    let layout = layout.diagonal(offset, axis1, axis2)?;
    return unsafe { Ok(TensorBase::new_unchecked(data, layout)) };
}

/// Mutable diagonal views of a tensor.
///
/// See also [`diagonal`].
pub fn into_diagonal_mut<S, D>(
    tensor: TensorBase<S, D>,
    diagonal_args: impl Into<DiagonalArgs>,
) -> TensorBase<S, D::SmallerOne>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    into_diagonal_mut_f(tensor, diagonal_args).rstsr_unwrap()
}

/// Mutable diagonal views of a tensor.
///
/// See also [`diagonal`].
pub fn diagonal_mut_f<R, T, B, D>(
    tensor: &mut TensorAny<R, T, B, D>,
    diagonal_args: impl Into<DiagonalArgs>,
) -> Result<TensorMut<'_, T, B, D::SmallerOne>>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    R: DataMutAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_diagonal_mut_f(tensor.view_mut(), diagonal_args)
}

/// Mutable diagonal views of a tensor.
///
/// See also [`diagonal`].
pub fn diagonal_mut<R, T, B, D>(
    tensor: &mut TensorAny<R, T, B, D>,
    diagonal_args: impl Into<DiagonalArgs>,
) -> TensorMut<'_, T, B, D::SmallerOne>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    R: DataMutAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    diagonal_mut_f(tensor, diagonal_args).rstsr_unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataMutAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    pub fn into_diagonal_mut_f(
        self,
        diagonal_args: impl Into<DiagonalArgs>,
    ) -> Result<TensorAny<R, T, B, D::SmallerOne>> {
        into_diagonal_mut_f(self, diagonal_args)
    }

    pub fn into_diagonal_mut(self, diagonal_args: impl Into<DiagonalArgs>) -> TensorAny<R, T, B, D::SmallerOne> {
        into_diagonal_mut(self, diagonal_args)
    }

    pub fn diagonal_mut_f(
        &mut self,
        diagonal_args: impl Into<DiagonalArgs>,
    ) -> Result<TensorMut<'_, T, B, D::SmallerOne>> {
        diagonal_mut_f(self, diagonal_args)
    }

    pub fn diagonal_mut(&mut self, diagonal_args: impl Into<DiagonalArgs>) -> TensorMut<'_, T, B, D::SmallerOne> {
        diagonal_mut(self, diagonal_args)
    }
}

/* #endregion */

/* #region indexing */

// It seems that implementing Index for TensorBase is not possible because of
// the lifetime issue. However, directly implementing each struct can avoid such
// problem.

#[duplicate_item(
    TensorStruct;
    [Tensor<T, B, D>];
    [TensorView<'_, T, B, D>];
    [TensorViewMut<'_, T, B, D>];
    [TensorCow<'_, T, B, D>];
)]
impl<T, D, B, I> Index<I> for TensorStruct
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T, Raw = Vec<T>>,
    I: AsRef<[usize]>,
{
    type Output = T;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        let index = index.as_ref().iter().map(|&v| v as isize).collect::<Vec<_>>();
        let i = self.layout().index(index.as_ref());
        let raw = self.raw();
        raw.index(i)
    }
}

#[duplicate_item(
    TensorStruct;
    [Tensor<T, B, D>];
    [TensorViewMut<'_, T, B, D>];
)]
impl<T, D, B, I> IndexMut<I> for TensorStruct
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T, Raw = Vec<T>>,
    I: AsRef<[usize]>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        let index = index.as_ref().iter().map(|&v| v as isize).collect::<Vec<_>>();
        let i = self.layout().index(index.as_ref());
        let raw = self.raw_mut();
        raw.index_mut(i)
    }
}

/* #endregion */

/* #region indexing (unchecked) */

#[duplicate_item(
    TensorStruct;
    [Tensor<T, B, D>];
    [TensorView<'_, T, B, D>];
    [TensorViewMut<'_, T, B, D>];
    [TensorCow<'_, T, B, D>];
)]
impl<T, B, D> TensorStruct
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    /// # Safety
    ///
    /// This function is unsafe because it does not check the validity of the
    /// index.
    #[inline]
    pub unsafe fn index_uncheck<I>(&self, index: I) -> &T
    where
        I: AsRef<[usize]>,
    {
        let index = index.as_ref();
        let i = unsafe { self.layout().index_uncheck(index) } as usize;
        let raw = self.raw();
        raw.index(i)
    }
}

#[duplicate_item(
    TensorStruct;
    [Tensor<T, B, D>];
    [TensorViewMut<'_, T, B, D>];
)]
impl<T, B, D> TensorStruct
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    /// # Safety
    ///
    /// This function is unsafe because it does not check the validity of the
    /// index.
    #[inline]
    pub unsafe fn index_mut_uncheck<I>(&mut self, index: I) -> &mut T
    where
        I: AsRef<[usize]>,
    {
        let index = index.as_ref();
        let i = unsafe { self.layout().index_uncheck(index) } as usize;
        let raw = self.raw_mut();
        raw.index_mut(i)
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_tensor_slice_1d() {
        let tensor = asarray(vec![1, 2, 3, 4, 5]);
        let tensor_slice = tensor.slice(s![1..4]);
        println!("{tensor_slice:?}");
        let tensor_slice = tensor.slice(s![1..4, None]);
        println!("{tensor_slice:?}");
        let tensor_slice = tensor.slice(1);
        println!("{tensor_slice:?}");
        let tensor_slice = tensor.slice(slice!(2, 7, 2));
        println!("{tensor_slice:?}");

        let mut tensor = asarray(vec![1, 2, 3, 4, 5]);
        let mut tensor_slice = tensor.slice_mut(s![1..4]);
        tensor_slice += 10;
        println!("{tensor:?}");
        *&mut tensor.slice_mut(s![1..4]) += 10;
        println!("{tensor:?}");
    }

    #[test]
    fn test_tensor_nd() {
        let tensor = arange(24.0).into_shape([2, 3, 4]);
        let tensor_slice = tensor.slice(s![1..2, 1..3, 1..4]);
        println!("{tensor_slice:?}");
        let tensor_slice = tensor.slice(s![1]);
        println!("{tensor_slice:?}");
    }

    #[test]
    fn test_tensor_index() {
        let mut tensor = asarray(vec![1, 2, 3, 4, 5]);
        let value = tensor[[1]];
        println!("{value:?}");
        let tensor_view = tensor.view();
        let value = tensor_view[[2]];
        {
            let tensor_view = tensor.view();
            let value = tensor_view[[3]];
            println!("{value:?}");
            let mut tensor_view_mut = tensor.view_mut();
            tensor_view_mut[[4]] += 1;
            *&mut tensor_view_mut.slice_mut(4) += 1;
        }
        println!("{value:?}");
        println!("{tensor:?}");
    }

    #[test]
    fn test_diagonal_compiles() {
        let a = arange(24.0).into_shape([2, 3, 4]);
        // this should reads input as i32
        let b = a.diagonal(1);
        println!("{b:?}");
    }
}
