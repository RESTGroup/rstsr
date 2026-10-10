use crate::prelude_dev::*;
use core::mem::transmute;

/// Resolve a [`AxesPairIndex`] into two normalized, pairwise-aligned,
/// non-negative axis lists for `tensordot`.
fn resolve_tensordot_axes(
    axes: &AxesPairIndex<isize>,
    ndim_a: usize,
    ndim_b: usize,
) -> Result<(Vec<isize>, Vec<isize>)> {
    match axes {
        AxesPairIndex::None => unreachable!("`None` is normalized to `Val(2)` before this call"),
        AxesPairIndex::Val(n) => {
            let n = *n;
            rstsr_assert!(n >= 0, InvalidValue, "when given as an integer, `axes` must be non-negative")?;
            let n = n as usize;
            rstsr_assert!(
                n <= ndim_a && n <= ndim_b,
                InvalidLayout,
                "`axes` must not exceed the number of dimensions of either input"
            )?;
            // last `n` axes of `a` against first `n` axes of `b`, in order
            let axes_a = (ndim_a - n..ndim_a).map(|x| x as isize).collect();
            let axes_b = (0..n).map(|x| x as isize).collect();
            Ok((axes_a, axes_b))
        },
        AxesPairIndex::Pair(axes_a, axes_b) => {
            let axes_a = normalize_axes_index(axes_a.clone(), ndim_a, false, false)?;
            let axes_b = normalize_axes_index(axes_b.clone(), ndim_b, false, false)?;
            rstsr_assert_eq!(
                axes_a.len(),
                axes_b.len(),
                InvalidValue,
                "`axes_a` and `axes_b` must have the same length"
            )?;
            Ok((axes_a, axes_b))
        },
    }
}

/// Tensor contraction over specified axes.
///
/// Contracts the axes of `a` given by `axes` with the corresponding axes of `b`
/// (see the `axes` parameter below); the result holds the non-contracted axes of
/// `a` (in order) followed by the non-contracted axes of `b` (in order). No
/// conjugation is applied, unlike [`vecdot`](crate::prelude::vecdot).
///
/// # Parameters
///
/// - `a`, `b`: impl [`TensorViewAPI`]
///
///   - The input arrays. Corresponding contracted axes must have equal sizes.
///
/// - `axes`: `impl TryInto<AxesPairIndex<isize>>`
///
///   - Number of axes to contract, or the pair of explicit axis sequences.
///   - Default: `2`.
///   - Overloads:
///     - integer `n`: contract the last `n` axes of `a` with the first `n` axes of `b`, in order
///       (`n` must be non-negative and at most the smaller dimensionality).
///     - `None` (or `()`): same as `2`.
///     - `(axes_a, axes_b)`: contract `axes_a` of `a` with `axes_b` of `b`, where each side is an
///       integer or a collection of integers; negative axes count from the back.
///     - a bare collection (`[0, 1]`, `vec![0, 1]`): shorthand for the *same* axes on both sides
///       (rstsr extension; note a 2-tuple is the pair overload — this differs from NumPy, where
///       `tensordot(a, b, [0, 1])` means `a` axis 0 against `b` axis 1).
///
/// # Returns
///
/// [`Tensor`] — the contraction result.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // outer product (axes = 0)
/// let a = rt::tensor_from_nested!([1, 2], &device);
/// let b = rt::tensor_from_nested!([3, 4], &device);
/// let c = rt::tensordot(&a, &b, 0);
/// assert_eq!(c.shape(), &[2, 2]);
/// ```
pub fn tensordot<TA, TB, TC, DA, DB, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    axes: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
) -> Tensor<TC, B, IxD>
where
    DA: DimAPI,
    DB: DimAPI,
    B: DeviceTensordotAPI<TA, TB, TC, DA, DB, IxD>
        + DeviceAPI<TA>
        + DeviceAPI<TB>
        + DeviceAPI<TC>
        + DeviceCreationAnyAPI<TC>,
{
    tensordot_f(a, b, axes).rstsr_unwrap()
}

/// Tensor contraction over specified axes.
///
/// See also [`tensordot`].
pub fn tensordot_f<TA, TB, TC, DA, DB, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    axes: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
) -> Result<Tensor<TC, B, IxD>>
where
    DA: DimAPI,
    DB: DimAPI,
    B: DeviceTensordotAPI<TA, TB, TC, DA, DB, IxD>
        + DeviceAPI<TA>
        + DeviceAPI<TB>
        + DeviceAPI<TC>
        + DeviceCreationAnyAPI<TC>,
{
    let (a, b) = (a.view(), b.view());
    let device = a.device().clone();
    rstsr_assert!(device.same_device(b.device()), DeviceMismatch)?;

    let mut axes = axes.try_into().map_err(Into::into)?;
    if axes == AxesPairIndex::None {
        axes = AxesPairIndex::Val(2);
    }
    let (axes_a, axes_b) = resolve_tensordot_axes(&axes, a.ndim(), b.ndim())?;

    let (las, lam) = a.layout().dim_split_axes(&axes_a)?;
    let (lbs, lbm) = b.layout().dim_split_axes(&axes_b)?;
    rstsr_assert_eq!(
        las.shape(),
        lbs.shape(),
        InvalidLayout,
        "the dimensions of a and b along the contracted axes should be the same"
    )?;

    let mut shape_c = lam.shape().clone();
    shape_c.extend_from_slice(lbm.shape());
    let layout_c = shape_c.new_contig(None, device.default_order());
    let mut storage_c = device.uninit_impl(layout_c.bounds_index()?.1)?;
    device.tensordot(storage_c.raw_mut(), &layout_c, a.raw(), a.layout(), b.raw(), b.layout(), &axes_a, &axes_b)?;
    // SAFETY: `device.tensordot` wrote every element of `layout_c`.
    unsafe { Tensor::new_f(B::assume_init_impl(storage_c)?, layout_c) }
}

/// Tensor contraction over specified axes, writing into `c`.
///
/// See also [`tensordot`].
pub fn tensordot_from<TA, TB, TC, DA, DB, DC, B>(
    c: impl TensorViewMutAPI<Type = TC, Backend = B, Dim = DC>,
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    axes: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
) -> Result<()>
where
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    B: DeviceTensordotAPI<TA, TB, TC, DA, DB, DC> + DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
{
    tensordot_from_f(c, a, b, axes)
}

/// Tensor contraction over specified axes, writing into `c`.
///
/// See also [`tensordot`].
pub fn tensordot_from_f<TA, TB, TC, DA, DB, DC, B>(
    mut c: impl TensorViewMutAPI<Type = TC, Backend = B, Dim = DC>,
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    axes: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
) -> Result<()>
where
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    B: DeviceTensordotAPI<TA, TB, TC, DA, DB, DC> + DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
{
    let (a, b, mut c) = (a.view(), b.view(), c.view_mut());
    // writing through a broadcast layout would alias elements
    rstsr_assert!(!c.layout().is_broadcasted(), InvalidLayout, "cannot write into broadcasted tensor")?;

    let device = c.device().clone();
    rstsr_assert!(device.same_device(a.device()), DeviceMismatch)?;
    rstsr_assert!(device.same_device(b.device()), DeviceMismatch)?;

    let mut axes = axes.try_into().map_err(Into::into)?;
    if axes == AxesPairIndex::None {
        axes = AxesPairIndex::Val(2);
    }
    let (axes_a, axes_b) = resolve_tensordot_axes(&axes, a.ndim(), b.ndim())?;

    let (las, lam) = a.layout().dim_split_axes(&axes_a)?;
    let (lbs, lbm) = b.layout().dim_split_axes(&axes_b)?;
    rstsr_assert_eq!(
        las.shape(),
        lbs.shape(),
        InvalidLayout,
        "the dimensions of a and b along the contracted axes should be the same"
    )?;

    let mut shape_c_expect = lam.shape().clone();
    shape_c_expect.extend_from_slice(lbm.shape());
    let shape_c = c.shape();
    rstsr_assert_eq!(shape_c_expect, shape_c.as_ref(), InvalidLayout, "incompatible output shape in tensordot")?;

    let c_layout = c.layout().clone();
    // SAFETY: `Vec<TC>` -> `Vec<MaybeUninit<TC>>` reinterpretation (identical
    // layout); `c` is an initialized, writable output buffer.
    let c_raw_mut = unsafe {
        transmute::<&mut <B as DeviceRawAPI<TC>>::Raw, &mut <B as DeviceRawAPI<MaybeUninit<TC>>>::Raw>(c.raw_mut())
    };
    device.tensordot(c_raw_mut, &c_layout, a.raw(), a.layout(), b.raw(), b.layout(), &axes_a, &axes_b)
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Tensor contraction over specified axes.
    ///
    /// See also [`tensordot`].
    pub fn tensordot<TB, DB, TC>(
        &self,
        b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
        axes: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
    ) -> Tensor<TC, B, IxD>
    where
        DB: DimAPI,
        B: DeviceTensordotAPI<T, TB, TC, D, DB, IxD>
            + DeviceAPI<T>
            + DeviceAPI<TB>
            + DeviceAPI<TC>
            + DeviceCreationAnyAPI<TC>,
    {
        tensordot_f(self.view(), b, axes).rstsr_unwrap()
    }

    /// Tensor contraction over specified axes.
    ///
    /// See also [`tensordot`].
    pub fn tensordot_f<TB, DB, TC>(
        &self,
        b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
        axes: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
    ) -> Result<Tensor<TC, B, IxD>>
    where
        DB: DimAPI,
        B: DeviceTensordotAPI<T, TB, TC, D, DB, IxD>
            + DeviceAPI<T>
            + DeviceAPI<TB>
            + DeviceAPI<TC>
            + DeviceCreationAnyAPI<TC>,
    {
        tensordot_f(self.view(), b, axes)
    }

    /// Tensor contraction over specified axes, writing into `self`.
    ///
    /// See also [`tensordot`].
    pub fn tensordot_from<TA, TB, DA, DB>(
        &mut self,
        a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
        b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
        axes: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
    ) where
        DA: DimAPI,
        DB: DimAPI,
        B: DeviceTensordotAPI<TA, TB, T, DA, DB, D> + DeviceAPI<TA> + DeviceAPI<TB>,
        R: DataMutAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    {
        tensordot_from_f(self, a, b, axes).rstsr_unwrap()
    }

    /// Tensor contraction over specified axes, writing into `self`.
    ///
    /// See also [`tensordot`].
    pub fn tensordot_from_f<TA, TB, DA, DB>(
        &mut self,
        a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
        b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
        axes: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
    ) -> Result<()>
    where
        DA: DimAPI,
        DB: DimAPI,
        B: DeviceTensordotAPI<TA, TB, T, DA, DB, D> + DeviceAPI<TA> + DeviceAPI<TB>,
        R: DataMutAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    {
        tensordot_from_f(self, a, b, axes)
    }
}

#[cfg(test)]
mod test {
    use rstsr::prelude::*;

    #[test]
    fn test_tensordot_basic() {
        let mut device = DeviceCpu::default();
        device.set_default_order(RowMajor);

        // axes = 0 -> outer product
        let a = rt::tensor_from_nested!([[1.0, 2.0], [3.0, 4.0]], &device);
        let b = rt::tensor_from_nested!([[5.0, 6.0], [7.0, 8.0]], &device);
        let c = rt::tensordot(&a, &b, 0);
        let target = rt::tensor_from_nested!(
            [[[[5, 6], [7, 8]], [[10, 12], [14, 16]]], [[[15, 18], [21, 24]], [[20, 24], [28, 32]]]],
            &device
        );
        assert_eq!(c.shape(), &[2, 2, 2, 2]);
        assert!(rt::allclose(&c, &target, None));

        // axes = 1 -> matrix product (GEMM fast path)
        let c = rt::tensordot(&a, &b, 1);
        let target = rt::matmul(&a, &b);
        assert_eq!(c.shape(), &[2, 2]);
        assert!(rt::allclose(&c, &target, None));

        // axes = 2 -> full contraction (0-d)
        let c = rt::tensordot(&a, &b, 2);
        assert_eq!(c.shape(), &[] as &[usize]);
        let target = rt::asarray((70.0, &device));
        assert!(rt::allclose(&c, &target, None));

        // default (None) == axes = 2
        let c_def: Tensor<_, _, IxD> = rt::tensordot(&a, &b, ());
        assert!(rt::allclose(&c_def, &target, None));
    }

    #[test]
    fn test_tensordot_col_major() {
        let mut device = DeviceCpu::default();
        device.set_default_order(ColMajor);

        let a = rt::tensor_from_nested!([[1.0, 2.0], [3.0, 4.0]], &device);
        let b = rt::tensor_from_nested!([[5.0, 6.0], [7.0, 8.0]], &device);
        // result is order-independent
        let c = rt::tensordot(&a, &b, 1);
        let target = rt::tensor_from_nested!([[19.0, 22.0], [43.0, 50.0]], &device);
        assert!(rt::allclose(&c, &target, None));

        let c = rt::tensordot(&a, &b, 2);
        assert!(rt::allclose(&c, &rt::asarray((70.0, &device)), None));

        // swapped pairing: sum_{i,j} a[i,j] * b[j,i] = 5 + 14 + 18 + 32 = 69
        let c = rt::tensordot(&a, &b, ([1, 0], [0, 1]));
        assert!(rt::allclose(&c, &rt::asarray((69.0, &device)), None));
    }

    #[test]
    fn test_tensordot_pair_axes() {
        let mut device = DeviceCpu::default();
        device.set_default_order(RowMajor);

        let a = rt::arange((60.0, &device)).into_shape((3, 4, 5));
        let b = rt::arange((24.0, &device)).into_shape((4, 3, 2));
        let c = rt::tensordot(&a, &b, ([1, 0], [0, 1]));
        let target = rt::tensor_from_nested!(
            [[4400., 4730.], [4532., 4874.], [4664., 5018.], [4796., 5162.], [4928., 5306.]],
            &device
        );
        assert_eq!(c.shape(), &[5, 2]);
        assert!(rt::allclose(&c, &target, None));
    }
}
