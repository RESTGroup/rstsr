//! Element-wise select function `where` (NumPy three-argument form).
//!
//! The anchor documentation is on [`r#where`](where()).

use crate::prelude_dev::*;

/* #region shared layout helpers */

/// Output layout for element-wise select: the common elementwise-op policy of
/// `layout_for_array_copy` candidates over all operand layouts, falling back
/// to the device default order when the operands disagree.
fn layout_for_select_output<D>(layouts: &[&Layout<D>], order: FlagOrder) -> Result<Layout<D>>
where
    D: DimAPI,
{
    let candidates =
        layouts.iter().map(|l| layout_for_array_copy(l, TensorIterOrder::default())).collect::<Result<Vec<_>>>()?;
    if candidates.iter().all(|c| *c == candidates[0]) {
        Ok(candidates[0].clone())
    } else {
        match order {
            RowMajor => Ok(layouts[0].shape().c()),
            ColMajor => Ok(layouts[0].shape().f()),
        }
    }
}

/// Broadcast layouts of a select op: (cond, x, y, output).
type SelectLayouts = (Layout<IxD>, Layout<IxD>, Layout<IxD>, Layout<IxD>);

/// Broadcast the three operand layouts and derive the output layout.
///
/// The 3-way broadcast chains the associative pairwise broadcasts (dynamic dim
/// as the intermediary, since `DimMaxAPI` has no generic-projection impls).
fn select_layouts(la: &Layout<IxD>, lx: &Layout<IxD>, ly: &Layout<IxD>, order: FlagOrder) -> Result<SelectLayouts> {
    let (la_x, lx) = broadcast_layout(la, lx, order)?;
    let (la_f, ly_f) = broadcast_layout(&la_x, ly, order)?;
    let (lx_f, _) = broadcast_layout(&lx, &ly_f, order)?;
    let lc = layout_for_select_output(&[&la_f, &lx_f, &ly_f], order)?;
    Ok((la_f, lx_f, ly_f, lc))
}

/* #endregion */

/* #region tensor traits */

/// API trait for element-wise select; the anchor is the free function
/// [`r#where`](where()).
pub trait TensorWhereAPI<TRX, TRY> {
    type Output;
    /// Fallible method form of element-wise select.
    fn where_f(self, x: TRX, y: TRY) -> Result<Self::Output>;
    /// Panicking method form of element-wise select.
    fn r#where(self, x: TRX, y: TRY) -> Self::Output
    where
        Self: Sized,
    {
        self.where_f(x, y).rstsr_unwrap()
    }
}

impl<RA, RX, RY, DA, DX, DY, TX, TY, B> TensorWhereAPI<&TensorAny<RX, TX, B, DX>, &TensorAny<RY, TY, B, DY>>
    for &TensorAny<RA, bool, B, DA>
where
    RA: DataAPI<Data = <B as DeviceRawAPI<bool>>::Raw>,
    RX: DataAPI<Data = <B as DeviceRawAPI<TX>>::Raw>,
    RY: DataAPI<Data = <B as DeviceRawAPI<TY>>::Raw>,
    DA: DimAPI + DimMaxAPI<DX>,
    DX: DimAPI,
    DY: DimAPI,
    DA::Max: DimAPI + DimMaxAPI<DY>,
    <DA::Max as DimMaxAPI<DY>>::Max: DimAPI,
    B: OpWhereAPI<TX, TY, <DA::Max as DimMaxAPI<DY>>::Max>,
    B: DeviceAPI<bool> + DeviceAPI<TX> + DeviceAPI<TY> + DeviceAPI<B::TOut> + DeviceCreationAnyAPI<B::TOut>,
{
    type Output = Tensor<B::TOut, B, <DA::Max as DimMaxAPI<DY>>::Max>;

    fn where_f(self, x: &TensorAny<RX, TX, B, DX>, y: &TensorAny<RY, TY, B, DY>) -> Result<Self::Output> {
        // check device
        rstsr_assert!(self.device().same_device(x.device()), DeviceMismatch)?;
        rstsr_assert!(self.device().same_device(y.device()), DeviceMismatch)?;

        // check and broadcast layouts
        let default_order = self.device().default_order();
        let la = self.layout().to_dim::<IxD>()?;
        let lx = x.layout().to_dim::<IxD>()?;
        let ly = y.layout().to_dim::<IxD>()?;
        let (la_f, lx_f, ly_f, lc) = select_layouts(&la, &lx, &ly, default_order)?;
        let la_f = la_f.to_dim::<<DA::Max as DimMaxAPI<DY>>::Max>()?;
        let lx_f = lx_f.to_dim::<<DA::Max as DimMaxAPI<DY>>::Max>()?;
        let ly_f = ly_f.to_dim::<<DA::Max as DimMaxAPI<DY>>::Max>()?;
        let lc = lc.to_dim::<<DA::Max as DimMaxAPI<DY>>::Max>()?;

        // perform operation and return
        let device = self.device();
        let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
        device.op_mutd_refa_refb_refc(storage_c.raw_mut(), &lc, self.raw(), &la_f, x.raw(), &lx_f, y.raw(), &ly_f)?;
        // SAFETY: the op above wrote every element of the fresh `storage_c`.
        let storage_c = unsafe { B::assume_init_impl(storage_c) }?;
        Tensor::new_f(storage_c, lc)
    }
}

// By-value view forms of `x`/`y` delegate to the reference implementation.
#[duplicate_item(
    ImplGenerics TrX refX TrY refY;
    [RY: DataAPI<Data = <B as DeviceRawAPI<TY>>::Raw>, DA, DX, DY, TX, TY, B]
        [TensorView<'_, TX, B, DX>] [&x] [&TensorAny<RY, TY, B, DY>] [y];
    [RX: DataAPI<Data = <B as DeviceRawAPI<TX>>::Raw>, DA, DX, DY, TX, TY, B]
        [&TensorAny<RX, TX, B, DX>] [x] [TensorView<'_, TY, B, DY>] [&y];
    [DA, DX, DY, TX, TY, B]
        [TensorView<'_, TX, B, DX>] [&x] [TensorView<'_, TY, B, DY>] [&y];
)]
impl<RA, ImplGenerics> TensorWhereAPI<TrX, TrY> for &TensorAny<RA, bool, B, DA>
where
    RA: DataAPI<Data = <B as DeviceRawAPI<bool>>::Raw>,
    DA: DimAPI + DimMaxAPI<DX>,
    DX: DimAPI,
    DY: DimAPI,
    DA::Max: DimAPI + DimMaxAPI<DY>,
    <DA::Max as DimMaxAPI<DY>>::Max: DimAPI,
    B: OpWhereAPI<TX, TY, <DA::Max as DimMaxAPI<DY>>::Max>,
    B: DeviceAPI<bool> + DeviceAPI<TX> + DeviceAPI<TY> + DeviceAPI<B::TOut> + DeviceCreationAnyAPI<B::TOut>,
{
    type Output = Tensor<B::TOut, B, <DA::Max as DimMaxAPI<DY>>::Max>;

    fn where_f(self, x: TrX, y: TrY) -> Result<Self::Output> {
        TensorWhereAPI::where_f(self, refX, refY)
    }
}

impl<RA, RX, DA, DX, TX, TY, B> TensorWhereAPI<&TensorAny<RX, TX, B, DX>, TY> for &TensorAny<RA, bool, B, DA>
where
    RA: DataAPI<Data = <B as DeviceRawAPI<bool>>::Raw>,
    RX: DataAPI<Data = <B as DeviceRawAPI<TX>>::Raw>,
    DA: DimAPI + DimMaxAPI<DX>,
    DX: DimAPI,
    DA::Max: DimAPI,
    B: OpWhereAPI<TX, TY, DA::Max>,
    B: DeviceAPI<bool> + DeviceAPI<TX> + DeviceAPI<TY> + DeviceAPI<B::TOut> + DeviceCreationAnyAPI<B::TOut>,
    TY: num::Num,
{
    type Output = Tensor<B::TOut, B, DA::Max>;

    fn where_f(self, x: &TensorAny<RX, TX, B, DX>, y: TY) -> Result<Self::Output> {
        // check device
        rstsr_assert!(self.device().same_device(x.device()), DeviceMismatch)?;

        // check and broadcast layouts
        let default_order = self.device().default_order();
        let la = self.layout().to_dim::<IxD>()?;
        let lx = x.layout().to_dim::<IxD>()?;
        let (la_b, lx_b) = broadcast_layout(&la, &lx, default_order)?;
        let lc = layout_for_select_output(&[&la_b, &lx_b], default_order)?;
        let la_b = la_b.to_dim::<DA::Max>()?;
        let lx_b = lx_b.to_dim::<DA::Max>()?;
        let lc = lc.to_dim::<DA::Max>()?;

        // perform operation and return
        let device = self.device();
        let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
        device.op_mutd_refa_refb_numc(storage_c.raw_mut(), &lc, self.raw(), &la_b, x.raw(), &lx_b, y)?;
        // SAFETY: the op above wrote every element of the fresh `storage_c`.
        let storage_c = unsafe { B::assume_init_impl(storage_c) }?;
        Tensor::new_f(storage_c, lc)
    }
}

impl<RA, RY, DA, DY, TX, TY, B> TensorWhereAPI<TX, &TensorAny<RY, TY, B, DY>> for &TensorAny<RA, bool, B, DA>
where
    RA: DataAPI<Data = <B as DeviceRawAPI<bool>>::Raw>,
    RY: DataAPI<Data = <B as DeviceRawAPI<TY>>::Raw>,
    DA: DimAPI + DimMaxAPI<DY>,
    DY: DimAPI,
    DA::Max: DimAPI,
    B: OpWhereAPI<TX, TY, DA::Max>,
    B: DeviceAPI<bool> + DeviceAPI<TX> + DeviceAPI<TY> + DeviceAPI<B::TOut> + DeviceCreationAnyAPI<B::TOut>,
    TX: num::Num,
{
    type Output = Tensor<B::TOut, B, DA::Max>;

    fn where_f(self, x: TX, y: &TensorAny<RY, TY, B, DY>) -> Result<Self::Output> {
        // check device
        rstsr_assert!(self.device().same_device(y.device()), DeviceMismatch)?;

        // check and broadcast layouts
        let default_order = self.device().default_order();
        let la = self.layout().to_dim::<IxD>()?;
        let ly = y.layout().to_dim::<IxD>()?;
        let (la_b, ly_b) = broadcast_layout(&la, &ly, default_order)?;
        let lc = layout_for_select_output(&[&la_b, &ly_b], default_order)?;
        let la_b = la_b.to_dim::<DA::Max>()?;
        let ly_b = ly_b.to_dim::<DA::Max>()?;
        let lc = lc.to_dim::<DA::Max>()?;

        // perform operation and return
        let device = self.device();
        let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
        device.op_mutd_refa_numb_refc(storage_c.raw_mut(), &lc, self.raw(), &la_b, x, y.raw(), &ly_b)?;
        // SAFETY: the op above wrote every element of the fresh `storage_c`.
        let storage_c = unsafe { B::assume_init_impl(storage_c) }?;
        Tensor::new_f(storage_c, lc)
    }
}

/* #endregion */

/* #region function impl */

/// Returns elements chosen from `x` or `y` depending on `condition`.
///
/// Element-wise select: where `condition` is `true`, the element of `x` is
/// chosen; where it is `false`, the element of `y` is chosen. The condition
/// must be a boolean tensor, while `x` and `y` may each be a tensor or a
/// scalar; scalar arguments follow rstsr's usual (strong) promotion, like
/// [`maximum`](crate::tensor::operators::op_binary_common::maximum()).
/// The result is a newly allocated tensor of the promoted dtype and the
/// broadcast shape of the tensor operands.
///
/// <div class="warning">
///
/// **Row/Column Major Notice**
///
/// This function behaves differently on default orders ([`RowMajor`] and [`ColMajor`]) of device.
///
/// </div>
///
/// See [`order_semantics`](crate::order_semantics) for the two device default
/// orders: the tensor shapes are broadcast together from the last axis under
/// [`RowMajor`] (NumPy-like) and from the first axis under [`ColMajor`], so a
/// call accepted by one order may be rejected by the other.
///
/// # Parameters
///
/// - `cond`: [`&TensorAny<R, bool, B, D>`](TensorAny) - the boolean condition tensor. Other
///   condition dtypes are rejected at compile time; translate NumPy truthy masks with
///   [`ne`](crate::tensor::operators::op_binary_common::ne()).
/// - `x`: elements chosen where the condition is `true`.
///   - Overloads:
///     - `x`: [`&TensorAny<R, T, B, D>`](TensorAny) - tensor operand
///     - `x`: [`TensorView<'_, T, B, D>`](TensorView) - tensor operand by value
///     - `x`: `T` - scalar operand
/// - `y`: elements chosen where the condition is `false`.
///   - Overloads: (same forms as `x`)
///
/// # Returns
///
/// A newly allocated [`Tensor`]`<B::TOut, B, D>` with the broadcast shape and
/// the promoted dtype of the tensor operands; the `bool` condition dtype never
/// participates in promotion.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let cond = rt::tensor_from_nested!([true, false, true], &device);
/// let x = rt::tensor_from_nested!([1, 2, 3], &device);
/// let y = rt::tensor_from_nested!([10, 20, 30], &device);
/// println!("{}", rt::r#where(&cond, &x, &y));
/// // [ 1 20 3]
/// # assert_eq!(format!("{}", rt::r#where(&cond, &x, &y)), "[ 1 20 3]");
/// ```
///
/// Scalar arguments are allowed for `x` and `y`:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let cond = rt::tensor_from_nested!([[true, false], [false, true]], &device);
/// println!("{}", rt::r#where(&cond, &rt::arange(4.0).reshape([2, 2]), 0.0));
/// // [[ 0 0]
/// //  [ 0 3]]
/// # assert_eq!(format!("{}", rt::r#where(&cond, &rt::arange(4.0).reshape([2, 2]), 0.0)), "[[ 0 0]\n [ 0 3]]");
/// ```
///
/// # Elaborated examples
///
/// ## Difference between RowMajor and ColMajor
///
/// A 1-D condition of length 2 against `(2, 3)` operands is rejected under
/// [`RowMajor`] (trailing-axis alignment, NumPy-like: `2` vs `3` do not
/// broadcast), but accepted under [`ColMajor`] (leading-axis alignment; row
/// `i` of the result is selected by `cond[i]`):
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// // RowMajor (the default): non-broadcastable
/// device.set_default_order(RowMajor);
/// let cond = rt::tensor_from_nested!([true, false], &device);
/// let x = rt::tensor_from_nested!([[0, 1, 2], [3, 4, 5]], &device);
/// let y = rt::tensor_from_nested!([[0, -1, -2], [-3, -4, -5]], &device);
/// # assert!(rt::where_f(&cond, &x, &y).is_err());
/// // ColMajor: leading-axis alignment (operands recreated so their devices
/// // carry the new default order)
/// device.set_default_order(ColMajor);
/// let cond = rt::tensor_from_nested!([true, false], &device);
/// let x = rt::tensor_from_nested!([[0, 1, 2], [3, 4, 5]], &device);
/// let y = rt::tensor_from_nested!([[0, -1, -2], [-3, -4, -5]], &device);
/// let r = rt::r#where(&cond, &x, &y);
/// println!("{r}");
/// // [[ 0 1 2]
/// //  [ -3 -4 -5]]
/// # assert_eq!(format!("{r}"), "[[ 0 1 2]\n [ -3 -4 -5]]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `where(condition, x1, x2)` ([`where`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.where.html))
/// - NumPy: `numpy.where(condition, [x, y])` ([`numpy.where`](https://numpy.org/doc/stable/reference/generated/numpy.where.html))
/// - RSTSR: `rt::r#where(&cond, &x, &y)` or `cond.r#where(x, y)`
///
/// Please note the differences (recorded in the
/// [tracking registry](https://github.com/RESTGroup/rstsr/blob/master/rstsr-core/tests/tracking/numpy_differences.md)):
///
/// - The condition must have a boolean dtype (array-api aligned); NumPy also accepts arbitrary
///   truthy masks, which should be translated with `rt::ne(mask, 0)`.
/// - Scalar `x`/`y` are strongly promoted (rstsr house rule); NumPy's NEP 50 weak-scalar dtype
///   minimization (e.g. keeping `float32` against a NaN Python scalar) is not implemented.
/// - NumPy's one-argument `where(condition)` (equivalent to `nonzero`) is not provided.
///
/// # Panics
///
/// - Panics if `cond`, `x`, and `y` are on different devices.
/// - Panics if the shapes cannot be broadcast together under the current default order (see the
///   notice above).
///
/// For a fallible version, use [`where_f`](where_f()).
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - NumPy `numpy.where` and array-api `where` (see `# Notes of API accordance` above).
///
/// ## Related functions in RSTSR
///
/// - [`maximum`](crate::tensor::operators::op_binary_common::maximum()): element-wise maximum; the
///   scalar-promotion reference.
/// - [`ne`](crate::tensor::operators::op_binary_common::ne()): builds boolean conditions from
///   numeric masks.
///
/// ## Variants of this function
///
/// - [`where_f`](where_f()): fallible version, actual implementation.
/// - Method form on trait `TensorWhereAPI`: `cond.r#where(x, y)`.
pub fn r#where<TRC, TRX, TRY>(cond: TRC, x: TRX, y: TRY) -> TRC::Output
where
    TRC: TensorWhereAPI<TRX, TRY>,
{
    cond.r#where(x, y)
}

/// Element-wise select (fallible): chooses elements from `x` or `y` by a
/// boolean condition.
///
/// See also [`r#where`](where()).
pub fn where_f<TRC, TRX, TRY>(cond: TRC, x: TRX, y: TRY) -> Result<TRC::Output>
where
    TRC: TensorWhereAPI<TRX, TRY>,
{
    cond.where_f(x, y)
}

/* #endregion */
