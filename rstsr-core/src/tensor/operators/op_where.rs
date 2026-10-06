//! Element-wise select function: `where` ([`where_f`](where_f()) fallible)
//! chooses elements from `x` or `y` by a boolean condition, following NumPy's
//! three-argument `np.where` and the array API `where(condition, x1, x2)`.
//!
//! The condition must be a boolean tensor; `x` and `y` may be tensors or
//! scalars (scalars follow rstsr's usual promotion, like
//! [`maximum`](crate::tensor::operators::op_binary_common::maximum())).
//! All three tensor shapes are NumPy-broadcast together.
//!
//! # Examples
//!
//! ```rust
//! # use rstsr::prelude::*;
//! # let mut device = DeviceCpu::default();
//! # device.set_default_order(RowMajor);
//! let cond = rt::tensor_from_nested!([true, false, true], &device);
//! let x = rt::tensor_from_nested!([1, 2, 3], &device);
//! let y = rt::tensor_from_nested!([10, 20, 30], &device);
//! println!("{}", rt::r#where(&cond, &x, &y));
//! // [ 1 20 3]
//! # assert_eq!(format!("{}", rt::r#where(&cond, &x, &y)), "[ 1 20 3]");
//! ```
//!
//! Scalar arguments are allowed for `x` and `y`:
//!
//! ```rust
//! # use rstsr::prelude::*;
//! # let mut device = DeviceCpu::default();
//! # device.set_default_order(RowMajor);
//! let cond = rt::tensor_from_nested!([[true, false], [false, true]], &device);
//! println!("{}", rt::r#where(&cond, &rt::arange(4.0).reshape([2, 2]), 0.0));
//! // [[ 0 0]
//! //  [ 0 3]]
//! # assert_eq!(format!("{}", rt::r#where(&cond, &rt::arange(4.0).reshape([2, 2]), 0.0)), "[[ 0 0]\n [ 0 3]]");
//! ```

use crate::prelude_dev::*;

/* #region tensor traits */

pub trait TensorWhereAPI<TRX, TRY> {
    type Output;
    fn where_f(self, x: TRX, y: TRY) -> Result<Self::Output>;
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

        // check and broadcast layouts; NumPy 3-way broadcast by chaining the
        // associative pairwise broadcasts (dynamic dim as the intermediary,
        // since `DimMaxAPI` has no generic-projection impls)
        let la = self.layout();
        let lx = x.layout();
        let ly = y.layout();
        let default_order = self.device().default_order();
        let (la_cx, lx_cx) = broadcast_layout(&la.to_dim::<IxD>()?, &lx.to_dim::<IxD>()?, default_order)?;
        let (la_f, ly_f) = broadcast_layout(&la_cx, &ly.to_dim::<IxD>()?, default_order)?;
        let (lx_f, _) = broadcast_layout(&lx_cx, &ly_f, default_order)?;
        let la_f = la_f.to_dim::<<DA::Max as DimMaxAPI<DY>>::Max>()?;
        let lx_f = lx_f.to_dim::<<DA::Max as DimMaxAPI<DY>>::Max>()?;
        let ly_f = ly_f.to_dim::<<DA::Max as DimMaxAPI<DY>>::Max>()?;
        let lc = match default_order {
            RowMajor => la_f.shape().c(),
            ColMajor => la_f.shape().f(),
        };

        // perform operation and return
        let device = self.device();
        let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
        device.op_mutd_refa_refb_refc(
            storage_c.raw_mut(),
            &lc,
            self.raw(),
            &la_f,
            x.raw(),
            &lx_f,
            y.raw(),
            &ly_f,
        )?;
        // SAFETY: the op above wrote every element of the fresh `storage_c`.
        let storage_c = unsafe { B::assume_init_impl(storage_c) }?;
        Tensor::new_f(storage_c, lc)
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
        // check and broadcast layout
        let la = self.layout();
        let lx = x.layout();
        let default_order = self.device().default_order();
        let (la_b, lx_b) = broadcast_layout(la, lx, default_order)?;
        let lc = match default_order {
            RowMajor => la_b.shape().c(),
            ColMajor => la_b.shape().f(),
        };

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
        // check and broadcast layout
        let la = self.layout();
        let ly = y.layout();
        let default_order = self.device().default_order();
        let (la_b, ly_b) = broadcast_layout(la, ly, default_order)?;
        let lc = match default_order {
            RowMajor => la_b.shape().c(),
            ColMajor => la_b.shape().f(),
        };

        // perform operation and return
        let device = self.device();
        let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
        device.op_mutd_refa_numb_refc(storage_c.raw_mut(), &lc, self.raw(), &la_b, x, y.raw(), &ly_b)?;
        // SAFETY: the op above wrote every element of the fresh `storage_c`.
        let storage_c = unsafe { B::assume_init_impl(storage_c) }?;
        Tensor::new_f(storage_c, lc)
    }
}

#[duplicate_item(
    TensorViewType;
   [TensorView<'_, bool, B, DA>];
   [TensorMut<'_, bool, B, DA>];
)]
impl<RX, RY, DA, DX, DY, TX, TY, B>
    TensorWhereAPI<&TensorAny<RX, TX, B, DX>, &TensorAny<RY, TY, B, DY>> for TensorViewType
where
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
        TensorWhereAPI::where_f(&self.view(), x, y)
    }
}

/* #endregion */

/* #region function impl */

/// Element-wise select (fallible): chooses elements from `x` or `y` by the
/// boolean tensor `cond`. See module documentation for details.
pub fn where_f<TRC, TRX, TRY>(cond: TRC, x: TRX, y: TRY) -> Result<TRC::Output>
where
    TRC: TensorWhereAPI<TRX, TRY>,
{
    cond.where_f(x, y)
}

/// Element-wise select: chooses elements from `x` or `y` by the boolean
/// tensor `cond` (panicking variant of [`where_f`]).
pub fn r#where<TRC, TRX, TRY>(cond: TRC, x: TRX, y: TRY) -> TRC::Output
where
    TRC: TensorWhereAPI<TRX, TRY>,
{
    cond.r#where(x, y)
}

/* #endregion */
