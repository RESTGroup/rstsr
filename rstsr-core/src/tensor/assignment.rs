//! Assignment of values into tensors: [`assign`] (tensor to tensor, with
//! broadcasting and dtype casting) and [`fill`] (scalar to tensor).

use crate::prelude_dev::*;

/* #region assign */

/// API trait backing [`assign`].
pub trait TensorAssignAPI<TRB> {
    fn assign_f(a: &mut Self, b: TRB) -> Result<()>;
    fn assign(a: &mut Self, b: TRB) {
        Self::assign_f(a, b).rstsr_unwrap()
    }
}

pub fn assign_f<TRA, TRB>(a: &mut TRA, b: TRB) -> Result<()>
where
    TRA: TensorAssignAPI<TRB>,
{
    TRA::assign_f(a, b)
}

/// Assign the values of `b` into `a` (`a <- b`), broadcasting `b` against
/// `a`'s shape and casting the dtype as needed.
///
/// <div class="warning">
///
/// **Row/Column Major Notice**
///
/// This function behaves differently on default orders ([`RowMajor`] and [`ColMajor`]) of device.
///
/// </div>
///
/// The source `b` is broadcast against `a`'s shape following the device
/// default order (see [`order_semantics`](crate::order_semantics)); same-shape
/// assignment is unaffected. `a` must not be a broadcasted (read-only) view.
///
/// # Parameters
///
/// - `a`: the destination tensor (mutable).
/// - `b`: the source; a tensor (view or owned). Values are cast from `TB` to `TA` on the fly.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let mut a: Tensor<f32, _> = rt::zeros(([2, 3], &device));
/// let b = rt::arange((6, &device)).into_shape([2, 3]);
/// a.assign(&b);
/// println!("{a}");
/// // [[ 0 1 2]
/// //  [ 3 4 5]]
/// # assert_eq!(format!("{a}"), "[[ 0 1 2]\n [ 3 4 5]]");
/// ```
///
/// Broadcastable sources are repeated element-wise:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let mut c: Tensor<f32, _> = rt::zeros(([2, 3], &device));
/// let row = rt::tensor_from_nested!([[1.0, 2.0, 3.0]], &device);
/// c.assign(row.view().into_shape([1, 3]));
/// println!("{c}");
/// // [[ 1 2 3]
/// //  [ 1 2 3]]
/// # assert_eq!(format!("{c}"), "[[ 1 2 3]\n [ 1 2 3]]");
/// ```
///
/// # Notes of API accordance
///
/// - NumPy: `a[...] = b` (broadcast assignment) ([`numpy.indexing`](https://numpy.org/doc/stable/user/basics.indexing.html))
/// - RSTSR: `rt::assign(&mut a, &b)` or method `a.assign(&b)`; the dtype cast is implicit.
///
/// # Panics
///
/// - Panics if `b` cannot be broadcast to `a`'s shape, if the devices differ, or if `a` is a
///   broadcasted (read-only) view.
///
/// For a fallible version, use [`assign_f`].
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`fill`]: set every element to a scalar.
/// - [`add_assign`](crate::tensor::operators::exports::add_assign()): in-place arithmetic
///   assignment.
///
/// ## Variants of this function
///
/// - [`assign_f`]: fallible version.
/// - [`TensorAssignAPI`]: trait backing this function.
/// - Associated methods on [`TensorBase`]: [`TensorBase::assign`] / [`TensorBase::assign_f`].
pub fn assign<TRA, TRB>(a: &mut TRA, b: TRB)
where
    TRA: TensorAssignAPI<TRB>,
{
    TRA::assign(a, b)
}

impl<RA, DA, RB, DB, TA, TB, B> TensorAssignAPI<TensorAny<RB, TB, B, DB>> for TensorAny<RA, TA, B, DA>
where
    RA: DataMutAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
    DA: DimAPI,
    DB: DimAPI,
    B: DeviceAPI<TA> + DeviceAPI<TB> + OpAssignAPI<TA, DA, TB>,
{
    fn assign_f(a: &mut Self, b: TensorAny<RB, TB, B, DB>) -> Result<()> {
        // get tensor views
        let mut a = a.view_mut();
        let b = b.view();
        // check device
        rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
        let device = a.device().clone();
        // check layout
        rstsr_assert!(!a.layout().is_broadcasted(), InvalidLayout, "cannot assign to broadcasted tensor")?;
        let la = a.layout().to_dim::<IxD>()?;
        let lb = b.layout().to_dim::<IxD>()?;
        let default_order = a.device().default_order();
        let (la_b, lb_b) = broadcast_layout_to_first(&la, &lb, default_order)?;
        let la_b = la_b.into_dim::<DA>()?;
        let lb_b = lb_b.into_dim::<DA>()?;
        // assign
        device.assign(a.raw_mut(), &la_b, b.raw(), &lb_b)
    }
}

impl<RA, DA, RB, DB, TA, TB, B> TensorAssignAPI<&TensorAny<RB, TB, B, DB>> for TensorAny<RA, TA, B, DA>
where
    RA: DataMutAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
    DA: DimAPI,
    DB: DimAPI,
    B: DeviceAPI<TA> + DeviceAPI<TB> + OpAssignAPI<TA, DA, TB>,
{
    fn assign_f(a: &mut Self, b: &TensorAny<RB, TB, B, DB>) -> Result<()> {
        TensorAssignAPI::assign_f(a, b.view())
    }
}

impl<S, D> TensorBase<S, D>
where
    D: DimAPI,
{
    pub fn assign_f<TRB>(&mut self, b: TRB) -> Result<()>
    where
        Self: TensorAssignAPI<TRB>,
    {
        assign_f(self, b)
    }

    pub fn assign<TRB>(&mut self, b: TRB)
    where
        Self: TensorAssignAPI<TRB>,
    {
        assign(self, b)
    }
}

/* #endregion */

/* #region fill */

/// API trait backing [`fill`].
pub trait TensorFillAPI<T> {
    fn fill_f(a: &mut Self, b: T) -> Result<()>;
    fn fill(a: &mut Self, b: T) {
        Self::fill_f(a, b).rstsr_unwrap()
    }
}

pub fn fill_f<TRA, T>(a: &mut TRA, b: T) -> Result<()>
where
    TRA: TensorFillAPI<T>,
{
    TRA::fill_f(a, b)
}

/// Fill the tensor with a scalar value.
///
/// Every element of `a` is set to `b` (cast from `TB` to `TA` as needed). `a`
/// must not be a broadcasted (read-only) view.
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
///
/// # Parameters
///
/// - `a`: the destination tensor (mutable).
/// - `b`: the scalar value to fill with.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let mut d: Tensor<i32, _> = rt::zeros(([2, 2], &device));
/// d.fill(7);
/// println!("{d}");
/// // [[ 7 7]
/// //  [ 7 7]]
/// # assert_eq!(format!("{d}"), "[[ 7 7]\n [ 7 7]]");
/// ```
///
/// # Notes of API accordance
///
/// - NumPy: `a.fill(value)` ([`ndarray.fill`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.fill.html))
/// - RSTSR: `rt::fill(&mut a, v)` or method `a.fill(v)`.
///
/// # Panics
///
/// - Panics if `a` is a broadcasted (read-only) view.
///
/// For a fallible version, use [`fill_f`].
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`assign`]: assign a whole tensor.
/// - [`full`]: create a new tensor filled with a value.
///
/// ## Variants of this function
///
/// - [`fill_f`]: fallible version.
/// - [`TensorFillAPI`]: trait backing this function.
/// - Associated methods on [`TensorBase`]: [`TensorBase::fill`] / [`TensorBase::fill_f`].
pub fn fill<TRA, T>(a: &mut TRA, b: T)
where
    TRA: TensorFillAPI<T>,
{
    TRA::fill(a, b)
}

impl<RA, DA, TA, TB, B> TensorFillAPI<TB> for TensorAny<RA, TA, B, DA>
where
    RA: DataMutAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    DA: DimAPI,
    B: DeviceAPI<TA> + OpAssignAPI<TA, DA, TB>,
{
    fn fill_f(a: &mut Self, b: TB) -> Result<()> {
        // check layout
        rstsr_assert!(!a.layout().is_broadcasted(), InvalidLayout, "cannot fill broadcasted tensor")?;
        let la = a.layout().clone();
        let device = a.device().clone();
        device.fill(a.raw_mut(), &la, b)
    }
}

impl<S, D> TensorBase<S, D>
where
    D: DimAPI,
{
    pub fn fill_f<T>(&mut self, b: T) -> Result<()>
    where
        Self: TensorFillAPI<T>,
    {
        fill_f(self, b)
    }

    pub fn fill<T>(&mut self, b: T)
    where
        Self: TensorFillAPI<T>,
    {
        fill(self, b)
    }
}

/* #endregion */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assign_with_cast() {
        let mut device = DeviceCpuSerial::default();
        device.set_default_order(RowMajor);
        let mut a: Tensor<f32, _> = zeros(([2, 3], &device));
        let b = arange((6i32, &device)).into_shape((2, 3));
        a.assign(&b);
        assert_eq!(a.raw(), &vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]);

        let c: i32 = 10;
        a.fill(c);
        assert_eq!(a.raw(), &vec![10.0f32; 6]);
    }

    #[test]
    #[cfg(feature = "faer")]
    fn test_assign_with_cast_faer() {
        let mut device = DeviceFaer::default();
        device.set_default_order(RowMajor);
        let mut a: Tensor<f32, _> = zeros(([2, 3], &device));
        let b = arange((6i32, &device)).into_shape((2, 3));
        a.assign(&b);
        assert_eq!(a.raw(), &vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]);

        let c: i32 = 10;
        a.fill(c);
        assert_eq!(a.raw(), &vec![10.0f32; 6]);
    }
}
