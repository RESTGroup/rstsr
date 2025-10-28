use crate::prelude_dev::*;

/* #region assign */

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
