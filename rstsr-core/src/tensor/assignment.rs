use crate::prelude_dev::*;

pub trait TensorAssignAPI<TRB>: Sized {
    fn assign_f(a: &mut Self, b: TRB) -> Result<()>;
    fn assign(a: &mut Self, b: TRB) {
        Self::assign_f(a, b).unwrap()
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

impl<RA, DA, RB, DB, T, B> TensorAssignAPI<TensorBase<RB, DB>> for TensorBase<RA, DA>
where
    RA: DataMutAPI<Data = Storage<T, B>>,
    RB: DataAPI<Data = Storage<T, B>>,
    DA: DimAPI,
    DB: DimAPI,
    B: DeviceAPI<T> + OpAssignAPI<T, DA>,
{
    fn assign_f(a: &mut Self, b: TensorBase<RB, DB>) -> Result<()> {
        // get tensor views
        let mut a = a.view_mut();
        let b = b.view();
        // check device
        rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
        let device = a.device().clone();
        // check layout
        rstsr_assert!(
            !a.layout().is_broadcasted(),
            InvalidLayout,
            "cannot assign to broadcasted tensor"
        )?;
        let la = a.layout().to_dim::<IxD>()?;
        let lb = b.layout().to_dim::<IxD>()?;
        let (la_b, lb_b) = broadcast_layout_to_first(&la, &lb)?;
        let la_b = la_b.into_dim::<DA>()?;
        let lb_b = lb_b.into_dim::<DA>()?;
        // assign
        device.assign(a.data_mut().storage_mut(), &la_b, b.data().storage(), &lb_b)
    }
}
