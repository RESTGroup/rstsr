use crate::prelude_dev::*;

/* #region op_func */

#[allow(non_camel_case_types)]
#[allow(clippy::too_many_arguments)]
pub trait DeviceOp_MutC_RefA_RefB_API<TA, TB, TC, D, F>
where
    D: DimAPI,
    F: FnMut(&mut MaybeUninit<TC>, &TA, &TB) + ?Sized,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<MaybeUninit<TC>>,
{
    fn op_mutc_refa_refb_func(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<TC>>>::Raw,
        lc: &Layout<D>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<D>,
        f: &mut F,
    ) -> Result<()>;
}

#[allow(non_camel_case_types)]
pub trait DeviceOp_MutC_RefA_NumB_API<TA, TB, TC, D, F>
where
    D: DimAPI,
    F: FnMut(&mut MaybeUninit<TC>, &TA, &TB) + ?Sized,
    Self: DeviceAPI<TA> + DeviceAPI<MaybeUninit<TC>>,
{
    fn op_mutc_refa_numb_func(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<TC>>>::Raw,
        lc: &Layout<D>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
        b: TB,
        f: &mut F,
    ) -> Result<()>;
}

#[allow(non_camel_case_types)]
pub trait DeviceOp_MutC_NumA_RefB_API<TA, TB, TC, D, F>
where
    D: DimAPI,
    F: FnMut(&mut MaybeUninit<TC>, &TA, &TB) + ?Sized,
    Self: DeviceAPI<TB> + DeviceAPI<MaybeUninit<TC>>,
{
    fn op_mutc_numa_refb_func(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<TC>>>::Raw,
        lc: &Layout<D>,
        a: TA,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<D>,
        f: &mut F,
    ) -> Result<()>;
}

#[allow(non_camel_case_types)]
pub trait DeviceOp_MutA_RefB_API<TA, TB, D, F>
where
    D: DimAPI,
    F: FnMut(&mut MaybeUninit<TA>, &TB) + ?Sized,
    Self: DeviceAPI<MaybeUninit<TA>> + DeviceAPI<TB>,
{
    fn op_muta_refb_func(
        &self,
        a: &mut <Self as DeviceRawAPI<MaybeUninit<TA>>>::Raw,
        la: &Layout<D>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<D>,
        f: &mut F,
    ) -> Result<()>;
}

#[allow(non_camel_case_types)]
pub trait DeviceOp_MutA_NumB_API<TA, TB, D, F>
where
    D: DimAPI,
    F: FnMut(&mut MaybeUninit<TA>, &TB) + ?Sized,
    Self: DeviceAPI<MaybeUninit<TA>>,
{
    fn op_muta_numb_func(
        &self,
        a: &mut <Self as DeviceRawAPI<MaybeUninit<TA>>>::Raw,
        la: &Layout<D>,
        b: TB,
        f: &mut F,
    ) -> Result<()>;
}

#[allow(non_camel_case_types)]
pub trait DeviceOp_MutA_API<T, D, F>
where
    D: DimAPI,
    F: FnMut(&mut MaybeUninit<T>) + ?Sized,
    Self: DeviceAPI<MaybeUninit<T>>,
{
    fn op_muta_func(
        &self,
        a: &mut <Self as DeviceRawAPI<MaybeUninit<T>>>::Raw,
        la: &Layout<D>,
        f: &mut F,
    ) -> Result<()>;
}

/* #endregion */
