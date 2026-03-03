use crate::prelude_dev::*;
use core::ops::Div;
use num::complex::ComplexFloat;
use num::{Float, Signed};
use rstsr_dtype_traits::{DTypeIntoFloatAPI, ExtNum};

// TODO: log1p

/* #region same type */

#[duplicate_item(
     DeviceOpAPI           NumTrait       func_inner;
    [DeviceAcosAPI      ] [ComplexFloat] [b.acos()  ];
    [DeviceAcoshAPI     ] [ComplexFloat] [b.acosh() ];
    [DeviceAsinAPI      ] [ComplexFloat] [b.asin()  ];
    [DeviceAsinhAPI     ] [ComplexFloat] [b.asinh() ];
    [DeviceAtanAPI      ] [ComplexFloat] [b.atan()  ];
    [DeviceAtanhAPI     ] [ComplexFloat] [b.atanh() ];
    [DeviceCeilAPI      ] [Float       ] [b.ceil()  ];
    [DeviceConjAPI      ] [ComplexFloat] [b.conj()  ];
    [DeviceCosAPI       ] [ComplexFloat] [b.cos()   ];
    [DeviceCoshAPI      ] [ComplexFloat] [b.cosh()  ];
    [DeviceExpAPI       ] [ComplexFloat] [b.exp()   ];
    [DeviceExpm1API     ] [Float       ] [b.exp_m1()];
    [DeviceFloorAPI     ] [Float       ] [b.floor() ];
    [DeviceInvAPI       ] [ComplexFloat] [b.recip() ];
    [DeviceLogAPI       ] [ComplexFloat] [b.ln()    ];
    [DeviceLog2API      ] [ComplexFloat] [b.log2()  ];
    [DeviceLog10API     ] [ComplexFloat] [b.log10() ];
    [DeviceReciprocalAPI] [ComplexFloat] [b.recip() ];
    [DeviceRoundAPI     ] [Float       ] [b.round() ];
    [DeviceSinAPI       ] [ComplexFloat] [b.sin()   ];
    [DeviceSinhAPI      ] [ComplexFloat] [b.sinh()  ];
    [DeviceSqrtAPI      ] [ComplexFloat] [b.sqrt()  ];
    [DeviceTanAPI       ] [ComplexFloat] [b.tan()   ];
    [DeviceTanhAPI      ] [ComplexFloat] [b.tanh()  ];
    [DeviceTruncAPI     ] [Float       ] [b.trunc() ];
)]
impl<T, D> DeviceOpAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync + DTypeIntoFloatAPI<FloatType: NumTrait + Send + Sync>,
    D: DimAPI,
{
    type TOut = T::FloatType;

    fn op_muta_refb(
        &self,
        a: &mut Vec<MaybeUninit<Self::TOut>>,
        la: &Layout<D>,
        b: &Vec<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let mut func = |a: &mut MaybeUninit<Self::TOut>, b: &T| {
            let b = b.clone().into_float();
            a.write(func_inner);
        };
        self.op_muta_refb_func(a, la, b, lb, &mut func)
    }

    fn op_muta(&self, a: &mut Vec<MaybeUninit<Self::TOut>>, la: &Layout<D>) -> Result<()> {
        let mut func = |a: &mut MaybeUninit<Self::TOut>| {
            let b = unsafe { a.assume_init_read() };
            a.write(func_inner);
        };
        self.op_muta_func(a, la, &mut func)
    }
}

impl<T, D> DeviceSquareAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync + Mul<Output = T>,
    D: DimAPI,
{
    type TOut = T;

    fn op_muta_refb(&self, a: &mut Vec<MaybeUninit<T>>, la: &Layout<D>, b: &Vec<T>, lb: &Layout<D>) -> Result<()> {
        let mut func = |a: &mut MaybeUninit<T>, b: &T| {
            a.write(b.clone() * b.clone());
        };
        self.op_muta_refb_func(a, la, b, lb, &mut func)
    }

    fn op_muta(&self, a: &mut Vec<MaybeUninit<T>>, la: &Layout<D>) -> Result<()> {
        let mut func = |a: &mut MaybeUninit<T>| {
            let b = unsafe { a.assume_init_read() };
            a.write(b.clone() * b);
        };
        self.op_muta_func(a, la, &mut func)
    }
}

/* #endregion */

/* #region boolean output */

#[duplicate_item(
     DeviceOpAPI         NumTrait       func                         ;
    [DeviceSignBitAPI ] [Signed      ] [|a, b| { a.write(b.is_positive()); } ];
    [DeviceIsFiniteAPI] [ComplexFloat] [|a, b| { a.write(b.is_finite()  ); } ];
    [DeviceIsInfAPI   ] [ComplexFloat] [|a, b| { a.write(b.is_infinite()); } ];
    [DeviceIsNanAPI   ] [ComplexFloat] [|a, b| { a.write(b.is_nan()     ); } ];
)]
impl<T, D> DeviceOpAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + NumTrait + Send + Sync,
    D: DimAPI,
{
    type TOut = bool;

    fn op_muta_refb(&self, a: &mut Vec<MaybeUninit<bool>>, la: &Layout<D>, b: &Vec<T>, lb: &Layout<D>) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut func)
    }

    fn op_muta(&self, _a: &mut Vec<MaybeUninit<bool>>, _la: &Layout<D>) -> Result<()> {
        let type_b = core::any::type_name::<T>();
        unreachable!("{:?} is not supported in this function.", type_b);
    }
}

/* #endregion */

/* #region complex specific implementation */

impl<T, D> DeviceAbsAPI<T, D> for DeviceRayonAutoImpl
where
    T: ExtNum + Send + Sync,
    T::AbsOut: Send + Sync,
    D: DimAPI,
{
    type TOut = T::AbsOut;

    fn op_muta_refb(
        &self,
        a: &mut Vec<MaybeUninit<T::AbsOut>>,
        la: &Layout<D>,
        b: &Vec<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut |a, b| {
            a.write(b.clone().ext_abs());
        })
    }

    fn op_muta(&self, a: &mut Vec<MaybeUninit<T::AbsOut>>, la: &Layout<D>) -> Result<()> {
        if T::ABS_UNCHANGED {
            return Ok(());
        } else if T::ABS_SAME_TYPE {
            return self.op_muta_func(a, la, &mut |a| unsafe {
                a.write(a.assume_init_read().ext_abs());
            });
        } else {
            let type_b = core::any::type_name::<T>();
            unreachable!("{:?} is not supported in this function.", type_b);
        }
    }
}

impl<T, D> DeviceImagAPI<T, D> for DeviceRayonAutoImpl
where
    T: ExtNum + Send + Sync,
    T::AbsOut: Send + Sync,
    D: DimAPI,
{
    type TOut = T::AbsOut;

    fn op_muta_refb(
        &self,
        a: &mut Vec<MaybeUninit<T::AbsOut>>,
        la: &Layout<D>,
        b: &Vec<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut |a, b| {
            a.write(b.clone().ext_imag());
        })
    }

    fn op_muta(&self, a: &mut Vec<MaybeUninit<T::AbsOut>>, la: &Layout<D>) -> Result<()> {
        if T::ABS_SAME_TYPE {
            return self.op_muta_func(a, la, &mut |a| unsafe {
                a.write(a.assume_init_read().ext_imag());
            });
        } else {
            let type_b = core::any::type_name::<T>();
            unreachable!("{:?} is not supported in this function.", type_b);
        }
    }
}

impl<T, D> DeviceRealAPI<T, D> for DeviceRayonAutoImpl
where
    T: ExtNum + Send + Sync,
    T::AbsOut: Send + Sync,
    D: DimAPI,
{
    type TOut = T::AbsOut;

    fn op_muta_refb(
        &self,
        a: &mut Vec<MaybeUninit<T::AbsOut>>,
        la: &Layout<D>,
        b: &Vec<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut |a, b| {
            a.write(b.clone().ext_real());
        })
    }

    fn op_muta(&self, _a: &mut Vec<MaybeUninit<T::AbsOut>>, _la: &Layout<D>) -> Result<()> {
        if T::ABS_SAME_TYPE {
            return Ok(());
        } else {
            let type_b = core::any::type_name::<T>();
            unreachable!("{:?} is not supported in this function.", type_b);
        }
    }
}

impl<T, D> DeviceSignAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync + ComplexFloat + Div<T::Real, Output = T>,
    D: DimAPI,
{
    type TOut = T;

    fn op_muta_refb(&self, a: &mut Vec<MaybeUninit<T>>, la: &Layout<D>, b: &Vec<T>, lb: &Layout<D>) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut |a, b| {
            a.write(*b / b.abs());
        })
    }

    fn op_muta(&self, a: &mut Vec<MaybeUninit<T>>, la: &Layout<D>) -> Result<()> {
        self.op_muta_func(a, la, &mut |a| unsafe {
            a.write(a.assume_init_read() / a.assume_init_read().abs());
        })
    }
}

/* #endregion */
