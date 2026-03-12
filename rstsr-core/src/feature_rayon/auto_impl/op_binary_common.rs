use crate::prelude_dev::*;
use core::ops::Div;
use num::complex::ComplexFloat;
use num::{Float, Signed};
use rstsr_dtype_traits::{DTypeIntoFloatAPI, ExtNum};

// TODO: log1p

/* #region same type */

#[duplicate_item(
     OpAPI             NumTrait       func_inner;
    [OpAcosAPI      ] [ComplexFloat] [b.acos()  ];
    [OpAcoshAPI     ] [ComplexFloat] [b.acosh() ];
    [OpAsinAPI      ] [ComplexFloat] [b.asin()  ];
    [OpAsinhAPI     ] [ComplexFloat] [b.asinh() ];
    [OpAtanAPI      ] [ComplexFloat] [b.atan()  ];
    [OpAtanhAPI     ] [ComplexFloat] [b.atanh() ];
    [OpCeilAPI      ] [Float       ] [b.ceil()  ];
    [OpConjAPI      ] [ComplexFloat] [b.conj()  ];
    [OpCosAPI       ] [ComplexFloat] [b.cos()   ];
    [OpCoshAPI      ] [ComplexFloat] [b.cosh()  ];
    [OpExpAPI       ] [ComplexFloat] [b.exp()   ];
    [OpExpm1API     ] [Float       ] [b.exp_m1()];
    [OpFloorAPI     ] [Float       ] [b.floor() ];
    [OpInvAPI       ] [ComplexFloat] [b.recip() ];
    [OpLogAPI       ] [ComplexFloat] [b.ln()    ];
    [OpLog2API      ] [ComplexFloat] [b.log2()  ];
    [OpLog10API     ] [ComplexFloat] [b.log10() ];
    [OpReciprocalAPI] [ComplexFloat] [b.recip() ];
    [OpRoundAPI     ] [Float       ] [b.round() ];
    [OpSinAPI       ] [ComplexFloat] [b.sin()   ];
    [OpSinhAPI      ] [ComplexFloat] [b.sinh()  ];
    [OpSqrtAPI      ] [ComplexFloat] [b.sqrt()  ];
    [OpTanAPI       ] [ComplexFloat] [b.tan()   ];
    [OpTanhAPI      ] [ComplexFloat] [b.tanh()  ];
    [OpTruncAPI     ] [Float       ] [b.trunc() ];
)]
impl<T, D> OpAPI<T, D> for DeviceRayonAutoImpl
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

impl<T, D> OpSquareAPI<T, D> for DeviceRayonAutoImpl
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
     OpAPI           NumTrait       func                         ;
    [OpSignBitAPI ] [Signed      ] [|a, b| { a.write(b.is_positive()); } ];
    [OpIsFiniteAPI] [ComplexFloat] [|a, b| { a.write(b.is_finite()  ); } ];
    [OpIsInfAPI   ] [ComplexFloat] [|a, b| { a.write(b.is_infinite()); } ];
    [OpIsNanAPI   ] [ComplexFloat] [|a, b| { a.write(b.is_nan()     ); } ];
)]
impl<T, D> OpAPI<T, D> for DeviceRayonAutoImpl
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

impl<T, D> OpAbsAPI<T, D> for DeviceRayonAutoImpl
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

impl<T, D> OpImagAPI<T, D> for DeviceRayonAutoImpl
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

impl<T, D> OpRealAPI<T, D> for DeviceRayonAutoImpl
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

impl<T, D> OpSignAPI<T, D> for DeviceRayonAutoImpl
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
