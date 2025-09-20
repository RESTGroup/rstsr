use crate::prelude_dev::*;
use core::ops::Div;
use num::complex::ComplexFloat;
use num::{Float, Num, Signed};
use rstsr_dtype_traits::{AbsAPI, ReImAPI};

// TODO: log1p

/* #region same type */

#[duplicate_item(
     DeviceOpAPI       NumTrait       func                                         func_inplace                   ;
    [DeviceAcosAPI  ] [ComplexFloat] [|a, b| { a.write(b.acos()             ); }] [|a| unsafe { a.write(a.assume_init_read().acos()                ); }];
    [DeviceAcoshAPI ] [ComplexFloat] [|a, b| { a.write(b.acosh()            ); }] [|a| unsafe { a.write(a.assume_init_read().acosh()               ); }];
    [DeviceAsinAPI  ] [ComplexFloat] [|a, b| { a.write(b.asin()             ); }] [|a| unsafe { a.write(a.assume_init_read().asin()                ); }];
    [DeviceAsinhAPI ] [ComplexFloat] [|a, b| { a.write(b.asinh()            ); }] [|a| unsafe { a.write(a.assume_init_read().asinh()               ); }];
    [DeviceAtanAPI  ] [ComplexFloat] [|a, b| { a.write(b.atan()             ); }] [|a| unsafe { a.write(a.assume_init_read().atan()                ); }];
    [DeviceAtanhAPI ] [ComplexFloat] [|a, b| { a.write(b.atanh()            ); }] [|a| unsafe { a.write(a.assume_init_read().atanh()               ); }];
    [DeviceCeilAPI  ] [Float       ] [|a, b| { a.write(b.ceil()             ); }] [|a| unsafe { a.write(a.assume_init_read().ceil()                ); }];
    [DeviceConjAPI  ] [ComplexFloat] [|a, b| { a.write(b.conj()             ); }] [|a| unsafe { a.write(a.assume_init_read().conj()                ); }];
    [DeviceCosAPI   ] [ComplexFloat] [|a, b| { a.write(b.cos()              ); }] [|a| unsafe { a.write(a.assume_init_read().cos()                 ); }];
    [DeviceCoshAPI  ] [ComplexFloat] [|a, b| { a.write(b.cosh()             ); }] [|a| unsafe { a.write(a.assume_init_read().cosh()                ); }];
    [DeviceExpAPI   ] [ComplexFloat] [|a, b| { a.write(b.exp()              ); }] [|a| unsafe { a.write(a.assume_init_read().exp()                 ); }];
    [DeviceExpm1API ] [Float       ] [|a, b| { a.write(b.exp_m1()           ); }] [|a| unsafe { a.write(a.assume_init_read().exp_m1()              ); }];
    [DeviceFloorAPI ] [Float       ] [|a, b| { a.write(b.floor()            ); }] [|a| unsafe { a.write(a.assume_init_read().floor()               ); }];
    [DeviceInvAPI   ] [ComplexFloat] [|a, b| { a.write(b.recip()            ); }] [|a| unsafe { a.write(a.assume_init_read().recip()               ); }];
    [DeviceLogAPI   ] [ComplexFloat] [|a, b| { a.write(b.ln()               ); }] [|a| unsafe { a.write(a.assume_init_read().ln()                  ); }];
    [DeviceLog2API  ] [ComplexFloat] [|a, b| { a.write(b.log2()             ); }] [|a| unsafe { a.write(a.assume_init_read().log2()                ); }];
    [DeviceLog10API ] [ComplexFloat] [|a, b| { a.write(b.log10()            ); }] [|a| unsafe { a.write(a.assume_init_read().log10()               ); }];
    [DeviceRoundAPI ] [Float       ] [|a, b| { a.write(b.round()            ); }] [|a| unsafe { a.write(a.assume_init_read().round()               ); }];
    [DeviceSinAPI   ] [ComplexFloat] [|a, b| { a.write(b.sin()              ); }] [|a| unsafe { a.write(a.assume_init_read().sin()                 ); }];
    [DeviceSinhAPI  ] [ComplexFloat] [|a, b| { a.write(b.sinh()             ); }] [|a| unsafe { a.write(a.assume_init_read().sinh()                ); }];
    [DeviceSqrtAPI  ] [ComplexFloat] [|a, b| { a.write(b.sqrt()             ); }] [|a| unsafe { a.write(a.assume_init_read().sqrt()                ); }];
    [DeviceSquareAPI] [Num         ] [|a, b| { a.write(b.clone() * b.clone()); }] [|a| unsafe { a.write(a.assume_init_read() * a.assume_init_read()); }];
    [DeviceTanAPI   ] [ComplexFloat] [|a, b| { a.write(b.tan()              ); }] [|a| unsafe { a.write(a.assume_init_read().tan()                 ); }];
    [DeviceTanhAPI  ] [ComplexFloat] [|a, b| { a.write(b.tanh()             ); }] [|a| unsafe { a.write(a.assume_init_read().tanh()                ); }];
    [DeviceTruncAPI ] [Float       ] [|a, b| { a.write(b.trunc()            ); }] [|a| unsafe { a.write(a.assume_init_read().trunc()               ); }];
)]
impl<T, D> DeviceOpAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync + NumTrait,
    D: DimAPI,
{
    type TOut = T;

    fn op_muta_refb(&self, a: &mut Vec<MaybeUninit<T>>, la: &Layout<D>, b: &Vec<T>, lb: &Layout<D>) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut func)
    }

    fn op_muta(&self, a: &mut Vec<MaybeUninit<T>>, la: &Layout<D>) -> Result<()> {
        self.op_muta_func(a, la, &mut func_inplace)
    }
}

/* #endregion */

/* #region boolean output */

#[duplicate_item(
     DeviceOpAPI         NumTrait       func                        ;
    [DeviceSignBitAPI ] [Signed      ] [|a, b| { a.write(b.is_positive()); }];
    [DeviceIsFiniteAPI] [ComplexFloat] [|a, b| { a.write(b.is_finite()  ); }];
    [DeviceIsInfAPI   ] [ComplexFloat] [|a, b| { a.write(b.is_infinite()); }];
    [DeviceIsNanAPI   ] [ComplexFloat] [|a, b| { a.write(b.is_nan()     ); }];
)]
impl<T, D> DeviceOpAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync + NumTrait,
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
    T: Clone + Send + Sync + AbsAPI,
    T::Out: Send + Sync,
    D: DimAPI,
{
    type TOut = T::Out;

    fn op_muta_refb(&self, a: &mut Vec<MaybeUninit<T::Out>>, la: &Layout<D>, b: &Vec<T>, lb: &Layout<D>) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut |a, b| {
            a.write(b.clone().abs());
        })
    }

    fn op_muta(&self, a: &mut Vec<MaybeUninit<T::Out>>, la: &Layout<D>) -> Result<()> {
        if T::UNCHANGED {
            return Ok(());
        } else if T::SAME_TYPE {
            return self.op_muta_func(a, la, &mut |a| unsafe {
                a.write(a.assume_init_read().abs());
            });
        } else {
            let type_b = core::any::type_name::<T>();
            unreachable!("{:?} is not supported in this function.", type_b);
        }
    }
}

impl<T, D> DeviceImagAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync + ReImAPI,
    T::Out: Send + Sync,
    D: DimAPI,
{
    type TOut = T::Out;

    fn op_muta_refb(&self, a: &mut Vec<MaybeUninit<T::Out>>, la: &Layout<D>, b: &Vec<T>, lb: &Layout<D>) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut |a, b| {
            a.write(b.clone().imag());
        })
    }

    fn op_muta(&self, a: &mut Vec<MaybeUninit<T::Out>>, la: &Layout<D>) -> Result<()> {
        if T::SAME_TYPE {
            return self.op_muta_func(a, la, &mut |a| unsafe {
                a.write(a.assume_init_read().imag());
            });
        } else {
            let type_b = core::any::type_name::<T>();
            unreachable!("{:?} is not supported in this function.", type_b);
        }
    }
}

impl<T, D> DeviceRealAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync + ReImAPI,
    T::Out: Send + Sync,
    D: DimAPI,
{
    type TOut = T::Out;

    fn op_muta_refb(&self, a: &mut Vec<MaybeUninit<T::Out>>, la: &Layout<D>, b: &Vec<T>, lb: &Layout<D>) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut |a, b| {
            a.write(b.clone().real());
        })
    }

    fn op_muta(&self, a: &mut Vec<MaybeUninit<T::Out>>, la: &Layout<D>) -> Result<()> {
        if T::REALIDENT {
            return Ok(());
        } else if T::SAME_TYPE {
            return self.op_muta_func(a, la, &mut |a| unsafe {
                a.write(a.assume_init_read().real());
            });
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
