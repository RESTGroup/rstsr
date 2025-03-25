use crate::prelude_dev::*;
use core::ops::Div;
use num::complex::ComplexFloat;
use num::{Float, Num, Signed};
use rstsr_dtype_traits::{AbsAPI, ReImAPI};

// TODO: log1p

/* #region same type */

#[duplicate_item(
     DeviceOpAPI       NumTrait       func                                func_inplace                   ;
    [DeviceAcosAPI  ] [ComplexFloat] [|a, b| *a = b.acos()             ] [|a| *a = a.acos()             ];
    [DeviceAcoshAPI ] [ComplexFloat] [|a, b| *a = b.acosh()            ] [|a| *a = a.acosh()            ];
    [DeviceAsinAPI  ] [ComplexFloat] [|a, b| *a = b.asin()             ] [|a| *a = a.asin()             ];
    [DeviceAsinhAPI ] [ComplexFloat] [|a, b| *a = b.asinh()            ] [|a| *a = a.asinh()            ];
    [DeviceAtanAPI  ] [ComplexFloat] [|a, b| *a = b.atan()             ] [|a| *a = a.atan()             ];
    [DeviceAtanhAPI ] [ComplexFloat] [|a, b| *a = b.atanh()            ] [|a| *a = a.atanh()            ];
    [DeviceCeilAPI  ] [Float       ] [|a, b| *a = b.ceil()             ] [|a| *a = a.ceil()             ];
    [DeviceConjAPI  ] [ComplexFloat] [|a, b| *a = b.conj()             ] [|a| *a = a.conj()             ];
    [DeviceCosAPI   ] [ComplexFloat] [|a, b| *a = b.cos()              ] [|a| *a = a.cos()              ];
    [DeviceCoshAPI  ] [ComplexFloat] [|a, b| *a = b.cosh()             ] [|a| *a = a.cosh()             ];
    [DeviceExpAPI   ] [ComplexFloat] [|a, b| *a = b.exp()              ] [|a| *a = a.exp()              ];
    [DeviceExpm1API ] [Float       ] [|a, b| *a = b.exp_m1()           ] [|a| *a = a.exp_m1()           ];
    [DeviceFloorAPI ] [Float       ] [|a, b| *a = b.floor()            ] [|a| *a = a.floor()            ];
    [DeviceInvAPI   ] [ComplexFloat] [|a, b| *a = b.recip()            ] [|a| *a = a.recip()            ];
    [DeviceLogAPI   ] [ComplexFloat] [|a, b| *a = b.ln()               ] [|a| *a = a.ln()               ];
    [DeviceLog2API  ] [ComplexFloat] [|a, b| *a = b.log2()             ] [|a| *a = a.log2()             ];
    [DeviceLog10API ] [ComplexFloat] [|a, b| *a = b.log10()            ] [|a| *a = a.log10()            ];
    [DeviceRoundAPI ] [Float       ] [|a, b| *a = b.round()            ] [|a| *a = a.round()            ];
    [DeviceSinAPI   ] [ComplexFloat] [|a, b| *a = b.sin()              ] [|a| *a = a.sin()              ];
    [DeviceSinhAPI  ] [ComplexFloat] [|a, b| *a = b.sinh()             ] [|a| *a = a.sinh()             ];
    [DeviceSqrtAPI  ] [ComplexFloat] [|a, b| *a = b.sqrt()             ] [|a| *a = a.sqrt()             ];
    [DeviceSquareAPI] [Num         ] [|a, b| *a = b.clone() * b.clone()] [|a| *a = a.clone() * a.clone()];
    [DeviceTanAPI   ] [ComplexFloat] [|a, b| *a = b.tan()              ] [|a| *a = a.tan()              ];
    [DeviceTanhAPI  ] [ComplexFloat] [|a, b| *a = b.tanh()             ] [|a| *a = a.tanh()             ];
    [DeviceTruncAPI ] [Float       ] [|a, b| *a = b.trunc()            ] [|a| *a = a.trunc()            ];
)]
impl<T, D> DeviceOpAPI<T, D> for DeviceCpuSerial
where
    T: Clone + NumTrait,
    D: DimAPI,
{
    type TOut = T;

    fn op_muta_refb(
        &self,
        a: &mut Vec<T>,
        la: &Layout<D>,
        b: &Vec<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut func)
    }

    fn op_muta(&self, a: &mut Vec<T>, la: &Layout<D>) -> Result<()> {
        self.op_muta_func(a, la, &mut func_inplace)
    }
}

/* #endregion */

/* #region boolean output */

#[duplicate_item(
     DeviceOpAPI         NumTrait       func                         ;
    [DeviceSignBitAPI ] [Signed      ] [|a, b| *a = b.is_positive() ];
    [DeviceIsFiniteAPI] [ComplexFloat] [|a, b| *a = b.is_finite()   ];
    [DeviceIsInfAPI   ] [ComplexFloat] [|a, b| *a = b.is_infinite() ];
    [DeviceIsNanAPI   ] [ComplexFloat] [|a, b| *a = b.is_nan()      ];
)]
impl<T, D> DeviceOpAPI<T, D> for DeviceCpuSerial
where
    T: Clone + NumTrait,
    D: DimAPI,
{
    type TOut = bool;

    fn op_muta_refb(
        &self,
        a: &mut Vec<bool>,
        la: &Layout<D>,
        b: &Vec<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut func)
    }

    fn op_muta(&self, _a: &mut Vec<bool>, _la: &Layout<D>) -> Result<()> {
        let type_b = core::any::type_name::<T>();
        unreachable!("{:?} is not supported in this function.", type_b);
    }
}

/* #endregion */

/* #region complex specific implementation */

impl<T, D> DeviceAbsAPI<T, D> for DeviceCpuSerial
where
    T: Clone + AbsAPI,
    D: DimAPI,
{
    type TOut = T::Out;

    fn op_muta_refb(
        &self,
        a: &mut Vec<T::Out>,
        la: &Layout<D>,
        b: &Vec<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut |a, b| *a = b.clone().abs())
    }

    fn op_muta(&self, a: &mut Vec<T::Out>, la: &Layout<D>) -> Result<()> {
        if T::UNCHANGED {
            return Ok(());
        } else if T::SAME_TYPE {
            return self.op_muta_func(a, la, &mut |a| *a = a.clone().abs());
        } else {
            let type_b = core::any::type_name::<T>();
            unreachable!("{:?} is not supported in this function.", type_b);
        }
    }
}

impl<T, D> DeviceImagAPI<T, D> for DeviceCpuSerial
where
    T: Clone + ReImAPI,
    D: DimAPI,
{
    type TOut = T::Out;

    fn op_muta_refb(
        &self,
        a: &mut Vec<T::Out>,
        la: &Layout<D>,
        b: &Vec<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut |a, b| *a = b.clone().imag())
    }

    fn op_muta(&self, a: &mut Vec<T::Out>, la: &Layout<D>) -> Result<()> {
        if T::SAME_TYPE {
            return self.op_muta_func(a, la, &mut |a| *a = a.clone().imag());
        } else {
            let type_b = core::any::type_name::<T>();
            unreachable!("{:?} is not supported in this function.", type_b);
        }
    }
}

impl<T, D> DeviceRealAPI<T, D> for DeviceCpuSerial
where
    T: Clone + ReImAPI,
    D: DimAPI,
{
    type TOut = T::Out;

    fn op_muta_refb(
        &self,
        a: &mut Vec<T::Out>,
        la: &Layout<D>,
        b: &Vec<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut |a, b| *a = b.clone().real())
    }

    fn op_muta(&self, a: &mut Vec<T::Out>, la: &Layout<D>) -> Result<()> {
        if T::REALIDENT {
            return Ok(());
        } else if T::SAME_TYPE {
            return self.op_muta_func(a, la, &mut |a| *a = a.clone().real());
        } else {
            let type_b = core::any::type_name::<T>();
            unreachable!("{:?} is not supported in this function.", type_b);
        }
    }
}

impl<T, D> DeviceSignAPI<T, D> for DeviceCpuSerial
where
    T: Clone + ComplexFloat + Div<T::Real, Output = T>,
    D: DimAPI,
{
    type TOut = T;

    fn op_muta_refb(
        &self,
        a: &mut Vec<T>,
        la: &Layout<D>,
        b: &Vec<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut |a, b| *a = *b / b.abs())
    }

    fn op_muta(&self, a: &mut Vec<T>, la: &Layout<D>) -> Result<()> {
        self.op_muta_func(a, la, &mut |a| *a = *a / a.abs())
    }
}

/* #endregion */
