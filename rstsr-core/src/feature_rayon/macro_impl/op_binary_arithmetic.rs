#[macro_export]
macro_rules! macro_impl_rayon_op_binary_arithmetic {
    ($Device: ident) => {

use $crate::prelude_dev::*;

macro_rules! impl_op_muta_refb_assign {
    ($DeviceOpAPI:ident, $Op:ident, $func:expr) => {
        impl<TA, TB, D> $DeviceOpAPI<TA, TB, D> for $Device
        where
            TA: Clone + Send + Sync + $Op<TB>,
            TB: Clone + Send + Sync,
            D: DimAPI,
        {
            fn op_muta_refb(
                &self,
                a: &mut Vec<TA>,
                la: &Layout<D>,
                b: &Vec<TB>,
                lb: &Layout<D>,
            ) -> Result<()> {
                self.op_muta_refb_func(a, la, b, lb, &mut $func)
            }

            fn op_muta_numb(&self, a: &mut Vec<TA>, la: &Layout<D>, b: TB) -> Result<()> {
                self.op_muta_numb_func(a, la, b, &mut $func)
            }
        }
    };
}

#[rustfmt::skip]
mod impl_op_muta_refb_assign {
    use super::*;
    use core::ops::*;
    impl_op_muta_refb_assign!(DeviceAddAssignAPI   , AddAssign   , |a, b| *a +=  b.clone());
    impl_op_muta_refb_assign!(DeviceSubAssignAPI   , SubAssign   , |a, b| *a -=  b.clone());
    impl_op_muta_refb_assign!(DeviceMulAssignAPI   , MulAssign   , |a, b| *a *=  b.clone());
    impl_op_muta_refb_assign!(DeviceDivAssignAPI   , DivAssign   , |a, b| *a /=  b.clone());
    impl_op_muta_refb_assign!(DeviceRemAssignAPI   , RemAssign   , |a, b| *a %=  b.clone());
    impl_op_muta_refb_assign!(DeviceBitOrAssignAPI , BitOrAssign , |a, b| *a |=  b.clone());
    impl_op_muta_refb_assign!(DeviceBitAndAssignAPI, BitAndAssign, |a, b| *a &=  b.clone());
    impl_op_muta_refb_assign!(DeviceBitXorAssignAPI, BitXorAssign, |a, b| *a ^=  b.clone());
    impl_op_muta_refb_assign!(DeviceShlAssignAPI   , ShlAssign   , |a, b| *a <<= b.clone());
    impl_op_muta_refb_assign!(DeviceShrAssignAPI   , ShrAssign   , |a, b| *a >>= b.clone());
}

macro_rules! impl_op_muta_refb_l_consume {
    ($DeviceOpAPI:ident, $Op:ident, $func:expr) => {
        impl<TA, TB, D> $DeviceOpAPI<TA, TB, D> for $Device
        where
            TA: Clone + Send + Sync + $Op<TB, Output = TA>,
            TB: Clone + Send + Sync,
            D: DimAPI,
        {
            fn op_muta_refb(
                &self,
                a: &mut Vec<TA>,
                la: &Layout<D>,
                b: &Vec<TB>,
                lb: &Layout<D>,
            ) -> Result<()> {
                self.op_muta_refb_func(a, la, b, lb, &mut $func)
            }

            fn op_muta_numb(&self, a: &mut Vec<TA>, la: &Layout<D>, b: TB) -> Result<()> {
                self.op_muta_numb_func(a, la, b, &mut $func)
            }
        }
    };
}

#[rustfmt::skip]
mod impl_op_muta_refb_l_consume {
    use super::*;
    use core::ops::*;
    impl_op_muta_refb_l_consume!(DeviceLConsumeAddAPI   , Add   , |a, b| *a = a.clone() +  b.clone());
    impl_op_muta_refb_l_consume!(DeviceLConsumeSubAPI   , Sub   , |a, b| *a = a.clone() -  b.clone());
    impl_op_muta_refb_l_consume!(DeviceLConsumeMulAPI   , Mul   , |a, b| *a = a.clone() *  b.clone());
    impl_op_muta_refb_l_consume!(DeviceLConsumeDivAPI   , Div   , |a, b| *a = a.clone() /  b.clone());
    impl_op_muta_refb_l_consume!(DeviceLConsumeRemAPI   , Rem   , |a, b| *a = a.clone() %  b.clone());
    impl_op_muta_refb_l_consume!(DeviceLConsumeBitOrAPI , BitOr , |a, b| *a = a.clone() |  b.clone());
    impl_op_muta_refb_l_consume!(DeviceLConsumeBitAndAPI, BitAnd, |a, b| *a = a.clone() &  b.clone());
    impl_op_muta_refb_l_consume!(DeviceLConsumeBitXorAPI, BitXor, |a, b| *a = a.clone() ^  b.clone());
    impl_op_muta_refb_l_consume!(DeviceLConsumeShlAPI   , Shl   , |a, b| *a = a.clone() << b.clone());
    impl_op_muta_refb_l_consume!(DeviceLConsumeShrAPI   , Shr   , |a, b| *a = a.clone() >> b.clone());
}

macro_rules! impl_op_muta_refb_r_consume {
    ($DeviceOpAPI:ident, $Op:ident, $func:expr) => {
        impl<TA, TB, D> $DeviceOpAPI<TA, TB, D> for $Device
        where
            TA: Clone + Send + Sync + $Op<TB, Output = TB>,
            TB: Clone + Send + Sync,
            D: DimAPI,
        {
            fn op_muta_refb(
                &self,
                b: &mut Vec<TB>,
                lb: &Layout<D>,
                a: &Vec<TA>,
                la: &Layout<D>,
            ) -> Result<()> {
                self.op_muta_refb_func(b, lb, a, la, &mut $func)
            }

            fn op_muta_numb(&self, b: &mut Vec<TB>, lb: &Layout<D>, a: TA) -> Result<()> {
                self.op_muta_numb_func(b, lb, a, &mut $func)
            }
        }
    };
}

#[rustfmt::skip]
mod impl_op_muta_refb_r_consume {
    use super::*;
    use core::ops::*;
    impl_op_muta_refb_r_consume!(DeviceRConsumeAddAPI   , Add   , |a, b| *a = b.clone() +  a.clone());
    impl_op_muta_refb_r_consume!(DeviceRConsumeSubAPI   , Sub   , |a, b| *a = b.clone() -  a.clone());
    impl_op_muta_refb_r_consume!(DeviceRConsumeMulAPI   , Mul   , |a, b| *a = b.clone() *  a.clone());
    impl_op_muta_refb_r_consume!(DeviceRConsumeDivAPI   , Div   , |a, b| *a = b.clone() /  a.clone());
    impl_op_muta_refb_r_consume!(DeviceRConsumeRemAPI   , Rem   , |a, b| *a = b.clone() %  a.clone());
    impl_op_muta_refb_r_consume!(DeviceRConsumeBitOrAPI , BitOr , |a, b| *a = b.clone() |  a.clone());
    impl_op_muta_refb_r_consume!(DeviceRConsumeBitAndAPI, BitAnd, |a, b| *a = b.clone() &  a.clone());
    impl_op_muta_refb_r_consume!(DeviceRConsumeBitXorAPI, BitXor, |a, b| *a = b.clone() ^  a.clone());
    impl_op_muta_refb_r_consume!(DeviceRConsumeShlAPI   , Shl   , |a, b| *a = b.clone() << a.clone());
    impl_op_muta_refb_r_consume!(DeviceRConsumeShrAPI   , Shr   , |a, b| *a = b.clone() >> a.clone());
}

macro_rules! impl_op_muta_refb_unary {
    ($DeviceOpAPI:ident, $Op:ident, $op_muta_refb_func:ident, $func:expr, $func_inplace:expr) => {
        impl<TA, TB, D> $DeviceOpAPI<TA, TB, D> for $Device
        where
            TA: Clone + Send + Sync,
            TB: Clone + Send + Sync,
            D: DimAPI,
        {
            fn op_muta_refb(
                &self,
                a: &mut Vec<TA>,
                la: &Layout<D>,
                b: &Vec<TB>,
                lb: &Layout<D>,
            ) -> Result<()>
            where
                TB: $Op<Output = TA>,
            {
                self.op_muta_refb_func(a, la, b, lb, &mut $func)
            }

            fn op_muta(&self, a: &mut Vec<TA>, la: &Layout<D>) -> Result<()>
            where
                TA: $Op<Output = TA>,
            {
                self.op_muta_func(a, la, &mut $func_inplace)
            }
        }
    };
}

#[rustfmt::skip]
mod impl_op_muta_refb_unary {
    use super::*;
    use core::ops::*;
    impl_op_muta_refb_unary!(DeviceNegAPI, Neg, op_muta_refb_neg, |a, b| *a = -b.clone(), |a| *a = -a.clone());
    impl_op_muta_refb_unary!(DeviceNotAPI, Not, op_muta_refb_not, |a, b| *a = !b.clone(), |a| *a = !a.clone());
}

    };
}
