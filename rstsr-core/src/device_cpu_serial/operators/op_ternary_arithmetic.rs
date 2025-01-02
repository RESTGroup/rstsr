use crate::prelude_dev::*;

macro_rules! impl_op_mutc_refa_refb_operator {
    ($DeviceOpAPI:ident, $Op:ident, $func:expr) => {
        impl<TA, TB, TC, D> $DeviceOpAPI<TA, TB, TC, D> for DeviceCpuSerial
        where
            TA: Clone + $Op<TB, Output = TC>,
            TB: Clone,
            TC: Clone,
            D: DimAPI,
        {
            fn op_mutc_refa_refb(
                &self,
                c: &mut Storage<TC, Self>,
                lc: &Layout<D>,
                a: &Storage<TA, Self>,
                la: &Layout<D>,
                b: &Storage<TB, Self>,
                lb: &Layout<D>,
            ) -> Result<()> {
                self.op_mutc_refa_refb_func(c, lc, a, la, b, lb, &mut $func)
            }

            fn op_mutc_refa_numb(
                &self,
                c: &mut Storage<TC, Self>,
                lc: &Layout<D>,
                a: &Storage<TA, Self>,
                la: &Layout<D>,
                b: TB,
            ) -> Result<()> {
                self.op_mutc_refa_numb_func(c, lc, a, la, b, &mut $func)
            }

            fn op_mutc_numa_refb(
                &self,
                c: &mut Storage<TC, Self>,
                lc: &Layout<D>,
                a: TA,
                b: &Storage<TB, Self>,
                lb: &Layout<D>,
            ) -> Result<()> {
                self.op_mutc_numa_refb_func(c, lc, a, b, lb, &mut $func)
            }
        }
    };
}

#[rustfmt::skip]
mod impl_op_mutc_refa_refb_operator {
    use super::*;
    use core::ops::*;
    impl_op_mutc_refa_refb_operator!(DeviceAddAPI   , Add   , |c, a, b| *c = a.clone() +  b.clone());
    impl_op_mutc_refa_refb_operator!(DeviceSubAPI   , Sub   , |c, a, b| *c = a.clone() -  b.clone());
    impl_op_mutc_refa_refb_operator!(DeviceMulAPI   , Mul   , |c, a, b| *c = a.clone() *  b.clone());
    impl_op_mutc_refa_refb_operator!(DeviceDivAPI   , Div   , |c, a, b| *c = a.clone() /  b.clone());
    impl_op_mutc_refa_refb_operator!(DeviceRemAPI   , Rem   , |c, a, b| *c = a.clone() %  b.clone());
    impl_op_mutc_refa_refb_operator!(DeviceBitOrAPI , BitOr , |c, a, b| *c = a.clone() |  b.clone());
    impl_op_mutc_refa_refb_operator!(DeviceBitAndAPI, BitAnd, |c, a, b| *c = a.clone() &  b.clone());
    impl_op_mutc_refa_refb_operator!(DeviceBitXorAPI, BitXor, |c, a, b| *c = a.clone() ^  b.clone());
    impl_op_mutc_refa_refb_operator!(DeviceShlAPI   , Shl   , |c, a, b| *c = a.clone() << b.clone());
    impl_op_mutc_refa_refb_operator!(DeviceShrAPI   , Shr   , |c, a, b| *c = a.clone() >> b.clone());
}
