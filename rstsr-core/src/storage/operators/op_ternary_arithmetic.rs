use crate::prelude_dev::*;

macro_rules! impl_op_mutc_refa_refb_operator {
    ($DeviceOpAPI:ident, $Op:ident) => {
        pub trait $DeviceOpAPI<TA, TB, TC, D>
        where
            TA: $Op<TB, Output = TC>,
            D: DimAPI,
            Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
        {
            fn op_mutc_refa_refb(
                &self,
                c: &mut Storage<TC, Self>,
                lc: &Layout<D>,
                a: &Storage<TA, Self>,
                la: &Layout<D>,
                b: &Storage<TB, Self>,
                lb: &Layout<D>,
            ) -> Result<()>;

            fn op_mutc_refa_numb(
                &self,
                c: &mut Storage<TC, Self>,
                lc: &Layout<D>,
                a: &Storage<TA, Self>,
                la: &Layout<D>,
                b: TB,
            ) -> Result<()>;

            fn op_mutc_numa_refb(
                &self,
                c: &mut Storage<TC, Self>,
                lc: &Layout<D>,
                a: TA,
                b: &Storage<TB, Self>,
                lb: &Layout<D>,
            ) -> Result<()>;
        }
    };
}

#[rustfmt::skip]
mod impl_op_mutc_refa_refb_operator {
    use super::*;
    use core::ops::*;
    impl_op_mutc_refa_refb_operator!(DeviceAddAPI   , Add   );
    impl_op_mutc_refa_refb_operator!(DeviceSubAPI   , Sub   );
    impl_op_mutc_refa_refb_operator!(DeviceMulAPI   , Mul   );
    impl_op_mutc_refa_refb_operator!(DeviceDivAPI   , Div   );
    impl_op_mutc_refa_refb_operator!(DeviceRemAPI   , Rem   );
    impl_op_mutc_refa_refb_operator!(DeviceBitOrAPI , BitOr );
    impl_op_mutc_refa_refb_operator!(DeviceBitAndAPI, BitAnd);
    impl_op_mutc_refa_refb_operator!(DeviceBitXorAPI, BitXor);
    impl_op_mutc_refa_refb_operator!(DeviceShlAPI   , Shl   );
    impl_op_mutc_refa_refb_operator!(DeviceShrAPI   , Shr   );
}
pub use impl_op_mutc_refa_refb_operator::*;
