use crate::prelude_dev::*;

macro_rules! impl_op_mutc_refa_refb_operator {
    ($DeviceOpAPI:ident, $Op:ident, $func:expr) => {
        impl<TA, TB, TC, D> $DeviceOpAPI<TA, TB, TC, D> for DeviceFaer
        where
            TA: Clone + Send + Sync + $Op<TB, Output = TC>,
            TB: Clone + Send + Sync,
            TC: Clone + Send + Sync,
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_add() {
        let device_serial = DeviceCpuSerial;
        let device_faer = DeviceFaer::default();
        let a1 = Tensor::linspace((1., 1024. * 1024., 1024 * 1024, &device_serial));
        let a1 = a1.into_shape_assume_contig_f([1024, 1024]).unwrap();
        let b1 = Tensor::linspace((1., 1024. * 1024., 1024 * 1024, &device_serial));
        let b1 = b1.into_shape_assume_contig_f([1024, 1024]).unwrap().into_reverse_axes();
        let a2 = Tensor::linspace((1., 1024. * 1024., 1024 * 1024, &device_faer));
        let a2 = a2.into_shape_assume_contig_f([1024, 1024]).unwrap();
        let b2 = Tensor::linspace((1., 1024. * 1024., 1024 * 1024, &device_faer));
        let b2 = b2.into_shape_assume_contig_f([1024, 1024]).unwrap().into_reverse_axes();

        let c1 = &a1 + &b1;
        let c2 = &a2 + &b2;
        assert!(allclose_f64(&c1, &c2));
    }
}
