use crate::prelude_dev::*;

macro_rules! impl_binary_with_output {
    ($op: ident, $DeviceOpAPI: ident, $Op: ident) => {
        pub fn $op<TRA, TRB, TRC, TA, TB, TC, DA, DB, DC, B>(
            a: TRA,
            b: TRB,
            mut c: TRC,
        ) -> Result<()>
        where
            // tensor types
            TRA: TensorRefOrOwnedAPI<Storage<TA, B>, DA>,
            TRB: TensorRefOrOwnedAPI<Storage<TB, B>, DB>,
            TRC: TensorRefMutAPI<Storage<TC, B>, DC>,
            // data constraints
            DA: DimAPI,
            DB: DimAPI,
            DC: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
            // broadcast constraints
            DC: DimMaxAPI<DA, Max = DC> + DimMaxAPI<DB, Max = DC>,
            // operation constraints
            TA: $Op<TB, Output = TC>,
            B: $DeviceOpAPI<TA, TB, TC, DC>,
        {
            // get tensor views
            let a = a.tsr_view();
            let b = b.tsr_view();
            let mut c = c.tsr_view_mut();
            // check device
            rstsr_assert!(c.device().same_device(a.device()), DeviceMismatch)?;
            rstsr_assert!(c.device().same_device(b.device()), DeviceMismatch)?;
            let lc = c.layout();
            let la = a.layout();
            let lb = b.layout();
            // all layouts should be broadcastable to lc
            // we can first generate broadcasted shape, then check this
            let (lc_b, la_b) = broadcast_layout_to_first(lc, la)?;
            rstsr_assert_eq!(lc_b, *lc, InvalidLayout)?;
            let (lc_b, lb_b) = broadcast_layout_to_first(lc, lb)?;
            rstsr_assert_eq!(lc_b, *lc, InvalidLayout)?;
            // op provided by device
            let device = c.device().clone();
            let storage_c = c.data_mut().storage_mut();
            let storage_a = a.data().storage();
            let storage_b = b.data().storage();
            device.op_mutc_refa_refb(storage_c, &lc_b, storage_a, &la_b, storage_b, &lb_b)
        }
    };
}

#[rustfmt::skip]
mod impl_binary_with_output{
    use super::*;
    use core::ops::*;
    impl_binary_with_output!(   add_with_output, DeviceAddAPI   , Add   );
    impl_binary_with_output!(   sub_with_output, DeviceSubAPI   , Sub   );
    impl_binary_with_output!(   mul_with_output, DeviceMulAPI   , Mul   );
    impl_binary_with_output!(   div_with_output, DeviceDivAPI   , Div   );
    impl_binary_with_output!(   rem_with_output, DeviceRemAPI   , Rem   );
    impl_binary_with_output!( bitor_with_output, DeviceBitOrAPI , BitOr );
    impl_binary_with_output!(bitand_with_output, DeviceBitAndAPI, BitAnd);
    impl_binary_with_output!(bitxor_with_output, DeviceBitXorAPI, BitXor);
    impl_binary_with_output!(   shl_with_output, DeviceShlAPI   , Shl   );
    impl_binary_with_output!(   shr_with_output, DeviceShrAPI   , Shr   );
}
pub use impl_binary_with_output::*;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_op_binary_with_output() {
        let a = Tensor::linspace((1.0, 10.0, 10)).into_shape_assume_contig([2, 5]);
        let b = Tensor::linspace((2.0, 10.0, 5));
        let mut c = Tensor::linspace((1.0, 10.0, 10)).into_shape_assume_contig([2, 5]);
        let c_view = c.view_mut();
        add_with_output(&a, b, c_view).unwrap();
        println!("{:?}", c);
    }
}
