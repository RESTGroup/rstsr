#[cfg(test)]
mod test {
    use rstsr_core::prelude_dev::*;
    use rstsr_openblas::DeviceOpenBLAS as DeviceBLAS;
    use rstsr_test_manifest::get_vec;

    #[test]
    pub fn test_argmin() {
        let device = DeviceBLAS::default();
        let la = [1024, 4096].c();
        let lb = [128, 129, 130].c();
        let a = Tensor::new(Storage::new(get_vec::<f64>('a').into(), device.clone()), la);
        let b = Tensor::new(Storage::new(get_vec::<f64>('b').into(), device.clone()), lb);
        let argmin_a = a.argmin_all();
        assert_eq!(argmin_a, 3772851);
        let unraveled_argmin_b = b.unraveled_argmin_all();
        assert_eq!(unraveled_argmin_b, [2, 7, 2]);
        let argmax_a = a.argmax_all();
        assert_eq!(argmax_a, 3706286);
        let unraveled_argmax_b = b.unraveled_argmax_all();
        assert_eq!(unraveled_argmax_b, [46, 63, 15]);
    }
}
