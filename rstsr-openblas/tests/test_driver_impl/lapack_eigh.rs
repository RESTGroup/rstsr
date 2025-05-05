use rstsr_blas_traits::lapack_eigh::*;
use rstsr_core::prelude_dev::*;
use rstsr_openblas::DeviceOpenBLAS as DeviceBLAS;
use rstsr_test_manifest::get_vec;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test() {
        let device = DeviceBLAS::default();
        let la = [2048, 2048].c();
        let a = Tensor::new(Storage::new(get_vec::<f64>('a').into(), device.clone()), la);

        // DSYEVD
        let driver = DSYEVD::default().a(a.view()).build().unwrap();
        let (w, v) = driver.run().unwrap();
        let v = v.into_owned();
        assert!((fingerprint(&v.abs()) - -15.79761028918105).abs() < 1e-8);
        assert!((fingerprint(&w) - -113.98166489712092).abs() < 1e-8);
    }
}
