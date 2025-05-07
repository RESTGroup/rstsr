use rstsr_core::prelude::*;
use rstsr_core::prelude_dev::fingerprint;
use rstsr_linalg_traits::traits_def::*;
use rstsr_openblas::DeviceOpenBLAS as DeviceBLAS;
use rstsr_test_manifest::get_vec;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_cholesky() {
        let device = DeviceBLAS::default();
        let b = rt::asarray((get_vec::<f64>('b'), [1024, 1024].c(), &device));

        // default
        let c = cholesky(b.view());
        assert!((fingerprint(&c) - 43.21904478556176).abs() < 1e-8);

        let c = cholesky((b.view(), Upper));
        assert!((fingerprint(&c) - -25.925655124816647).abs() < 1e-8);
    }
}
