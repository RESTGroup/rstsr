use num::Complex;
use rstsr_blas_traits::lapack_eig::*;
use rstsr_core::prelude::*;
use rstsr_openblas::DeviceOpenBLAS as DeviceBLAS;

#[allow(non_camel_case_types)]
type c64 = Complex<f64>;

#[cfg(test)]
mod test_ggev_c64 {
    use super::*;

    #[test]
    fn test_zggev_simple_colmajor() {
        // Simple 2x2 generalized eigenvalue problem (column-major)
        let device = DeviceBLAS::default();
        let data_a: Vec<c64> = vec![c64::new(1.0, 0.5), c64::new(2.0, -0.3), c64::new(3.0, 0.2), c64::new(4.0, -0.1)];
        let data_b: Vec<c64> = vec![c64::new(2.0, 0.1), c64::new(1.0, -0.2), c64::new(1.0, 0.3), c64::new(2.0, -0.4)];
        let a = rt::asarray((data_a, [2, 2].f(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((data_b, [2, 2].f(), &device)).into_dim::<Ix2>();

        println!("Running column-major test...");
        let driver = GGEV::default().a(a.view()).b(b.view()).left(false).right(true).build().unwrap();
        let (_alpha, _beta, _vl, _vr, _a_out, _b_out) = driver.run().unwrap();
        println!("Column-major test passed!");
    }

    #[test]
    fn test_zggev_simple_rowmajor() {
        // Simple 2x2 generalized eigenvalue problem (row-major)
        let device = DeviceBLAS::default();
        let data_a: Vec<c64> = vec![c64::new(1.0, 0.5), c64::new(2.0, -0.3), c64::new(3.0, 0.2), c64::new(4.0, -0.1)];
        let data_b: Vec<c64> = vec![c64::new(2.0, 0.1), c64::new(1.0, -0.2), c64::new(1.0, 0.3), c64::new(2.0, -0.4)];
        let a = rt::asarray((data_a, [2, 2].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((data_b, [2, 2].c(), &device)).into_dim::<Ix2>();

        println!("Running row-major test...");
        let driver = GGEV::default().a(a.view()).b(b.view()).left(false).right(true).build().unwrap();
        let (_alpha, _beta, _vl, _vr, _a_out, _b_out) = driver.run().unwrap();
        println!("Row-major test passed!");
    }
}
