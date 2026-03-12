//! Matrix multiplication for CPU backend.
//!
//! **This implementation is not optimized!**

use core::ops::{Add, Mul};

use crate::prelude_dev::*;

impl<TA, TB, TC, DA, DB, DC> DeviceMatMulAPI<TA, TB, TC, DA, DB, DC> for DeviceCpuSerial
where
    TA: Clone,
    TB: Clone,
    TC: Clone,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: Mul<TB, Output = TC>,
    TB: Mul<TA, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
    Self: DeviceAPI<TA, Raw = Vec<TA>> + DeviceAPI<TB, Raw = Vec<TB>> + DeviceAPI<TC, Raw = Vec<TC>>,
{
    fn matmul(
        &self,
        c: &mut Vec<TC>,
        lc: &Layout<DC>,
        a: &Vec<TA>,
        la: &Layout<DA>,
        b: &Vec<TB>,
        lb: &Layout<DB>,
        alpha: TC,
        beta: TC,
    ) -> Result<()> {
        let default_order = self.default_order();
        match default_order {
            RowMajor => matmul_naive_cpu_serial(c, lc, a, la, b, lb, alpha, beta),
            ColMajor => {
                let la = la.reverse_axes();
                let lb = lb.reverse_axes();
                let lc = lc.reverse_axes();
                matmul_naive_cpu_serial(c, &lc, b, &lb, a, &la, alpha, beta)
            },
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_row_major() {
        /* Python code
            a = np.linspace(1, 24, 24).reshape(2, 3, 4)
            b = np.linspace(1, 20, 20).reshape(4, 5)
            c = np.linspace(1, 30, 30).reshape(2, 3, 5)
            (1.5 * a @ b + 2.0 * c).reshape(-1)
        */
        let mut device = DeviceCpuSerial::default();
        device.set_default_order(RowMajor);
        let a = rt::linspace((1.0, 24.0, 24, &device)).into_shape((2, 3, 4));
        let b = rt::linspace((1.0, 20.0, 20, &device)).into_shape((4, 5));
        let mut c = rt::linspace((1.0, 30.0, 30, &device)).into_shape((2, 3, 5));
        let alpha = 1.5;
        let beta = 2.0;
        let la = a.layout();
        let lb = b.layout();
        let lc = c.layout().clone();
        device.matmul(c.raw_mut(), &lc, a.raw(), la, b.raw(), lb, alpha, beta).unwrap();
        println!("Result c: {c:?}");

        let c_ref = rt::asarray((
            vec![
                167., 184., 201., 218., 235., 381., 422., 463., 504., 545., 595., 660., 725., 790., 855., 809., 898.,
                987., 1076., 1165., 1023., 1136., 1249., 1362., 1475., 1237., 1374., 1511., 1648., 1785.,
            ],
            &device,
        ));
        assert!((&c.reshape(-1) - c_ref).l2_norm() < 1e-10);
    }

    #[test]
    fn test_col_major() {
        let mut device = DeviceCpuSerial::default();
        device.set_default_order(ColMajor);
        let a = rt::linspace((1.0, 20.0, 20, &device)).into_shape((5, 4));
        let b = rt::linspace((1.0, 24.0, 24, &device)).into_shape((4, 3, 2));
        let mut c = rt::linspace((1.0, 30.0, 30, &device)).into_shape((5, 3, 2));
        let alpha = 1.5;
        let beta = 2.0;
        let la = a.layout();
        let lb = b.layout();
        let lc = c.layout().clone();
        device.matmul(c.raw_mut(), &lc, a.raw(), la, b.raw(), lb, alpha, beta).unwrap();
        println!("Result c: {c:?}");

        let c_ref = rt::asarray((
            vec![
                167., 184., 201., 218., 235., 381., 422., 463., 504., 545., 595., 660., 725., 790., 855., 809., 898.,
                987., 1076., 1165., 1023., 1136., 1249., 1362., 1475., 1237., 1374., 1511., 1648., 1785.,
            ],
            &device,
        ));
        assert!((&c.reshape(-1) - c_ref).l2_norm() < 1e-10);
    }
}
