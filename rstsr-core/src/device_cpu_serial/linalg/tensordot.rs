use crate::prelude_dev::*;
use core::ops::Mul;
use num::{One, Zero};

impl<TA, TB, TC, DA, DB, DC> DeviceTensordotAPI<TA, TB, TC, DA, DB, DC> for DeviceCpuSerial
where
    TA: Clone,
    TB: Clone,
    TC: Clone + Zero + One,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: Mul<TB, Output = TC>,
    Self: DeviceAPI<TA, Raw = Vec<TA>> + DeviceAPI<TB, Raw = Vec<TB>> + DeviceAPI<TC, Raw = Vec<TC>>,
    Self: DeviceAPI<MaybeUninit<TC>, Raw = Vec<MaybeUninit<TC>>>,
    Self: DeviceMatMulAPI<TA, TB, TC, Ix2, Ix2, Ix2>,
{
    fn tensordot(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<DC>,
        a: &Vec<TA>,
        la: &Layout<DA>,
        b: &Vec<TB>,
        lb: &Layout<DB>,
        axes_a: &[isize],
        axes_b: &[isize],
    ) -> Result<()> {
        let order = self.default_order();
        // View-only GEMM fast path; never copies to enable GEMM.
        if let Some((la2, lb2, lc2)) = tensordot_gemm_layouts(la, axes_a, lb, axes_b, lc, order)? {
            return self.matmul_uninit(c, &lc2, a, &la2, b, &lb2, TC::one());
        }
        tensordot_naive_cpu_serial(c, lc, a, la, b, lb, axes_a, axes_b, order)
    }
}
