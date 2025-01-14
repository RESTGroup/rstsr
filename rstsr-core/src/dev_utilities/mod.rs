#![allow(dead_code)]

use crate::prelude_dev::*;

/// Compare two tensors with f64 data type.
///
/// This function assumes c-contiguous iteration, and will not check two
/// dimensions are broadcastable.
pub fn allclose_f64<RA, RB, DA, DB, BA, BB>(
    a: &TensorAny<RA, f64, BA, DA>,
    b: &TensorAny<RB, f64, BB, DB>,
) -> bool
where
    RA: DataAPI<Data = <BA as DeviceRawAPI<f64>>::Raw>,
    RB: DataAPI<Data = <BB as DeviceRawAPI<f64>>::Raw>,
    DA: DimAPI,
    DB: DimAPI,
    BA: DeviceAPI<f64, Raw = Vec<f64>>,
    BB: DeviceAPI<f64, Raw = Vec<f64>>,
{
    let la = a.layout().reverse_axes();
    let lb = b.layout().reverse_axes();
    if la.size() != lb.size() {
        return false;
    }
    let it_la = IterLayoutColMajor::new(&la).unwrap();
    let it_lb = IterLayoutColMajor::new(&lb).unwrap();
    let data_a = a.raw();
    let data_b = b.raw();
    let atol = 1e-8;
    let rtol = 1e-5;
    for (idx_a, idx_b) in izip!(it_la, it_lb) {
        let va = data_a[idx_a];
        let vb = data_b[idx_b];
        let comp = (va - vb).abs() <= atol + rtol * vb.abs();
        if !comp {
            return false;
        }
    }
    return true;
}
