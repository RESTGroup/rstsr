#![allow(dead_code)]

use crate::prelude::*;
use crate::prelude_dev::*;
use num::complex::ComplexFloat;

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

/// Get a somehow unique fingerprint of a tensor.
///
/// # See also
///
/// PySCF `pyscf.lib.misc.fingerprint`
/// https://github.com/pyscf/pyscf/blob/6f6d3741bf42543e02ccaa1d4ef43d9bf83b3dda/pyscf/lib/misc.py#L1249-L1253
pub fn fingerprint<R, T, B, D>(a: &TensorAny<R, T, B, D>) -> T
where
    T: ComplexFloat,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceCreationComplexFloatAPI<T>
        + DeviceCosAPI<T, IxD, TOut = T>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, IxD>
        + OpAssignArbitaryAPI<T, IxD, D>
        + DeviceMatMulAPI<T, T, T, IxD, IxD, IxD>,
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    for<'a> R: DataIntoCowAPI<'a>,
{
    let range = linspace((T::zero(), T::from(a.size()).unwrap(), a.size(), false, a.device()));
    let val = a.reshape(-1) % range.cos();
    val.to_scalar()
}
