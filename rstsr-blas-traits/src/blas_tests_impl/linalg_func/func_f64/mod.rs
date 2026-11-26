// per-device alias defined by the test entry
pub use crate::DeviceBLAS;
pub use rstsr::prelude::*;
pub use rstsr_core::prelude_dev::fingerprint;
pub use rstsr_test_manifest::get_vec;

mod cholesky;
mod det;
mod eig;
mod eigh;
mod eigvals;
mod eigvalsh;
mod inv;
mod lstsq;
mod lu;
mod lu_factor;
mod lu_solve;
mod pinv;
mod qr;
mod slogdet;
mod solve_general;
mod solve_symmetric;
mod solve_triangular;
mod svd;
mod svdvals;

/// Shape of a stack in the axis convention of the device order `order`
/// (matrix axes last under `RowMajor`, first under `ColMajor`).
pub(crate) fn stacked(batch: &[usize], inner: &[usize], order: FlagOrder) -> Vec<usize> {
    match order {
        RowMajor => [batch, inner].concat(),
        ColMajor => [inner, batch].concat(),
    }
}

/// A stack of logical shape [`stacked`]`(..)` holding the C-ordered data `v`,
/// stored contiguously in the memory order `stored` (`RowMajor`: C-contiguous,
/// `ColMajor`: Fortran-contiguous). The matrix slices are in order for a
/// device exactly when `stored` is the device order.
pub(crate) fn stack_tensor(
    v: &[f64],
    batch: &[usize],
    inner: &[usize],
    order: FlagOrder,
    stored: FlagOrder,
    device: &DeviceBLAS,
) -> Tensor<f64, DeviceBLAS, IxD> {
    let stack = rt::asarray((v.to_vec(), stacked(batch, inner, order).c(), device)).into_owned().into_dyn();
    match stored {
        RowMajor => stack,
        ColMajor => stack.to_contig_f(ColMajor).unwrap().into_owned().into_dyn(),
    }
}
