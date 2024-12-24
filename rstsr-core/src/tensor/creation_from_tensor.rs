//! Creation methods for `Tensor` struct from other tensors.
//!
//! Todo list:
//! - [ ] `diag`
//! - [ ] `tril`
//! - [ ] `triu`

use num::Num;

use crate::prelude_dev::*;

/* #region diag */

pub trait DiagAPI<Param>: Sized {
    fn diag_f(param: Param) -> Result<Self>;

    fn diag(param: Param) -> Self {
        Self::diag_f(param).unwrap()
    }
}

/// Extract a diagonal or construct a diagonal tensor.
///
/// - If input is a 2-D tensor, return a copy of its diagonal (with offset).
/// - If input is a 1-D tensor, construct a 2-D tensor with the input as its
///   diagonal.
///
/// # See also
///
/// - [numpy.diag](https://numpy.org/doc/stable/reference/generated/numpy.diag.html)
pub fn diag<Rhs, Param>(param: Param) -> Rhs
where
    Rhs: DiagAPI<Param>,
{
    Rhs::diag(param)
}

pub fn diag_f<Rhs, Param>(param: Param) -> Result<Rhs>
where
    Rhs: DiagAPI<Param>,
{
    Rhs::diag_f(param)
}

impl<R, T, B> TensorBase<R, Ix1>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Clone + Num,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, Ix1> + DeviceCreationNumAPI<T>,
{
    pub fn diag(&self) -> Tensor<T, Ix2, B> {
        return diag(self);
    }
}

impl<R, T, B> TensorBase<R, Ix2>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Clone + Num,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, Ix1>,
{
    pub fn diag(&self) -> Tensor<T, Ix1, B> {
        return diag(self);
    }
}

impl<R, T, B> DiagAPI<(&TensorBase<R, Ix2>, isize)> for Tensor<T, Ix1, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Clone,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, Ix1>,
{
    fn diag_f(param: (&TensorBase<R, Ix2>, isize)) -> Result<Self> {
        let (tensor, offset) = param;
        let layout_diag = tensor.layout().diagonal(Some(offset), Some(0), Some(1))?;
        let size = layout_diag.size();
        let device = tensor.device();
        let mut result = unsafe { Tensor::empty_f(([size], device))? };
        let layout_result = result.layout().clone();
        device.assign(result.storage_mut(), &layout_result, tensor.storage(), &layout_diag)?;
        return Ok(result);
    }
}

impl<R, T, B> DiagAPI<&TensorBase<R, Ix2>> for Tensor<T, Ix1, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Clone,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, Ix1>,
{
    fn diag_f(tensor: &TensorBase<R, Ix2>) -> Result<Self> {
        return diag_f((tensor, 0));
    }
}

impl<T, B, R> DiagAPI<(&TensorBase<R, Ix1>, isize)> for Tensor<T, Ix2, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Clone + Num,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + DeviceCreationNumAPI<T> + OpAssignAPI<T, Ix1>,
{
    fn diag_f(param: (&TensorBase<R, Ix1>, isize)) -> Result<Self> {
        let (tensor, offset) = param;
        let layout_diag = tensor.layout().clone();
        let n_row = tensor.size() + offset.unsigned_abs();
        let mut result = Tensor::zeros_f(([n_row, n_row], tensor.device()))?;
        let layout_result = result.layout().diagonal(Some(offset), Some(0), Some(1))?;
        let device = tensor.device();
        device.assign(result.storage_mut(), &layout_result, tensor.storage(), &layout_diag)?;
        return Ok(result);
    }
}

impl<T, B, R> DiagAPI<&TensorBase<R, Ix1>> for Tensor<T, Ix2, B>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Clone + Num,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + DeviceCreationNumAPI<T> + OpAssignAPI<T, Ix1>,
{
    fn diag_f(tensor: &TensorBase<R, Ix1>) -> Result<Self> {
        return diag_f((tensor, 0));
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_diag() {
        // let a = Tensor::arange(9).into_shape([3, 3]).into_owned();
        // let b = Tensor::<_, Ix2, _>::diag((&a, 0));
        // let c = a.diag();
        // println!("{:}", c);
        // let c = Tensor::arange(3);
        // let d = Tensor::diag((&c, -1));
        // println!("{:}", d);
    }
}
