//! Creation methods for `Tensor` struct from other tensors.
//!
//! Todo list:
//! - [ ] `diag`
//! - [ ] `tril`
//! - [ ] `triu`

use num::Num;

use crate::prelude_dev::*;

/* #region diag */

pub trait DiagAPI<Inp>: Sized {
    type Out;

    fn diag_f(self) -> Result<Self::Out>;

    fn diag(self) -> Self::Out {
        Self::diag_f(self).unwrap()
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
pub fn diag<Param, Inp, Rhs>(param: Param) -> Rhs
where
    Param: DiagAPI<Inp, Out = Rhs>,
{
    Param::diag(param)
}

pub fn diag_f<Param, Inp, Rhs>(param: Param) -> Result<Rhs>
where
    Param: DiagAPI<Inp, Out = Rhs>,
{
    Param::diag_f(param)
}

impl<R, T, B> DiagAPI<()> for (&TensorBase<R, Ix2>, isize)
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Clone,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, Ix1>,
{
    type Out = Tensor<T, Ix1, B>;

    fn diag_f(self) -> Result<Self::Out> {
        let (tensor, offset) = self;
        let layout_diag = tensor.layout().diagonal(Some(offset), Some(0), Some(1))?;
        let size = layout_diag.size();
        let device = tensor.device();
        let mut result = unsafe { empty_f(([size], device))? };
        let layout_result = result.layout().clone();
        device.assign(result.storage_mut(), &layout_result, tensor.storage(), &layout_diag)?;
        return Ok(result);
    }
}

impl<R, T, B> DiagAPI<()> for &TensorBase<R, Ix2>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Clone,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, Ix1>,
{
    type Out = Tensor<T, Ix1, B>;

    fn diag_f(self) -> Result<Self::Out> {
        return diag_f((self, 0));
    }
}

impl<T, B, R> DiagAPI<()> for (&TensorBase<R, Ix1>, isize)
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Clone + Num,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + DeviceCreationNumAPI<T> + OpAssignAPI<T, Ix1>,
{
    type Out = Tensor<T, Ix2, B>;

    fn diag_f(self) -> Result<Self::Out> {
        let (tensor, offset) = self;
        let layout_diag = tensor.layout().clone();
        let n_row = tensor.size() + offset.unsigned_abs();
        let mut result = zeros_f(([n_row, n_row], tensor.device()))?;
        let layout_result = result.layout().diagonal(Some(offset), Some(0), Some(1))?;
        let device = tensor.device();
        device.assign(result.storage_mut(), &layout_result, tensor.storage(), &layout_diag)?;
        return Ok(result);
    }
}

impl<T, B, R> DiagAPI<()> for &TensorBase<R, Ix1>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Clone + Num,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + DeviceCreationNumAPI<T> + OpAssignAPI<T, Ix1>,
{
    type Out = Tensor<T, Ix2, B>;

    fn diag_f(self) -> Result<Self::Out> {
        return diag_f((self, 0));
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_diag() {
        let a = arange(9).into_shape([3, 3]).into_owned();
        let b = diag((&a, 1));
        println!("{:}", b);
        let c = a.diag();
        println!("{:}", c);
        let c = arange(3) + 1;
        let d = diag((&c, -1));
        println!("{:}", d);
    }
}
