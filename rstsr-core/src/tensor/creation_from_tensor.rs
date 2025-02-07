//! Creation methods for `Tensor` struct from other tensors.
//!
//! Todo list:
//! - [ ] `diag`
//! - [ ] `tril`
//! - [ ] `triu`

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
pub fn diag<Args, Inp>(param: Args) -> Args::Out
where
    Args: DiagAPI<Inp>,
{
    Args::diag(param)
}

pub fn diag_f<Args, Inp>(param: Args) -> Result<Args::Out>
where
    Args: DiagAPI<Inp>,
{
    Args::diag_f(param)
}

impl<R, T, B, D> DiagAPI<()> for (&TensorAny<R, T, B, D>, isize)
where
    R: DataAPI<Data = B::Raw>,
    T: Clone + Default,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, Ix1>,
{
    type Out = Tensor<T, B, IxD>;

    fn diag_f(self) -> Result<Self::Out> {
        let (tensor, offset) = self;
        if tensor.ndim() == 1 {
            let layout_diag = tensor.layout().to_dim::<Ix1>()?;
            let n_row = tensor.size() + offset.unsigned_abs();
            let mut result = full_f(([n_row, n_row], T::default(), tensor.device()))?;
            let layout_result = result.layout().diagonal(Some(offset), Some(0), Some(1))?;
            let device = tensor.device();
            device.assign(
                result.raw_mut(),
                &layout_result.to_dim()?,
                tensor.raw(),
                &layout_diag,
            )?;
            return Ok(result);
        } else if tensor.ndim() == 2 {
            let layout = tensor.layout().to_dim::<Ix2>()?;
            let layout_diag = layout.diagonal(Some(offset), Some(0), Some(1))?;
            let size = layout_diag.size();
            let device = tensor.device();
            let mut result = unsafe { empty_f(([size], device))? };
            let layout_result = result.layout().to_dim()?;
            device.assign(result.raw_mut(), &layout_result, tensor.raw(), &layout_diag)?;
            return Ok(result);
        } else {
            return rstsr_raise!(InvalidLayout, "diag only support 1-D or 2-D tensor.");
        }
    }
}

impl<R, T, B, D> DiagAPI<()> for &TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    T: Clone + Default,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, Ix1>,
{
    type Out = Tensor<T, B, IxD>;

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
        let a = arange(9).into_shape([3, 3]);
        let b = diag((&a, 1));
        println!("{:}", b);
        let c = a.diag();
        println!("{:}", c);
        let c = arange(3) + 1;
        let d = diag((&c, -1));
        println!("{:}", d);
    }
}
