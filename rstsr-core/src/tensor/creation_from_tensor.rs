//! Creation methods for `Tensor` struct from other tensors.
//!
//! Todo list:
//! - [ ] `diag`
//! - [ ] `tril`
//! - [ ] `triu`

use crate::prelude_dev::*;

/* #region diag */

pub trait DiagAPI<Inp> {
    type Out;

    fn diag_f(self) -> Result<Self::Out>;

    fn diag(self) -> Self::Out
    where
        Self: Sized,
    {
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

/* #region meshgrid */

pub trait MeshgridAPI<Inp> {
    type Out;

    fn meshgrid_f(self) -> Result<Self::Out>;

    fn meshgrid(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::meshgrid_f(self).unwrap()
    }
}

/// Returns coordinate matrices from coordinate vectors.
///
/// # See also
///
/// - [Python Array Standard `meshgrid`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.meshgrid.html)
pub fn meshgrid<Args, Inp>(args: Args) -> Args::Out
where
    Args: MeshgridAPI<Inp>,
{
    Args::meshgrid(args)
}

pub fn meshgrid_f<Args, Inp>(args: Args) -> Result<Args::Out>
where
    Args: MeshgridAPI<Inp>,
{
    Args::meshgrid_f(args)
}

impl<R, T, B, D> MeshgridAPI<()> for (Vec<&TensorAny<R, T, B, D>>, &str, bool)
where
    R: DataAPI<Data = B::Raw> + DataCloneAPI,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, IxD>
        + OpAssignArbitaryAPI<T, IxD, IxD>,
    B::Raw: Clone,
{
    type Out = Vec<Tensor<T, B, IxD>>;

    fn meshgrid_f(self) -> Result<Self::Out> {
        let (tensors, indexing, copy) = self;

        match indexing {
            "ij" | "xy" => (),
            _ => rstsr_raise!(InvalidValue, "indexing must be 'ij' or 'xy'.")?,
        }

        // fast return for tensors with length 0/1
        if tensors.is_empty() {
            return Ok(vec![]);
        } else if tensors.len() == 1 {
            let tensor = tensors[0];
            rstsr_assert_eq!(tensor.ndim(), 1, InvalidLayout, "meshgrid only support 1-D tensor.")?;
            return Ok(vec![tensor.view().into_dim().into_owned()]);
        }

        // check
        // a. all tensors must have the same device
        // b. all tensors are 1-D
        let device = tensors[0].device();
        tensors.iter().try_for_each(|tensor| -> Result<()> {
            rstsr_assert_eq!(tensor.ndim(), 1, InvalidLayout, "meshgrid only support 1-D tensor.")?;
            rstsr_assert!(
                tensor.device().same_device(device),
                DeviceMismatch,
                "All tensors must be on the same device."
            )?;
            Ok(())
        })?;

        let ndim = tensors.len();
        let s0 = vec![1isize; ndim];

        // tensors to be broadcasted
        let tensors = tensors
            .iter()
            .enumerate()
            .map(|(i, tensor)| {
                let mut shape_new = s0.clone();
                if indexing == "xy" && i == 0 {
                    // special case for indexing="xy"
                    shape_new[1] = -1;
                } else if indexing == "xy" && i == 1 {
                    // special case for indexing="xy"
                    shape_new[0] = -1;
                } else {
                    // s0[:i] + (-1,) + so[i+1:]
                    shape_new[i] = -1;
                }
                tensor.view().into_dim::<IxD>().into_shape_f(shape_new)
            })
            .collect::<Result<Vec<_>>>()?;
        // tensors have been broadcasted to the same shape
        let tensors = broadcast_arrays_f(tensors)?;

        if !copy {
            Ok(tensors)
        } else {
            tensors.into_iter().map(|t| t.into_contig_f(device.default_order())).collect()
        }
    }
}

#[duplicate_item(
    ImplType         ImplStruct                                           tuple_args                  tuple_internal                             ;
   [              ] [(&Vec<&TensorAny<R, T, B, D>>, &str, bool)] [(tensors, indexing, copy)] [(tensors.to_vec(), indexing, copy)];
   [const N: usize] [([&TensorAny<R, T, B, D>; N] , &str, bool)] [(tensors, indexing, copy)] [(tensors.to_vec(), indexing, copy)];
   [              ] [(Vec<&TensorAny<R, T, B, D>> , &str,     )] [(tensors, indexing,     )] [(tensors.to_vec(), indexing, true)];
   [              ] [(&Vec<&TensorAny<R, T, B, D>>, &str,     )] [(tensors, indexing,     )] [(tensors.to_vec(), indexing, true)];
   [const N: usize] [([&TensorAny<R, T, B, D>; N] , &str,     )] [(tensors, indexing,     )] [(tensors.to_vec(), indexing, true)];
   [              ] [(Vec<&TensorAny<R, T, B, D>> ,       bool)] [(tensors,           copy)] [(tensors.to_vec(), "xy"    , copy)];
   [              ] [(&Vec<&TensorAny<R, T, B, D>>,       bool)] [(tensors,           copy)] [(tensors.to_vec(), "xy"    , copy)];
   [const N: usize] [([&TensorAny<R, T, B, D>; N] ,       bool)] [(tensors,           copy)] [(tensors.to_vec(), "xy"    , copy)];
   [              ] [ Vec<&TensorAny<R, T, B, D>>              ] [ tensors                 ] [(tensors.to_vec(), "xy"    , true)];
   [              ] [ &Vec<&TensorAny<R, T, B, D>>             ] [ tensors                 ] [(tensors.to_vec(), "xy"    , true)];
   [const N: usize] [ [&TensorAny<R, T, B, D>; N]              ] [ tensors                 ] [(tensors.to_vec(), "xy"    , true)];
)]
impl<R, T, B, D, ImplType> MeshgridAPI<()> for ImplStruct
where
    R: DataAPI<Data = B::Raw> + DataCloneAPI,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, IxD>
        + OpAssignArbitaryAPI<T, IxD, IxD>,
    B::Raw: Clone,
{
    type Out = Vec<Tensor<T, B, IxD>>;

    fn meshgrid_f(self) -> Result<Self::Out> {
        let tuple_args = self;
        let (tensors, indexing, copy) = tuple_internal;
        MeshgridAPI::meshgrid_f((tensors, indexing, copy))
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
        println!("{b:}");
        let c = a.diag();
        println!("{c:}");
        let c = arange(3) + 1;
        let d = diag((&c, -1));
        println!("{d:}");
    }

    #[test]
    fn test_meshgrid() {
        let a = arange((3i32, &DeviceFaer::default()));
        let b = arange((4i32, &DeviceFaer::default()));
        let c = meshgrid((&vec![&a, &b], "ij", true));
        println!("{c:?}");
        let d = meshgrid((&vec![&a, &b], "xy", true));
        println!("{d:?}");
    }
}
