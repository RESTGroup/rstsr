//! Advanced indexing related tensor manuplications.
//!
//! Currently, full support of advanced indexing is not available. However, it
//! is still possible to index one axis by list.

use crate::prelude_dev::*;

pub fn index_select_f<R, T, B, D, I>(
    tensor: &TensorAny<R, T, B, D>,
    axis: isize,
    indices: I,
) -> Result<Tensor<T, B, D>>
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceIndexSelectAPI<T, D> + DeviceCreationAnyAPI<T>,
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
{
    // TODO: output layout control (TensorIterOrder::K or default layout)
    let device = tensor.device().clone();
    let tensor_layout = tensor.layout();
    let ndim = tensor_layout.ndim();
    // check axis and index
    let axis = if axis < 0 { ndim as isize + axis } else { axis };
    rstsr_pattern!(axis, 0..ndim as isize, InvalidLayout, "Invalid axis that exceeds ndim.")?;
    let axis = axis as usize;
    let nshape: usize = tensor_layout.shape()[axis];
    let indices = indices.try_into()?;
    let indices = indices
        .as_ref()
        .iter()
        .map(|&i| -> Result<usize> {
            let i = if i < 0 { nshape as isize + i } else { i };
            rstsr_pattern!(
                i,
                0..nshape as isize,
                InvalidLayout,
                "Invalid index that exceeds shape length at axis {}.",
                axis
            )?;
            Ok(i as usize)
        })
        .collect::<Result<Vec<usize>>>()?;
    let mut out_shape = tensor_layout.shape().as_ref().to_vec();
    out_shape[axis] = indices.len();
    let out_layout = out_shape.new_contig(None, device.default_order()).into_dim()?;
    let mut out_storage = unsafe { device.empty_impl(out_layout.size())? };
    device.index_select(
        out_storage.raw_mut(),
        &out_layout,
        tensor.storage().raw(),
        tensor_layout,
        axis,
        &indices,
    )?;
    TensorBase::new_f(out_storage, out_layout)
}

/// Returns a new tensor, which indexes the input tensor along dimension `axis`
/// using the entries in `indices`.
///
/// # See also
///
/// This function should be similar to PyTorch's [`torch.index_select`](https://docs.pytorch.org/docs/stable/generated/torch.index_select.html).
pub fn index_select<R, T, B, D, I>(
    tensor: &TensorAny<R, T, B, D>,
    axis: isize,
    indices: I,
) -> Tensor<T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceIndexSelectAPI<T, D> + DeviceCreationAnyAPI<T>,
    I: TryInto<AxesIndex<isize>>,
    Error: From<I::Error>,
{
    index_select_f(tensor, axis, indices).unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceIndexSelectAPI<T, D> + DeviceCreationAnyAPI<T>,
{
    pub fn index_select_f<I>(&self, axis: isize, indices: I) -> Result<Tensor<T, B, D>>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        index_select_f(self, axis, indices)
    }

    /// Returns a new tensor, which indexes the input tensor along dimension
    /// `axis` using the entries in `indices`.
    ///
    /// # See also
    ///
    /// This function should be similar to PyTorch's [`torch.index_select`](https://docs.pytorch.org/docs/stable/generated/torch.index_select.html).
    pub fn index_select<I>(&self, axis: isize, indices: I) -> Tensor<T, B, D>
    where
        I: TryInto<AxesIndex<isize>>,
        Error: From<I::Error>,
    {
        index_select(self, axis, indices)
    }
}
