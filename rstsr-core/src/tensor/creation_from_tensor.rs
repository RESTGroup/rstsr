//! Creation methods for `Tensor` struct from other tensors.
//!
//! Todo list:
//! - [ ] `diag`
//! - [ ] `tril`
//! - [ ] `triu`

use core::mem::transmute;

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

// implementation for reference tensors
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

// implementation for non-reference tensors
#[duplicate_item(
    ImplType         ImplStruct                                           tuple_args           tuple_internal ;
   [              ] [(Vec<TensorAny<R, T, B, D>> , &str, bool)] [(tensors, indexing, copy)] [(indexing, copy)];
   [              ] [(&Vec<TensorAny<R, T, B, D>>, &str, bool)] [(tensors, indexing, copy)] [(indexing, copy)];
   [const N: usize] [([TensorAny<R, T, B, D>; N] , &str, bool)] [(tensors, indexing, copy)] [(indexing, copy)];
   [              ] [(Vec<TensorAny<R, T, B, D>> , &str,     )] [(tensors, indexing,     )] [(indexing, true)];
   [              ] [(&Vec<TensorAny<R, T, B, D>>, &str,     )] [(tensors, indexing,     )] [(indexing, true)];
   [const N: usize] [([TensorAny<R, T, B, D>; N] , &str,     )] [(tensors, indexing,     )] [(indexing, true)];
   [              ] [(Vec<TensorAny<R, T, B, D>> ,       bool)] [(tensors,           copy)] [("xy"    , copy)];
   [              ] [(&Vec<TensorAny<R, T, B, D>>,       bool)] [(tensors,           copy)] [("xy"    , copy)];
   [const N: usize] [([TensorAny<R, T, B, D>; N] ,       bool)] [(tensors,           copy)] [("xy"    , copy)];
   [              ] [ Vec<TensorAny<R, T, B, D>>              ] [ tensors                 ] [("xy"    , true)];
   [              ] [ &Vec<TensorAny<R, T, B, D>>             ] [ tensors                 ] [("xy"    , true)];
   [const N: usize] [ [TensorAny<R, T, B, D>; N]              ] [ tensors                 ] [("xy"    , true)];
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
        let (indexing, copy) = tuple_internal;
        let tensors = tensors.iter().collect::<Vec<_>>();
        MeshgridAPI::meshgrid_f((tensors, indexing, copy))
    }
}

/* #endregion */

/* #region concat */

pub trait ConcatAPI<Inp> {
    type Out;

    fn concat_f(self) -> Result<Self::Out>;
    fn concat(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::concat_f(self).unwrap()
    }
}

/// Join a sequence of arrays along an existing axis.
///
/// # See also
///
/// - [Python Array Standard `concatnate`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.concat.html)
pub fn concat<Args, Inp>(args: Args) -> Args::Out
where
    Args: ConcatAPI<Inp>,
{
    Args::concat(args)
}

pub fn concat_f<Args, Inp>(args: Args) -> Result<Args::Out>
where
    Args: ConcatAPI<Inp>,
{
    Args::concat_f(args)
}

pub use concat as concatenate;
pub use concat_f as concatenate_f;

impl<R, T, B, D> ConcatAPI<()> for (Vec<TensorAny<R, T, B, D>>, isize)
where
    R: DataAPI<Data = B::Raw>,
    T: Clone + Default,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    type Out = Tensor<T, B, IxD>;

    fn concat_f(self) -> Result<Self::Out> {
        let (tensors, axis) = self;

        // quick error for empty tensors
        rstsr_assert!(!tensors.is_empty(), InvalidValue, "concat requires at least one tensor.")?;

        // check same device and same ndim
        let device = tensors[0].device().clone();
        let ndim = tensors[0].ndim();

        rstsr_assert!(ndim > 0, InvalidLayout, "All tensors must have ndim > 0 in concat.")?;
        tensors.iter().try_for_each(|tensor| -> Result<()> {
            rstsr_assert_eq!(
                tensor.ndim(),
                ndim,
                InvalidLayout,
                "All tensors must have the same ndim."
            )?;
            rstsr_assert!(
                tensor.device().same_device(&device),
                DeviceMismatch,
                "All tensors must be on the same device."
            )?;
            Ok(())
        })?;

        // check and make axis positive
        let axis = if axis < 0 { ndim as isize + axis } else { axis };
        rstsr_pattern!(axis, 0..ndim as isize, InvalidLayout, "axis out of bounds")?;
        let axis = axis as usize;

        // - check shape compatibility (dimension other than axis must match)
        // - calculate the new shape
        let mut new_axis_size = 0;
        let mut shape_other = tensors[0].shape().as_ref().to_vec();
        shape_other.remove(axis);
        for tensor in &tensors {
            let mut shape_other_i = tensor.shape().as_ref().to_vec();
            new_axis_size += shape_other_i.remove(axis);
            rstsr_assert_eq!(
                shape_other_i,
                shape_other,
                InvalidLayout,
                "All tensors must have the same shape except for the concatenation axis."
            )?;
        }
        shape_other.insert(axis, new_axis_size);
        let new_shape = shape_other;

        // create the result tensor
        let mut result = unsafe { empty_f((new_shape, &device))? };

        // assign each tensor to the result tensor
        let mut offset = 0;
        for tensor in tensors {
            let layout = tensor.layout().to_dim::<IxD>()?;
            let axis_size = tensor.shape()[axis];
            let layout_result =
                result.layout().dim_narrow(axis as isize, slice!(offset, offset + axis_size))?;
            device.assign(result.raw_mut(), &layout_result, tensor.raw(), &layout)?;
            offset += axis_size;
        }

        Ok(result)
    }
}

#[duplicate_item(
    ImplType         ImplStruct                            ;
   [              ] [(&Vec<TensorAny<R, T, B, D>> , isize)];
   [const N: usize] [([TensorAny<R, T, B, D>; N]  , isize)];
   [              ] [(Vec<TensorAny<R, T, B, D>>  , usize)];
   [              ] [(&Vec<TensorAny<R, T, B, D>> , usize)];
   [const N: usize] [([TensorAny<R, T, B, D>; N]  , usize)];
   [              ] [(Vec<TensorAny<R, T, B, D>>  , i32  )];
   [              ] [(&Vec<TensorAny<R, T, B, D>> , i32  )];
   [const N: usize] [([TensorAny<R, T, B, D>; N]  , i32  )];
   [              ] [(Vec<&TensorAny<R, T, B, D>> , isize)];
   [              ] [(&Vec<&TensorAny<R, T, B, D>>, isize)];
   [const N: usize] [([&TensorAny<R, T, B, D>; N] , isize)];
   [              ] [(Vec<&TensorAny<R, T, B, D>> , usize)];
   [              ] [(&Vec<&TensorAny<R, T, B, D>>, usize)];
   [const N: usize] [([&TensorAny<R, T, B, D>; N] , usize)];
   [              ] [(Vec<&TensorAny<R, T, B, D>> , i32  )];
   [              ] [(&Vec<&TensorAny<R, T, B, D>>, i32  )];
   [const N: usize] [([&TensorAny<R, T, B, D>; N] , i32  )];
)]
impl<R, T, B, D, ImplType> ConcatAPI<()> for ImplStruct
where
    R: DataAPI<Data = B::Raw>,
    T: Clone + Default,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    type Out = Tensor<T, B, IxD>;

    fn concat_f(self) -> Result<Self::Out> {
        let (tensors, axis) = self;
        #[allow(clippy::unnecessary_cast)]
        let axis = axis as isize;
        let tensors = tensors.iter().map(|t| t.view()).collect::<Vec<_>>();
        ConcatAPI::concat_f((tensors, axis))
    }
}

#[duplicate_item(
    ImplType         ImplStruct                   ;
   [              ] [Vec<TensorAny<R, T, B, D>>  ];
   [              ] [&Vec<TensorAny<R, T, B, D>> ];
   [const N: usize] [[TensorAny<R, T, B, D>; N]  ];
   [              ] [Vec<&TensorAny<R, T, B, D>> ];
   [              ] [&Vec<&TensorAny<R, T, B, D>>];
   [const N: usize] [[&TensorAny<R, T, B, D>; N] ];
)]
impl<R, T, B, D, ImplType> ConcatAPI<()> for ImplStruct
where
    R: DataAPI<Data = B::Raw>,
    T: Clone + Default,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    type Out = Tensor<T, B, IxD>;

    fn concat_f(self) -> Result<Self::Out> {
        let tensors = self;
        #[allow(clippy::unnecessary_cast)]
        let axis = 0;
        let tensors = tensors.iter().map(|t| t.view()).collect::<Vec<_>>();
        ConcatAPI::concat_f((tensors, axis))
    }
}

/* #endregion */

/* #region hstack */

pub trait HStackAPI<Inp> {
    type Out;

    fn hstack_f(self) -> Result<Self::Out>;
    fn hstack(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::hstack_f(self).unwrap()
    }
}

/// Stack tensors in sequence horizontally (column-wise).
///
/// # See also
///
/// [NumPy `hstack`](https://numpy.org/doc/stable/reference/generated/numpy.hstack.html)
pub fn hstack<Args, Inp>(args: Args) -> Args::Out
where
    Args: HStackAPI<Inp>,
{
    Args::hstack(args)
}

pub fn hstack_f<Args, Inp>(args: Args) -> Result<Args::Out>
where
    Args: HStackAPI<Inp>,
{
    Args::hstack_f(args)
}

#[duplicate_item(
    ImplType         ImplStruct                   ;
   [              ] [Vec<TensorAny<R, T, B, D>>  ];
   [              ] [&Vec<TensorAny<R, T, B, D>> ];
   [const N: usize] [[TensorAny<R, T, B, D>; N]  ];
   [              ] [Vec<&TensorAny<R, T, B, D>> ];
   [              ] [&Vec<&TensorAny<R, T, B, D>>];
   [const N: usize] [[&TensorAny<R, T, B, D>; N] ];
)]
impl<R, T, B, D, ImplType> HStackAPI<()> for ImplStruct
where
    R: DataAPI<Data = B::Raw>,
    T: Clone + Default,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    type Out = Tensor<T, B, IxD>;

    fn hstack_f(self) -> Result<Self::Out> {
        let tensors = self;

        if tensors.is_empty() {
            return rstsr_raise!(InvalidValue, "hstack requires at least one tensor.");
        }

        if tensors[0].ndim() == 1 {
            ConcatAPI::concat_f((tensors, 0))
        } else {
            ConcatAPI::concat_f((tensors, 1))
        }
    }
}

/* #endregion */

/* #region vstack */

pub trait VStackAPI<Inp> {
    type Out;

    fn vstack_f(self) -> Result<Self::Out>;
    fn vstack(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::vstack_f(self).unwrap()
    }
}

/// Stack tensors in sequence horizontally (row-wise).
///
/// # See also
///
/// [NumPy `vstack`](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html)
pub fn vstack<Args, Inp>(args: Args) -> Args::Out
where
    Args: VStackAPI<Inp>,
{
    Args::vstack(args)
}

pub fn vstack_f<Args, Inp>(args: Args) -> Result<Args::Out>
where
    Args: VStackAPI<Inp>,
{
    Args::vstack_f(args)
}

#[duplicate_item(
    ImplType         ImplStruct                   ;
   [              ] [Vec<TensorAny<R, T, B, D>>  ];
   [              ] [&Vec<TensorAny<R, T, B, D>> ];
   [const N: usize] [[TensorAny<R, T, B, D>; N]  ];
   [              ] [Vec<&TensorAny<R, T, B, D>> ];
   [              ] [&Vec<&TensorAny<R, T, B, D>>];
   [const N: usize] [[&TensorAny<R, T, B, D>; N] ];
)]
impl<R, T, B, D, ImplType> VStackAPI<()> for ImplStruct
where
    R: DataAPI<Data = B::Raw>,
    T: Clone + Default,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    type Out = Tensor<T, B, IxD>;

    fn vstack_f(self) -> Result<Self::Out> {
        let tensors = self;

        if tensors.is_empty() {
            return rstsr_raise!(InvalidValue, "vstack requires at least one tensor.");
        }

        ConcatAPI::concat_f((tensors, 0))
    }
}

/* #endregion */

/* #region stack */

pub trait StackAPI<Inp> {
    type Out;

    fn stack_f(self) -> Result<Self::Out>;
    fn stack(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::stack_f(self).unwrap()
    }
}

/// Joins a sequence of arrays along a new axis.
///
/// # See also
///
/// [Python Array Standard `stack`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.stack.html)
pub fn stack<Args, Inp>(args: Args) -> Args::Out
where
    Args: StackAPI<Inp>,
{
    Args::stack(args)
}

pub fn stack_f<Args, Inp>(args: Args) -> Result<Args::Out>
where
    Args: StackAPI<Inp>,
{
    Args::stack_f(args)
}

impl<R, T, B, D> StackAPI<()> for (Vec<TensorAny<R, T, B, D>>, isize)
where
    R: DataAPI<Data = B::Raw>,
    T: Clone + Default,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    type Out = Tensor<T, B, IxD>;

    fn stack_f(self) -> Result<Self::Out> {
        let (tensors, axis) = self;

        // quick error for empty tensors
        rstsr_assert!(!tensors.is_empty(), InvalidValue, "stack requires at least one tensor.")?;

        // check same device and same ndim
        let device = tensors[0].device().clone();
        let ndim = tensors[0].ndim();
        let shape_orig = tensors[0].shape();

        rstsr_assert!(ndim > 0, InvalidLayout, "All tensors must have ndim > 0 in stack.")?;
        tensors.iter().try_for_each(|tensor| -> Result<()> {
            rstsr_assert_eq!(
                tensor.shape(),
                shape_orig,
                InvalidLayout,
                "All tensors must have the same shape."
            )?;
            rstsr_assert!(
                tensor.device().same_device(&device),
                DeviceMismatch,
                "All tensors must be on the same device."
            )?;
            Ok(())
        })?;

        // check and make axis positive
        let axis = if axis < 0 { ndim as isize + axis + 1 } else { axis };
        rstsr_pattern!(axis, 0..=ndim as isize, InvalidLayout, "axis out of bounds")?;
        let axis = axis as usize;

        // expand the shape of each tensor
        let tensors = tensors
            .into_iter()
            .map(|tensor| tensor.into_expand_dims_f(axis))
            .collect::<Result<Vec<_>>>()?;

        // use concat function to perform the stacking
        ConcatAPI::concat_f((tensors, axis as isize))
    }
}

#[duplicate_item(
    ImplType         ImplStruct                            ;
   [              ] [(&Vec<TensorAny<R, T, B, D>> , isize)];
   [const N: usize] [([TensorAny<R, T, B, D>; N]  , isize)];
   [              ] [(Vec<TensorAny<R, T, B, D>>  , usize)];
   [              ] [(&Vec<TensorAny<R, T, B, D>> , usize)];
   [const N: usize] [([TensorAny<R, T, B, D>; N]  , usize)];
   [              ] [(Vec<TensorAny<R, T, B, D>>  , i32  )];
   [              ] [(&Vec<TensorAny<R, T, B, D>> , i32  )];
   [const N: usize] [([TensorAny<R, T, B, D>; N]  , i32  )];
   [              ] [(Vec<&TensorAny<R, T, B, D>> , isize)];
   [              ] [(&Vec<&TensorAny<R, T, B, D>>, isize)];
   [const N: usize] [([&TensorAny<R, T, B, D>; N] , isize)];
   [              ] [(Vec<&TensorAny<R, T, B, D>> , usize)];
   [              ] [(&Vec<&TensorAny<R, T, B, D>>, usize)];
   [const N: usize] [([&TensorAny<R, T, B, D>; N] , usize)];
   [              ] [(Vec<&TensorAny<R, T, B, D>> , i32  )];
   [              ] [(&Vec<&TensorAny<R, T, B, D>>, i32  )];
   [const N: usize] [([&TensorAny<R, T, B, D>; N] , i32  )];
)]
impl<R, T, B, D, ImplType> StackAPI<()> for ImplStruct
where
    R: DataAPI<Data = B::Raw>,
    T: Clone + Default,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    type Out = Tensor<T, B, IxD>;

    fn stack_f(self) -> Result<Self::Out> {
        let (tensors, axis) = self;
        #[allow(clippy::unnecessary_cast)]
        let axis = axis as isize;
        let tensors = tensors.iter().map(|t| t.view()).collect::<Vec<_>>();
        StackAPI::stack_f((tensors, axis))
    }
}

#[duplicate_item(
    ImplType         ImplStruct                   ;
   [              ] [Vec<TensorAny<R, T, B, D>>  ];
   [              ] [&Vec<TensorAny<R, T, B, D>> ];
   [const N: usize] [[TensorAny<R, T, B, D>; N]  ];
   [              ] [Vec<&TensorAny<R, T, B, D>> ];
   [              ] [&Vec<&TensorAny<R, T, B, D>>];
   [const N: usize] [[&TensorAny<R, T, B, D>; N] ];
)]
impl<R, T, B, D, ImplType> StackAPI<()> for ImplStruct
where
    R: DataAPI<Data = B::Raw>,
    T: Clone + Default,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    type Out = Tensor<T, B, IxD>;

    fn stack_f(self) -> Result<Self::Out> {
        let tensors = self;
        #[allow(clippy::unnecessary_cast)]
        let axis = 0;
        let tensors = tensors.iter().map(|t| t.view()).collect::<Vec<_>>();
        StackAPI::stack_f((tensors, axis))
    }
}

/* #endregion */

/* #region unstack */

pub trait UnstackAPI<Inp> {
    type Out;

    fn unstack_f(self) -> Result<Self::Out>;
    fn unstack(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::unstack_f(self).unwrap()
    }
}

/// Splits an array into a sequence of arrays along the given axis.
///
/// # See also
///
/// [Python Array Standard `unstack`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.unstack.html)
pub fn unstack<Args, Inp>(args: Args) -> Args::Out
where
    Args: UnstackAPI<Inp>,
{
    Args::unstack(args)
}

pub fn unstack_f<Args, Inp>(args: Args) -> Result<Args::Out>
where
    Args: UnstackAPI<Inp>,
{
    Args::unstack_f(args)
}

impl<'a, T, B, D> UnstackAPI<()> for (TensorView<'a, T, B, D>, isize)
where
    T: Clone + Default,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: DeviceAPI<T>,
{
    type Out = Vec<TensorView<'a, T, B, D::SmallerOne>>;

    fn unstack_f(self) -> Result<Self::Out> {
        let (tensor, axis) = self;

        // check tensor ndim
        rstsr_assert!(
            tensor.ndim() > 0,
            InvalidLayout,
            "unstack requires a tensor with ndim > 0."
        )?;

        // check axis
        let ndim = tensor.ndim();
        let axis = if axis < 0 { ndim as isize + axis } else { axis };
        rstsr_pattern!(axis, 0..ndim as isize, InvalidLayout, "axis out of bounds")?;
        let axis = axis as usize;

        (0..tensor.layout().shape()[axis])
            .map(|i| {
                let view = tensor.view();
                let (storage, layout) = view.into_raw_parts();
                let layout = layout.dim_select(axis as isize, i as isize)?;
                // safety: transmute for lifetime annotation
                let storage = unsafe { transmute::<Storage<_, T, B>, Storage<_, T, B>>(storage) };
                unsafe { Ok(TensorBase::new_unchecked(storage, layout)) }
            })
            .collect()
    }
}

impl<'a, R, T, B, D> UnstackAPI<()> for (&'a TensorAny<R, T, B, D>, isize)
where
    T: Clone + Default,
    R: DataAPI<Data = B::Raw>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: DeviceAPI<T>,
{
    type Out = Vec<TensorView<'a, T, B, D::SmallerOne>>;

    fn unstack_f(self) -> Result<Self::Out> {
        let (tensor, axis) = self;
        UnstackAPI::unstack_f((tensor.view(), axis))
    }
}

impl<'a, T, B, D> UnstackAPI<()> for TensorView<'a, T, B, D>
where
    T: Clone + Default,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: DeviceAPI<T>,
{
    type Out = Vec<TensorView<'a, T, B, D::SmallerOne>>;

    fn unstack_f(self) -> Result<Self::Out> {
        UnstackAPI::unstack_f((self, 0))
    }
}

impl<'a, R, T, B, D> UnstackAPI<()> for &'a TensorAny<R, T, B, D>
where
    T: Clone + Default,
    R: DataAPI<Data = B::Raw>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: DeviceAPI<T>,
{
    type Out = Vec<TensorView<'a, T, B, D::SmallerOne>>;

    fn unstack_f(self) -> Result<Self::Out> {
        UnstackAPI::unstack_f((self, 0))
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
        let a = arange(3);
        let b = arange(4);
        let c = meshgrid((&vec![&a, &b], "ij", true));
        println!("{c:?}");
        let d = meshgrid((&vec![&a, &b], "xy", true));
        println!("{d:?}");
    }

    #[test]
    fn test_concat() {
        let a = arange(18).into_shape([2, 3, 3]);
        let b = arange(24).into_shape([2, 4, 3]);
        let c = arange(30).into_shape([2, 5, 3]);
        let d = concat(([a, b, c], -2));
        println!("{d:?}");
    }

    #[test]
    fn test_hstack() {
        let a = arange(18).into_shape([2, 3, 3]);
        let b = arange(24).into_shape([2, 4, 3]);
        let c = arange(30).into_shape([2, 5, 3]);
        let d = hstack([a, b, c]);
        println!("{d:?}");
    }

    #[test]
    fn test_stack() {
        let a = arange(8).into_shape([2, 4]);
        let b = arange(8).into_shape([2, 4]);
        let c = arange(8).into_shape([2, 4]);
        let d = stack([&a, &b, &c]);
        println!("{d:?}");
        let d = stack(([&a, &b, &c], -1));
        println!("{d:?}");
    }

    #[test]
    fn test_unstack() {
        let a = arange(24).into_shape([2, 3, 4]);
        let v = unstack((&a, 2));
        println!("{v:?}");
        let v = unstack(a.view());
        println!("{v:?}");
    }
}
