//! Creation functions that take other tensors as input: [`diag`],
//! [`meshgrid`], joining functions ([`concat`](concat()), [`stack`], [`hstack`],
//! [`vstack`], [`unstack`]), and dimension promotions ([`atleast_1d`],
//! [`atleast_2d`], [`atleast_3d`]).
//!
//! This module partly relates to the [Python array API standard
//! v2024.12](https://data-apis.org/array-api/2024.12/API_specification/creation_functions.html).

use core::mem::transmute;

use crate::prelude_dev::*;

/* #region diag */

/// API trait backing [`diag`].
pub trait DiagAPI<Inp> {
    type Out;

    fn diag_f(self) -> Result<Self::Out>;
    fn diag(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::diag_f(self).rstsr_unwrap()
    }
}

/// Extract a diagonal or construct a diagonal tensor.
///
/// - If the input is a 2-D tensor, return a copy of its k-th diagonal as a one-dimensional tensor.
/// - If the input is a 1-D tensor, return a new two-dimensional tensor with the input on its k-th
///   diagonal (the rest is filled with `T::default()`).
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders. (A constructed diagonal matrix is contiguous in the device
/// default order; its logical content does not depend on the order.)
///
/// # Overloads Table
///
/// Output is [`Tensor<T, B, IxD>`][`Tensor`].
///
/// - `diag((tensor: &TensorAny<R, T, B, D>, offset: isize)) -> Tensor<T, B, IxD>`
/// - `diag(tensor: &TensorAny<R, T, B, D>) -> Tensor<T, B, IxD>` (implicit `offset = 0`)
///
/// # Parameters
///
/// - `tensor`: input tensor; must be 1-D or 2-D.
/// - `offset`: diagonal index: `0` the main diagonal, positive above it, negative below it.
///   Defaults to `0` if omitted.
///
/// # Returns
///
/// - [`Tensor<T, B, IxD>`][`Tensor`]: the extracted diagonal (2-D input), or the constructed
///   diagonal matrix (1-D input). The result always owns its data.
///
/// # Examples
///
/// Extracting the diagonal of a 2-D tensor:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((9, &device)).into_shape([3, 3]);
/// println!("{}", rt::diag(&a));
/// // [ 0 4 8]
/// println!("{}", rt::diag((&a, 1)));
/// // [ 1 5]
/// # assert_eq!(format!("{}", rt::diag((&a, 1))), "[ 1 5]");
/// ```
///
/// Constructing a diagonal matrix from a 1-D tensor:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let v = rt::arange((3, &device));
/// println!("{}", rt::diag((&v, -1)));
/// // [[ 0 0 0 0]
/// //  [ 0 0 0 0]
/// //  [ 0 1 0 0]
/// //  [ 0 0 2 0]]
/// # assert_eq!(format!("{}", rt::diag((&v, -1))), "[[ 0 0 0 0]\n [ 0 0 0 0]\n [ 0 1 0 0]\n [ 0 0 2 0]]");
/// ```
///
/// # Notes of API accordance
///
/// - NumPy: `numpy.diag(v, k=0)` ([`numpy.diag`](https://numpy.org/doc/stable/reference/generated/numpy.diag.html))
/// - RSTSR: `rt::diag((tensor, offset))`; the two NumPy behaviors (extract / construct) are
///   dispatched by the dimensionality of the input.
///
/// # Panics
///
/// - Panics if the input tensor is neither 1-D nor 2-D.
///
/// For a fallible version, use [`diag_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - NumPy: [`numpy.diag`](https://numpy.org/doc/stable/reference/generated/numpy.diag.html)
///
/// ## Related functions in RSTSR
///
/// - [`diagonal`]: layout-level diagonal view (no copy).
/// - [`eye`]: identity-like matrix with ones on the k-th diagonal.
///
/// ## Variants of this function
///
/// - [`diag_f`]: fallible version.
pub fn diag<Args, Inp>(param: Args) -> Args::Out
where
    Args: DiagAPI<Inp>,
{
    Args::diag(param)
}

/// Extract a diagonal or construct a diagonal tensor.
///
/// See also [`diag`].
pub fn diag_f<Args, Inp>(param: Args) -> Result<Args::Out>
where
    Args: DiagAPI<Inp>,
{
    Args::diag_f(param)
}

impl<R, T, B, D> DiagAPI<()> for (&TensorAny<R, T, B, D>, isize)
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
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
            device.assign(result.raw_mut(), &layout_result.to_dim()?, tensor.raw(), &layout_diag)?;
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
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
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

/// API trait backing [`meshgrid`].
pub trait MeshgridAPI<Inp> {
    type Out;

    fn meshgrid_f(self) -> Result<Self::Out>;
    fn meshgrid(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::meshgrid_f(self).rstsr_unwrap()
    }
}

/// Returns coordinate matrices from coordinate vectors.
///
/// Makes N-D grid tensors for vectorized evaluation of functions on a grid. For
/// `N` one-dimensional input tensors of lengths `n0, ..., nN-1`, returns `N`
/// tensors of shape `(n0, ..., nN-1)` such that `grids[i][idx] == tensors[i]`
/// broadcast along the grid.
///
/// With `indexing = "xy"` (cartesian convention, NumPy's default), the first
/// two grid dimensions are swapped compared to `"ij"`.
///
/// The `copy` flag controls data sharing:
///
/// - `copy = true` (the default): each grid is a fresh owned tensor, contiguous in the device
///   default order (as in NumPy's `copy = True`).
/// - `copy = false`: no data is copied. For reference inputs (`Vec<&TensorAny>`, `[&TensorAny; N]`,
///   ...), the grids are broadcast views sharing the inputs' memory, as in NumPy's `copy = False`.
///   For consumed owned inputs (`Vec<Tensor>`, ...), the grids are owned tensors whose stride-0
///   layouts alias the inputs' own storage.
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders. (Only the memory arrangement of copied grids follows the
/// device default order.)
///
/// # Overloads Table
///
/// Reference-input forms output `Vec<TensorCow<'a, T, B, IxD>>` (one grid per
/// input):
///
/// - `meshgrid(tensors: Vec<&'a TensorAny<R, T, B, D>>) -> Vec<TensorCow<'a, T, B, IxD>>` (implicit
///   `"xy"`, `copy = true`)
/// - `meshgrid((tensors, indexing: &str)) -> Vec<TensorCow<'a, T, B, IxD>>` (implicit `copy =
///   true`)
/// - `meshgrid((tensors, copy: bool)) -> Vec<TensorCow<'a, T, B, IxD>>` (implicit `"xy"`)
/// - `meshgrid((tensors, indexing: &str, copy: bool)) -> Vec<TensorCow<'a, T, B, IxD>>`
///
/// Also, overloads of `&Vec<...>` and `[&TensorAny; N]` behave the same.
/// Consumed owned inputs output `Vec<Tensor<T, B, IxD>>` instead:
///
/// - `meshgrid((tensors: Vec<Tensor<T, B, D>>, indexing: &str, copy: bool)) -> Vec<Tensor<T, B,
///   IxD>>`
///
/// and `[Tensor; N]` forms of the same shapes; `&Vec<TensorAny<R, T, B, D>>` forms are also
/// accepted and return owned grids.
///
/// # Parameters
///
/// - `tensors`: one-dimensional input tensors.
/// - `indexing`: `"ij"` (matrix convention) or `"xy"` (cartesian convention); defaults to `"xy"` if
///   omitted.
/// - `copy`: whether the returned grids are fresh copies; defaults to `true` if omitted.
///
/// # Returns
///
/// - `Vec<TensorCow<'a, T, B, IxD>>` (reference inputs) or `Vec<Tensor<T, B, IxD>>` (owned inputs):
///   one grid tensor per input. With `copy = true` the grids are owned fresh copies; with `copy =
///   false` they share the inputs' memory (views for reference inputs).
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let x = rt::arange((3, &device));
/// let y = rt::arange((2, &device));
/// let grids = rt::meshgrid(([&x, &y], "ij"));
/// println!("{}", grids[0]);
/// // [[ 0 0]
/// //  [ 1 1]
/// //  [ 2 2]]
/// println!("{}", grids[1]);
/// // [[ 0 1]
/// //  [ 0 1]
/// //  [ 0 1]]
/// # assert_eq!(format!("{}", grids[1]), "[[ 0 1]\n [ 0 1]\n [ 0 1]]");
/// ```
///
/// With `copy = false`, the grids are broadcast views over the inputs
/// (stride-0 axes), as in NumPy:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let x = rt::arange((3, &device));
/// let y = rt::arange((2, &device));
/// let grids = rt::meshgrid(([&x, &y], "ij", false));
/// println!("{}", grids[0]);
/// // [[ 0 0]
/// //  [ 1 1]
/// //  [ 2 2]]
/// println!("{:?}", grids[0].layout());
/// // 2-Dim (dyn), contiguous: Custom
/// // shape: [3, 2], stride: [1, 0], offset: 0
/// # assert!(grids.iter().all(|grid| !grid.is_owned()));
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `meshgrid(*arrays, indexing='xy')` ([`meshgrid`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.meshgrid.html))
/// - NumPy: `numpy.meshgrid(*xi, indexing='xy', sparse=False, copy=True)` ([`numpy.meshgrid`](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html))
/// - RSTSR: `rt::meshgrid((tensors, indexing, copy))`; `sparse` is not supported.
///
/// # Panics
///
/// - Panics if `indexing` is neither `"ij"` nor `"xy"`.
/// - Panics if any input tensor is not 1-D, or if inputs are on different devices.
///
/// For a fallible version, use [`meshgrid_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - Python Array API standard: [`meshgrid`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.meshgrid.html)
/// - NumPy: [`numpy.meshgrid`](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html)
///
/// ## Related functions in RSTSR
///
/// - [`broadcast_arrays`](crate::tensor::manipulation::exports::broadcast_arrays()): broadcast
///   tensors against each other (analogous arrangement of the grids).
///
/// ## Variants of this function
///
/// - [`meshgrid_f`]: fallible version.
pub fn meshgrid<Args, Inp>(args: Args) -> Args::Out
where
    Args: MeshgridAPI<Inp>,
{
    Args::meshgrid(args)
}

/// Returns coordinate matrices from coordinate vectors.
///
/// See also [`meshgrid`].
pub fn meshgrid_f<Args, Inp>(args: Args) -> Result<Args::Out>
where
    Args: MeshgridAPI<Inp>,
{
    Args::meshgrid_f(args)
}

/// Compute the output shape and the varying-axis position of each input.
///
/// For `indexing = "xy"` the first two entries are swapped (NumPy convention);
/// `positions[i]` is the axis of the output grid that varies with input `i`.
fn meshgrid_out_shape_and_pos(lens: &[usize], indexing: &str) -> (Vec<usize>, Vec<usize>) {
    let ndim = lens.len();
    let mut shape_out: Vec<usize> = lens.to_vec();
    let mut positions: Vec<usize> = (0..ndim).collect();
    if indexing == "xy" && ndim >= 2 {
        shape_out.swap(0, 1);
        positions.swap(0, 1);
    }
    (shape_out, positions)
}

/// Layout of the grid for one input: the input's own stride is kept on axis
/// `pos`, all other axes have stride 0 (broadcast convention).
fn meshgrid_grid_layout(layout_in: &Layout<Ix1>, pos: usize, shape_out: &[usize]) -> Result<Layout<IxD>> {
    let stride_var = layout_in.stride()[0];
    let stride: Vec<isize> = (0..shape_out.len()).map(|j| if j == pos { stride_var } else { 0 }).collect();
    let layout: Layout<IxD> = Layout::new(shape_out.into(), stride, layout_in.offset())?;
    return Ok(layout);
}

impl<'a, R, T, B, D> MeshgridAPI<()> for (Vec<&'a TensorAny<R, T, B, D>>, &str, bool)
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD> + OpAssignArbitaryAPI<T, IxD, IxD>,
    <B as DeviceRawAPI<T>>::Raw: Clone,
{
    type Out = Vec<TensorCow<'a, T, B, IxD>>;

    fn meshgrid_f(self) -> Result<Self::Out> {
        let (tensors, indexing, copy) = self;

        match indexing {
            "ij" | "xy" => (),
            _ => rstsr_raise!(InvalidValue, "indexing must be 'ij' or 'xy'.")?,
        }

        // fast return for empty input
        if tensors.is_empty() {
            return Ok(vec![]);
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

        let lens = tensors.iter().map(|tensor| tensor.shape()[0]).collect::<Vec<_>>();
        let (shape_out, positions) = meshgrid_out_shape_and_pos(&lens, indexing);

        // each grid is built layout-only from its input; `copy` decides
        // whether the result is a fresh owned tensor or a view of the input
        tensors
            .iter()
            .enumerate()
            .map(|(i, tensor)| -> Result<TensorCow<'a, T, B, IxD>> {
                // copy the reference out so the view borrows the input
                // directly, not the local vector of references
                let tensor_ref: &'a TensorAny<R, T, B, D> = tensor;
                let view: TensorView<'a, T, B, Ix1> = tensor_ref.view().into_dim::<Ix1>();
                let layout_grid = meshgrid_grid_layout(view.layout(), positions[i], &shape_out)?;
                let (storage, _) = view.into_raw_parts();
                // safety: `layout_grid` references only elements within the
                // input's own bounds (strides scaled from the input's layout)
                let grid: TensorView<'a, T, B, IxD> = unsafe { TensorBase::new_unchecked(storage, layout_grid) };
                if copy {
                    // copy = true: fresh owned grid, contiguous in device default order
                    Ok(grid.into_contig_f(device.default_order())?.into_cow())
                } else {
                    // copy = false: broadcast view sharing the input's memory
                    Ok(grid.into_cow())
                }
            })
            .collect()
    }
}

impl<T, B, D> MeshgridAPI<()> for (Vec<Tensor<T, B, D>>, &str, bool)
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD> + OpAssignArbitaryAPI<T, IxD, IxD>,
    <B as DeviceRawAPI<T>>::Raw: Clone,
{
    type Out = Vec<Tensor<T, B, IxD>>;

    fn meshgrid_f(self) -> Result<Self::Out> {
        let (tensors, indexing, copy) = self;

        match indexing {
            "ij" | "xy" => (),
            _ => rstsr_raise!(InvalidValue, "indexing must be 'ij' or 'xy'.")?,
        }

        // fast return for empty input
        if tensors.is_empty() {
            return Ok(vec![]);
        }

        // check
        // a. all tensors must have the same device
        // b. all tensors are 1-D
        let device = tensors[0].device().clone();
        tensors.iter().try_for_each(|tensor| -> Result<()> {
            rstsr_assert_eq!(tensor.ndim(), 1, InvalidLayout, "meshgrid only support 1-D tensor.")?;
            rstsr_assert!(
                tensor.device().same_device(&device),
                DeviceMismatch,
                "All tensors must be on the same device."
            )?;
            Ok(())
        })?;

        let lens = tensors.iter().map(|tensor| tensor.shape()[0]).collect::<Vec<_>>();
        let (shape_out, positions) = meshgrid_out_shape_and_pos(&lens, indexing);

        tensors
            .into_iter()
            .enumerate()
            .map(|(i, tensor)| -> Result<Tensor<T, B, IxD>> {
                let tensor = tensor.into_dim::<Ix1>();
                let layout_grid = meshgrid_grid_layout(tensor.layout(), positions[i], &shape_out)?;
                let (storage, _) = tensor.into_raw_parts();
                // safety: `layout_grid` references only elements within the
                // input's own bounds (strides scaled from the input's layout)
                let grid: Tensor<T, B, IxD> = unsafe { TensorBase::new_unchecked(storage, layout_grid) };
                if copy {
                    // copy = true: fresh owned grid, contiguous in device default order
                    grid.into_contig_f(device.default_order())
                } else {
                    // copy = false: owned grid with stride-0 layout aliasing
                    // the input's own storage (no copy is performed)
                    Ok(grid)
                }
            })
            .collect()
    }
}

// implementation for reference tensors
#[duplicate_item(
    ImplType         ImplStruct                                                       tuple_args                  tuple_internal                             ;
   [              ] [(&Vec<&'a TensorAny<R, T, B, D>>, &str, bool)] [(tensors, indexing, copy)] [(tensors.to_vec(), indexing, copy)];
   [const N: usize] [([&'a TensorAny<R, T, B, D>; N] , &str, bool)] [(tensors, indexing, copy)] [(tensors.to_vec(), indexing, copy)];
   [              ] [(Vec<&'a TensorAny<R, T, B, D>> , &str,     )] [(tensors, indexing,     )] [(tensors.to_vec(), indexing, true)];
   [              ] [(&Vec<&'a TensorAny<R, T, B, D>>, &str,     )] [(tensors, indexing,     )] [(tensors.to_vec(), indexing, true)];
   [const N: usize] [([&'a TensorAny<R, T, B, D>; N] , &str,     )] [(tensors, indexing,     )] [(tensors.to_vec(), indexing, true)];
   [              ] [(Vec<&'a TensorAny<R, T, B, D>> ,       bool)] [(tensors,           copy)] [(tensors.to_vec(), "xy"    , copy)];
   [              ] [(&Vec<&'a TensorAny<R, T, B, D>>,       bool)] [(tensors,           copy)] [(tensors.to_vec(), "xy"    , copy)];
   [const N: usize] [([&'a TensorAny<R, T, B, D>; N] ,       bool)] [(tensors,           copy)] [(tensors.to_vec(), "xy"    , copy)];
   [              ] [ Vec<&'a TensorAny<R, T, B, D>>              ] [ tensors                 ] [(tensors.to_vec(), "xy"    , true)];
   [              ] [ &Vec<&'a TensorAny<R, T, B, D>>             ] [ tensors                 ] [(tensors.to_vec(), "xy"    , true)];
   [const N: usize] [ [&'a TensorAny<R, T, B, D>; N]              ] [ tensors                 ] [(tensors.to_vec(), "xy"    , true)];
)]
impl<'a, R, T, B, D, ImplType> MeshgridAPI<()> for ImplStruct
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD> + OpAssignArbitaryAPI<T, IxD, IxD>,
    <B as DeviceRawAPI<T>>::Raw: Clone,
{
    type Out = Vec<TensorCow<'a, T, B, IxD>>;

    fn meshgrid_f(self) -> Result<Self::Out> {
        let tuple_args = self;
        let (tensors, indexing, copy) = tuple_internal;
        MeshgridAPI::meshgrid_f((tensors, indexing, copy))
    }
}

// implementation for owned tensors consumed by value: grids are built from
// the inputs' own storages without an intermediate copy
#[duplicate_item(
    ImplType         ImplStruct                             tuple_args                  tuple_internal                             ;
   [const N: usize] [([Tensor<T, B, D>; N] , &str, bool)] [(tensors, indexing, copy)] [(Vec::from(tensors), indexing, copy)];
   [              ] [(Vec<Tensor<T, B, D>> , &str,     )] [(tensors, indexing,     )] [(tensors, indexing, true)];
   [const N: usize] [([Tensor<T, B, D>; N] , &str,     )] [(tensors, indexing,     )] [(Vec::from(tensors), indexing, true)];
   [              ] [(Vec<Tensor<T, B, D>> ,       bool)] [(tensors,           copy)] [(tensors, "xy"    , copy)];
   [const N: usize] [([Tensor<T, B, D>; N] ,       bool)] [(tensors,           copy)] [(Vec::from(tensors), "xy"    , copy)];
   [              ] [ Vec<Tensor<T, B, D>>              ] [ tensors                 ] [(tensors, "xy"    , true)];
   [const N: usize] [ [Tensor<T, B, D>; N]              ] [ tensors                 ] [(Vec::from(tensors), "xy"    , true)];
)]
impl<T, B, D, ImplType> MeshgridAPI<()> for ImplStruct
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD> + OpAssignArbitaryAPI<T, IxD, IxD>,
    <B as DeviceRawAPI<T>>::Raw: Clone,
{
    type Out = Vec<Tensor<T, B, IxD>>;

    fn meshgrid_f(self) -> Result<Self::Out> {
        let tuple_args = self;
        let (tensors, indexing, copy) = tuple_internal;
        MeshgridAPI::meshgrid_f((tensors, indexing, copy))
    }
}

// implementation for owned tensors borrowed by reference: the reference-input
// grids are converted into owned tensors (moving when already owned)
#[duplicate_item(
    ImplType         ImplStruct                                         tuple_args                  tuple_internal            ;
   [              ] [(&'a Vec<TensorAny<R, T, B, D>>, &str, bool)] [(tensors, indexing, copy)] [(tensors, indexing, copy)];
   [              ] [(&'a Vec<TensorAny<R, T, B, D>>, &str,     )] [(tensors, indexing,     )] [(tensors, indexing, true)];
   [              ] [(&'a Vec<TensorAny<R, T, B, D>>,       bool)] [(tensors,           copy)] [(tensors, "xy"    , copy)];
   [              ] [ &'a Vec<TensorAny<R, T, B, D>>             ] [ tensors                 ] [(tensors, "xy"    , true)];
)]
impl<'a, R, T, B, D, ImplType> MeshgridAPI<()> for ImplStruct
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD> + OpAssignArbitaryAPI<T, IxD, IxD>,
    <B as DeviceRawAPI<T>>::Raw: Clone,
{
    type Out = Vec<Tensor<T, B, IxD>>;

    fn meshgrid_f(self) -> Result<Self::Out> {
        let tuple_args = self;
        let (tensors, indexing, copy) = tuple_internal;
        meshgrid_into_owned_from_ref(tensors, indexing, copy)
    }
}

/// Meshgrid over borrowed owned tensors, returning owned grids.
///
/// The reference-input grids are converted into owned tensors: moved when the
/// cow buffer is already owned (`copy = true`), copied only when the grids are
/// views (`copy = false`).
fn meshgrid_into_owned_from_ref<'a, R, T, B, D>(
    tensors: &'a [TensorAny<R, T, B, D>],
    indexing: &str,
    copy: bool,
) -> Result<Vec<Tensor<T, B, IxD>>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD> + OpAssignArbitaryAPI<T, IxD, IxD>,
    <B as DeviceRawAPI<T>>::Raw: Clone,
{
    let refs = tensors.iter().collect::<Vec<_>>();
    let grids = MeshgridAPI::meshgrid_f((refs, indexing, copy))?;
    return Ok(grids.into_iter().map(|grid| grid.into_owned()).collect());
}

/* #endregion */

/* #region concat */

/// API trait backing [`concat`](concat()) and [`concatenate`](concatenate()).
pub trait ConcatAPI<Inp> {
    type Out;

    fn concat_f(self) -> Result<Self::Out>;
    fn concat(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::concat_f(self).rstsr_unwrap()
    }
}

/// Join a sequence of tensors along an existing axis.
///
/// All inputs must have the same shape except on the concatenation axis, and
/// live on the same device. The result is an owned tensor, contiguous in the
/// device default order.
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders. (Only the memory arrangement of the new tensor follows the
/// device default order.)
///
/// # Overloads Table
///
/// Output is [`Tensor<T, B, IxD>`][`Tensor`]; `tensors` also accepts owned
/// `Vec<TensorAny>` / `[TensorAny; N]` forms.
///
/// - `concat(tensors: Vec<&TensorAny<R, T, B, D>>) -> Tensor<T, B, IxD>` (implicit `axis = 0`)
/// - `concat((tensors, axis)) -> Tensor<T, B, IxD>` where `axis` is `isize`, `usize`, or `i32`
///
/// # Parameters
///
/// - `tensors`: the tensors to join; at least one is required.
/// - `axis`: the axis along which to join; negative values count from the back. Defaults to `0` if
///   omitted.
///
/// # Returns
///
/// - [`Tensor<T, B, IxD>`][`Tensor`]: the joined tensor, owning its data.
///
/// # Examples
///
/// Joining along the default axis (0):
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((3, &device));
/// let b = rt::arange((3, 6, &device));
/// println!("{}", rt::concat([&a, &b]));
/// // [ 0 1 2 3 4 5]
/// # assert_eq!(format!("{}", rt::concat([&a, &b])), "[ 0 1 2 3 4 5]");
/// ```
///
/// Joining 2-D tensors along axis 1:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((6, &device)).into_shape([2, 3]);
/// let b = rt::full(([2, 2], 9, &device));
/// println!("{}", rt::concat(([a, b], 1)));
/// // [[ 0 1 2 9 9]
/// //  [ 3 4 5 9 9]]
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `concat(arrays, /, *, axis=0)` ([`concat`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.concat.html))
/// - NumPy: `numpy.concatenate(arrays, axis=0, out=None)` ([`numpy.concatenate`](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html))
/// - RSTSR: `rt::concat((tensors, axis))`; `out` is not supported.
///
/// # Panics
///
/// - Panics if `tensors` is empty, or if the inputs have mismatching ndim, shape (outside `axis`),
///   or device.
///
/// For a fallible version, use [`concat_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - Python Array API standard: [`concat`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.concat.html)
/// - NumPy: [`numpy.concatenate`](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html)
///
/// ## Related functions in RSTSR
///
/// - [`stack`]: join along a new axis.
/// - [`hstack`] / [`vstack`]: horizontal / vertical joining conventions.
/// - [`unstack`]: split along an axis (reverse operation).
///
/// ## Variants of this function
///
/// - [`concat_f`]: fallible version.
/// - [`concatenate`](concatenate()) / [`concatenate_f`]: aliases of [`concat`](concat()) /
///   [`concat_f`].
pub fn concat<Args, Inp>(args: Args) -> Args::Out
where
    Args: ConcatAPI<Inp>,
{
    Args::concat(args)
}

/// Join a sequence of tensors along an existing axis.
///
/// See also [`concat`](concat()).
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
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
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
            rstsr_assert_eq!(tensor.ndim(), ndim, InvalidLayout, "All tensors must have the same ndim.")?;
            rstsr_assert!(
                tensor.device().same_device(&device),
                DeviceMismatch,
                "All tensors must be on the same device."
            )?;
            Ok(())
        })?;

        // check and make axis positive
        let axis = rstsr_check_axis!(axis, ndim)?;

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
            let layout_result = result.layout().dim_narrow(axis as isize, slice!(offset, offset + axis_size))?;
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
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
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
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
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

/// API trait backing [`hstack`].
pub trait HStackAPI<Inp> {
    type Out;

    fn hstack_f(self) -> Result<Self::Out>;
    fn hstack(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::hstack_f(self).rstsr_unwrap()
    }
}

/// Stack tensors in sequence horizontally (column-wise).
///
/// Equivalent to NumPy `hstack`: each input is promoted with [`atleast_1d`] (0-D ->
/// 1-D), then concatenated along axis 0 for 1-D inputs, or along axis 1 otherwise.
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders. (Only the memory arrangement of the new tensor follows the
/// device default order.)
///
/// # Overloads Table
///
/// Output is [`Tensor<T, B, IxD>`][`Tensor`]; `tensors` also accepts owned
/// `Vec<TensorAny>` / `[TensorAny; N]` forms.
///
/// - `hstack(tensors: Vec<&TensorAny<R, T, B, D>>) -> Tensor<T, B, IxD>`
///
/// # Parameters
///
/// - `tensors`: the tensors to join; at least one is required.
///
/// # Returns
///
/// - [`Tensor<T, B, IxD>`][`Tensor`]: the joined tensor, owning its data.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((3, &device));
/// let b = rt::arange((3, 6, &device));
/// println!("{}", rt::hstack([&a, &b]));
/// // [ 0 1 2 3 4 5]
/// # assert_eq!(format!("{}", rt::hstack([&a, &b])), "[ 0 1 2 3 4 5]");
/// ```
///
/// For 2-D inputs, joining happens along axis 1:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((4, &device)).into_shape([2, 2]);
/// let b = rt::full(([2, 1], 9, &device));
/// println!("{}", rt::hstack([&a, &b]));
/// // [[ 0 1 9]
/// //  [ 2 3 9]]
/// # assert_eq!(format!("{}", rt::hstack([&a, &b])), "[[ 0 1 9]\n [ 2 3 9]]");
/// ```
///
/// # Notes of API accordance
///
/// - NumPy: `numpy.hstack(tup)` ([`numpy.hstack`](https://numpy.org/doc/stable/reference/generated/numpy.hstack.html))
/// - RSTSR: `rt::hstack(tensors)`.
///
/// # Panics
///
/// - Panics if `tensors` is empty, or if the promoted inputs are not concatenable (mismatching
///   ndim, shape, or device).
///
/// For a fallible version, use [`hstack_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - NumPy: [`numpy.hstack`](https://numpy.org/doc/stable/reference/generated/numpy.hstack.html)
///
/// ## Related functions in RSTSR
///
/// - [`vstack`]: vertical joining.
/// - [`concat`](concat()): joining along an explicit axis.
///
/// ## Variants of this function
///
/// - [`hstack_f`]: fallible version.
pub fn hstack<Args, Inp>(args: Args) -> Args::Out
where
    Args: HStackAPI<Inp>,
{
    Args::hstack(args)
}

/// Stack tensors in sequence horizontally (column-wise).
///
/// See also [`hstack`].
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
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
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

        // NumPy hstack: promote each input with `atleast_1d` (0-D -> (1,)), then
        // concatenate along axis 0 for 1-D inputs, else along axis 1. See [`atleast_1d`].
        let tensors = tensors.iter().map(|t| into_atleast_1d_f(t.view())).collect::<Result<Vec<_>>>()?;
        if tensors[0].ndim() == 1 {
            ConcatAPI::concat_f((tensors, 0))
        } else {
            ConcatAPI::concat_f((tensors, 1))
        }
    }
}

/* #endregion */

/* #region vstack */

/// API trait backing [`vstack`].
pub trait VStackAPI<Inp> {
    type Out;

    fn vstack_f(self) -> Result<Self::Out>;
    fn vstack(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::vstack_f(self).rstsr_unwrap()
    }
}

/// Stack tensors in sequence vertically (row-wise).
///
/// Equivalent to NumPy `vstack`: each input is promoted with [`atleast_2d`] (0-D ->
/// `(1, 1)`, 1-D `(N,)` -> `(1, N)`) and concatenated along axis 0, so the result is
/// always at least 2-D.
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders. (Only the memory arrangement of the new tensor follows the
/// device default order.)
///
/// # Overloads Table
///
/// Output is [`Tensor<T, B, IxD>`][`Tensor`]; `tensors` also accepts owned
/// `Vec<TensorAny>` / `[TensorAny; N]` forms.
///
/// - `vstack(tensors: Vec<&TensorAny<R, T, B, D>>) -> Tensor<T, B, IxD>`
///
/// # Parameters
///
/// - `tensors`: the tensors to join; at least one is required.
///
/// # Returns
///
/// - [`Tensor<T, B, IxD>`][`Tensor`]: the joined tensor (at least 2-D), owning its data.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((3, &device));
/// let b = rt::arange((3, 6, &device));
/// println!("{}", rt::vstack([&a, &b]));
/// // [[ 0 1 2]
/// //  [ 3 4 5]]
/// # assert_eq!(format!("{}", rt::vstack([&a, &b])), "[[ 0 1 2]\n [ 3 4 5]]");
/// ```
///
/// # Notes of API accordance
///
/// - NumPy: `numpy.vstack(tup)` ([`numpy.vstack`](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html))
/// - RSTSR: `rt::vstack(tensors)`.
///
/// # Panics
///
/// - Panics if `tensors` is empty, or if the promoted inputs are not concatenable (mismatching
///   shape or device).
///
/// For a fallible version, use [`vstack_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - NumPy: [`numpy.vstack`](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html)
///
/// ## Related functions in RSTSR
///
/// - [`hstack`]: horizontal joining.
/// - [`concat`](concat()): joining along an explicit axis.
///
/// ## Variants of this function
///
/// - [`vstack_f`]: fallible version.
pub fn vstack<Args, Inp>(args: Args) -> Args::Out
where
    Args: VStackAPI<Inp>,
{
    Args::vstack(args)
}

/// Stack tensors in sequence vertically (row-wise).
///
/// See also [`vstack`].
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
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
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

        // NumPy vstack: promote each input with `atleast_2d` (0-D -> (1, 1),
        // 1-D (N,) -> (1, N)), then concatenate along axis 0. The result is at
        // least 2-D. See [`atleast_2d`].
        let tensors = tensors.iter().map(|t| into_atleast_2d_f(t.view())).collect::<Result<Vec<_>>>()?;
        ConcatAPI::concat_f((tensors, 0))
    }
}

/* #endregion */

/* #region stack */

/// API trait backing [`stack`].
pub trait StackAPI<Inp> {
    type Out;

    fn stack_f(self) -> Result<Self::Out>;
    fn stack(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::stack_f(self).rstsr_unwrap()
    }
}

/// Joins a sequence of tensors along a new axis.
///
/// Equivalent to NumPy `stack`: a new axis of size one is inserted at `axis` in each
/// input, then the results are concatenated along that axis. 0-D inputs are supported
/// (they stack into a 1-D tensor). All inputs must have the same shape and device.
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders. (Only the memory arrangement of the new tensor follows the
/// device default order.)
///
/// # Overloads Table
///
/// Output is [`Tensor<T, B, IxD>`][`Tensor`]; `tensors` also accepts owned
/// `Vec<TensorAny>` / `[TensorAny; N]` forms.
///
/// - `stack(tensors: Vec<&TensorAny<R, T, B, D>>) -> Tensor<T, B, IxD>` (implicit `axis = 0`)
/// - `stack((tensors, axis)) -> Tensor<T, B, IxD>` where `axis` is `isize`, `usize`, or `i32`
///
/// # Parameters
///
/// - `tensors`: the tensors to join; at least one is required.
/// - `axis`: the position of the new axis; valid range is `0..=ndim` (negative values count from
///   the back). Defaults to `0` if omitted.
///
/// # Returns
///
/// - [`Tensor<T, B, IxD>`][`Tensor`]: the stacked tensor, owning its data.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((4, &device));
/// let b = rt::full(([4], 9, &device));
/// println!("{}", rt::stack([&a, &b]));
/// // [[ 0 1 2 3]
/// //  [ 9 9 9 9]]
/// # assert_eq!(format!("{}", rt::stack([&a, &b])), "[[ 0 1 2 3]\n [ 9 9 9 9]]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `stack(arrays, /, *, axis=0)` ([`stack`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.stack.html))
/// - NumPy: `numpy.stack(arrays, axis=0, out=None)` ([`numpy.stack`](https://numpy.org/doc/stable/reference/generated/numpy.stack.html))
/// - RSTSR: `rt::stack((tensors, axis))`; `out` is not supported.
///
/// # Panics
///
/// - Panics if `tensors` is empty, if the inputs have mismatching shape or device, or if `axis` is
///   out of range.
///
/// For a fallible version, use [`stack_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - Python Array API standard: [`stack`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.stack.html)
/// - NumPy: [`numpy.stack`](https://numpy.org/doc/stable/reference/generated/numpy.stack.html)
///
/// ## Related functions in RSTSR
///
/// - [`concat`](concat()): join along an existing axis.
/// - [`unstack`]: split along an axis (reverse operation).
///
/// ## Variants of this function
///
/// - [`stack_f`]: fallible version.
pub fn stack<Args, Inp>(args: Args) -> Args::Out
where
    Args: StackAPI<Inp>,
{
    Args::stack(args)
}

/// Joins a sequence of tensors along a new axis.
///
/// See also [`stack`].
pub fn stack_f<Args, Inp>(args: Args) -> Result<Args::Out>
where
    Args: StackAPI<Inp>,
{
    Args::stack_f(args)
}

impl<R, T, B, D> StackAPI<()> for (Vec<TensorAny<R, T, B, D>>, isize)
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
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

        // NumPy allows 0-D inputs to `stack` (they stack into a 1-D array); the
        // `into_expand_dims_f` path below handles 0-D by inserting the new axis.
        tensors.iter().try_for_each(|tensor| -> Result<()> {
            rstsr_assert_eq!(tensor.shape(), shape_orig, InvalidLayout, "All tensors must have the same shape.")?;
            rstsr_assert!(
                tensor.device().same_device(&device),
                DeviceMismatch,
                "All tensors must be on the same device."
            )?;
            Ok(())
        })?;

        // check and make axis positive (stack inserts a new axis, so 0..=ndim is valid)
        let axis = rstsr_check_axis_insert!(axis, ndim)?;

        // expand the shape of each tensor
        let tensors = tensors.into_iter().map(|tensor| tensor.into_expand_dims_f(axis)).collect::<Result<Vec<_>>>()?;

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
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
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
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
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

/// API trait backing [`unstack`].
pub trait UnstackAPI<Inp> {
    type Out;

    fn unstack_f(self) -> Result<Self::Out>;
    fn unstack(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::unstack_f(self).rstsr_unwrap()
    }
}

/// Splits a tensor into a sequence of views along the given axis.
///
/// The reverse of [`stack`]: a tensor of shape `(n0, ..., nk, ..., nN-1)` is
/// split into `nk` views of shape `(n0, ..., n_k_removed, ..., nN-1)`. The
/// returned tensors are views sharing the input's data (no copy).
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
///
/// # Overloads Table
///
/// Output is [`Vec<TensorView<'a, T, B, D::SmallerOne>>`][`TensorView`] (one
/// view per slice along `axis`; the output dimensionality is one less than the
/// input's).
///
/// - `unstack(tensor: &'a TensorAny<R, T, B, D>) -> Vec<TensorView<...>>` (implicit `axis = 0`)
/// - `unstack((tensor: &'a TensorAny<R, T, B, D>, axis: isize)) -> Vec<TensorView<...>>`
/// - `unstack(tensor: TensorView<'a, T, B, D>) -> Vec<TensorView<...>>` / `unstack((view, axis))`
///
/// # Parameters
///
/// - `tensor`: the tensor to split; must have `ndim > 0`.
/// - `axis`: the axis to split along; negative values count from the back. Defaults to `0` if
///   omitted.
///
/// # Returns
///
/// - A vector of [`TensorView`] slices along `axis`, sharing the input's data.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((6, &device)).into_shape([2, 3]);
/// let v = rt::unstack(&a);
/// println!("{}", v.len());
/// // 2
/// println!("{}", v[0]);
/// // [ 0 1 2]
/// println!("{}", v[1]);
/// // [ 3 4 5]
/// # assert_eq!(format!("{}", v[1]), "[ 3 4 5]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `unstack(x, /, *, axis=0)` ([`unstack`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.unstack.html))
/// - PyTorch: `torch.unstack(input, dim=0)` ([`torch.unstack`](https://docs.pytorch.org/docs/stable/generated/torch.unstack.html))
/// - RSTSR: `rt::unstack((tensor, axis))`; the results are views, not copies.
///
/// # Panics
///
/// - Panics if the input has `ndim == 0`, or if `axis` is out of range.
///
/// For a fallible version, use [`unstack_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - Python Array API standard: [`unstack`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.unstack.html)
/// - PyTorch: [`torch.unstack`](https://docs.pytorch.org/docs/stable/generated/torch.unstack.html)
///
/// ## Related functions in RSTSR
///
/// - [`stack`]: join along a new axis (reverse operation).
/// - [`diagonal`]: extract diagonal views.
///
/// ## Variants of this function
///
/// - [`unstack_f`]: fallible version.
pub fn unstack<Args, Inp>(args: Args) -> Args::Out
where
    Args: UnstackAPI<Inp>,
{
    Args::unstack(args)
}

/// Splits a tensor into a sequence of views along the given axis.
///
/// See also [`unstack`].
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
        rstsr_assert!(tensor.ndim() > 0, InvalidLayout, "unstack requires a tensor with ndim > 0.")?;

        // check axis
        let ndim = tensor.ndim();
        let axis = rstsr_check_axis!(axis, ndim)?;

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

/* #region atleast */

/// View a tensor as having at least one dimension.
///
/// See also [`atleast_1d`].
pub fn into_atleast_1d_f<S, D>(tensor: TensorBase<S, D>) -> Result<TensorBase<S, IxD>>
where
    D: DimAPI,
{
    match tensor.ndim() {
        // 0-D -> (1,): insert a new axis 0
        0 => into_expand_dims_f(tensor, vec![0]),
        _ => into_dim_f::<_, _, IxD>(tensor),
    }
}

/// View a tensor as having at least two dimensions.
///
/// See also [`atleast_2d`].
pub fn into_atleast_2d_f<S, D>(tensor: TensorBase<S, D>) -> Result<TensorBase<S, IxD>>
where
    D: DimAPI,
{
    match tensor.ndim() {
        // 0-D -> (1, 1)
        0 => into_expand_dims_f(tensor, vec![0, 1]),
        // 1-D (N,) -> (1, N): insert a new axis 0
        1 => into_expand_dims_f(tensor, vec![0]),
        _ => into_dim_f::<_, _, IxD>(tensor),
    }
}

/// View a tensor as having at least three dimensions.
///
/// See also [`atleast_3d`].
pub fn into_atleast_3d_f<S, D>(tensor: TensorBase<S, D>) -> Result<TensorBase<S, IxD>>
where
    D: DimAPI,
{
    match tensor.ndim() {
        // 0-D -> (1, 1, 1)
        0 => into_expand_dims_f(tensor, vec![0, 1, 2]),
        // 1-D (N,) -> (1, N, 1): insert axis 0, then axis 2
        1 => into_expand_dims_f(tensor, vec![0, 2]),
        // 2-D (M, N) -> (M, N, 1): insert axis 2
        2 => into_expand_dims_f(tensor, vec![2]),
        _ => into_dim_f::<_, _, IxD>(tensor),
    }
}

/// View a tensor as having at least one dimension.
///
/// See also [`atleast_1d`].
pub fn into_atleast_1d<S, D>(tensor: TensorBase<S, D>) -> TensorBase<S, IxD>
where
    D: DimAPI,
{
    into_atleast_1d_f(tensor).rstsr_unwrap()
}

/// View a tensor as having at least two dimensions.
///
/// See also [`atleast_2d`].
pub fn into_atleast_2d<S, D>(tensor: TensorBase<S, D>) -> TensorBase<S, IxD>
where
    D: DimAPI,
{
    into_atleast_2d_f(tensor).rstsr_unwrap()
}

/// View a tensor as having at least three dimensions.
///
/// See also [`atleast_3d`].
pub fn into_atleast_3d<S, D>(tensor: TensorBase<S, D>) -> TensorBase<S, IxD>
where
    D: DimAPI,
{
    into_atleast_3d_f(tensor).rstsr_unwrap()
}

/// View a tensor as having at least one dimension.
///
/// See also [`atleast_1d`].
pub fn atleast_1d_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> Result<TensorView<'_, T, B, IxD>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    into_atleast_1d_f(tensor.view())
}

/// View a tensor as having at least 1 dimension.
///
/// Equivalent to NumPy `atleast_1d`: a 0-D (scalar) tensor is reshaped to `(1,)`.
/// Tensors with `ndim >= 1` are returned unchanged. The result is a view
/// (no copy).
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
///
/// # Overloads Table
///
/// Output is [`TensorView<'_, T, B, IxD>`][`TensorView`].
///
/// - `atleast_1d(tensor: &TensorAny<R, T, B, D>) -> TensorView<'_, T, B, IxD>`
///
/// Also, ownership-consuming forms [`into_atleast_1d`] / [`into_atleast_1d_f`] behave
/// the same for any [`TensorBase`] input and preserve its ownership.
///
/// # Parameters
///
/// - `tensor`: the input tensor.
///
/// # Returns
///
/// - [`TensorView<'_, T, B, IxD>`][`TensorView`]: view of the input with at least 1 dimension.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((3, &device));
/// println!("{}", rt::atleast_1d(&a));
/// // [ 0 1 2]
/// # assert_eq!(format!("{}", rt::atleast_1d(&a)), "[ 0 1 2]");
/// ```
///
/// # Notes of API accordance
///
/// - NumPy: `numpy.atleast_1d(x)` ([`numpy.atleast_1d`](https://numpy.org/doc/stable/reference/generated/numpy.atleast_1d.html))
/// - RSTSR: `rt::atleast_1d(tensor)`; dimension promotion follows NumPy's convention, and the
///   result is always a view with dynamic dimensionality.
///
/// # Panics
///
/// This function does not panic; layout conversions are always valid. For a
/// fallible version, use [`atleast_1d_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - NumPy: [`numpy.atleast_1d`](https://numpy.org/doc/stable/reference/generated/numpy.atleast_1d.html)
///
/// ## Related functions in RSTSR
///
/// - [`expand_dims`]: insert axes at arbitrary positions.
/// - [`squeeze`]: remove size-one axes.
///
/// ## Variants of this function
///
/// - [`atleast_1d_f`]: fallible version.
/// - [`into_atleast_1d`] / [`into_atleast_1d_f`]: ownership-consuming forms.
/// - Associated methods on [`TensorAny`]: [`TensorAny::atleast_1d`] / [`TensorAny::atleast_1d_f`].
pub fn atleast_1d<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> TensorView<'_, T, B, IxD>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    atleast_1d_f(tensor).rstsr_unwrap()
}

/// View a tensor as having at least two dimensions.
///
/// See also [`atleast_2d`].
pub fn atleast_2d_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> Result<TensorView<'_, T, B, IxD>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    into_atleast_2d_f(tensor.view())
}

/// View a tensor as having at least 2 dimensions.
///
/// Equivalent to NumPy `atleast_2d`: a 0-D tensor becomes `(1, 1)`, and a 1-D tensor `(N,)` becomes
/// `(1, N)`. Tensors with `ndim >= 2` are returned unchanged. The result is a view
/// (no copy).
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
///
/// # Overloads Table
///
/// Output is [`TensorView<'_, T, B, IxD>`][`TensorView`].
///
/// - `atleast_2d(tensor: &TensorAny<R, T, B, D>) -> TensorView<'_, T, B, IxD>`
///
/// Also, ownership-consuming forms [`into_atleast_2d`] / [`into_atleast_2d_f`] behave
/// the same for any [`TensorBase`] input and preserve its ownership.
///
/// # Parameters
///
/// - `tensor`: the input tensor.
///
/// # Returns
///
/// - [`TensorView<'_, T, B, IxD>`][`TensorView`]: view of the input with at least 2 dimensions.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((3, &device));
/// println!("{}", rt::atleast_2d(&a));
/// // [[ 0 1 2]]
/// # assert_eq!(format!("{}", rt::atleast_2d(&a)), "[[ 0 1 2]]");
/// ```
///
/// # Notes of API accordance
///
/// - NumPy: `numpy.atleast_2d(x)` ([`numpy.atleast_2d`](https://numpy.org/doc/stable/reference/generated/numpy.atleast_2d.html))
/// - RSTSR: `rt::atleast_2d(tensor)`; dimension promotion follows NumPy's convention, and the
///   result is always a view with dynamic dimensionality.
///
/// # Panics
///
/// This function does not panic; layout conversions are always valid. For a
/// fallible version, use [`atleast_2d_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - NumPy: [`numpy.atleast_2d`](https://numpy.org/doc/stable/reference/generated/numpy.atleast_2d.html)
///
/// ## Related functions in RSTSR
///
/// - [`expand_dims`]: insert axes at arbitrary positions.
/// - [`squeeze`]: remove size-one axes.
///
/// ## Variants of this function
///
/// - [`atleast_2d_f`]: fallible version.
/// - [`into_atleast_2d`] / [`into_atleast_2d_f`]: ownership-consuming forms.
/// - Associated methods on [`TensorAny`]: [`TensorAny::atleast_2d`] / [`TensorAny::atleast_2d_f`].
pub fn atleast_2d<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> TensorView<'_, T, B, IxD>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    atleast_2d_f(tensor).rstsr_unwrap()
}

/// View a tensor as having at least three dimensions.
///
/// See also [`atleast_3d`].
pub fn atleast_3d_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> Result<TensorView<'_, T, B, IxD>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    into_atleast_3d_f(tensor.view())
}

/// View a tensor as having at least 3 dimensions.
///
/// Equivalent to NumPy `atleast_3d`: a 0-D tensor becomes `(1, 1, 1)`, a 1-D tensor `(N,)` becomes
/// `(1, N, 1)`, and a 2-D tensor `(M, N)` becomes `(M, N, 1)`. Tensors with `ndim >= 3` are
/// returned unchanged. The result is a view (no copy).
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
///
/// # Overloads Table
///
/// Output is [`TensorView<'_, T, B, IxD>`][`TensorView`].
///
/// - `atleast_3d(tensor: &TensorAny<R, T, B, D>) -> TensorView<'_, T, B, IxD>`
///
/// Also, ownership-consuming forms [`into_atleast_3d`] / [`into_atleast_3d_f`] behave
/// the same for any [`TensorBase`] input and preserve its ownership.
///
/// # Parameters
///
/// - `tensor`: the input tensor.
///
/// # Returns
///
/// - [`TensorView<'_, T, B, IxD>`][`TensorView`]: view of the input with at least 3 dimensions.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((3, &device));
/// println!("{}", rt::atleast_3d(&a));
/// // [[[ 0]
/// //   [ 1]
/// //   [ 2]]]
/// # assert_eq!(format!("{}", rt::atleast_3d(&a)), "[[[ 0]\n  [ 1]\n  [ 2]]]");
/// ```
///
/// # Notes of API accordance
///
/// - NumPy: `numpy.atleast_3d(x)` ([`numpy.atleast_3d`](https://numpy.org/doc/stable/reference/generated/numpy.atleast_3d.html))
/// - RSTSR: `rt::atleast_3d(tensor)`; dimension promotion follows NumPy's convention, and the
///   result is always a view with dynamic dimensionality.
///
/// # Panics
///
/// This function does not panic; layout conversions are always valid. For a
/// fallible version, use [`atleast_3d_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - NumPy: [`numpy.atleast_3d`](https://numpy.org/doc/stable/reference/generated/numpy.atleast_3d.html)
///
/// ## Related functions in RSTSR
///
/// - [`expand_dims`]: insert axes at arbitrary positions.
/// - [`squeeze`]: remove size-one axes.
///
/// ## Variants of this function
///
/// - [`atleast_3d_f`]: fallible version.
/// - [`into_atleast_3d`] / [`into_atleast_3d_f`]: ownership-consuming forms.
/// - Associated methods on [`TensorAny`]: [`TensorAny::atleast_3d`] / [`TensorAny::atleast_3d_f`].
pub fn atleast_3d<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> TensorView<'_, T, B, IxD>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    atleast_3d_f(tensor).rstsr_unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// View the tensor as having at least one dimension. See also [`atleast_1d`].
    pub fn atleast_1d(&self) -> TensorView<'_, T, B, IxD> {
        atleast_1d(self)
    }
    /// Fallible variant of [`atleast_1d`](Self::atleast_1d).
    pub fn atleast_1d_f(&self) -> Result<TensorView<'_, T, B, IxD>> {
        atleast_1d_f(self)
    }
    /// View the tensor as having at least two dimensions. See also [`atleast_2d`].
    pub fn atleast_2d(&self) -> TensorView<'_, T, B, IxD> {
        atleast_2d(self)
    }
    /// Fallible variant of [`atleast_2d`](Self::atleast_2d).
    pub fn atleast_2d_f(&self) -> Result<TensorView<'_, T, B, IxD>> {
        atleast_2d_f(self)
    }
    /// View the tensor as having at least three dimensions. See also [`atleast_3d`].
    pub fn atleast_3d(&self) -> TensorView<'_, T, B, IxD> {
        atleast_3d(self)
    }
    /// Fallible variant of [`atleast_3d`](Self::atleast_3d).
    pub fn atleast_3d_f(&self) -> Result<TensorView<'_, T, B, IxD>> {
        atleast_3d_f(self)
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
