//! Implementation of function `asarray`.

use crate::prelude_dev::*;
use core::mem::ManuallyDrop;
use num::complex::{Complex32, Complex64};

/// Trait for function [`asarray`] impl: converting the input to an array.
///
/// This trait can be implemented for different backends. For usual CPU backends, we refer to
/// function [`asarray`] for API documentation details.
pub trait AsArrayAPI<Inp> {
    type Out;

    fn asarray_f(self) -> Result<Self::Out>;

    fn asarray(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::asarray_f(self).rstsr_unwrap()
    }
}

/// Convert the input to an array.
///
/// **This function is overloaded.**
///
/// <div class="warning">
///
/// **Row/Column Major Notice**
///
/// When passing shape into this function, the layout of output tensor will be different on default
/// orders ([`RowMajor`] and [`ColMajor`]) of device.
///
/// </div>
///
/// Note that this function always returns a dynamic-dimensional ([`IxD`]) tensor. To convert it
/// into a fixed-dimensional tensor, you can use [`.into_dim::<D>()`](Tensor::into_dim) method on
/// the output tensor without explicit data copy.
///
/// # Overloads Table
///
/// ## Output owned tensor [`Tensor`]
///
/// Input vector [`Vec<T>`] as raw data:
///
/// - `asarray((input: Vec<T>, layout: Layout<D>, device: &B)) -> Tensor<T, B, IxD>`
/// - `asarray((input: Vec<T>, shape: D, device: &B)) -> Tensor<T, B, IxD>`
/// - `asarray((input: Vec<T>, device: &B)) -> Tensor<T, B, IxD>`
/// - `asarray((input: Vec<T>, layout: Layout<D>)) -> Tensor<T, DeviceCpu, IxD>`
/// - `asarray((input: Vec<T>, shape: D)) -> Tensor<T, DeviceCpu, IxD>`
/// - `asarray(input: Vec<T>) -> Tensor<T, DeviceCpu, IxD>`
///
/// Input scalar `T` as raw data:
///
/// - `asarray((input: T, device: &B)) -> Tensor<T, B, IxD>`
/// - `asarray(input: T) -> Tensor<T, DeviceCpu, IxD>`
///
/// Input tensor as raw data and change its layout:
///
/// - `asarray(input: &TensorAny<R, T, B, D>) -> Tensor<T, B, D>`
/// - `asarray((input: &TensorAny<R, T, B, D>, order: TensorIterOrder)) -> Tensor<T, B, D>`
/// - `asarray(input: Tensor<T, B, D>) -> Tensor<T, B, D>`
/// - `asarray((input: Tensor<T, B, D>, order: TensorIterOrder)) -> Tensor<T, B, D>`
///
/// ## Output tensor view [`TensorView`]
///
/// - `asarray((input: &[T], layout: Layout<D>, device: &B)) -> TensorView<'a, T, B, IxD>`
/// - `asarray((input: &[T], shape: D, device: &B)) -> TensorView<'a, T, B, IxD>`
/// - `asarray((input: &[T], device: &B)) -> TensorView<'a, T, B, IxD>`
/// - `asarray((input: &[T], layout: Layout<D>)) -> TensorView<'a, T, DeviceCpu, IxD>`
/// - `asarray(input: &[T]) -> TensorView<'a, T, DeviceCpu, IxD>`
///
/// Also, overloads for `&Vec<T>` that behave the same as `&[T]`.
///
/// ## Output mutable tensor view [`TensorMut`]
///
/// All overloads for `&[T]` and `&Vec<T>` above also have mutable versions for `&mut [T]` and `&mut
/// Vec<T>`, which output [`TensorMut<'a, T, B, IxD>`] and [`TensorMut<'a, T, DeviceCpu, IxD>`],
/// respectively.
///
/// # Examples
///
/// ## Vector as input
///
/// The most usual usage is to convert a vector into a tensor. You can also specify the shape /
/// layout and device.
///
/// **The following example assumes that the device's default order is row-major**. The input shape
/// `[2, 3]` corresponds to a row-major layout `[2, 3].c()`.
///
/// ```rust
/// use rstsr::prelude::*;
/// let mut device = DeviceCpu::default();
/// device.set_default_order(RowMajor);
///
/// // vector as input, row-major layout by default
/// let input = vec![1, 2, 3, 4, 5, 6];
/// let a = rt::asarray((input, [2, 3], &device));
/// println!("{a:?}");
/// // [[1, 2, 3],
/// //  [4, 5, 6]]
/// // 2-Dim (dyn), contiguous: Cc, shape: [2, 3], stride: [3, 1], offset: 0
/// # let expected = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6]], &device);
/// # assert!(rt::allclose(&a, &expected, None));
/// ```
///
/// If you want to use column-major layout, you can specify the layout explicitly. But be cautious
/// that **row-major and column-major layouts will lead to different arrangements of data**.
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // vector as input, column-major layout
/// let input = vec![1, 2, 3, 4, 5, 6];
/// let a = rt::asarray((input, [2, 3].f(), &device));
/// println!("{a:?}");
/// // [[ 1 3 5]
/// //  [ 2 4 6]]
/// // 2-Dim (dyn), contiguous: Ff, shape: [2, 3], stride: [1, 2], offset: 0
/// # let expected = rt::tensor_from_nested!([[1, 3, 5], [2, 4, 6]], &device);
/// # assert!(rt::allclose(&a, &expected, None));
/// ```
///
/// Also, **if the device's default order is column-major**, the shape input (`[2, 3]`) will also
/// lead to a column-major layout (`[2, 3].f()`):
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // vector as input, column layout by default
/// device.set_default_order(ColMajor);
/// let input = vec![1, 2, 3, 4, 5, 6];
/// let a = rt::asarray((input, [2, 3], &device));
/// println!("{a:?}");
/// // [[ 1 3 5]
/// //  [ 2 4 6]]
/// // 2-Dim (dyn), contiguous: Ff, shape: [2, 3], stride: [1, 2], offset: 0
/// # let expected = rt::tensor_from_nested!([[1, 3, 5], [2, 4, 6]], &device);
/// # assert!(rt::allclose(&a, &expected, None));
/// ```
///
/// Finally, you can omit the device argument to use the default CPU device, and omit the
/// layout/shape to get an 1-D tensor:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let input = vec![1, 2, 3, 4, 5, 6];
/// let a = rt::asarray(input);;
/// println!("{a:?}");
/// // [ 1 2 3 4 5 6]
/// // 1-Dim (dyn), contiguous: Cc, shape: [6], stride: [1], offset: 0
/// # let expected = rt::tensor_from_nested!([1, 2, 3, 4, 5, 6]);
/// # assert!(rt::allclose(&a, &expected, None));
/// ```
///
/// ## `&[T]` or `&mut [T]` as input
///
/// You can also convert a slice into a tensor view. Please note, `asarray` accepts `&[T]` and
/// `&Vec<T>`, but do not accept other slice-like types such as `&[T; N]`. You may need to convert
/// them by `.as_ref()` first.
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // Slice &[T] as input
/// let input = &[1, 2, 3, 4, 5, 6];
/// let a = rt::asarray((input.as_ref(), [2, 3].c(), &device));
/// println!("{a:?}");
/// // [[ 1 2 3]
/// //  [ 4 5 6]]
/// # let expected = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6]], &device);
/// # assert!(rt::allclose(&a, &expected, None));
/// ```
///
/// Also, mutable slices `&mut [T]` and `&mut Vec<T>` are supported. You can modify the original
/// data via the output tensor mutable view.
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // Slice &mut [T] as input
/// let mut input = vec![1, 2, 3, 4, 5, 6];
/// let mut a = rt::asarray((&mut input, [2, 3].c(), &device));
/// // change `input` via tensor view `a`
/// a[[0, 0]] = 10;
/// println!("{a:2?}");
/// // [[ 10  2  3]
/// //  [  4  5  6]]
/// # let expected = rt::tensor_from_nested!([[10, 2, 3], [4, 5, 6]], &device);
/// # assert!(rt::allclose(&a, &expected, None));
/// println!("{input:?}");
/// // [10, 2, 3, 4, 5, 6]
/// # assert_eq!(input, vec![10, 2, 3, 4, 5, 6]);
/// ```
///
/// You can also specify a sub-view via layout:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let input = (0..30).collect::<Vec<i32>>();
/// let layout = Layout::new([3, 2], [2, 7], 5).unwrap();
/// let a = rt::asarray((&input, layout, &device));
/// println!("{a:2?}");
/// // [[  5 12]
/// //  [  7 14]
/// //  [  9 16]]
/// # let expected = rt::tensor_from_nested!([[5, 12], [7, 14], [9, 16]], &device);
/// # assert!(rt::allclose(&a, &expected, None));
/// ```
///
/// Finally, you can also omit the device argument to use the default CPU device.
///
/// ## Scalar as input
///
/// You can also convert a scalar into a tensor with zero dimensions.
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::asarray((42, &device));
/// println!("{a:?}");
/// // 42
/// // 0-Dim (dyn), contiguous: CcFf, shape: [], stride: [], offset: 0
/// ```
pub fn asarray<Args, Inp>(param: Args) -> Args::Out
where
    Args: AsArrayAPI<Inp>,
{
    return AsArrayAPI::asarray(param);
}

pub fn asarray_f<Args, Inp>(param: Args) -> Result<Args::Out>
where
    Args: AsArrayAPI<Inp>,
{
    return AsArrayAPI::asarray_f(param);
}

/* #region tensor input */

impl<R, T, B, D> AsArrayAPI<()> for (&TensorAny<R, T, B, D>, TensorIterOrder)
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
{
    type Out = Tensor<T, B, D>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, order) = self;
        let device = input.device();
        let layout_a = input.layout();
        let layout_c = layout_for_array_copy(layout_a, order)?;
        let mut storage_c = device.uninit_impl(layout_c.size())?;
        device.assign_uninit(storage_c.raw_mut(), &layout_c, input.raw(), layout_a)?;
        let storage_c = unsafe { B::assume_init_impl(storage_c) }?;
        let tensor = unsafe { Tensor::new_unchecked(storage_c, layout_c) };
        return Ok(tensor);
    }
}

impl<R, T, B, D> AsArrayAPI<()> for &TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
{
    type Out = Tensor<T, B, D>;

    fn asarray_f(self) -> Result<Self::Out> {
        asarray_f((self, TensorIterOrder::default()))
    }
}

impl<T, B, D> AsArrayAPI<()> for (Tensor<T, B, D>, TensorIterOrder)
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
{
    type Out = Tensor<T, B, D>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, order) = self;
        let storage_a = input.storage();
        let layout_a = input.layout();
        let device = storage_a.device();
        let layout_c = layout_for_array_copy(layout_a, order)?;
        if layout_c == *layout_a {
            return Ok(input);
        } else {
            let mut storage_c = device.uninit_impl(layout_c.size())?;
            device.assign_uninit(storage_c.raw_mut(), &layout_c, storage_a.raw(), layout_a)?;
            let storage_c = unsafe { B::assume_init_impl(storage_c) }?;
            let tensor = unsafe { Tensor::new_unchecked(storage_c, layout_c) };
            return Ok(tensor);
        }
    }
}

impl<T, B, D> AsArrayAPI<()> for Tensor<T, B, D>
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
{
    type Out = Tensor<T, B, D>;

    fn asarray_f(self) -> Result<Self::Out> {
        asarray_f((self, TensorIterOrder::default()))
    }
}

/* #endregion */

/* #region vec-like input */

impl<T, B> AsArrayAPI<()> for (Vec<T>, &B)
where
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, device) = self;
        let layout = vec![input.len()].c();
        let storage = device.outof_cpu_vec(input)?;
        let tensor = unsafe { Tensor::new_unchecked(storage, layout) };
        return Ok(tensor);
    }
}

impl<T, B, D> AsArrayAPI<D> for (Vec<T>, Layout<D>, &B)
where
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout, device) = self;
        rstsr_assert_eq!(
            layout.bounds_index()?,
            (0, layout.size()),
            InvalidLayout,
            "This constructor assumes compact memory layout."
        )?;
        rstsr_assert_eq!(
            layout.size(),
            input.len(),
            InvalidLayout,
            "This constructor assumes that the layout size is equal to the input size."
        )?;
        let storage = device.outof_cpu_vec(input)?;
        let tensor = unsafe { Tensor::new_unchecked(storage, layout.into_dim()?) };
        return Ok(tensor);
    }
}

impl<T, B, D> AsArrayAPI<D> for (Vec<T>, D, &B)
where
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    type Out = Tensor<T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, shape, device) = self;
        let default_order = device.default_order();
        let layout = match default_order {
            RowMajor => shape.c(),
            ColMajor => shape.f(),
        };
        asarray_f((input, layout, device))
    }
}

impl<T> AsArrayAPI<()> for Vec<T>
where
    T: Clone,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        asarray_f((self, &DeviceCpu::default()))
    }
}

#[duplicate_item(L; [D]; [Layout<D>])]
impl<T, D> AsArrayAPI<D> for (Vec<T>, L)
where
    T: Clone,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceCpu, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout) = self;
        asarray_f((input, layout, &DeviceCpu::default()))
    }
}

impl<T> From<Vec<T>> for Tensor<T, DeviceCpu, IxD>
where
    T: Clone,
{
    fn from(input: Vec<T>) -> Self {
        asarray_f(input).rstsr_unwrap()
    }
}

/* #endregion */

/* #region slice-like input */

impl<'a, T, B, D> AsArrayAPI<D> for (&'a [T], Layout<D>, &B)
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>>,
    D: DimAPI,
{
    type Out = TensorView<'a, T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout, device) = self;
        let ptr = input.as_ptr();
        let len = input.len();
        let raw = unsafe {
            let ptr = ptr as *mut T;
            Vec::from_raw_parts(ptr, len, len)
        };
        let device = device.clone();
        let data = DataRef::from_manually_drop(ManuallyDrop::new(raw));
        let storage = Storage::new(data, device);
        let tensor = unsafe { TensorView::new_unchecked(storage, layout.into_dim()?) };
        return Ok(tensor);
    }
}

impl<'a, T, B, D> AsArrayAPI<D> for (&'a [T], D, &B)
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>>,
    D: DimAPI,
{
    type Out = TensorView<'a, T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, shape, device) = self;
        let default_order = device.default_order();
        let layout = match default_order {
            RowMajor => shape.c(),
            ColMajor => shape.f(),
        };
        asarray_f((input, layout, device))
    }
}

impl<'a, T, B> AsArrayAPI<()> for (&'a [T], &B)
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    type Out = TensorView<'a, T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, device) = self;
        let layout = vec![input.len()].c();
        let device = device.clone();

        let ptr = input.as_ptr();
        let len = input.len();
        let raw = unsafe {
            let ptr = ptr as *mut T;
            Vec::from_raw_parts(ptr, len, len)
        };
        let data = DataRef::from_manually_drop(ManuallyDrop::new(raw));
        let storage = Storage::new(data, device);
        let tensor = unsafe { TensorView::new_unchecked(storage, layout) };
        return Ok(tensor);
    }
}

#[duplicate_item(L; [D]; [Layout<D>])]
impl<'a, T, D> AsArrayAPI<D> for (&'a [T], L)
where
    T: Clone,
    D: DimAPI,
{
    type Out = TensorView<'a, T, DeviceCpu, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout) = self;
        asarray_f((input, layout, &DeviceCpu::default()))
    }
}

impl<'a, T> AsArrayAPI<()> for &'a [T]
where
    T: Clone,
{
    type Out = TensorView<'a, T, DeviceCpu, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        asarray_f((self, &DeviceCpu::default()))
    }
}

#[duplicate_item(L; [D]; [Layout<D>])]
impl<'a, T, B, D> AsArrayAPI<D> for (&'a Vec<T>, L, &B)
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>> + 'a,
    D: DimAPI,
{
    type Out = TensorView<'a, T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout, device) = self;
        asarray_f((input.as_slice(), layout, device))
    }
}

impl<'a, T, B> AsArrayAPI<()> for (&'a Vec<T>, &B)
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    type Out = TensorView<'a, T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, device) = self;
        asarray_f((input.as_slice(), device))
    }
}

#[duplicate_item(L; [D]; [Layout<D>])]
impl<'a, T, D> AsArrayAPI<D> for (&'a Vec<T>, L)
where
    T: Clone,
    D: DimAPI,
{
    type Out = TensorView<'a, T, DeviceCpu, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout) = self;
        asarray_f((input.as_slice(), layout, &DeviceCpu::default()))
    }
}

impl<'a, T> AsArrayAPI<()> for &'a Vec<T>
where
    T: Clone,
{
    type Out = TensorView<'a, T, DeviceCpu, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        asarray_f((self.as_slice(), &DeviceCpu::default()))
    }
}

impl<'a, T> From<&'a [T]> for TensorView<'a, T, DeviceCpu, IxD>
where
    T: Clone,
{
    fn from(input: &'a [T]) -> Self {
        asarray(input)
    }
}

impl<'a, T> From<&'a Vec<T>> for TensorView<'a, T, DeviceCpu, IxD>
where
    T: Clone,
{
    fn from(input: &'a Vec<T>) -> Self {
        asarray(input)
    }
}

/* #endregion */

/* #region slice-like mutable input */

impl<'a, T, B, D> AsArrayAPI<D> for (&'a mut [T], Layout<D>, &B)
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>>,
    D: DimAPI,
{
    type Out = TensorMut<'a, T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout, device) = self;
        let ptr = input.as_ptr();
        let len = input.len();
        let raw = unsafe {
            let ptr = ptr as *mut T;
            Vec::from_raw_parts(ptr, len, len)
        };
        let device = device.clone();
        let data = DataMut::from_manually_drop(ManuallyDrop::new(raw));
        let storage = Storage::new(data, device);
        let tensor = unsafe { TensorMut::new_unchecked(storage, layout.into_dim()?) };
        return Ok(tensor);
    }
}

impl<'a, T, B, D> AsArrayAPI<D> for (&'a mut [T], D, &B)
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>>,
    D: DimAPI,
{
    type Out = TensorMut<'a, T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, shape, device) = self;
        let default_order = device.default_order();
        let layout = match default_order {
            RowMajor => shape.c(),
            ColMajor => shape.f(),
        };
        asarray_f((input, layout, device))
    }
}

impl<'a, T, B> AsArrayAPI<()> for (&'a mut [T], &B)
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    type Out = TensorMut<'a, T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, device) = self;
        let layout = [input.len()].c();
        let device = device.clone();

        let ptr = input.as_ptr();
        let len = input.len();
        let raw = unsafe {
            let ptr = ptr as *mut T;
            Vec::from_raw_parts(ptr, len, len)
        };
        let data = DataMut::from_manually_drop(ManuallyDrop::new(raw));
        let storage = Storage::new(data, device);
        let tensor = unsafe { TensorMut::new_unchecked(storage, layout.into_dim()?) };
        return Ok(tensor);
    }
}

#[duplicate_item(L; [D]; [Layout<D>])]
impl<'a, T, D> AsArrayAPI<D> for (&'a mut [T], L)
where
    T: Clone,
    D: DimAPI,
{
    type Out = TensorMut<'a, T, DeviceCpu, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout) = self;
        asarray_f((input, layout, &DeviceCpu::default()))
    }
}

impl<'a, T> AsArrayAPI<()> for &'a mut [T]
where
    T: Clone,
{
    type Out = TensorMut<'a, T, DeviceCpu, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        asarray_f((self, &DeviceCpu::default()))
    }
}

#[duplicate_item(L; [D]; [Layout<D>])]
impl<'a, T, B, D> AsArrayAPI<D> for (&'a mut Vec<T>, L, &B)
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>>,
    D: DimAPI,
{
    type Out = TensorMut<'a, T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout, device) = self;
        asarray_f((input.as_mut_slice(), layout, device))
    }
}

impl<'a, T, B> AsArrayAPI<()> for (&'a mut Vec<T>, &B)
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    type Out = TensorMut<'a, T, B, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, device) = self;
        asarray_f((input.as_mut_slice(), device))
    }
}

#[duplicate_item(L; [D]; [Layout<D>])]
impl<'a, T, D> AsArrayAPI<D> for (&'a mut Vec<T>, L)
where
    T: Clone,
    D: DimAPI,
{
    type Out = TensorMut<'a, T, DeviceCpu, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        let (input, layout) = self;
        asarray_f((input.as_mut_slice(), layout, &DeviceCpu::default()))
    }
}

impl<'a, T> AsArrayAPI<()> for &'a mut Vec<T>
where
    T: Clone,
{
    type Out = TensorMut<'a, T, DeviceCpu, IxD>;

    fn asarray_f(self) -> Result<Self::Out> {
        asarray_f((self.as_mut_slice(), &DeviceCpu::default()))
    }
}

impl<'a, T> From<&'a mut [T]> for TensorMut<'a, T, DeviceCpu, IxD>
where
    T: Clone,
{
    fn from(input: &'a mut [T]) -> Self {
        asarray(input)
    }
}

impl<'a, T> From<&'a mut Vec<T>> for TensorMut<'a, T, DeviceCpu, IxD>
where
    T: Clone,
{
    fn from(input: &'a mut Vec<T>) -> Self {
        asarray(input)
    }
}

/* #endregion */

/* #region scalar input */

macro_rules! impl_asarray_scalar {
    ($($t:ty),*) => {
        $(
            impl<B> AsArrayAPI<()> for ($t, &B)
            where
                B: DeviceAPI<$t> + DeviceCreationAnyAPI<$t>,
            {
                type Out = Tensor<$t, B, IxD>;

                fn asarray_f(self) -> Result<Self::Out> {
                    let (input, device) = self;
                    let layout = Layout::new(vec![], vec![], 0)?;
                    let storage = device.outof_cpu_vec(vec![input])?;
                    let tensor = unsafe { Tensor::new_unchecked(storage, layout) };
                    return Ok(tensor);
                }
            }

            impl AsArrayAPI<()> for $t {
                type Out = Tensor<$t, DeviceCpu, IxD>;

                fn asarray_f(self) -> Result<Self::Out> {
                    asarray_f((self, &DeviceCpu::default()))
                }
            }
        )*
    };
}

impl_asarray_scalar!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64, Complex32, Complex64);

/* #endregion */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asarray() {
        let input = vec![1, 2, 3];
        let tensor = asarray_f(input).unwrap();
        println!("{tensor:?}");
        let input = [1, 2, 3];
        let tensor = asarray_f(input.as_ref()).unwrap();
        println!("{tensor:?}");

        let input = vec![1, 2, 3];
        let tensor = asarray_f(&input).unwrap();
        println!("{:?}", tensor.raw().as_ptr());
        println!("{tensor:?}");

        let tensor = asarray_f((&tensor, TensorIterOrder::K)).unwrap();
        println!("{tensor:?}");

        let tensor = asarray_f((tensor, TensorIterOrder::K)).unwrap();
        println!("{tensor:?}");
    }

    #[test]
    fn test_asarray_scalar() {
        let tensor = asarray_f(1).unwrap();
        println!("{tensor:?}");
        let tensor = asarray_f((Complex64::new(0., 1.), &DeviceCpuSerial::default())).unwrap();
        println!("{tensor:?}");
    }

    #[test]
    fn doc_asarray() {
        use rstsr::prelude::*;
        let mut device = DeviceCpu::default();
        device.set_default_order(RowMajor);

        // vector as input, row-major layout by default
        let input = vec![1, 2, 3, 4, 5, 6];
        let a = rt::asarray((input, [2, 3], &device));
        println!("{a:?}");
        // [[ 1 2 3]
        //  [ 4 5 6]]
        // 2-Dim (dyn), contiguous: Cc, shape: [2, 3], stride: [3, 1], offset: 0
        let expected = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6]], &device);
        assert!(rt::allclose(&a, &expected, None));

        // vector as input, column-major layout
        let input = vec![1, 2, 3, 4, 5, 6];
        let a = rt::asarray((input, [2, 3].f(), &device));
        println!("{a:?}");
        // [[ 1 3 5]
        //  [ 2 4 6]]
        // 2-Dim (dyn), contiguous: Ff, shape: [2, 3], stride: [1, 2], offset: 0
        let expected = rt::tensor_from_nested!([[1, 3, 5], [2, 4, 6]], &device);
        assert!(rt::allclose(&a, &expected, None));

        // vector as input, column layout by default
        device.set_default_order(ColMajor);
        let input = vec![1, 2, 3, 4, 5, 6];
        let a = rt::asarray((input, [2, 3], &device));
        println!("{a:?}");
        // [[ 1 3 5]
        //  [ 2 4 6]]
        // 2-Dim (dyn), contiguous: Ff, shape: [2, 3], stride: [1, 2], offset: 0
        let expected = rt::tensor_from_nested!([[1, 3, 5], [2, 4, 6]], &device);
        assert!(rt::allclose(&a, &expected, None));

        // 1-D vector, default CPU device
        let input = vec![1, 2, 3, 4, 5, 6];
        let a = rt::asarray(input);
        println!("{a:?}");
        // [ 1 2 3 4 5 6]
        // 1-Dim (dyn), contiguous: Cc, shape: [6], stride: [1], offset: 0
        let expected = rt::tensor_from_nested!([1, 2, 3, 4, 5, 6]);
        assert!(rt::allclose(&a, &expected, None));

        // Slice &[T] as input
        device.set_default_order(RowMajor);
        let input = &[1, 2, 3, 4, 5, 6];
        let a = rt::asarray((input.as_ref(), [2, 3].c(), &device));
        println!("{a:?}");
        // [[ 1 2 3]
        //  [ 4 5 6]]
        let expected = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6]], &device);
        assert!(rt::allclose(&a, &expected, None));

        // Slice &mut [T] as input
        let mut input = vec![1, 2, 3, 4, 5, 6];
        let mut a = rt::asarray((&mut input, [2, 3].c(), &device));
        // change `input` via tensor view `a`
        a[[0, 0]] = 10;
        println!("{a:2?}");
        // [[10 2 3]
        //  [ 4 5 6]]
        let expected = rt::tensor_from_nested!([[10, 2, 3], [4, 5, 6]], &device);
        assert!(rt::allclose(&a, &expected, None));
        println!("{input:?}");
        // [10, 2, 3, 4, 5, 6]
        assert_eq!(input, vec![10, 2, 3, 4, 5, 6]);

        // Sub-view from &Vec<T>
        let input = (0..30).collect::<Vec<i32>>();
        let layout = Layout::new([3, 2], [2, 7], 5).unwrap();
        let a = rt::asarray((&input, layout, &device));
        println!("{a:2?}");
        // [[  5 12]
        //  [  7 14]
        //  [  9 16]]
        let expected = rt::tensor_from_nested!([[5, 12], [7, 14], [9, 16]], &device);
        assert!(rt::allclose(&a, &expected, None));

        // Scalar as input
        let a = rt::asarray((42, &device));
        println!("{a:?}");
        // 42
        // 0-Dim (dyn), contiguous: CcFf, shape: [], stride: [], offset: 0
    }
}
