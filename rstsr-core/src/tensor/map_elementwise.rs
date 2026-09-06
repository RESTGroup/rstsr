//! Element-wise mapping over tensors: [`TensorAny::map`], [`TensorAny::mapv`],
//! [`TensorAny::mapi`], [`TensorAny::mapvi`], and their binary counterparts
//! [`TensorAny::mapb`] / [`TensorAny::mapvb`], plus `*_fnmut` variants for
//! non-`Send` closures.
//!
//! Naming: `mapv` passes elements by value (cloned), `mapi`/`mapvi` modify the
//! tensor in place, and `mapb`/`mapvb` map two tensors (broadcast against each
//! other) into a new one.

use crate::prelude_dev::*;
use core::mem::transmute;

/* #region map_fnmut */

// map, mapv, mapi, mapvi

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    /// Call `f` by reference on each element and create a new tensor with the
    /// new values.
    pub fn map_fnmut_f<'f, TOut>(&self, mut f: impl FnMut(&T) -> TOut + 'f) -> Result<Tensor<TOut, B, D>>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        B: Op_MutA_RefB_API<TOut, T, D, dyn FnMut(&mut MaybeUninit<TOut>, &T) + 'f>,
    {
        let la = self.layout();
        let lc = layout_for_array_copy(la, TensorIterOrder::default())?;
        let device = self.device();
        let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
        let mut f_inner = move |c: &mut MaybeUninit<TOut>, a: &T| {
            c.write(f(a));
        };
        device.op_muta_refb_func(storage_c.raw_mut(), &lc, self.raw(), la, &mut f_inner)?;
        let storage_c = unsafe { B::assume_init_impl(storage_c) }?;
        return Tensor::new_f(storage_c, lc);
    }

    /// Map a `FnMut` function over every element, producing a new tensor.
    ///
    /// Non-`Send` (`FnMut`) counterpart of [`TensorAny::map`], for
    /// closures that capture non-thread-safe state; single-threaded devices
    /// only.
    ///
    /// # See also
    ///
    /// [`TensorAny::map`].
    pub fn map_fnmut<'f, TOut>(&self, f: impl FnMut(&T) -> TOut + 'f) -> Tensor<TOut, B, D>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        B: Op_MutA_RefB_API<TOut, T, D, dyn FnMut(&mut MaybeUninit<TOut>, &T) + 'f>,
    {
        self.map_fnmut_f(f).rstsr_unwrap()
    }

    /// Call `f` by value on each element and create a new tensor with the new
    /// values.
    pub fn mapv_fnmut_f<'f, TOut>(&self, mut f: impl FnMut(T) -> TOut + 'f) -> Result<Tensor<TOut, B, D>>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        T: Clone,
        B: Op_MutA_RefB_API<TOut, T, D, dyn FnMut(&mut MaybeUninit<TOut>, &T) + 'f>,
    {
        self.map_fnmut_f(move |x| f(x.clone()))
    }

    /// Map a by-value `FnMut` function over every element.
    ///
    /// Non-`Send` (`FnMut`) counterpart of [`TensorAny::mapv`], for
    /// closures that capture non-thread-safe state; single-threaded devices
    /// only.
    ///
    /// # See also
    ///
    /// [`TensorAny::mapv`].
    pub fn mapv_fnmut<'f, TOut>(&self, mut f: impl FnMut(T) -> TOut + 'f) -> Tensor<TOut, B, D>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        T: Clone,
        B: Op_MutA_RefB_API<TOut, T, D, dyn FnMut(&mut MaybeUninit<TOut>, &T) + 'f>,
    {
        self.map_fnmut_f(move |x| f(x.clone())).rstsr_unwrap()
    }

    /// Modify the tensor in place by calling `f` by mutable reference on each
    /// element.
    pub fn mapi_fnmut_f<'f>(&mut self, mut f: impl FnMut(&mut T) + 'f) -> Result<()>
    where
        R: DataMutAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
        B: Op_MutA_API<T, D, dyn FnMut(&mut MaybeUninit<T>) + 'f>,
    {
        let (la, _) = greedy_layout(self.layout(), false);
        let device = self.device().clone();
        let self_raw_mut = unsafe {
            transmute::<&mut <B as DeviceRawAPI<T>>::Raw, &mut <B as DeviceRawAPI<MaybeUninit<T>>>::Raw>(self.raw_mut())
        };
        let mut f_inner = move |x: &mut MaybeUninit<T>| {
            let x_ref = unsafe { x.assume_init_mut() };
            f(x_ref);
        };
        device.op_muta_func(self_raw_mut, &la, &mut f_inner)
    }

    /// Modify the tensor in place with a `FnMut` function.
    ///
    /// Non-`Send` (`FnMut`) counterpart of [`TensorAny::mapi`], for
    /// closures that capture non-thread-safe state; single-threaded devices
    /// only.
    ///
    /// # See also
    ///
    /// [`TensorAny::mapi`].
    pub fn mapi_fnmut<'f>(&mut self, f: impl FnMut(&mut T) + 'f)
    where
        R: DataMutAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
        B: Op_MutA_API<T, D, dyn FnMut(&mut MaybeUninit<T>) + 'f>,
    {
        self.mapi_fnmut_f(f).rstsr_unwrap()
    }

    /// Modify the tensor in place by calling `f` by value on each
    /// element.
    pub fn mapvi_fnmut_f<'f>(&mut self, mut f: impl FnMut(T) -> T + 'f) -> Result<()>
    where
        R: DataMutAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
        T: Clone,
        B: Op_MutA_API<T, D, dyn FnMut(&mut MaybeUninit<T>) + 'f>,
    {
        self.mapi_fnmut_f(move |x| *x = f(x.clone()))
    }

    /// Modify the tensor in place with a by-value `FnMut` function.
    ///
    /// Non-`Send` (`FnMut`) counterpart of [`TensorAny::mapvi`], for
    /// closures that capture non-thread-safe state; single-threaded devices
    /// only.
    ///
    /// # See also
    ///
    /// [`TensorAny::mapvi`].
    pub fn mapvi_fnmut<'f>(&mut self, f: impl FnMut(T) -> T + 'f)
    where
        R: DataMutAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
        T: Clone,
        B: Op_MutA_API<T, D, dyn FnMut(&mut MaybeUninit<T>) + 'f>,
    {
        self.mapvi_fnmut_f(f).rstsr_unwrap()
    }
}

// map_binary, mapv_binary

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T>,
    T: Clone,
{
    pub fn mapb_fnmut_f<'f, R2, T2, D2, DOut, TOut>(
        &self,
        other: &TensorAny<R2, T2, B, D2>,
        mut f: impl FnMut(&T, &T2) -> TOut + 'f,
    ) -> Result<Tensor<TOut, B, DOut>>
    where
        R2: DataAPI<Data = <B as DeviceRawAPI<T2>>::Raw>,
        D2: DimAPI,
        DOut: DimAPI,
        D: DimMaxAPI<D2, Max = DOut>,
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        B: Op_MutC_RefA_RefB_API<T, T2, TOut, DOut, dyn FnMut(&mut MaybeUninit<TOut>, &T, &T2) + 'f>,
    {
        // get tensor views
        let a = self.view();
        let b = other.view();
        // check device and layout
        rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
        let la = a.layout();
        let lb = b.layout();
        let default_order = a.device().default_order();
        let (la_b, lb_b) = broadcast_layout(la, lb, default_order)?;
        // generate output layout
        let lc = match TensorIterOrder::default() {
            TensorIterOrder::C => la_b.shape().c(),
            TensorIterOrder::F => la_b.shape().f(),
            _ => get_layout_for_binary_op(&la_b, &lb_b, default_order)?,
        };
        // generate empty c
        let device = self.device();
        let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
        let mut f_inner = move |c: &mut MaybeUninit<TOut>, a: &T, b: &T2| {
            c.write(f(a, b));
        };
        device.op_mutc_refa_refb_func(storage_c.raw_mut(), &lc, self.raw(), &la_b, other.raw(), &lb_b, &mut f_inner)?;
        let storage_c = unsafe { B::assume_init_impl(storage_c) }?;
        Tensor::new_f(storage_c, lc)
    }

    /// Map a two-argument `FnMut` function over two tensors.
    ///
    /// Non-`Send` (`FnMut`) counterpart of [`TensorAny::mapb`], for
    /// closures that capture non-thread-safe state; single-threaded devices
    /// only.
    ///
    /// # See also
    ///
    /// [`TensorAny::mapb`].
    pub fn mapb_fnmut<'f, R2, T2, D2, DOut, TOut>(
        &self,
        other: &TensorAny<R2, T2, B, D2>,
        f: impl FnMut(&T, &T2) -> TOut + 'f,
    ) -> Tensor<TOut, B, DOut>
    where
        R2: DataAPI<Data = <B as DeviceRawAPI<T2>>::Raw>,
        D2: DimAPI,
        DOut: DimAPI,
        D: DimMaxAPI<D2, Max = DOut>,
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        B: Op_MutC_RefA_RefB_API<T, T2, TOut, DOut, dyn FnMut(&mut MaybeUninit<TOut>, &T, &T2) + 'f>,
    {
        self.mapb_fnmut_f(other, f).rstsr_unwrap()
    }

    pub fn mapvb_fnmut_f<'f, R2, T2, D2, DOut, TOut>(
        &self,
        other: &TensorAny<R2, T2, B, D2>,
        mut f: impl FnMut(T, T2) -> TOut + 'f,
    ) -> Result<Tensor<TOut, B, DOut>>
    where
        R2: DataAPI<Data = <B as DeviceRawAPI<T2>>::Raw>,
        D2: DimAPI,
        DOut: DimAPI,
        D: DimMaxAPI<D2, Max = DOut>,
        T: Clone,
        T2: Clone,
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        B: Op_MutC_RefA_RefB_API<T, T2, TOut, DOut, dyn FnMut(&mut MaybeUninit<TOut>, &T, &T2) + 'f>,
    {
        self.mapb_fnmut_f(other, move |x, y| f(x.clone(), y.clone()))
    }

    /// Map a by-value two-argument `FnMut` function over two tensors.
    ///
    /// Non-`Send` (`FnMut`) counterpart of [`TensorAny::mapvb`], for
    /// closures that capture non-thread-safe state; single-threaded devices
    /// only.
    ///
    /// # See also
    ///
    /// [`TensorAny::mapvb`].
    pub fn mapvb_fnmut<'f, R2, T2, D2, DOut, TOut>(
        &self,
        other: &TensorAny<R2, T2, B, D2>,
        mut f: impl FnMut(T, T2) -> TOut + 'f,
    ) -> Tensor<TOut, B, DOut>
    where
        R2: DataAPI<Data = <B as DeviceRawAPI<T2>>::Raw>,
        D2: DimAPI,
        DOut: DimAPI,
        D: DimMaxAPI<D2, Max = DOut>,
        T: Clone,
        T2: Clone,
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        B: Op_MutC_RefA_RefB_API<T, T2, TOut, DOut, dyn FnMut(&mut MaybeUninit<TOut>, &T, &T2) + 'f>,
    {
        self.mapb_fnmut_f(other, move |x, y| f(x.clone(), y.clone())).rstsr_unwrap()
    }
}

/* #endregion */

/* #region map sync */

// map, mapv, mapi, mapvi

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    /// Call `f` by reference on each element and create a new tensor with the
    /// new values.
    pub fn map_f<'f, TOut>(&self, f: impl Fn(&T) -> TOut + Send + Sync + 'f) -> Result<Tensor<TOut, B, D>>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        B: Op_MutA_RefB_API<TOut, T, D, dyn Fn(&mut MaybeUninit<TOut>, &T) + Send + Sync + 'f>,
    {
        let la = self.layout();
        let lc = layout_for_array_copy(la, TensorIterOrder::default())?;
        let device = self.device();
        let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
        let mut f_inner = move |c: &mut MaybeUninit<TOut>, a: &T| {
            c.write(f(a));
        };
        device.op_muta_refb_func(storage_c.raw_mut(), &lc, self.raw(), la, &mut f_inner)?;
        let storage_c = unsafe { B::assume_init_impl(storage_c) }?;
        return Tensor::new_f(storage_c, lc);
    }

    /// Map a function over every element, producing a new tensor.
    ///
    /// The closure takes each element by reference and returns the mapped
    /// value; the output dtype follows the closure's return type. The closure
    /// must be `Send + Sync` (for parallel devices); for `FnMut` closures see
    /// [`TensorAny::map_fnmut`].
    ///
    /// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
    ///
    /// # Parameters
    ///
    /// - `f`: the element-wise mapping function.
    ///
    /// # Returns
    ///
    /// - [`Tensor<TOut, B, D>`][`Tensor`]: the mapped tensor (same shape).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// let a = rt::arange((6, &device)).into_shape([2, 3]);
    /// let b = a.map(|x| x * 2);
    /// println!("{b}");
    /// // [[ 0 2 4]
    /// //  [ 6 8 10]]
    /// # assert_eq!(format!("{b}"), "[[ 0 2 4]\n [ 6 8 10]]");
    /// ```
    ///
    /// # Panics
    ///
    /// - Panics if the output storage cannot be allocated or written.
    ///
    /// For a fallible version, use [`TensorAny::map_f`].
    ///
    /// # See also
    ///
    /// ## Related functions in RSTSR
    ///
    /// - [`TensorAny::mapv`]: elements passed by value.
    /// - [`TensorAny::mapi`]: in-place modification.
    /// - [`TensorAny::mapb`]: two-tensor mapping.
    /// - [`TensorAny::map_fnmut`]: `FnMut` (non-`Send`) closures.
    ///
    /// ## Variants of this function
    ///
    /// - [`TensorAny::map_f`]: fallible version.
    pub fn map<'f, TOut>(&self, f: impl Fn(&T) -> TOut + Send + Sync + 'f) -> Tensor<TOut, B, D>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        B: Op_MutA_RefB_API<TOut, T, D, dyn Fn(&mut MaybeUninit<TOut>, &T) + Send + Sync + 'f>,
    {
        self.map_f(f).rstsr_unwrap()
    }

    /// Call `f` by value on each element and create a new tensor with the new
    /// values.
    pub fn mapv_f<'f, TOut>(&self, f: impl Fn(T) -> TOut + Send + Sync + 'f) -> Result<Tensor<TOut, B, D>>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        T: Clone,
        B: Op_MutA_RefB_API<TOut, T, D, dyn Fn(&mut MaybeUninit<TOut>, &T) + Send + Sync + 'f>,
    {
        self.map_f(move |x| f(x.clone()))
    }

    /// Call `f` by value on each element and create a new tensor with the new
    /// values.
    /// Map a function over every element by value, producing a new tensor;
    /// see [`TensorAny::map`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// let a = rt::arange((6, &device)).into_shape([2, 3]);
    /// let c = a.mapv(|x| x as f64 / 2.0);
    /// println!("{c}");
    /// // [[ 0 0.5 1]
    /// //  [ 1.5 2 2.5]]
    /// # assert_eq!(format!("{c}"), "[[ 0 0.5 1]\n [ 1.5 2 2.5]]");
    /// ```
    ///
    /// # See also
    ///
    /// [`TensorAny::map`].
    pub fn mapv<'f, TOut>(&self, f: impl Fn(T) -> TOut + Send + Sync + 'f) -> Tensor<TOut, B, D>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        T: Clone,
        B: Op_MutA_RefB_API<TOut, T, D, dyn Fn(&mut MaybeUninit<TOut>, &T) + Send + Sync + 'f>,
    {
        self.map_f(move |x| f(x.clone())).rstsr_unwrap()
    }

    /// Modify the tensor in place by calling `f` by mutable reference on each
    /// element.
    pub fn mapi_f<'f>(&mut self, f: impl Fn(&mut T) + Send + Sync + 'f) -> Result<()>
    where
        R: DataMutAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
        B: Op_MutA_API<T, D, dyn Fn(&mut MaybeUninit<T>) + Send + Sync + 'f>,
    {
        let (la, _) = greedy_layout(self.layout(), false);
        let device = self.device().clone();
        let self_raw_mut = unsafe {
            transmute::<&mut <B as DeviceRawAPI<T>>::Raw, &mut <B as DeviceRawAPI<MaybeUninit<T>>>::Raw>(self.raw_mut())
        };
        let mut f_inner = move |x: &mut MaybeUninit<T>| {
            let x_ref = unsafe { x.assume_init_mut() };
            f(x_ref);
        };
        device.op_muta_func(self_raw_mut, &la, &mut f_inner)
    }

    /// Modify the tensor in place by calling `f` by mutable reference on each
    /// element.
    /// Modify the tensor in place by mapping a function over every element;
    /// see [`TensorAny::map`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// let mut d = rt::arange((6, &device)).into_shape([2, 3]);
    /// d.mapi(|x| *x += 1);
    /// println!("{d}");
    /// // [[ 1 2 3]
    /// //  [ 4 5 6]]
    /// # assert_eq!(format!("{d}"), "[[ 1 2 3]\n [ 4 5 6]]");
    /// ```
    ///
    /// # See also
    ///
    /// [`TensorAny::map`].
    pub fn mapi<'f>(&mut self, f: impl Fn(&mut T) + Send + Sync + 'f)
    where
        R: DataMutAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
        B: Op_MutA_API<T, D, dyn Fn(&mut MaybeUninit<T>) + Send + Sync + 'f>,
    {
        self.mapi_f(f).rstsr_unwrap()
    }

    /// Modify the tensor in place by calling `f` by value on each
    /// element.
    pub fn mapvi_f<'f>(&mut self, f: impl Fn(T) -> T + Send + Sync + 'f) -> Result<()>
    where
        R: DataMutAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
        T: Clone,
        B: Op_MutA_API<T, D, dyn Fn(&mut MaybeUninit<T>) + Send + Sync + 'f>,
    {
        self.mapi_f(move |x| *x = f(x.clone()))
    }

    /// Modify the tensor in place by calling `f` by value on each
    /// element.
    /// Modify the tensor in place by mapping a by-value function over every
    /// element; see [`TensorAny::map`] and [`TensorAny::mapi`].
    ///
    /// # See also
    ///
    /// [`TensorAny::map`].
    pub fn mapvi<'f>(&mut self, f: impl Fn(T) -> T + Send + Sync + 'f)
    where
        R: DataMutAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
        T: Clone,
        B: Op_MutA_API<T, D, dyn Fn(&mut MaybeUninit<T>) + Send + Sync + 'f>,
    {
        self.mapvi_f(f).rstsr_unwrap()
    }
}

// map_binary, mapv_binary

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T>,
    T: Clone,
{
    pub fn mapb_f<'f, R2, T2, D2, DOut, TOut>(
        &self,
        other: &TensorAny<R2, T2, B, D2>,
        f: impl Fn(&T, &T2) -> TOut + Send + Sync + 'f,
    ) -> Result<Tensor<TOut, B, DOut>>
    where
        R2: DataAPI<Data = <B as DeviceRawAPI<T2>>::Raw>,
        D2: DimAPI,
        DOut: DimAPI,
        D: DimMaxAPI<D2, Max = DOut>,
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        B: Op_MutC_RefA_RefB_API<T, T2, TOut, DOut, dyn Fn(&mut MaybeUninit<TOut>, &T, &T2) + Send + Sync + 'f>,
    {
        // get tensor views
        let a = self.view();
        let b = other.view();
        // check device and layout
        rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
        let la = a.layout();
        let lb = b.layout();
        let default_order = a.device().default_order();
        let (la_b, lb_b) = broadcast_layout(la, lb, default_order)?;
        // generate output layout
        let lc = match TensorIterOrder::default() {
            TensorIterOrder::C => la_b.shape().c(),
            TensorIterOrder::F => la_b.shape().f(),
            _ => get_layout_for_binary_op(&la_b, &lb_b, default_order)?,
        };
        // generate empty c
        let device = self.device();
        let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
        let mut f_inner = move |c: &mut MaybeUninit<TOut>, a: &T, b: &T2| {
            c.write(f(a, b));
        };
        device.op_mutc_refa_refb_func(storage_c.raw_mut(), &lc, self.raw(), &la_b, other.raw(), &lb_b, &mut f_inner)?;
        let storage_c = unsafe { B::assume_init_impl(storage_c) }?;
        Tensor::new_f(storage_c, lc)
    }

    /// Map a two-argument function over the elements of two tensors
    /// (broadcast against each other), producing a new tensor; see
    /// [`TensorAny::map`].
    ///
    /// # See also
    ///
    /// [`TensorAny::map`].
    pub fn mapb<'f, R2, T2, D2, DOut, TOut>(
        &self,
        other: &TensorAny<R2, T2, B, D2>,
        f: impl Fn(&T, &T2) -> TOut + Send + Sync + 'f,
    ) -> Tensor<TOut, B, DOut>
    where
        R2: DataAPI<Data = <B as DeviceRawAPI<T2>>::Raw>,
        D2: DimAPI,
        DOut: DimAPI,
        D: DimMaxAPI<D2, Max = DOut>,
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        B: Op_MutC_RefA_RefB_API<T, T2, TOut, DOut, dyn Fn(&mut MaybeUninit<TOut>, &T, &T2) + Send + Sync + 'f>,
    {
        self.mapb_f(other, f).rstsr_unwrap()
    }

    pub fn mapvb_f<'f, R2, T2, D2, DOut, TOut>(
        &self,
        other: &TensorAny<R2, T2, B, D2>,
        f: impl Fn(T, T2) -> TOut + Send + Sync + 'f,
    ) -> Result<Tensor<TOut, B, DOut>>
    where
        R2: DataAPI<Data = <B as DeviceRawAPI<T2>>::Raw>,
        D2: DimAPI,
        DOut: DimAPI,
        D: DimMaxAPI<D2, Max = DOut>,
        T: Clone,
        T2: Clone,
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        B: Op_MutC_RefA_RefB_API<T, T2, TOut, DOut, dyn Fn(&mut MaybeUninit<TOut>, &T, &T2) + Send + Sync + 'f>,
    {
        self.mapb_f(other, move |x, y| f(x.clone(), y.clone()))
    }

    /// Map a by-value two-argument function over the elements of two tensors;
    /// see [`TensorAny::map`] and [`TensorAny::mapb`].
    ///
    /// # See also
    ///
    /// [`TensorAny::map`].
    pub fn mapvb<'f, R2, T2, D2, DOut, TOut>(
        &self,
        other: &TensorAny<R2, T2, B, D2>,
        f: impl Fn(T, T2) -> TOut + Send + Sync + 'f,
    ) -> Tensor<TOut, B, DOut>
    where
        R2: DataAPI<Data = <B as DeviceRawAPI<T2>>::Raw>,
        D2: DimAPI,
        DOut: DimAPI,
        D: DimMaxAPI<D2, Max = DOut>,
        T: Clone,
        T2: Clone,
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        B: Op_MutC_RefA_RefB_API<T, T2, TOut, DOut, dyn Fn(&mut MaybeUninit<TOut>, &T, &T2) + Send + Sync + 'f>,
    {
        self.mapb_f(other, move |x, y| f(x.clone(), y.clone())).rstsr_unwrap()
    }
}

/* #endregion */

#[cfg(test)]
mod tests_fnmut {
    use super::*;

    #[test]
    fn test_mapv() {
        let device = DeviceCpuSerial::default();
        let mut i = 0;
        let f = |x| {
            i += 1;
            x * 2.0
        };
        let a = asarray((vec![1., 2., 3., 4.], &device));
        let b = a.mapv_fnmut(f);
        assert!(allclose_f64(&b, &vec![2., 4., 6., 8.].into()));
        assert_eq!(i, 4);
        println!("{b:?}");
    }

    #[test]
    fn test_mapv_binary() {
        let device = DeviceCpuSerial::default();
        let mut i = 0;
        let f = |x, y| {
            i += 1;
            2.0 * x + 3.0 * y
        };
        #[cfg(not(feature = "col_major"))]
        {
            // a = np.arange(1, 7).reshape(2, 3)
            // b = np.arange(1, 4)
            // (2 * a + 3 * b).reshape(-1)
            let a = linspace((1., 6., 6, &device)).into_shape([2, 3]);
            let b = linspace((1., 3., 3, &device));
            let c = a.mapvb_fnmut(&b, f);
            assert_eq!(i, 6);
            println!("{c:?}");
            assert!(allclose_f64(&c.raw().into(), &vec![5., 10., 15., 11., 16., 21.].into()));
        }
        #[cfg(feature = "col_major")]
        {
            // a = reshape(range(1, 6), (3, 2))
            // b = reshape(range(1, 3), 3)
            // 2 * a .+ 3 * b
            let a = linspace((1., 6., 6, &device)).into_shape([3, 2]);
            let b = linspace((1., 3., 3, &device));
            let c = a.mapvb_fnmut(&b, f);
            assert_eq!(i, 6);
            println!("{c:?}");
            assert!(allclose_f64(&c.raw().into(), &vec![5., 10., 15., 11., 16., 21.].into()));
        }
    }
}

#[cfg(test)]
mod tests_sync {
    use super::*;

    #[test]
    fn test_mapv() {
        let f = |x| x * 2.0;
        let a = asarray(vec![1., 2., 3., 4.]);
        let b = a.mapv(f);
        assert!(allclose_f64(&b, &vec![2., 4., 6., 8.].into()));
        println!("{b:?}");
    }

    #[test]
    fn test_mapv_binary() {
        let f = |x, y| 2.0 * x + 3.0 * y;
        #[cfg(not(feature = "col_major"))]
        {
            // a = np.arange(1, 7).reshape(2, 3)
            // b = np.arange(1, 4)
            // (2 * a + 3 * b).reshape(-1)
            let a = linspace((1., 6., 6)).into_shape([2, 3]);
            let b = linspace((1., 3., 3));
            let c = a.mapvb(&b, f);
            assert!(allclose_f64(&c.raw().into(), &vec![5., 10., 15., 11., 16., 21.].into()));
        }
        #[cfg(feature = "col_major")]
        {
            // a = reshape(range(1, 6), (3, 2))
            // b = reshape(range(1, 3), 3)
            // 2 * a .+ 3 * b
            let a = linspace((1., 6., 6)).into_shape([3, 2]);
            let b = linspace((1., 3., 3));
            let c = a.mapvb(&b, f);
            assert!(allclose_f64(&c.raw().into(), &vec![5., 10., 15., 11., 16., 21.].into()));
        }
    }
}
