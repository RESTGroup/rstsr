use crate::prelude_dev::*;

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
    pub fn map_fnmut_f<'f, TOut>(
        &self,
        mut f: impl FnMut(&T) -> TOut + 'f,
    ) -> Result<Tensor<TOut, B, D>>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        B: DeviceOp_MutA_RefB_API<TOut, T, D, dyn FnMut(&mut TOut, &T) + 'f>,
    {
        let la = self.layout();
        let lc = layout_for_array_copy(la, TensorIterOrder::default())?;
        let device = self.device();
        let mut storage_c = unsafe { device.empty_impl(lc.bounds_index()?.1)? };
        let mut f_inner = move |c: &mut TOut, a: &T| *c = f(a);
        device.op_muta_refb_func(storage_c.raw_mut(), &lc, self.raw(), la, &mut f_inner)?;
        return Tensor::new_f(storage_c, lc);
    }

    /// Call `f` by reference on each element and create a new tensor with the
    /// new values.
    pub fn map_fnmut<'f, TOut>(&self, f: impl FnMut(&T) -> TOut + 'f) -> Tensor<TOut, B, D>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        B: DeviceOp_MutA_RefB_API<TOut, T, D, dyn FnMut(&mut TOut, &T) + 'f>,
    {
        self.map_fnmut_f(f).unwrap()
    }

    /// Call `f` by value on each element and create a new tensor with the new
    /// values.
    pub fn mapv_fnmut_f<'f, TOut>(
        &self,
        mut f: impl FnMut(T) -> TOut + 'f,
    ) -> Result<Tensor<TOut, B, D>>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        T: Clone,
        B: DeviceOp_MutA_RefB_API<TOut, T, D, dyn FnMut(&mut TOut, &T) + 'f>,
    {
        self.map_fnmut_f(move |x| f(x.clone()))
    }

    /// Call `f` by value on each element and create a new tensor with the new
    /// values.
    pub fn mapv_fnmut<'f, TOut>(&self, mut f: impl FnMut(T) -> TOut + 'f) -> Tensor<TOut, B, D>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        T: Clone,
        B: DeviceOp_MutA_RefB_API<TOut, T, D, dyn FnMut(&mut TOut, &T) + 'f>,
    {
        self.map_fnmut_f(move |x| f(x.clone())).unwrap()
    }

    /// Modify the tensor in place by calling `f` by mutable reference on each
    /// element.
    pub fn mapi_fnmut_f<'f>(&mut self, mut f: impl FnMut(&mut T) + 'f) -> Result<()>
    where
        R: DataMutAPI<Data = B::Raw>,
        B: DeviceOp_MutA_API<T, D, dyn FnMut(&mut T) + 'f>,
    {
        let (la, _) = greedy_layout(self.layout(), false);
        let device = self.device().clone();
        device.op_muta_func(self.raw_mut(), &la, &mut f)
    }

    /// Modify the tensor in place by calling `f` by mutable reference on each
    /// element.
    pub fn mapi_fnmut<'f>(&mut self, f: impl FnMut(&mut T) + 'f)
    where
        R: DataMutAPI<Data = B::Raw>,
        B: DeviceOp_MutA_API<T, D, dyn FnMut(&mut T) + 'f>,
    {
        self.mapi_fnmut_f(f).unwrap()
    }

    /// Modify the tensor in place by calling `f` by value on each
    /// element.
    pub fn mapvi_fnmut_f<'f>(&mut self, mut f: impl FnMut(T) -> T + 'f) -> Result<()>
    where
        R: DataMutAPI<Data = B::Raw>,
        T: Clone,
        B: DeviceOp_MutA_API<T, D, dyn FnMut(&mut T) + 'f>,
    {
        self.mapi_fnmut_f(move |x| *x = f(x.clone()))
    }

    /// Modify the tensor in place by calling `f` by value on each
    /// element.
    pub fn mapvi_fnmut<'f>(&mut self, f: impl FnMut(T) -> T + 'f)
    where
        R: DataMutAPI<Data = B::Raw>,
        T: Clone,
        B: DeviceOp_MutA_API<T, D, dyn FnMut(&mut T) + 'f>,
    {
        self.mapvi_fnmut_f(f).unwrap()
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
        B: DeviceOp_MutC_RefA_RefB_API<T, T2, TOut, DOut, dyn FnMut(&mut TOut, &T, &T2) + 'f>,
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
        let lc_from_a = layout_for_array_copy(&la_b, TensorIterOrder::default())?;
        let lc_from_b = layout_for_array_copy(&lb_b, TensorIterOrder::default())?;
        let lc = if lc_from_a == lc_from_b {
            lc_from_a
        } else {
            match self.device().default_order() {
                RowMajor => la_b.shape().c(),
                ColMajor => la_b.shape().f(),
            }
        };
        let device = self.device();
        let mut storage_c = unsafe { device.empty_impl(lc.bounds_index()?.1)? };
        let mut f_inner = move |c: &mut TOut, a: &T, b: &T2| *c = f(a, b);
        device.op_mutc_refa_refb_func(
            storage_c.raw_mut(),
            &lc,
            self.raw(),
            &la_b,
            other.raw(),
            &lb_b,
            &mut f_inner,
        )?;
        Tensor::new_f(storage_c, lc)
    }

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
        B: DeviceOp_MutC_RefA_RefB_API<T, T2, TOut, DOut, dyn FnMut(&mut TOut, &T, &T2) + 'f>,
    {
        self.mapb_fnmut_f(other, f).unwrap()
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
        B: DeviceOp_MutC_RefA_RefB_API<T, T2, TOut, DOut, dyn FnMut(&mut TOut, &T, &T2) + 'f>,
    {
        self.mapb_fnmut_f(other, move |x, y| f(x.clone(), y.clone()))
    }

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
        B: DeviceOp_MutC_RefA_RefB_API<T, T2, TOut, DOut, dyn FnMut(&mut TOut, &T, &T2) + 'f>,
    {
        self.mapb_fnmut_f(other, move |x, y| f(x.clone(), y.clone())).unwrap()
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
    pub fn map_f<'f, TOut>(
        &self,
        f: impl Fn(&T) -> TOut + Send + Sync + 'f,
    ) -> Result<Tensor<TOut, B, D>>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        B: DeviceOp_MutA_RefB_API<TOut, T, D, dyn Fn(&mut TOut, &T) + Send + Sync + 'f>,
    {
        let la = self.layout();
        let lc = layout_for_array_copy(la, TensorIterOrder::default())?;
        let device = self.device();
        let mut storage_c = unsafe { device.empty_impl(lc.bounds_index()?.1)? };
        let mut f_inner = move |c: &mut TOut, a: &T| *c = f(a);
        device.op_muta_refb_func(storage_c.raw_mut(), &lc, self.raw(), la, &mut f_inner)?;
        return Tensor::new_f(storage_c, lc);
    }

    /// Call `f` by reference on each element and create a new tensor with the
    /// new values.
    pub fn map<'f, TOut>(&self, f: impl Fn(&T) -> TOut + Send + Sync + 'f) -> Tensor<TOut, B, D>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        B: DeviceOp_MutA_RefB_API<TOut, T, D, dyn Fn(&mut TOut, &T) + Send + Sync + 'f>,
    {
        self.map_f(f).unwrap()
    }

    /// Call `f` by value on each element and create a new tensor with the new
    /// values.
    pub fn mapv_f<'f, TOut>(
        &self,
        f: impl Fn(T) -> TOut + Send + Sync + 'f,
    ) -> Result<Tensor<TOut, B, D>>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        T: Clone,
        B: DeviceOp_MutA_RefB_API<TOut, T, D, dyn Fn(&mut TOut, &T) + Send + Sync + 'f>,
    {
        self.map_f(move |x| f(x.clone()))
    }

    /// Call `f` by value on each element and create a new tensor with the new
    /// values.
    pub fn mapv<'f, TOut>(&self, f: impl Fn(T) -> TOut + Send + Sync + 'f) -> Tensor<TOut, B, D>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        T: Clone,
        B: DeviceOp_MutA_RefB_API<TOut, T, D, dyn Fn(&mut TOut, &T) + Send + Sync + 'f>,
    {
        self.map_f(move |x| f(x.clone())).unwrap()
    }

    /// Modify the tensor in place by calling `f` by mutable reference on each
    /// element.
    pub fn mapi_f<'f>(&mut self, mut f: impl Fn(&mut T) + Send + Sync + 'f) -> Result<()>
    where
        R: DataMutAPI<Data = B::Raw>,
        B: DeviceOp_MutA_API<T, D, dyn Fn(&mut T) + Send + Sync + 'f>,
    {
        let (la, _) = greedy_layout(self.layout(), false);
        let device = self.device().clone();
        device.op_muta_func(self.raw_mut(), &la, &mut f)
    }

    /// Modify the tensor in place by calling `f` by mutable reference on each
    /// element.
    pub fn mapi<'f>(&mut self, f: impl Fn(&mut T) + Send + Sync + 'f)
    where
        R: DataMutAPI<Data = B::Raw>,
        B: DeviceOp_MutA_API<T, D, dyn Fn(&mut T) + Send + Sync + 'f>,
    {
        self.mapi_f(f).unwrap()
    }

    /// Modify the tensor in place by calling `f` by value on each
    /// element.
    pub fn mapvi_f<'f>(&mut self, f: impl Fn(T) -> T + Send + Sync + 'f) -> Result<()>
    where
        R: DataMutAPI<Data = B::Raw>,
        T: Clone,
        B: DeviceOp_MutA_API<T, D, dyn Fn(&mut T) + Send + Sync + 'f>,
    {
        self.mapi_f(move |x| *x = f(x.clone()))
    }

    /// Modify the tensor in place by calling `f` by value on each
    /// element.
    pub fn mapvi<'f>(&mut self, f: impl Fn(T) -> T + Send + Sync + 'f)
    where
        R: DataMutAPI<Data = B::Raw>,
        T: Clone,
        B: DeviceOp_MutA_API<T, D, dyn Fn(&mut T) + Send + Sync + 'f>,
    {
        self.mapvi_f(f).unwrap()
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
        B: DeviceOp_MutC_RefA_RefB_API<
            T,
            T2,
            TOut,
            DOut,
            dyn Fn(&mut TOut, &T, &T2) + Send + Sync + 'f,
        >,
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
        let lc_from_a = layout_for_array_copy(&la_b, TensorIterOrder::default())?;
        let lc_from_b = layout_for_array_copy(&lb_b, TensorIterOrder::default())?;
        let lc = if lc_from_a == lc_from_b {
            lc_from_a
        } else {
            match self.device().default_order() {
                RowMajor => la_b.shape().c(),
                ColMajor => la_b.shape().f(),
            }
        };
        let device = self.device();
        let mut storage_c = unsafe { device.empty_impl(lc.bounds_index()?.1)? };
        let mut f_inner = move |c: &mut TOut, a: &T, b: &T2| *c = f(a, b);
        device.op_mutc_refa_refb_func(
            storage_c.raw_mut(),
            &lc,
            self.raw(),
            &la_b,
            other.raw(),
            &lb_b,
            &mut f_inner,
        )?;
        Tensor::new_f(storage_c, lc)
    }

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
        B: DeviceOp_MutC_RefA_RefB_API<
            T,
            T2,
            TOut,
            DOut,
            dyn Fn(&mut TOut, &T, &T2) + Send + Sync + 'f,
        >,
    {
        self.mapb_f(other, f).unwrap()
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
        B: DeviceOp_MutC_RefA_RefB_API<
            T,
            T2,
            TOut,
            DOut,
            dyn Fn(&mut TOut, &T, &T2) + Send + Sync + 'f,
        >,
    {
        self.mapb_f(other, move |x, y| f(x.clone(), y.clone()))
    }

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
        B: DeviceOp_MutC_RefA_RefB_API<
            T,
            T2,
            TOut,
            DOut,
            dyn Fn(&mut TOut, &T, &T2) + Send + Sync + 'f,
        >,
    {
        self.mapb_f(other, move |x, y| f(x.clone(), y.clone())).unwrap()
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
            let a = linspace((1., 6., 6, &device)).into_shape_assume_contig([2, 3]);
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
            let a = linspace((1., 6., 6, &device)).into_shape_assume_contig([3, 2]);
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
            let a = linspace((1., 6., 6)).into_shape_assume_contig([2, 3]);
            let b = linspace((1., 3., 3));
            let c = a.mapvb(&b, f);
            assert!(allclose_f64(&c.raw().into(), &vec![5., 10., 15., 11., 16., 21.].into()));
        }
        #[cfg(feature = "col_major")]
        {
            // a = reshape(range(1, 6), (3, 2))
            // b = reshape(range(1, 3), 3)
            // 2 * a .+ 3 * b
            let a = linspace((1., 6., 6)).into_shape_assume_contig([3, 2]);
            let b = linspace((1., 3., 3));
            let c = a.mapvb(&b, f);
            assert!(allclose_f64(&c.raw().into(), &vec![5., 10., 15., 11., 16., 21.].into()));
        }
    }
}
