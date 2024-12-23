use crate::prelude_dev::*;

impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    /// Call `f` by reference on each element and create a new array with the
    /// new values.
    pub fn map<'f, TOut>(&self, mut f: impl FnMut(&T) -> TOut + 'f) -> Tensor<TOut, D, B>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        B: DeviceOp_MutA_RefB_API<TOut, T, D, dyn FnMut(&mut TOut, &T) + 'f>,
    {
        let la = self.layout();
        let lc = layout_for_array_copy(la, TensorIterOrder::default()).unwrap();
        let device = self.device();
        let mut storage_c = unsafe { device.empty_impl(lc.bounds_index().unwrap().1).unwrap() };
        let storage_a = self.data().storage();
        let mut f_inner = move |c: &mut TOut, a: &T| *c = f(a);
        device.op_muta_refb_func(&mut storage_c, &lc, storage_a, la, &mut f_inner).unwrap();
        return Tensor::new_f(DataOwned::from(storage_c), lc).unwrap();
    }

    /// Call `f` by value on each element and create a new array with the new
    /// values.
    pub fn mapv<'f, TOut>(&self, mut f: impl FnMut(T) -> TOut + 'f) -> Tensor<TOut, D, B>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        T: Clone,
        B: DeviceOp_MutA_RefB_API<TOut, T, D, dyn FnMut(&mut TOut, &T) + 'f>,
    {
        self.map(move |x| f(x.clone()))
    }

    /// Call `f` by reference on each element and create a new array with the
    /// new values.
    pub fn map_tmp<'f, TOut>(
        &self,
        f: impl for<'a> Fn(&'a T) -> TOut + Send + Sync + 'f,
    ) -> Tensor<TOut, D, B>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        B: DeviceOp_MutA_RefB_API<TOut, T, D, dyn Fn(&mut TOut, &T) + Send + Sync + 'f>,
    {
        let la = self.layout();
        let lc = layout_for_array_copy(la, TensorIterOrder::default()).unwrap();
        let device = self.device();
        let mut storage_c = unsafe { device.empty_impl(lc.bounds_index().unwrap().1).unwrap() };
        let storage_a = self.data().storage();
        let mut f_inner = move |c: &mut TOut, a: &T| *c = f(a);
        device.op_muta_refb_func(&mut storage_c, &lc, storage_a, la, &mut f_inner).unwrap();
        return Tensor::new_f(DataOwned::from(storage_c), lc).unwrap();
    }

    /// Modify the array in place by calling `f` by mutable reference on each
    /// element.
    pub fn map_inplace<'f>(&mut self, mut f: impl FnMut(&mut T) + 'f)
    where
        R: DataMutAPI<Data = Storage<T, B>>,
        B: DeviceOp_MutA_API<T, D, dyn FnMut(&mut T) + 'f>,
    {
        let (la, _) = greedy_layout(self.layout(), false);
        let device = self.device().clone();
        let storage_a = self.data_mut().storage_mut();
        device.op_muta_func(storage_a, &la, &mut f).unwrap();
    }

    /// Modify the array in place by calling `f` by mutable reference on each
    /// element.
    pub fn mapv_inplace<'f>(&mut self, mut f: impl FnMut(T) -> T + 'f)
    where
        R: DataMutAPI<Data = Storage<T, B>>,
        T: Clone,
        B: DeviceOp_MutA_API<T, D, dyn FnMut(&mut T) + 'f>,
    {
        self.map_inplace(move |x| *x = f(x.clone()));
    }

    pub fn map_binary<'f, R2, T2, D2, DOut, TOut>(
        &self,
        other: &TensorBase<R2, D2>,
        mut f: impl FnMut(&T, &T2) -> TOut + 'f,
    ) -> Tensor<TOut, DOut, B>
    where
        R2: DataAPI<Data = Storage<T2, B>>,
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
        rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch).unwrap();
        let la = a.layout();
        let lb = b.layout();
        let (la_b, lb_b) = broadcast_layout(la, lb).unwrap();
        // generate output layout
        let lc_from_a = layout_for_array_copy(&la_b, TensorIterOrder::default()).unwrap();
        let lc_from_b = layout_for_array_copy(&lb_b, TensorIterOrder::default()).unwrap();
        let lc = if lc_from_a == lc_from_b {
            lc_from_a
        } else {
            match TensorOrder::default() {
                TensorOrder::C => la_b.shape().c(),
                TensorOrder::F => la_b.shape().f(),
            }
        };
        let device = self.device();
        let mut storage_c = unsafe { device.empty_impl(lc.bounds_index().unwrap().1).unwrap() };
        let storage_a = self.data().storage();
        let storage_b = other.data().storage();
        let mut f_inner = move |c: &mut TOut, a: &T, b: &T2| *c = f(a, b);
        device
            .op_mutc_refa_refb_func(
                &mut storage_c,
                &lc,
                storage_a,
                &la_b,
                storage_b,
                &lb_b,
                &mut f_inner,
            )
            .unwrap();
        return Tensor::new_f(DataOwned::from(storage_c), lc).unwrap();
    }

    pub fn mapv_binary<'f, R2, T2, D2, DOut, TOut>(
        &self,
        other: &TensorBase<R2, D2>,
        mut f: impl FnMut(T, T2) -> TOut + 'f,
    ) -> Tensor<TOut, DOut, B>
    where
        R2: DataAPI<Data = Storage<T2, B>>,
        D2: DimAPI,
        DOut: DimAPI,
        D: DimMaxAPI<D2, Max = DOut>,
        T: Clone,
        T2: Clone,
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        B: DeviceOp_MutC_RefA_RefB_API<T, T2, TOut, DOut, dyn FnMut(&mut TOut, &T, &T2) + 'f>,
    {
        self.map_binary(other, move |x, y| f(x.clone(), y.clone()))
    }
}

impl<T, D, B> Tensor<T, D, B>
where
    D: DimAPI,
    B: DeviceAPI<T>,
{
    /// Call `f` by value on each element, update the array with the new values
    /// and return it.
    pub fn mapv_into<'f>(mut self, mut f: impl FnMut(T) -> T + 'f) -> Tensor<T, D, B>
    where
        T: Clone,
        B: DeviceOp_MutA_API<T, D, dyn FnMut(&mut T) + 'f>,
    {
        let (la, _) = greedy_layout(self.layout(), false);
        let device = self.device().clone();
        let storage_a = self.data_mut().storage_mut();
        let mut f_inner = move |x: &mut T| *x = f(x.clone());
        device.op_muta_func(storage_a, &la, &mut f_inner).unwrap();
        return self;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapv() {
        let device = DeviceCpuSerial;
        let f = |x| x * 2.0;
        let a = Tensor::asarray_f((vec![1., 2., 3., 4.], &device)).unwrap();
        let b = a.mapv(f);
        assert!(allclose_f64(&b, &vec![2., 4., 6., 8.].into()));
        println!("{:?}", b);
    }

    #[test]
    fn test_mapv_binary() {
        let device = DeviceCpuSerial;
        let f = |x, y| 2.0 * x + 3.0 * y;
        let a = Tensor::linspace(1., 6., 6, &device).into_shape_assume_contig_f([2, 3]).unwrap();
        let b = Tensor::linspace(1., 3., 3, &device);
        let c = a.mapv_binary(&b, f);
        assert!(allclose_f64(&c, &vec![5., 10., 15., 11., 16., 21.].into()));
    }

    #[test]
    fn test_mapv_rayon() {
        let device = DeviceFaer::default();
        let f = |&x: &f64| x * 2.0;
        let a = Tensor::asarray_f((vec![1., 2., 3., 4.], &device)).unwrap();
        let b = a.map_tmp(f);
        assert!(allclose_f64(&b, &vec![2., 4., 6., 8.].into()));
        println!("{:?}", b);
    }
}
