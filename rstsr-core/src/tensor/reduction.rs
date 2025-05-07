use crate::prelude_dev::*;

macro_rules! trait_reduction {
    ($OpReduceAPI: ident, $fn: ident, $fn_f: ident, $fn_axes: ident, $fn_axes_f: ident, $fn_all: ident, $fn_all_f: ident) => {
        pub fn $fn_all_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> Result<B::TOut>
        where
            R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
            D: DimAPI,
            B: $OpReduceAPI<T, D>,
        {
            tensor.device().$fn_all(tensor.raw(), tensor.layout())
        }

        pub fn $fn_axes_f<R, T, B, D, I>(
            tensor: &TensorAny<R, T, B, D>,
            axes: I,
        ) -> Result<Tensor<B::TOut, B, IxD>>
        where
            R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
            D: DimAPI,
            B: $OpReduceAPI<T, D> + DeviceCreationAnyAPI<B::TOut>,
            I: TryInto<AxesIndex<isize>>,
            Error: From<I::Error>,
        {
            let axes = axes.try_into()?;

            // special case for summing all axes
            if axes.as_ref().is_empty() {
                let sum = tensor.device().$fn_all(tensor.raw(), tensor.layout())?;
                let storage = tensor.device().outof_cpu_vec(vec![sum])?;
                let layout = Layout::new(vec![], vec![], 0)?;
                return Tensor::new_f(storage, layout);
            }

            let (storage, layout) =
                tensor.device().$fn_axes(tensor.raw(), tensor.layout(), axes.as_ref())?;
            Tensor::new_f(storage, layout)
        }

        pub fn $fn_all<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> B::TOut
        where
            R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
            D: DimAPI,
            B: $OpReduceAPI<T, D>,
        {
            $fn_all_f(tensor).unwrap()
        }

        pub fn $fn_axes<R, T, B, D, I>(
            tensor: &TensorAny<R, T, B, D>,
            axes: I,
        ) -> Tensor<B::TOut, B, IxD>
        where
            R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
            D: DimAPI,
            B: $OpReduceAPI<T, D> + DeviceCreationAnyAPI<B::TOut>,
            I: TryInto<AxesIndex<isize>>,
            Error: From<I::Error>,
        {
            $fn_axes_f(tensor, axes).unwrap()
        }

        pub fn $fn_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> Result<B::TOut>
        where
            R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
            D: DimAPI,
            B: $OpReduceAPI<T, D>,
        {
            $fn_all_f(tensor)
        }

        pub fn $fn<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> B::TOut
        where
            R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
            D: DimAPI,
            B: $OpReduceAPI<T, D>,
        {
            $fn_all(tensor)
        }

        impl<R, T, B, D> TensorAny<R, T, B, D>
        where
            R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
            D: DimAPI,
            B: $OpReduceAPI<T, D>,
        {
            pub fn $fn_all_f(&self) -> Result<B::TOut> {
                $fn_all_f(self)
            }

            pub fn $fn_all(&self) -> B::TOut {
                $fn_all(self)
            }

            pub fn $fn_axes_f<I>(&self, axes: I) -> Result<Tensor<B::TOut, B, IxD>>
            where
                B: DeviceCreationAnyAPI<B::TOut>,
                I: TryInto<AxesIndex<isize>>,
                Error: From<I::Error>,
            {
                $fn_axes_f(self, axes)
            }

            pub fn $fn_axes<I>(&self, axes: I) -> Tensor<B::TOut, B, IxD>
            where
                B: DeviceCreationAnyAPI<B::TOut>,
                I: TryInto<AxesIndex<isize>>,
                Error: From<I::Error>,
            {
                $fn_axes(self, axes)
            }

            pub fn $fn_f(&self) -> Result<B::TOut> {
                $fn_f(self)
            }

            pub fn $fn(&self) -> B::TOut {
                $fn(self)
            }
        }
    };
}

#[rustfmt::skip]
mod impl_trait_reduction {
    use super::*;
    trait_reduction!(OpSumAPI, sum, sum_f, sum_axes, sum_axes_f, sum_all, sum_all_f);
    trait_reduction!(OpMinAPI, min, min_f, min_axes, min_axes_f, min_all, min_all_f);
    trait_reduction!(OpMaxAPI, max, max_f, max_axes, max_axes_f, max_all, max_all_f);
    trait_reduction!(OpProdAPI, prod, prod_f, prod_axes, prod_axes_f, prod_all, prod_all_f);
    trait_reduction!(OpMeanAPI, mean, mean_f, mean_axes, mean_axes_f, mean_all, mean_all_f);
    trait_reduction!(OpVarAPI, var, var_f, var_axes, var_axes_f, var_all, var_all_f);
    trait_reduction!(OpStdAPI, std, std_f, std_axes, std_axes_f, std_all, std_all_f);
    trait_reduction!(OpL2NormAPI, l2_norm, l2_norm_f, l2_norm_axes, l2_norm_axes_f, l2_norm_all, l2_norm_all_f);
    trait_reduction!(OpArgMinAPI, argmin, argmin_f, argmin_axes, argmin_axes_f, argmin_all, argmin_all_f);
    trait_reduction!(OpArgMaxAPI, argmax, argmax_f, argmax_axes, argmax_axes_f, argmax_all, argmax_all_f);
}
pub use impl_trait_reduction::*;

macro_rules! trait_reduction_arg {
    ($OpReduceAPI: ident, $fn: ident, $fn_f: ident, $fn_axes: ident, $fn_axes_f: ident, $fn_all: ident, $fn_all_f: ident) => {
        pub fn $fn_all_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> Result<D>
        where
            R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
            D: DimAPI,
            B: $OpReduceAPI<T, D>,
        {
            tensor.device().$fn_all(tensor.raw(), tensor.layout())
        }

        pub fn $fn_axes_f<R, T, B, D, I>(
            tensor: &TensorAny<R, T, B, D>,
            axes: I,
        ) -> Result<Tensor<IxD, B, IxD>>
        where
            R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
            D: DimAPI,
            B: $OpReduceAPI<T, D> + DeviceAPI<IxD>,
            I: TryInto<AxesIndex<isize>>,
            Error: From<I::Error>,
        {
            let axes = axes.try_into()?;

            let (storage, layout) =
                tensor.device().$fn_axes(tensor.raw(), tensor.layout(), axes.as_ref())?;
            Tensor::new_f(storage, layout)
        }

        pub fn $fn_all<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> D
        where
            R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
            D: DimAPI,
            B: $OpReduceAPI<T, D>,
        {
            $fn_all_f(tensor).unwrap()
        }

        pub fn $fn_axes<R, T, B, D, I>(
            tensor: &TensorAny<R, T, B, D>,
            axes: I,
        ) -> Tensor<IxD, B, IxD>
        where
            R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
            D: DimAPI,
            B: $OpReduceAPI<T, D> + DeviceAPI<IxD>,
            I: TryInto<AxesIndex<isize>>,
            Error: From<I::Error>,
        {
            $fn_axes_f(tensor, axes).unwrap()
        }

        pub fn $fn_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> Result<D>
        where
            R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
            D: DimAPI,
            B: $OpReduceAPI<T, D>,
        {
            $fn_all_f(tensor)
        }

        pub fn $fn<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> D
        where
            R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
            D: DimAPI,
            B: $OpReduceAPI<T, D>,
        {
            $fn_all(tensor)
        }

        impl<R, T, B, D> TensorAny<R, T, B, D>
        where
            R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
            D: DimAPI,
            B: $OpReduceAPI<T, D>,
        {
            pub fn $fn_all_f(&self) -> Result<D> {
                $fn_all_f(self)
            }

            pub fn $fn_all(&self) -> D {
                $fn_all(self)
            }

            pub fn $fn_axes_f<I>(&self, axes: I) -> Result<Tensor<IxD, B, IxD>>
            where
                B: DeviceAPI<IxD>,
                I: TryInto<AxesIndex<isize>>,
                Error: From<I::Error>,
            {
                $fn_axes_f(self, axes)
            }

            pub fn $fn_axes<I>(&self, axes: I) -> Tensor<IxD, B, IxD>
            where
                B: DeviceAPI<IxD>,
                I: TryInto<AxesIndex<isize>>,
                Error: From<I::Error>,
            {
                $fn_axes(self, axes)
            }

            pub fn $fn_f(&self) -> Result<D> {
                $fn_f(self)
            }

            pub fn $fn(&self) -> D {
                $fn(self)
            }
        }
    };
}

trait_reduction_arg!(
    OpUnraveledArgMinAPI,
    unraveled_argmin,
    unraveled_argmin_f,
    unraveled_argmin_axes,
    unraveled_argmin_axes_f,
    unraveled_argmin_all,
    unraveled_argmin_all_f
);
trait_reduction_arg!(
    OpUnraveledArgMaxAPI,
    unraveled_argmax,
    unraveled_argmax_f,
    unraveled_argmax_axes,
    unraveled_argmax_axes_f,
    unraveled_argmax_all,
    unraveled_argmax_all_f
);

#[cfg(test)]
mod test {
    use num::ToPrimitive;

    use super::*;

    #[test]
    #[cfg(not(feature = "col_major"))]
    fn test_sum_all_row_major() {
        // DeviceCpuSerial
        let a = arange((24, &DeviceCpuSerial::default()));
        let s = sum_all(&a);
        assert_eq!(s, 276);

        // np.arange(3240).reshape(12, 15, 18)
        //   .swapaxes(-1, -2)[2:-3, 1:-4:2, -1:3:-2].sum()
        let a_owned = arange((3240, &DeviceCpuSerial::default()))
            .into_shape([12, 15, 18])
            .into_swapaxes(-1, -2);
        let a = a_owned.i((slice!(2, -3), slice!(1, -4, 2), slice!(-1, 3, -2)));
        let s = a.sum_all();
        assert_eq!(s, 446586);

        let s = a.sum_axes(());
        println!("{s:?}");
        assert_eq!(s.to_scalar(), 446586);

        // DeviceFaer
        let a = arange(24);
        let s = sum_all(&a);
        assert_eq!(s, 276);

        // np.arange(3240).reshape(12, 15, 18)
        //   .swapaxes(-1, -2)[2:-3, 1:-4:2, -1:3:-2].sum()
        let a_owned: Tensor<usize> = arange(3240).into_shape([12, 15, 18]).into_swapaxes(-1, -2);
        let a = a_owned.i((slice!(2, -3), slice!(1, -4, 2), slice!(-1, 3, -2)));
        let s = a.sum_all();
        assert_eq!(s, 446586);

        let s = a.sum_axes(());
        println!("{s:?}");
        assert_eq!(s.to_scalar(), 446586);
    }

    #[test]
    #[cfg(feature = "col_major")]
    fn test_sum_all_col_major() {
        // DeviceCpuSerial
        let a = arange((24, &DeviceCpuSerial::default()));
        let s = sum_all(&a);
        assert_eq!(s, 276);

        // a = reshape(range(0, 3239), (12, 15, 18));
        // a = permutedims(a, (1, 3, 2));
        // a = a[3:9, 2:2:15, 15:-2:4];
        // sum(a)
        let a_owned = arange((3240, &DeviceCpuSerial::default()))
            .into_shape([12, 15, 18])
            .into_swapaxes(-1, -2);
        let a = a_owned.i((slice!(2, -3), slice!(1, -4, 2), slice!(-1, 3, -2)));
        let s = a.sum_all();
        assert_eq!(s, 403662);

        let s = a.sum_axes(());
        println!("{s:?}");
        assert_eq!(s.to_scalar(), 403662);

        // DeviceFaer
        let a = arange(24);
        let s = sum_all(&a);
        assert_eq!(s, 276);

        let a_owned: Tensor<usize> = arange(3240).into_shape([12, 15, 18]).into_swapaxes(-1, -2);
        let a = a_owned.i((slice!(2, -3), slice!(1, -4, 2), slice!(-1, 3, -2)));
        let s = a.sum_all();
        assert_eq!(s, 403662);

        let s = a.sum_axes(());
        println!("{s:?}");
        assert_eq!(s.to_scalar(), 403662);
    }

    #[test]
    fn test_sum_axes() {
        #[cfg(not(feature = "col_major"))]
        {
            // a = np.arange(3240).reshape(4, 6, 15, 9).transpose(2, 0, 3, 1)
            // a.sum(axis=(0, -2))
            // DeviceCpuSerial
            let a = arange((3240, &DeviceCpuSerial::default()))
                .into_shape([4, 6, 15, 9])
                .into_transpose([2, 0, 3, 1]);
            let s = a.sum_axes([0, -2]);
            println!("{s:?}");
            assert_eq!(s[[0, 1]], 27270);
            assert_eq!(s[[1, 2]], 154845);
            assert_eq!(s[[3, 5]], 428220);

            // DeviceFaer
            let a: Tensor<usize> =
                arange(3240).into_shape([4, 6, 15, 9]).into_transpose([2, 0, 3, 1]);
            let s = a.sum_axes([0, -2]);
            println!("{s:?}");
            assert_eq!(s[[0, 1]], 27270);
            assert_eq!(s[[1, 2]], 154845);
            assert_eq!(s[[3, 5]], 428220);
        }
        #[cfg(feature = "col_major")]
        {
            // a = reshape(range(0, 3239), (4, 6, 15, 9));
            // a = permutedims(a, (3, 1, 4, 2));
            // sum(a, dims=(1, 3))
            // DeviceCpuSerial
            let a = arange((3240, &DeviceCpuSerial::default()))
                .into_shape([4, 6, 15, 9])
                .into_transpose([2, 0, 3, 1]);
            let s = a.sum_axes([0, -2]);
            println!("{s:?}");
            assert_eq!(s[[0, 1]], 217620);
            assert_eq!(s[[1, 2]], 218295);
            assert_eq!(s[[3, 5]], 220185);

            // DeviceFaer
            let a: Tensor<usize> =
                arange(3240).into_shape([4, 6, 15, 9]).into_transpose([2, 0, 3, 1]);
            let s = a.sum_axes([0, -2]);
            println!("{s:?}");
            assert_eq!(s[[0, 1]], 217620);
            assert_eq!(s[[1, 2]], 218295);
            assert_eq!(s[[3, 5]], 220185);
        }
    }

    #[test]
    fn test_min() {
        // DeviceCpuSerial
        let v = vec![8, 4, 2, 9, 3, 7, 2, 8, 1, 6, 10, 5];
        let a = asarray((&v, [4, 3].c(), &DeviceCpuSerial::default()));
        println!("{a:}");
        let m = a.min_axes(0);
        assert_eq!(m.to_vec(), vec![2, 3, 1]);
        let m = a.min_axes(1);
        assert_eq!(m.to_vec(), vec![2, 3, 1, 5]);
        let m = a.min_all();
        assert_eq!(m, 1);

        // DeviceFaer
        let v = vec![8, 4, 2, 9, 3, 7, 2, 8, 1, 6, 10, 5];
        let a = asarray((&v, [4, 3].c()));
        println!("{a:}");
        let m = a.min_axes(0);
        assert_eq!(m.to_vec(), vec![2, 3, 1]);
        let m = a.min_axes(1);
        assert_eq!(m.to_vec(), vec![2, 3, 1, 5]);
        let m = a.min_all();
        assert_eq!(m, 1);
    }

    #[test]
    fn test_mean() {
        #[cfg(not(feature = "col_major"))]
        {
            // DeviceCpuSerial
            let a = arange((24.0, &DeviceCpuSerial::default())).into_shape((2, 3, 4));
            let m = a.mean_all();
            assert_eq!(m, 11.5);

            let m = a.mean_axes((0, 2));
            println!("{m:}");
            assert_eq!(m.to_vec(), vec![7.5, 11.5, 15.5]);

            let m = a.i((slice!(None, None, -1), .., slice!(None, None, -2))).mean_axes((-1, 1));
            println!("{m:}");
            assert_eq!(m.to_vec(), vec![18.0, 6.0]);

            // DeviceFaer
            let a: Tensor<f64> = arange(24.0).into_shape((2, 3, 4));
            let m = a.mean_all();
            assert_eq!(m, 11.5);

            let m = a.mean_axes((0, 2));
            println!("{m:}");
            assert_eq!(m.to_vec(), vec![7.5, 11.5, 15.5]);

            let m = a.i((slice!(None, None, -1), .., slice!(None, None, -2))).mean_axes((-1, 1));
            println!("{m:}");
            assert_eq!(m.to_vec(), vec![18.0, 6.0]);
        }
        #[cfg(feature = "col_major")]
        {
            // DeviceCpuSerial
            let a = arange((24.0, &DeviceCpuSerial::default())).into_shape((2, 3, 4));
            let m = a.mean_all();
            assert_eq!(m, 11.5);

            let m = a.mean_axes((0, 2));
            println!("{m:}");
            assert_eq!(m.to_vec(), vec![9.5, 11.5, 13.5]);

            let m = a.i((slice!(None, None, -1), .., slice!(None, None, -2))).mean_axes((-1, 1));
            println!("{m:}");
            assert_eq!(m.to_vec(), vec![15.0, 14.0]);

            // // DeviceFaer
            let a: Tensor<f64> = arange(24.0).into_shape((2, 3, 4));
            let m = a.mean_all();
            assert_eq!(m, 11.5);

            let m = a.mean_axes((0, 2));
            println!("{m:}");
            assert_eq!(m.to_vec(), vec![9.5, 11.5, 13.5]);

            let m = a.i((slice!(None, None, -1), .., slice!(None, None, -2))).mean_axes((-1, 1));
            println!("{m:}");
            assert_eq!(m.to_vec(), vec![15.0, 14.0]);
        }
    }

    #[test]
    fn test_var() {
        // DeviceCpuSerial
        let v = vec![8, 4, 2, 9, 3, 7, 2, 8, 1, 6, 10, 5];
        let a = asarray((&v, [4, 3].c(), &DeviceCpuSerial::default())).mapv(|x| x as f64);

        let m = a.var_all();
        println!("{m:}");
        assert!((m - 8.409722222222221).abs() < 1e-10);

        let m = a.var_axes(0);
        println!("{m:}");
        assert!(allclose_f64(&m, &asarray(vec![7.1875, 8.1875, 5.6875])));

        let m = a.var_axes(1);
        println!("{m:}");
        assert!(allclose_f64(&m, &asarray(vec![6.22222222, 6.22222222, 9.55555556, 4.66666667])));

        // DeviceFaer
        let v = vec![8, 4, 2, 9, 3, 7, 2, 8, 1, 6, 10, 5];
        let a = asarray((&v, [4, 3].c())).mapv(|x| x as f64);

        let m = a.var_all();
        println!("{m:}");
        assert!((m - 8.409722222222221).abs() < 1e-10);

        let m = a.var_axes(0);
        println!("{m:}");
        assert!(allclose_f64(&m, &asarray(vec![7.1875, 8.1875, 5.6875])));

        let m = a.var_axes(1);
        println!("{m:}");
        assert!(allclose_f64(&m, &asarray(vec![6.22222222, 6.22222222, 9.55555556, 4.66666667])));
    }

    #[test]
    fn test_std() {
        // DeviceCpuSerial
        let v = vec![8, 4, 2, 9, 3, 7, 2, 8, 1, 6, 10, 5];
        let a = asarray((&v, [4, 3].c(), &DeviceCpuSerial::default())).mapv(|x| x as f64);

        let m = a.std_all();
        println!("{m:}");
        assert!((m - 2.899952106884219).abs() < 1e-10);

        let m = a.std_axes(0);
        println!("{m:}");
        assert!(allclose_f64(&m, &asarray(vec![2.68095132, 2.86138079, 2.384848])));

        let m = a.std_axes(1);
        println!("{m:}");
        assert!(allclose_f64(&m, &asarray(vec![2.49443826, 2.49443826, 3.09120617, 2.1602469])));

        let vr = [8, 4, 2, 9, 3, 7, 2, 8, 1, 6, 10, 5];
        let vi = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let v = vr
            .iter()
            .zip(vi.iter())
            .map(|(r, i)| num::Complex::new(r.to_f64().unwrap(), i.to_f64().unwrap()))
            .collect::<Vec<_>>();
        let a = asarray((&v, [4, 3].c(), &DeviceCpuSerial::default()));

        let m = a.std_all();
        println!("{m:}");
        assert!((m - 4.508479664907993).abs() < 1e-10);

        // DeviceFaer
        let v = vec![8, 4, 2, 9, 3, 7, 2, 8, 1, 6, 10, 5];
        let a = asarray((&v, [4, 3].c())).mapv(|x| x as f64);

        let m = a.std_all();
        println!("{m:}");
        assert!((m - 2.899952106884219).abs() < 1e-10);

        let m = a.std_axes(0);
        println!("{m:}");
        assert!(allclose_f64(&m, &asarray(vec![2.68095132, 2.86138079, 2.384848])));

        let m = a.std_axes(1);
        println!("{m:}");
        assert!(allclose_f64(&m, &asarray(vec![2.49443826, 2.49443826, 3.09120617, 2.1602469])));

        let vr = [8, 4, 2, 9, 3, 7, 2, 8, 1, 6, 10, 5];
        let vi = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let v = vr
            .iter()
            .zip(vi.iter())
            .map(|(r, i)| num::Complex::new(r.to_f64().unwrap(), i.to_f64().unwrap()))
            .collect::<Vec<_>>();
        let a = asarray((&v, [4, 3].c()));

        let m = a.std_all();
        println!("{m:}");
        assert!((m - 4.508479664907993).abs() < 1e-10);
    }

    #[test]
    fn test_l2_norm() {
        // DeviceCpuSerial
        let vr = [8, 4, 2, 9, 3, 7, 2, 8, 1, 6, 10, 5];
        let vi = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let v = vr
            .iter()
            .zip(vi.iter())
            .map(|(r, i)| num::Complex::new(r.to_f64().unwrap(), i.to_f64().unwrap()))
            .collect::<Vec<_>>();
        let a = asarray((&v, [4, 3].c(), &DeviceCpuSerial::default()));

        let m = a.l2_norm_all();
        println!("{m:}");
        assert!((m - 33.21144381083123).abs() < 1e-10);

        // DeviceFaer
        let vr = [8, 4, 2, 9, 3, 7, 2, 8, 1, 6, 10, 5];
        let vi = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let v = vr
            .iter()
            .zip(vi.iter())
            .map(|(r, i)| num::Complex::new(r.to_f64().unwrap(), i.to_f64().unwrap()))
            .collect::<Vec<_>>();
        let a = asarray((&v, [4, 3].c()));

        let m = a.l2_norm_all();
        println!("{m:}");
        assert!((m - 33.21144381083123).abs() < 1e-10);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_large_std() {
        #[cfg(not(feature = "col_major"))]
        {
            // a = np.linspace(0, 1, 1048576).reshape(16, 256, 256)
            // b = np.linspace(1, 2, 1048576).reshape(16, 256, 256)
            // c = a @ b
            // print(c.mean(), c.std())
            // print(c.std(axis=(0, 1))[[0, -1]])
            // print(c.std(axis=(1, 2))[[0, -1]])
            let a = linspace((0.0, 1.0, 1048576)).into_shape([16, 256, 256]);
            let b = linspace((1.0, 2.0, 1048576)).into_shape([16, 256, 256]);
            let c: Tensor<f64> = &a % &b;

            let c_mean = c.mean_all();
            println!("{c_mean:?}");
            assert!((c_mean - 213.2503660477036) < 1e-6);

            let c_std = c.std_all();
            println!("{c_std:?}");
            assert!((c_std - 148.88523481701804) < 1e-6);

            let c_std_1 = c.std_axes((0, 1));
            println!("{c_std_1}");
            assert!(c_std_1[[0]] - 148.8763226818815 < 1e-6);
            assert!(c_std_1[[255]] - 148.8941462322758 < 1e-6);

            let c_std_2 = c.std_axes((1, 2));
            println!("{c_std_2}");
            assert!(c_std_2[[0]] - 4.763105902995575 < 1e-6);
            assert!(c_std_2[[15]] - 9.093224903569157 < 1e-6);
        }
        #[cfg(feature = "col_major")]
        {
            // a = reshape(LinRange(0, 1, 1048576), (256, 256, 16));
            // b = reshape(LinRange(1, 2, 1048576), (256, 256, 16));
            // c = Array{Float64}(undef, 256, 256, 16);
            // for i in 1:16
            //     c[:, :, i] = a[:, :, i] * b[:, :, i]
            // end
            // mean(c), std(c)
            // std(c, dims=(2, 3))
            // std(c, dims=(1, 2))
            let a = linspace((0.0, 1.0, 1048576)).into_shape([256, 256, 16]);
            let b = linspace((1.0, 2.0, 1048576)).into_shape([256, 256, 16]);
            let mut c: Tensor<f64> = zeros([256, 256, 16]);
            for i in 0..16 {
                c.i_mut((.., .., i)).assign(&a.i((.., .., i)) % &b.i((.., .., i)));
            }

            let c_mean = c.mean_all();
            println!("{c_mean:?}");
            assert!((c_mean - 213.25036604770355) < 1e-6);

            let c_std = c.std_all();
            println!("{c_std:?}");
            assert!((c_std - 148.7419537312827) < 1e-6);

            let c_std_1 = c.std_axes((1, 2));
            println!("{c_std_1}");
            assert!(c_std_1[[0]] - 148.75113653867191 < 1e-6);
            assert!(c_std_1[[255]] - 148.7689445622776 < 1e-6);

            let c_std_2 = c.std_axes((0, 1));
            println!("{c_std_2}");
            assert!(c_std_2[[0]] - 0.145530296246335 < 1e-6);
            assert!(c_std_2[[15]] - 4.474611918106057 < 1e-6);
        }
    }

    #[test]
    fn test_unraveled_argmin() {
        // DeviceCpuSerial
        let v = vec![8, 4, 2, 9, 7, 1, 2, 1, 8, 6, 10, 5];
        let a = asarray((&v, [4, 3].c(), &DeviceCpuSerial::default()));
        println!("{a:}");
        // [[ 8 4 2]
        //  [ 9 7 1]
        //  [ 2 1 8]
        //  [ 6 10 5]]

        let m = a.unraveled_argmin_all();
        println!("{m:?}");
        assert_eq!(m, vec![1, 2]);

        let m = a.unraveled_argmin_axes(-1);
        println!("{m:?}");
        let m_vec = m.raw();
        assert_eq!(m_vec, &vec![vec![2], vec![2], vec![1], vec![2]]);

        let m = a.unraveled_argmin_axes(0);
        println!("{m:?}");
        let m_vec = m.raw();
        assert_eq!(m_vec, &vec![vec![2], vec![2], vec![1]]);
    }

    #[test]
    fn test_argmin() {
        // DeviceCpuSerial
        let v = vec![8, 4, 2, 9, 7, 1, 2, 1, 8, 6, 10, 5];
        let a = asarray((&v, [4, 3].c(), &DeviceCpuSerial::default()));
        println!("{a:}");
        // [[ 8 4 2]
        //  [ 9 7 1]
        //  [ 2 1 8]
        //  [ 6 10 5]]

        let m = a.argmin_all();
        println!("{m:?}");
        assert_eq!(m, 5);

        let m = a.argmin_axes(-1);
        println!("{m:?}");
        let m_vec = m.raw();
        assert_eq!(m_vec, &vec![2, 2, 1, 2]);

        let m = a.argmin_axes(0);
        println!("{m:?}");
        let m_vec = m.raw();
        assert_eq!(m_vec, &vec![2, 2, 1]);
    }
}
