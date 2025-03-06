use crate::prelude_dev::*;

macro_rules! trait_reduction {
    ($OpReduceAPI: ident, $fn: ident, $fn_f: ident, $fn_all: ident, $fn_all_f: ident) => {
        pub fn $fn_all_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> Result<B::TOut>
        where
            R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
            D: DimAPI,
            B: $OpReduceAPI<T, D>,
        {
            tensor.device().$fn_all(tensor.raw(), tensor.layout())
        }

        pub fn $fn_f<R, T, B, D, I>(
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
                tensor.device().$fn(tensor.raw(), tensor.layout(), axes.as_ref())?;
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

        pub fn $fn<R, T, B, D, I>(
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
            $fn_f(tensor, axes).unwrap()
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

            pub fn $fn_f<I>(&self, axes: I) -> Result<Tensor<B::TOut, B, IxD>>
            where
                B: DeviceCreationAnyAPI<B::TOut>,
                I: TryInto<AxesIndex<isize>>,
                Error: From<I::Error>,
            {
                $fn_f(self, axes)
            }

            pub fn $fn<I>(&self, axes: I) -> Tensor<B::TOut, B, IxD>
            where
                B: DeviceCreationAnyAPI<B::TOut>,
                I: TryInto<AxesIndex<isize>>,
                Error: From<I::Error>,
            {
                $fn(self, axes)
            }
        }
    };
}

trait_reduction!(OpSumAPI, sum, sum_f, sum_all, sum_all_f);
trait_reduction!(OpMinAPI, min, min_f, min_all, min_all_f);
trait_reduction!(OpMaxAPI, max, max_f, max_all, max_all_f);
trait_reduction!(OpProdAPI, prod, prod_f, prod_all, prod_all_f);
trait_reduction!(OpMeanAPI, mean, mean_f, mean_all, mean_all_f);
trait_reduction!(OpVarAPI, var, var_f, var_all, var_all_f);
trait_reduction!(OpStdAPI, std, std_f, std_all, std_all_f);
trait_reduction!(OpL2NormAPI, l2_norm, l2_norm_f, l2_norm_all, l2_norm_all_f);

macro_rules! trait_reduction_arg {
    ($OpReduceAPI: ident, $fn: ident, $fn_f: ident, $fn_all: ident, $fn_all_f: ident) => {
        pub fn $fn_all_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> Result<D>
        where
            R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
            D: DimAPI,
            B: $OpReduceAPI<T, D>,
        {
            tensor.device().$fn_all(tensor.raw(), tensor.layout())
        }

        pub fn $fn_f<R, T, B, D, I>(
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
                tensor.device().$fn(tensor.raw(), tensor.layout(), axes.as_ref())?;
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

        pub fn $fn<R, T, B, D, I>(tensor: &TensorAny<R, T, B, D>, axes: I) -> Tensor<IxD, B, IxD>
        where
            R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
            D: DimAPI,
            B: $OpReduceAPI<T, D> + DeviceAPI<IxD>,
            I: TryInto<AxesIndex<isize>>,
            Error: From<I::Error>,
        {
            $fn_f(tensor, axes).unwrap()
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

            pub fn $fn_f<I>(&self, axes: I) -> Result<Tensor<IxD, B, IxD>>
            where
                B: DeviceAPI<IxD>,
                I: TryInto<AxesIndex<isize>>,
                Error: From<I::Error>,
            {
                $fn_f(self, axes)
            }

            pub fn $fn<I>(&self, axes: I) -> Tensor<IxD, B, IxD>
            where
                B: DeviceAPI<IxD>,
                I: TryInto<AxesIndex<isize>>,
                Error: From<I::Error>,
            {
                $fn(self, axes)
            }
        }
    };
}

trait_reduction_arg!(OpArgMinAPI, argmin, argmin_f, argmin_all, argmin_all_f);
trait_reduction_arg!(OpArgMaxAPI, argmax, argmax_f, argmax_all, argmax_all_f);

#[cfg(test)]
mod test {
    use num::ToPrimitive;

    use super::*;

    #[test]
    fn test_sum_all() {
        // DeviceCpuSerial
        let a = arange((24, &DeviceCpuSerial));
        let s = sum_all(&a);
        assert_eq!(s, 276);

        // np.arange(3240).reshape(12, 15, 18)
        //   .swapaxes(-1, -2)[2:-3, 1:-4:2, -1:3:-2].sum()
        let a_owned =
            arange((3240, &DeviceCpuSerial)).into_shape([12, 15, 18]).into_swapaxes(-1, -2);
        let a = a_owned.i((slice!(2, -3), slice!(1, -4, 2), slice!(-1, 3, -2)));
        let s = a.sum_all();
        assert_eq!(s, 446586);

        let s = a.sum(());
        println!("{:?}", s);
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

        let s = a.sum(());
        println!("{:?}", s);
        assert_eq!(s.to_scalar(), 446586);
    }

    #[test]
    fn test_sum_axes() {
        // DeviceCpuSerial
        let a =
            arange((3240, &DeviceCpuSerial)).into_shape([4, 6, 15, 9]).into_transpose([2, 0, 3, 1]);
        let s = a.sum([0, -2]);
        println!("{:?}", s);
        assert_eq!(s[[0, 1]], 27270);
        assert_eq!(s[[1, 2]], 154845);
        assert_eq!(s[[3, 5]], 428220);

        // DeviceFaer
        let a: Tensor<usize> = arange(3240).into_shape([4, 6, 15, 9]).into_transpose([2, 0, 3, 1]);
        let s = a.sum([0, -2]);
        println!("{:?}", s);
        assert_eq!(s[[0, 1]], 27270);
        assert_eq!(s[[1, 2]], 154845);
        assert_eq!(s[[3, 5]], 428220);
    }

    #[test]
    fn test_min() {
        // DeviceCpuSerial
        let v = vec![8, 4, 2, 9, 3, 7, 2, 8, 1, 6, 10, 5];
        let a = asarray((&v, [4, 3].c(), &DeviceCpuSerial));
        println!("{:}", a);
        let m = a.min(0);
        assert_eq!(m.to_vec(), vec![2, 3, 1]);
        let m = a.min(1);
        assert_eq!(m.to_vec(), vec![2, 3, 1, 5]);
        let m = a.min_all();
        assert_eq!(m, 1);

        // DeviceFaer
        let v = vec![8, 4, 2, 9, 3, 7, 2, 8, 1, 6, 10, 5];
        let a = asarray((&v, [4, 3].c()));
        println!("{:}", a);
        let m = a.min(0);
        assert_eq!(m.to_vec(), vec![2, 3, 1]);
        let m = a.min(1);
        assert_eq!(m.to_vec(), vec![2, 3, 1, 5]);
        let m = a.min_all();
        assert_eq!(m, 1);
    }

    #[test]
    fn test_mean() {
        // DeviceCpuSerial
        let a = arange((24.0, &DeviceCpuSerial)).into_shape((2, 3, 4));
        let m = a.mean_all();
        assert_eq!(m, 11.5);

        let m = a.mean((0, 2));
        println!("{:}", m);
        assert_eq!(m.to_vec(), vec![7.5, 11.5, 15.5]);

        let m = a.i((slice!(None, None, -1), .., slice!(None, None, -2))).mean((-1, 1));
        println!("{:}", m);
        assert_eq!(m.to_vec(), vec![18.0, 6.0]);

        // DeviceFaer
        let a: Tensor<f64> = arange(24.0).into_shape((2, 3, 4));
        let m = a.mean_all();
        assert_eq!(m, 11.5);

        let m = a.mean((0, 2));
        println!("{:}", m);
        assert_eq!(m.to_vec(), vec![7.5, 11.5, 15.5]);

        let m = a.i((slice!(None, None, -1), .., slice!(None, None, -2))).mean((-1, 1));
        println!("{:}", m);
        assert_eq!(m.to_vec(), vec![18.0, 6.0]);
    }

    #[test]
    fn test_var() {
        // DeviceCpuSerial
        let v = vec![8, 4, 2, 9, 3, 7, 2, 8, 1, 6, 10, 5];
        let a = asarray((&v, [4, 3].c(), &DeviceCpuSerial)).mapv(|x| x as f64);

        let m = a.var_all();
        println!("{:}", m);
        assert!((m - 8.409722222222221).abs() < 1e-10);

        let m = a.var(0);
        println!("{:}", m);
        assert!(allclose_f64(&m, &asarray(vec![7.1875, 8.1875, 5.6875])));

        let m = a.var(1);
        println!("{:}", m);
        assert!(allclose_f64(&m, &asarray(vec![6.22222222, 6.22222222, 9.55555556, 4.66666667])));

        // DeviceFaer
        let v = vec![8, 4, 2, 9, 3, 7, 2, 8, 1, 6, 10, 5];
        let a = asarray((&v, [4, 3].c())).mapv(|x| x as f64);

        let m = a.var_all();
        println!("{:}", m);
        assert!((m - 8.409722222222221).abs() < 1e-10);

        let m = a.var(0);
        println!("{:}", m);
        assert!(allclose_f64(&m, &asarray(vec![7.1875, 8.1875, 5.6875])));

        let m = a.var(1);
        println!("{:}", m);
        assert!(allclose_f64(&m, &asarray(vec![6.22222222, 6.22222222, 9.55555556, 4.66666667])));
    }

    #[test]
    fn test_std() {
        // DeviceCpuSerial
        let v = vec![8, 4, 2, 9, 3, 7, 2, 8, 1, 6, 10, 5];
        let a = asarray((&v, [4, 3].c(), &DeviceCpuSerial)).mapv(|x| x as f64);

        let m = a.std_all();
        println!("{:}", m);
        assert!((m - 2.899952106884219).abs() < 1e-10);

        let m = a.std(0);
        println!("{:}", m);
        assert!(allclose_f64(&m, &asarray(vec![2.68095132, 2.86138079, 2.384848])));

        let m = a.std(1);
        println!("{:}", m);
        assert!(allclose_f64(&m, &asarray(vec![2.49443826, 2.49443826, 3.09120617, 2.1602469])));

        let vr = [8, 4, 2, 9, 3, 7, 2, 8, 1, 6, 10, 5];
        let vi = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let v = vr
            .iter()
            .zip(vi.iter())
            .map(|(r, i)| num::Complex::new(r.to_f64().unwrap(), i.to_f64().unwrap()))
            .collect::<Vec<_>>();
        let a = asarray((&v, [4, 3].c(), &DeviceCpuSerial));

        let m = a.std_all();
        println!("{:}", m);
        assert!((m - 4.508479664907993).abs() < 1e-10);

        // DeviceFaer
        let v = vec![8, 4, 2, 9, 3, 7, 2, 8, 1, 6, 10, 5];
        let a = asarray((&v, [4, 3].c())).mapv(|x| x as f64);

        let m = a.std_all();
        println!("{:}", m);
        assert!((m - 2.899952106884219).abs() < 1e-10);

        let m = a.std(0);
        println!("{:}", m);
        assert!(allclose_f64(&m, &asarray(vec![2.68095132, 2.86138079, 2.384848])));

        let m = a.std(1);
        println!("{:}", m);
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
        println!("{:}", m);
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
        let a = asarray((&v, [4, 3].c(), &DeviceCpuSerial));

        let m = a.l2_norm_all();
        println!("{:}", m);
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
        println!("{:}", m);
        assert!((m - 33.21144381083123).abs() < 1e-10);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_large_std() {
        let a = linspace((0.0, 1.0, 1048576)).into_shape([16, 256, 256]);
        let b = linspace((1.0, 2.0, 1048576)).into_shape([16, 256, 256]);
        let c: Tensor<f64> = &a % &b;

        let c_mean = c.mean_all();
        println!("{:?}", c_mean);
        assert!((c_mean - 213.2503660477036) < 1e-6);

        let c_std = c.std_all();
        println!("{:?}", c_std);
        assert!((c_std - 148.88523481701804) < 1e-6);

        let c_std_1 = c.std((0, 1));
        println!("{}", c_std_1);

        let c_std_2 = c.std((1, 2));
        println!("{}", c_std_2);
    }

    #[test]
    fn test_argmin() {
        // DeviceCpuSerial
        let v = vec![8, 4, 2, 9, 7, 1, 2, 1, 8, 6, 10, 5];
        let a = asarray((&v, [4, 3].c(), &DeviceCpuSerial));
        println!("{:}", a);
        // [[ 8 4 2]
        //  [ 9 7 1]
        //  [ 2 1 8]
        //  [ 6 10 5]]

        let m = a.argmin_all();
        println!("{:?}", m);
        assert_eq!(m, vec![1, 2]);

        let m = a.argmin(-1);
        println!("{:?}", m);
        let m_vec = m.raw();
        assert_eq!(m_vec, &vec![vec![2], vec![2], vec![1], vec![2]]);

        let m = a.argmin(0);
        println!("{:?}", m);
        let m_vec = m.raw();
        assert_eq!(m_vec, &vec![vec![2], vec![2], vec![1]]);
    }
}
