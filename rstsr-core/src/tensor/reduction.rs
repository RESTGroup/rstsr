use crate::prelude_dev::*;

macro_rules! trait_reduction {
    ($OpReduceAPI: ident, $fn: ident, $fn_f: ident, $fn_all: ident, $fn_all_f: ident) => {
        pub fn $fn_all_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> Result<T>
        where
            R: DataAPI<Data = B::Raw>,
            D: DimAPI,
            B: $OpReduceAPI<T, D>,
        {
            tensor.device().$fn_all(tensor.raw(), tensor.layout())
        }

        pub fn $fn_f<R, T, B, D, I>(
            tensor: &TensorAny<R, T, B, D>,
            axes: I,
        ) -> Result<Tensor<T, B, IxD>>
        where
            R: DataAPI<Data = B::Raw>,
            D: DimAPI,
            B: $OpReduceAPI<T, D> + DeviceAPI<T> + DeviceCreationAnyAPI<T>,
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

        pub fn $fn_all<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> T
        where
            R: DataAPI<Data = B::Raw>,
            D: DimAPI,
            B: $OpReduceAPI<T, D>,
        {
            $fn_all_f(tensor).unwrap()
        }

        pub fn $fn<R, T, B, D, I>(tensor: &TensorAny<R, T, B, D>, axes: I) -> Tensor<T, B, IxD>
        where
            R: DataAPI<Data = B::Raw>,
            D: DimAPI,
            B: $OpReduceAPI<T, D> + DeviceCreationAnyAPI<T>,
            I: TryInto<AxesIndex<isize>>,
            Error: From<I::Error>,
        {
            $fn_f(tensor, axes).unwrap()
        }

        impl<R, T, B, D> TensorAny<R, T, B, D>
        where
            R: DataAPI<Data = B::Raw>,
            D: DimAPI,
            B: $OpReduceAPI<T, D>,
        {
            pub fn $fn_all_f(&self) -> Result<T> {
                $fn_all_f(self)
            }

            pub fn $fn_all(&self) -> T {
                $fn_all(self)
            }

            pub fn $fn_f<I>(&self, axes: I) -> Result<Tensor<T, B, IxD>>
            where
                B: DeviceCreationAnyAPI<T>,
                I: TryInto<AxesIndex<isize>>,
                Error: From<I::Error>,
            {
                $fn_f(self, axes)
            }

            pub fn $fn<I>(&self, axes: I) -> Tensor<T, B, IxD>
            where
                B: DeviceCreationAnyAPI<T>,
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

#[cfg(test)]
mod test {
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
        let a_owned = arange(3240).into_shape([12, 15, 18]).into_swapaxes(-1, -2);
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
        let s = arange((3240, &DeviceCpuSerial))
            .into_shape([4, 6, 15, 9])
            .transpose([2, 0, 3, 1])
            .sum([0, -2]);
        println!("{:?}", s);
        assert_eq!(s[[0, 1]], 27270);
        assert_eq!(s[[1, 2]], 154845);
        assert_eq!(s[[3, 5]], 428220);

        // DeviceFaer
        let s = arange(3240).into_shape([4, 6, 15, 9]).transpose([2, 0, 3, 1]).sum([0, -2]);
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
}
