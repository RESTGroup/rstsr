#[cfg(test)]
mod test {

    #[test]
    fn workable() {
        use rstsr_core::prelude::*;
        use rstsr_openblas::DeviceOpenBLAS;

        // specify the number of threads of 16
        let device = DeviceOpenBLAS::new(16);
        // if you want to use the default number of threads, use the following line
        // let device = DeviceOpenBLAS::default();

        let a = rt::linspace((0.0, 1.0, 1048576, &device)).into_shape([16, 256, 256]);
        let b = rt::linspace((1.0, 2.0, 1048576, &device)).into_shape([16, 256, 256]);

        // by optimized BLAS, the following operation is very fast
        let c = &a % &b;

        // mean of all elements is also performed in parallel
        let c_mean = c.mean_all();
        println!("{:?}", c_mean);
        assert!((c_mean - 213.2503660477036) < 1e-6);

        let c_std = c.std_all();
        println!("{:?}", c_std);
        assert!((c_std - 148.88523481701804) < 1e-6);

        let c_std_1 = c.std_axes((0, 1));
        println!("{}", c_std_1);

        let c_std_2 = c.std_axes((1, 2));
        println!("{}", c_std_2);
    }
}
