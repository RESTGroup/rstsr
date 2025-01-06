use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_tensor_sum() {
        use rstsr_core::prelude::rstsr_traits::*;

        let n = 256;
        // let time = Instant::now();
        let t_rstsr = {
            use rstsr_core::prelude::*;
            let mut rng = StdRng::seed_from_u64(42);

            let vec_b: Vec<f64> = (0..4 * n * n).map(|_| rng.gen()).collect::<_>();
            let b_full = rt::asarray((vec_b, [4, n, n], &DeviceCpuSerial));
            b_full.sum(0)
        };
        // println!("{:8.3?}", t);
        // println!("{:} msec", time.elapsed().as_millis());
        let t_rstsr = t_rstsr.reshape(-1).to_vec();

        // let time = Instant::now();
        let t_ndarray = {
            use ndarray::prelude::*;

            let mut rng = StdRng::seed_from_u64(42);
            let vec_b: Vec<f64> = (0..4 * n * n).map(|_| rng.gen()).collect::<_>();
            let b_full = Array::from_shape_vec((4, n, n), vec_b).unwrap();
            b_full.sum_axis(Axis(0))
        };
        // println!("{:8.3?}", t);
        // println!("{:} msec", time.elapsed().as_millis());
        let t_ndarray = t_ndarray.into_raw_vec();

        let diff = t_rstsr
            .iter()
            .zip(t_ndarray.iter())
            .map(|(a, b)| (a - b).abs())
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();
        assert!(diff < 1e-6);
    }
}
