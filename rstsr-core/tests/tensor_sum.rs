use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[cfg(test)]
mod test {
    use std::time::Instant;

    use super::*;

    #[test]
    #[cfg(not(feature = "col_major"))]
    fn test_tensor_sum_leading() {
        let n = 512;
        let time = Instant::now();
        let t_rstsr = {
            use rstsr_core::prelude::*;
            let mut rng = StdRng::seed_from_u64(42);

            let vec_b: Vec<f64> = (0..4 * n * n).map(|_| rng.gen()).collect::<_>();
            let b_full = rt::asarray((vec_b, [4, n, n], &DeviceCpuSerial::default()));
            b_full.sum_axes(0)
        };
        println!("{:} usec", time.elapsed().as_micros());
        let t_rstsr = t_rstsr.reshape(-1).to_vec();

        let time = Instant::now();
        let t_rayon = {
            use rstsr_core::prelude::*;
            let mut rng = StdRng::seed_from_u64(42);

            let vec_b: Vec<f64> = (0..4 * n * n).map(|_| rng.gen()).collect::<_>();
            let b_full = rt::asarray((vec_b, [4, n, n]));
            b_full.sum_axes(0)
        };
        println!("{:} usec", time.elapsed().as_micros());
        let t_rayon = t_rayon.reshape(-1).to_vec();

        // let time = Instant::now();
        let t_ndarray = {
            use ndarray::prelude::*;

            let mut rng = StdRng::seed_from_u64(42);
            let vec_b: Vec<f64> = (0..4 * n * n).map(|_| rng.gen()).collect::<_>();
            let b_full = Array::from_shape_vec((4, n, n), vec_b).unwrap();
            b_full.sum_axis(Axis(0))
        };
        println!("{:} usec", time.elapsed().as_micros());
        let t_ndarray = t_ndarray.into_raw_vec();

        let diff = t_rstsr
            .iter()
            .zip(t_ndarray.iter())
            .map(|(a, b)| (a - b).abs())
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();
        assert!(diff < 1e-6);

        let diff = t_rayon
            .iter()
            .zip(t_ndarray.iter())
            .map(|(a, b)| (a - b).abs())
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();
        assert!(diff < 1e-6);
    }

    #[test]
    #[cfg(not(feature = "col_major"))]
    fn test_tensor_sum_last() {
        let n = 512;
        let time = Instant::now();
        let t_rstsr = {
            use rstsr_core::prelude::*;
            let mut rng = StdRng::seed_from_u64(42);

            let vec_b: Vec<f64> = (0..4 * n * n).map(|_| rng.gen()).collect::<_>();
            let b_full = rt::asarray((vec_b, [4, n, n], &DeviceCpuSerial::default()));
            b_full.sum_axes([-1, -2])
        };
        println!("{:} usec", time.elapsed().as_micros());
        let t_rstsr = t_rstsr.reshape(-1).to_vec();

        let time = Instant::now();
        let t_rayon = {
            use rstsr_core::prelude::*;
            let mut rng = StdRng::seed_from_u64(42);

            let vec_b: Vec<f64> = (0..4 * n * n).map(|_| rng.gen()).collect::<_>();
            let b_full = rt::asarray((vec_b, [4, n, n]));
            b_full.sum_axes([-1, -2])
        };
        println!("{:} usec", time.elapsed().as_micros());
        let t_rayon = t_rayon.reshape(-1).to_vec();

        let time = Instant::now();
        let t_ndarray = {
            use ndarray::prelude::*;

            let mut rng = StdRng::seed_from_u64(42);
            let vec_b: Vec<f64> = (0..4 * n * n).map(|_| rng.gen()).collect::<_>();
            let b_full = Array::from_shape_vec((4, n, n), vec_b).unwrap();
            b_full.sum_axis(Axis(2)).sum_axis(Axis(1))
        };
        println!("{:} usec", time.elapsed().as_micros());
        let t_ndarray = t_ndarray.into_raw_vec();

        let diff = t_rstsr
            .iter()
            .zip(t_ndarray.iter())
            .map(|(a, b)| (a - b).abs())
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();
        assert!(diff < 1e-6);

        let diff = t_rayon
            .iter()
            .zip(t_ndarray.iter())
            .map(|(a, b)| (a - b).abs())
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();
        assert!(diff < 1e-6);
    }
}
