use criterion::{black_box, criterion_group, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;

fn rstsr_serial_4096(criterion: &mut Criterion) {
    use rstsr_core::prelude::*;

    let n = 4096;
    let mut rng = StdRng::seed_from_u64(42);

    let vec_a: Vec<f64> = (0..2 * n * 2 * n).map(|_| rng.gen()).collect::<_>();
    let vec_b: Vec<f64> = (0..2 * n * 2 * n).map(|_| rng.gen()).collect::<_>();
    let a_full = rt::asarray((vec_a, [2 * n, 2 * n], &DeviceCpuSerial));
    let b_full = rt::asarray((vec_b, [2 * n, 2 * n], &DeviceCpuSerial));

    criterion.bench_function("rstsr serial add 8192 contiguous", |bencher| {
        bencher.iter(|| {
            let c = &a_full + &b_full;
            black_box(c);
        })
    });

    let a = a_full.slice((..n, ..n)).into_dim::<Ix2>();
    let b = b_full.slice((..n, ..n)).into_dim::<Ix2>();

    criterion.bench_function("rstsr serial add 4096 c-prefer", |bencher| {
        bencher.iter(|| {
            let c = &a + &b;
            black_box(c);
        })
    });

    criterion.bench_function("rstsr serial add 4096 c-prefer transpose", |bencher| {
        bencher.iter(|| {
            let c = &a + &b.t();
            black_box(c);
        })
    });

    let a = a_full.i((slice!(0, 2 * n, 2), slice!(0, 2 * n, 2))).into_dim::<Ix2>();
    let b = b_full.i((slice!(0, 2 * n, 2), slice!(0, 2 * n, 2))).into_dim::<Ix2>();

    criterion.bench_function("rstsr serial add 4096 strided", |bencher| {
        bencher.iter(|| {
            let c = &a + &b;
            black_box(c);
        })
    });
}

fn rstsr_rayon_4096(criterion: &mut Criterion) {
    use rstsr_core::prelude::*;

    let n = 4096;
    let mut rng = StdRng::seed_from_u64(42);

    let vec_a: Vec<f64> = (0..2 * n * 2 * n).map(|_| rng.gen()).collect::<_>();
    let vec_b: Vec<f64> = (0..2 * n * 2 * n).map(|_| rng.gen()).collect::<_>();
    let a_full = rt::asarray((vec_a, [2 * n, 2 * n]));
    let b_full = rt::asarray((vec_b, [2 * n, 2 * n]));

    let a = a_full.slice((..n, ..n)).into_dim::<Ix2>();
    let b = b_full.slice((..n, ..n)).into_dim::<Ix2>();

    criterion.bench_function("rstsr rayon add 8192 contiguous", |bencher| {
        bencher.iter(|| {
            let c = &a_full + &b_full;
            black_box(c);
        })
    });

    criterion.bench_function("rstsr rayon add 4096 c-prefer", |bencher| {
        bencher.iter(|| {
            let c = &a + &b;
            black_box(c);
        })
    });

    criterion.bench_function("rstsr rayon add 4096 c-prefer transpose", |bencher| {
        bencher.iter(|| {
            let c = &a + &b.t();
            black_box(c);
        })
    });

    let a = a_full.i((slice!(0, 2 * n, 2), slice!(0, 2 * n, 2))).into_dim::<Ix2>();
    let b = b_full.i((slice!(0, 2 * n, 2), slice!(0, 2 * n, 2))).into_dim::<Ix2>();

    criterion.bench_function("rstsr rayon add 4096 strided", |bencher| {
        bencher.iter(|| {
            let c = &a + &b;
            black_box(c);
        })
    });
}

fn ndarray_4096(criterion: &mut Criterion) {
    use ndarray::prelude::*;

    let n = 4096;
    let mut rng = StdRng::seed_from_u64(42);

    let vec_a: Vec<f64> = (0..2 * n * 2 * n).map(|_| rng.gen()).collect::<_>();
    let vec_b: Vec<f64> = (0..2 * n * 2 * n).map(|_| rng.gen()).collect::<_>();
    let a_full = Array2::from_shape_vec((2 * n, 2 * n), vec_a).unwrap();
    let b_full = Array2::from_shape_vec((2 * n, 2 * n), vec_b).unwrap();

    let a = a_full.slice(s![0..n, 0..n]);
    let b = b_full.slice(s![0..n, 0..n]);

    criterion.bench_function("ndarray add 8192 contiguous", |bencher| {
        bencher.iter(|| {
            let c = &a_full + &b_full;
            black_box(c);
        })
    });

    criterion.bench_function("ndarray add 4096 c-prefer", |bencher| {
        bencher.iter(|| {
            let c = &a + &b;
            black_box(c);
        })
    });

    criterion.bench_function("ndarray add 4096 c-prefer transpose", |bencher| {
        bencher.iter(|| {
            let c = &a + &b.t();
            black_box(c);
        })
    });

    let a = a_full.slice(s![0..2 * n;2, 0..2 * n;2]);
    let b = b_full.slice(s![0..2 * n;2, 0..2 * n;2]);

    criterion.bench_function("ndarray add 4096 strided", |bencher| {
        bencher.iter(|| {
            let c = &a + &b;
            black_box(c);
        })
    });
}

criterion_group! {
    name = bench;
    config = Criterion::default().warm_up_time(Duration::from_millis(200)).measurement_time(Duration::from_millis(2000)).sample_size(10);
    targets = rstsr_serial_4096, rstsr_rayon_4096, ndarray_4096
}
