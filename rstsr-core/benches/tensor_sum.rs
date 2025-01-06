use criterion::{black_box, criterion_group, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;

fn rstsr_4096(criterion: &mut Criterion) {
    use rstsr_core::prelude::*;

    let n = 4096;
    let mut rng = StdRng::seed_from_u64(42);

    let vec_a: Vec<f64> = (0..n * n).map(|_| rng.gen()).collect::<_>();
    let a = rt::asarray((vec_a, [n, n], &DeviceCpuSerial));

    criterion.bench_function("rstsr simple sum 4096", |bencher| {
        bencher.iter(|| {
            let c = a.sum_all();
            black_box(c);
        })
    });

    let vec_b: Vec<f64> = (0..4 * n * n).map(|_| rng.gen()).collect::<_>();
    let b_full = rt::asarray((vec_b, [2 * n, 2 * n].f(), &DeviceCpuSerial));

    // contiguous slice
    let b = b_full.slice([0..n, 0..n]);
    criterion.bench_function("rstsr simple sum 4096 slice", |bencher| {
        bencher.iter(|| {
            let c = b.sum_all();
            black_box(c);
        })
    });

    // strided
    let b = b_full.slice([slice!(0, 2 * n, 2), slice!(0, 2 * n, 2)]);
    criterion.bench_function("rstsr simple sum 4096 strided", |bencher| {
        bencher.iter(|| {
            let c = b.sum_all();
            black_box(c);
        })
    });
}

fn ndarray_4096(criterion: &mut Criterion) {
    use ndarray::prelude::*;

    let n = 4096;
    let mut rng = StdRng::seed_from_u64(42);

    let vec_a: Vec<f64> = (0..n * n).map(|_| rng.gen()).collect::<_>();
    let a = Array2::from_shape_vec((n, n).f(), vec_a).unwrap();

    criterion.bench_function("ndarray simple sum 4096", |bencher| {
        bencher.iter(|| {
            let c = a.sum();
            black_box(c);
        })
    });

    let vec_b: Vec<f64> = (0..4 * n * n).map(|_| rng.gen()).collect::<_>();
    let b_full = Array2::from_shape_vec((2 * n, 2 * n), vec_b).unwrap();

    // contiguous slice
    let b = b_full.slice(s![..n, ..n]);
    criterion.bench_function("ndarray simple sum 4096 slice", |bencher| {
        bencher.iter(|| {
            let c = b.sum();
            black_box(c);
        })
    });

    // strided
    let b = b_full.slice(s![..;2, ..;2]);
    criterion.bench_function("ndarray simple sum 4096 strided", |bencher| {
        bencher.iter(|| {
            let c = b.sum();
            black_box(c);
        })
    });
}

criterion_group! {
    name = bench;
    config = Criterion::default().warm_up_time(Duration::from_millis(200)).measurement_time(Duration::from_millis(2000));
    targets = rstsr_4096, ndarray_4096
}
