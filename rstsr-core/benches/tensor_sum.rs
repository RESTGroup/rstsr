use criterion::{black_box, criterion_group, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;

fn rstsr_serial_4096(criterion: &mut Criterion) {
    use rstsr_core::prelude::*;

    let n = 4096;
    let mut rng = StdRng::seed_from_u64(42);

    let vec_a: Vec<f64> = (0..n * n).map(|_| rng.gen()).collect::<_>();
    let a = rt::asarray((vec_a, [n, n], &DeviceCpuSerial));

    criterion.bench_function("rstsr serial simple sum 4096", |bencher| {
        bencher.iter(|| {
            let c = a.sum_all();
            black_box(c);
        })
    });

    let vec_b: Vec<f64> = (0..4 * n * n).map(|_| rng.gen()).collect::<_>();
    let b_full = rt::asarray((vec_b, [2 * n, 2 * n], &DeviceCpuSerial));

    // contiguous slice
    let b = b_full.slice([0..n, 0..n]);
    criterion.bench_function("rstsr serial simple sum 4096 slice", |bencher| {
        bencher.iter(|| {
            let c = b.sum_all();
            black_box(c);
        })
    });

    // strided
    let b = b_full.slice([slice!(0, 2 * n, 2), slice!(0, 2 * n, 2)]);
    criterion.bench_function("rstsr serial simple sum 4096 strided", |bencher| {
        bencher.iter(|| {
            let c = b.sum_all();
            black_box(c);
        })
    });

    // add leading dimension
    let b = b_full.reshape([4, n, n]);
    criterion.bench_function("rstsr serial simple sum 4096 leading dimension", |bencher| {
        bencher.iter(|| {
            // let c = b.i(0) + b.i(1) + b.i(2) + b.i(3);
            let c = b.sum(0);
            black_box(c);
        })
    });

    // add last dimension
    let b = b_full.reshape([4, n, n]);
    criterion.bench_function("rstsr serial simple sum 4096 last dimension", |bencher| {
        bencher.iter(|| {
            let c = b.sum([-1, -2]);
            black_box(c);
        })
    });
}

fn rstsr_rayon_4096(criterion: &mut Criterion) {
    use rstsr_core::prelude::*;

    let n = 4096;
    let mut rng = StdRng::seed_from_u64(42);

    let vec_a: Vec<f64> = (0..n * n).map(|_| rng.gen()).collect::<_>();
    let a = rt::asarray((vec_a, [n, n]));

    criterion.bench_function("rstsr rayon simple sum 4096", |bencher| {
        bencher.iter(|| {
            let c = a.sum_all();
            black_box(c);
        })
    });

    let vec_b: Vec<f64> = (0..4 * n * n).map(|_| rng.gen()).collect::<_>();
    let b_full = rt::asarray((vec_b, [2 * n, 2 * n]));

    // contiguous slice
    let b = b_full.slice([0..n, 0..n]);
    criterion.bench_function("rstsr rayon simple sum 4096 slice", |bencher| {
        bencher.iter(|| {
            let c = b.sum_all();
            black_box(c);
        })
    });

    // strided
    let b = b_full.slice([slice!(0, 2 * n, 2), slice!(0, 2 * n, 2)]);
    criterion.bench_function("rstsr rayon simple sum 4096 strided", |bencher| {
        bencher.iter(|| {
            let c = b.sum_all();
            black_box(c);
        })
    });

    // add leading dimension
    let b = b_full.reshape([4, n, n]);
    criterion.bench_function("rstsr rayon simple sum 4096 leading dimension", |bencher| {
        bencher.iter(|| {
            // let c = b.i(0) + b.i(1) + b.i(2) + b.i(3);
            let c = b.sum(0);
            black_box(c);
        })
    });

    // add last dimension
    let b = b_full.reshape([4, n, n]);
    criterion.bench_function("rstsr serial simple sum 4096 last dimension", |bencher| {
        bencher.iter(|| {
            let c = b.sum([-1, -2]);
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

    // add leading dimension
    let b = b_full.to_shape((4, n, n)).unwrap();
    criterion.bench_function("ndarray simple sum 4096 leading dimension", |bencher| {
        bencher.iter(|| {
            let c = b.sum_axis(Axis(0));
            black_box(c);
        })
    });

    // add last dimension
    let b = b_full.to_shape((4, n * n)).unwrap();
    criterion.bench_function("ndarray simple sum 4096 last dimension", |bencher| {
        bencher.iter(|| {
            let c = b.sum_axis(Axis(1));
            black_box(c);
        })
    });
}

criterion_group! {
    name = bench;
    config = Criterion::default().warm_up_time(Duration::from_millis(200)).measurement_time(Duration::from_millis(2000)).sample_size(10);
    targets = rstsr_serial_4096, rstsr_rayon_4096, ndarray_4096
}
