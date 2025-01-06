use criterion::{black_box, criterion_group, Criterion};
use rstsr_core::prelude_dev::*;
use std::time::Duration;

pub fn bench_faer_gemm(crit: &mut Criterion) {
    let m = 4096;
    let n = 4096;
    let k = 4096;
    let device = DeviceFaer::default();
    let a = linspace((0.0, 1.0, m * k, &device)).into_shape_assume_contig([m, k]);
    let b = linspace((0.0, 1.0, k * n, &device)).into_shape_assume_contig([k, n]);
    crit.bench_function("gemm 4096", |ben| ben.iter(|| black_box(&a % &b)));
    crit.bench_function("syrk 4096", |ben| ben.iter(|| black_box(&a % &a.reverse_axes())));
}

criterion_group! {
    name = bench;
    config = Criterion::default().warm_up_time(Duration::from_secs(1)).measurement_time(Duration::from_secs(10));
    targets = bench_faer_gemm
}
