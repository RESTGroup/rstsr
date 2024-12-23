use criterion::{criterion_group, criterion_main, Criterion};
use rstsr_core::prelude_dev::*;

pub fn bench_faer_gemm(crit: &mut Criterion) {
    let m = 4096;
    let n = 4096;
    let k = 4096;
    let device = DeviceFaer::default();
    let a = Tensor::linspace((0.0, 1.0, m * k, &device)).into_shape_assume_contig([m, k]);
    let b = Tensor::linspace((0.0, 1.0, k * n, &device)).into_shape_assume_contig([k, n]);
    crit.bench_function("gemm 4096", |ben| ben.iter(|| &a % &b));
    crit.bench_function("syrk 4096", |ben| ben.iter(|| &a % &a.reverse_axes()));
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = bench_faer_gemm
}
criterion_main!(benches);
