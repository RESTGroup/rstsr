use criterion::criterion_main;

mod faer_gemm;
mod tensor_sum;

criterion_main!(tensor_sum::bench, faer_gemm::bench);
