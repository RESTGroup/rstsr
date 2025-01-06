use criterion::criterion_main;

mod faer_gemm;
mod tensor_add;
mod tensor_sum;

criterion_main!(tensor_sum::bench, tensor_add::bench, faer_gemm::bench);
