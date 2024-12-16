use rstsr_core::prelude_dev::*;

fn main() {
    let vec = vec![1, 2, 3, 4, 5];
    let tensor = Tensor::from(vec);
    println!("{:?}", tensor);
}
