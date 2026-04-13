#[test]
fn fatal_code() {
    use rstsr_core::prelude::*;
    use rstsr_core::prelude_dev::*;

    let vec_a = vec![vec![0]; 12];
    let a: Tensor<Vec<usize>> = rt::asarray(vec_a);
    println!("{:?}", a);
    let mut a: Tensor<MaybeUninit<Vec<usize>>> = rt::uninit([12]);
    a.iter_mut().enumerate().for_each(|(idx, x)| {
        *x = MaybeUninit::new(vec![idx]);
    });
    let a = unsafe { rt::assume_init(a) };
    println!("{:?}", a);
}
