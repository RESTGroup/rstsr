#![doc = include_str!("../notes_to_api_doc.md")]
#![doc = include_str!("../readme.md")]
pub mod prelude;

#[test]
fn test() {
    use crate::prelude::*;

    // 3x2 matrix with c-contiguous memory layout
    let a = rt::asarray((vec![6., 2., 7., 4., 8., 5.], [3, 2].c()));

    // 2x4x3 matrix by arange and reshaping
    let b = rt::arange(24.);
    let b = b.reshape((-1, 4, 3));
    // in one line, you can also
    // let b = rt::arange(24.).into_shape((-1, 4, 3));

    // broadcasted matrix multiplication
    let c = b % a;

    // print layout of the result
    println!("{:?}", c.layout());
    // output:
    // 3-Dim (dyn), contiguous: Cc
    // shape: [2, 4, 2], stride: [8, 2, 1], offset: 0

    // print the result
    println!("{:6.1}", c);
    // output:
    // [[[   23.0   14.0]
    //   [   86.0   47.0]
    //   [  149.0   80.0]
    //   [  212.0  113.0]]
    //
    //  [[  275.0  146.0]
    //   [  338.0  179.0]
    //   [  401.0  212.0]
    //   [  464.0  245.0]]]
}
