//! BLAS-2 `GEMV` container tests (OpenBLAS device, issue-independent).
//!
//! The container is the only place a device crate can exercise
//! `GEMVDriverAPI`, so these tests cover the layout paths the `matmul`
//! dispatch does not: `Trans`/`ConjTrans`, `beta != 0` with a caller output,
//! strided `x`/`y`, column-major input, zero contraction length, and f32.

use num::Complex;
use rstsr::prelude::*;
use rstsr_blas_traits::prelude::*;

type Dev = DeviceOpenBLAS;

fn device() -> Dev {
    Dev::default()
}

fn close(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-12)
}

fn close_c(a: &[Complex<f64>], b: &[Complex<f64>]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).norm() < 1e-12)
}

fn mat(v: &[f64], shape: [usize; 2], device: &Dev) -> Tensor<f64, Dev, Ix2> {
    rt::asarray((v, shape.c(), device)).into_dim().into_owned()
}

fn vec1(v: &[f64], device: &Dev) -> Tensor<f64, Dev, Ix1> {
    rt::asarray((v, [v.len()].c(), device)).into_dim().into_owned()
}

#[test]
fn gemv_matvec_matches_matmul() {
    let device = device();
    let a: Vec<f64> = (0..12).map(|v| v as f64).collect();
    let x = vec![1.0, -1.0, 2.0, 0.5];
    let la = mat(&a, [3, 4], &device);
    let lx = vec1(&x, &device);

    let got = DGEMV::default().a(la.view()).x(lx.view()).build().unwrap().run().unwrap().into_owned();
    let want = &la % &lx;
    assert!(close(&got.to_vec(), &want.to_vec()), "got {:?} want {:?}", got.to_vec(), want.to_vec());
}

#[test]
fn gemv_trans_matches_vecmat() {
    let device = device();
    let a: Vec<f64> = (0..12).map(|v| v as f64).collect();
    let x = vec![2.0, 0.5, -1.0];
    let la = mat(&a, [3, 4], &device);
    let lx = vec1(&x, &device);

    let got = DGEMV::default().a(la.view()).x(lx.view()).transa(Trans).build().unwrap().run().unwrap().into_owned();
    let want = &lx % &la;
    assert!(close(&got.to_vec(), &want.to_vec()), "got {:?} want {:?}", got.to_vec(), want.to_vec());
}

#[test]
fn gemv_beta_with_caller_output() {
    let device = device();
    let a: Vec<f64> = (1..=12).map(|v| v as f64).collect();
    let x = vec![1.0, 0.0, -1.0, 2.0];
    let la = mat(&a, [3, 4], &device);
    let lx = vec1(&x, &device);
    let y0 = vec![1.0, 2.0, 3.0];
    let mut ly = vec1(&y0, &device);

    let got = DGEMV::default()
        .a(la.view())
        .x(lx.view())
        .y(ly.view_mut())
        .alpha(1.0)
        .beta(2.0)
        .build()
        .unwrap()
        .run()
        .unwrap()
        .into_owned();

    let base = (&la % &lx).to_vec();
    let want: Vec<f64> = base.iter().zip(y0.iter()).map(|(v, y)| v + 2.0 * y).collect();
    assert!(close(&got.to_vec(), &want), "got {:?} want {:?}", got.to_vec(), want);
}

#[test]
fn gemv_strided_y_is_written_through() {
    let device = device();
    let a: Vec<f64> = (0..12).map(|v| v as f64).collect();
    let x = vec![1.0, 1.0, 1.0, 1.0];
    let la = mat(&a, [3, 4], &device);
    let lx = vec1(&x, &device);
    let mut base: Tensor<f64, Dev, Ix2> = rt::zeros(([3, 2].c(), &device)).into_dim();
    {
        let mut column = base.i_mut((.., 0));
        assert_eq!(column.stride()[0], 2);

        let _ = DGEMV::default()
            .a(la.view())
            .x(lx.view())
            .y(column.view_mut().into_dim::<Ix1>())
            .build()
            .unwrap()
            .run()
            .unwrap();
    }

    let want = (&la % &lx).to_vec();
    let got = base.i((.., 0)).to_vec();
    assert!(close(&got, &want), "got {:?} want {:?}", got, want);
    // the other column must be untouched
    assert!(base.i((.., 1)).to_vec().iter().all(|v| *v == 0.0));
}

#[test]
fn gemv_strided_x_is_copied() {
    let device = device();
    let a: Vec<f64> = (0..12).map(|v| v as f64).collect();
    let la = mat(&a, [3, 4], &device);
    let interleaved = [1.0, 9.0, 2.0, 9.0, 3.0, 9.0, 4.0, 9.0];
    let base_x: Tensor<f64, Dev, Ix2> = rt::asarray((&interleaved[..], [4, 2].c(), &device)).into_dim().into_owned();
    let x = base_x.i((.., 0));
    assert_eq!(x.stride()[0], 2);

    let got = DGEMV::default().a(la.view()).x(x.into_dim::<Ix1>()).build().unwrap().run().unwrap().into_owned();
    let want = &la % &vec1(&[1.0, 2.0, 3.0, 4.0], &device);
    assert!(close(&got.to_vec(), &want.to_vec()));
}

#[test]
fn gemv_column_major_input() {
    let device = device();
    let a: Vec<f64> = (0..12).map(|v| v as f64).collect();
    let x = vec![1.0, -2.0, 0.5, 3.0];
    // f-prefer buffer: the same 12 elements read column-major as a 3x4 matrix
    let la: Tensor<f64, Dev, Ix2> = rt::asarray((&a[..], [3, 4].f(), &device)).into_dim().into_owned();
    let lx = vec1(&x, &device);
    assert!(la.f_prefer());
    assert!(!la.c_prefer());

    let got = DGEMV::default().a(la.view()).x(lx.view()).build().unwrap().run().unwrap().into_owned();
    let want = &la % &lx;
    assert!(close(&got.to_vec(), &want.to_vec()));
}

#[test]
fn gemv_zero_contraction_scales_y() {
    let device = device();
    let la: Tensor<f64, Dev, Ix2> = rt::zeros(([3, 0].c(), &device)).into_dim();
    let lx: Tensor<f64, Dev, Ix1> = rt::zeros(([0].c(), &device)).into_dim();
    let y0 = vec![1.0, 2.0, 3.0];
    let mut ly = vec1(&y0, &device);

    let got = DGEMV::default()
        .a(la.view())
        .x(lx.view())
        .y(ly.view_mut())
        .beta(2.0)
        .build()
        .unwrap()
        .run()
        .unwrap()
        .into_owned();
    assert!(close(&got.to_vec(), &[2.0, 4.0, 6.0]));
}

#[test]
fn gemv_conj_trans_complex() {
    let device = device();
    let a: Vec<Complex<f64>> = vec![
        Complex::new(1.0, 2.0),
        Complex::new(0.0, -1.0),
        Complex::new(3.0, 0.0),
        Complex::new(-2.0, 1.0),
        Complex::new(0.5, 0.5),
        Complex::new(4.0, 0.0),
    ];
    let x: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 2.0)];
    let la: Tensor<Complex<f64>, Dev, Ix2> = rt::asarray((&a[..], [2, 3].c(), &device)).into_dim().into_owned();
    let lx: Tensor<Complex<f64>, Dev, Ix1> = rt::asarray((&x[..], [2].c(), &device)).into_dim().into_owned();

    let got = ZGEMV::default().a(la.view()).x(lx.view()).transa(ConjTrans).build().unwrap().run().unwrap().into_owned();

    // y(j) = sum_i conj(a[i, j]) x(i)
    let mut want = vec![Complex::new(0.0, 0.0); 3];
    for j in 0..3 {
        for i in 0..2 {
            want[j] += a[i * 3 + j].conj() * x[i];
        }
    }
    assert!(close_c(&got.to_vec(), &want), "got {:?} want {:?}", got.to_vec(), want);
}

#[test]
fn gemv_f32_matvec() {
    let device = device();
    let a: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let x = [1.0f32, -1.0, 2.0, 0.5];
    let la: Tensor<f32, Dev, Ix2> = rt::asarray((&a[..], [3, 4].c(), &device)).into_dim().into_owned();
    let lx: Tensor<f32, Dev, Ix1> = rt::asarray((&x[..], [4].c(), &device)).into_dim().into_owned();

    let got = SGEMV::default().a(la.view()).x(lx.view()).build().unwrap().run().unwrap().into_owned();
    let want = (&la % &lx).to_vec();
    let got = got.to_vec();
    assert_eq!(got.len(), want.len());
    assert!(got.iter().zip(want.iter()).all(|(x, y)| (x - y).abs() < 1e-5));
}

#[test]
fn gemv_conj_no_trans_rejected() {
    let device = device();
    let la: Tensor<f64, Dev, Ix2> = rt::zeros(([2, 3].c(), &device)).into_dim();
    let lx: Tensor<f64, Dev, Ix1> = rt::zeros(([3].c(), &device)).into_dim();
    let res = DGEMV::default().a(la.view()).x(lx.view()).transa(FlagTrans::CN).build().unwrap().run();
    assert!(res.is_err(), "CN must be rejected like GEMM's run()");
}
