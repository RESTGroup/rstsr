//! BLAS-2 `GEMV` container: layout normalisation, `beta` handling and the
//! degenerate contraction length.
//!
//! References are computed elementwise so the assertions do not depend on the
//! `matmul` dispatch that will later route through this container.

use crate::DeviceBLAS;
use num::Complex;
use rstsr::prelude::*;
use rstsr_blas_traits::prelude::*;

fn device() -> DeviceBLAS {
    DeviceBLAS::default()
}

fn close(got: &[f64], want: &[f64]) -> bool {
    got.len() == want.len() && got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-12)
}

fn close_c(got: &[Complex<f64>], want: &[Complex<f64>]) -> bool {
    got.len() == want.len() && got.iter().zip(want).all(|(g, w)| (g - w).norm() < 1e-12)
}

fn mat_c(v: &[f64], shape: [usize; 2], device: &DeviceBLAS) -> Tensor<f64, DeviceBLAS, Ix2> {
    rt::asarray((v, shape.c(), device)).into_dim().into_owned()
}

fn mat_f(v: &[f64], shape: [usize; 2], device: &DeviceBLAS) -> Tensor<f64, DeviceBLAS, Ix2> {
    rt::asarray((v, shape.f(), device)).into_dim().into_owned()
}

fn vec1(v: &[f64], device: &DeviceBLAS) -> Tensor<f64, DeviceBLAS, Ix1> {
    rt::asarray((v, [v.len()].c(), device)).into_dim().into_owned()
}

/// `y = op(A) x + beta * y0` for a row-major `m`-by-`n` matrix `a`.
fn gemv_ref(a: &[f64], m: usize, n: usize, x: &[f64], beta: f64, y0: &[f64], trans: bool) -> Vec<f64> {
    let (len_x, len_y) = if trans { (m, n) } else { (n, m) };
    assert_eq!(a.len(), m * n);
    assert_eq!(x.len(), len_x);
    assert_eq!(y0.len(), len_y);
    let mut y: Vec<f64> = y0.iter().map(|v| beta * v).collect();
    for i in 0..m {
        for j in 0..n {
            let aij = a[i * n + j];
            if trans {
                y[j] += aij * x[i];
            } else {
                y[i] += aij * x[j];
            }
        }
    }
    y
}

/// Column-major flattening of a row-major `m`-by-`n` matrix.
fn to_col_major(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut v = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            v[i + j * m] = a[i * n + j];
        }
    }
    v
}

#[test]
fn gemv_no_trans() {
    let device = device();
    let a: Vec<f64> = (0..12).map(|v| v as f64).collect();
    let x = [1.0, -1.0, 2.0, 0.5];
    let la = mat_c(&a, [3, 4], &device);
    let lx = vec1(&x, &device);

    let got = DGEMV::default().a(la.view()).x(lx.view()).build().unwrap().run().unwrap().into_owned();
    let want = gemv_ref(&a, 3, 4, &x, 0.0, &[0.0, 0.0, 0.0], false);
    assert!(close(&got.to_vec(), &want), "got {:?} want {:?}", got.to_vec(), want);
}

#[test]
fn gemv_trans() {
    let device = device();
    let a: Vec<f64> = (0..12).map(|v| v as f64).collect();
    let x = [2.0, 0.5, -1.0];
    let la = mat_c(&a, [3, 4], &device);
    let lx = vec1(&x, &device);

    let got = DGEMV::default().a(la.view()).x(lx.view()).transa(Trans).build().unwrap().run().unwrap().into_owned();
    let want = gemv_ref(&a, 3, 4, &x, 0.0, &[0.0, 0.0, 0.0, 0.0], true);
    assert!(close(&got.to_vec(), &want), "got {:?} want {:?}", got.to_vec(), want);
}

#[test]
fn gemv_beta_with_caller_output() {
    let device = device();
    let a: Vec<f64> = (1..=12).map(|v| v as f64).collect();
    let x = [1.0, 0.0, -1.0, 2.0];
    let y0 = [1.0, 2.0, 3.0];
    let la = mat_c(&a, [3, 4], &device);
    let lx = vec1(&x, &device);
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

    let want = gemv_ref(&a, 3, 4, &x, 2.0, &y0, false);
    assert!(close(&got.to_vec(), &want), "got {:?} want {:?}", got.to_vec(), want);
}

#[test]
fn gemv_strided_y_is_written_through() {
    let device = device();
    let a: Vec<f64> = (0..12).map(|v| v as f64).collect();
    let x = [1.0, 1.0, 1.0, 1.0];
    let la = mat_c(&a, [3, 4], &device);
    let lx = vec1(&x, &device);
    // c-prefer 3x2: column 0 is [1, 3, 5] at stride 2, column 1 is [2, 4, 6]
    let y0 = [1.0, 3.0, 5.0];
    let mut base = mat_c(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2], &device);
    let want = gemv_ref(&a, 3, 4, &x, 2.0, &y0, false);
    {
        let mut column = base.i_mut((.., 0));
        assert_eq!(column.stride()[0], 2);

        let got = DGEMV::default()
            .a(la.view())
            .x(lx.view())
            .y(column.view_mut().into_dim::<Ix1>())
            .beta(2.0)
            .build()
            .unwrap()
            .run()
            .unwrap()
            .into_owned();
        assert!(close(&got.to_vec(), &want), "got {:?} want {:?}", got.to_vec(), want);
    }

    assert!(close(&base.i((.., 0)).to_vec(), &want));
    // the other column must be untouched
    assert!(close(&base.i((.., 1)).to_vec(), &[2.0, 4.0, 6.0]));
}

#[test]
fn gemv_strided_x_is_copied() {
    let device = device();
    let a: Vec<f64> = (0..12).map(|v| v as f64).collect();
    let la = mat_c(&a, [3, 4], &device);
    let base_x = mat_c(&[1.0, 9.0, 2.0, 9.0, 3.0, 9.0, 4.0, 9.0], [4, 2], &device);
    let x = base_x.i((.., 0));
    assert_eq!(x.stride()[0], 2);

    let got = DGEMV::default().a(la.view()).x(x.into_dim::<Ix1>()).build().unwrap().run().unwrap().into_owned();
    let want = gemv_ref(&a, 3, 4, &[1.0, 2.0, 3.0, 4.0], 0.0, &[0.0, 0.0, 0.0], false);
    assert!(close(&got.to_vec(), &want), "got {:?} want {:?}", got.to_vec(), want);
}

#[test]
fn gemv_column_major_input() {
    let device = device();
    let a: Vec<f64> = (0..12).map(|v| v as f64).collect();
    let x = [1.0, -2.0, 0.5, 3.0];
    let la = mat_f(&to_col_major(&a, 3, 4), [3, 4], &device);
    let lx = vec1(&x, &device);
    assert!(la.f_prefer());
    assert!(!la.c_prefer());

    let got = DGEMV::default().a(la.view()).x(lx.view()).build().unwrap().run().unwrap().into_owned();
    let want = gemv_ref(&a, 3, 4, &x, 0.0, &[0.0, 0.0, 0.0], false);
    assert!(close(&got.to_vec(), &want), "got {:?} want {:?}", got.to_vec(), want);
}

#[test]
fn gemv_zero_contraction() {
    let device = device();
    let la: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros(([3, 0].c(), &device)).into_dim();
    let lx: Tensor<f64, DeviceBLAS, Ix1> = rt::zeros(([0].c(), &device)).into_dim();

    // beta scales the incoming y
    let mut ly = vec1(&[1.0, 2.0, 3.0], &device);
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
    assert!(close(&got.to_vec(), &[2.0, 4.0, 6.0]), "got {:?}", got.to_vec());

    // beta == 0 zeroes it
    let mut ly = vec1(&[1.0, 2.0, 3.0], &device);
    let got = DGEMV::default()
        .a(la.view())
        .x(lx.view())
        .y(ly.view_mut())
        .beta(0.0)
        .build()
        .unwrap()
        .run()
        .unwrap()
        .into_owned();
    assert!(close(&got.to_vec(), &[0.0, 0.0, 0.0]), "got {:?}", got.to_vec());
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
    let la: Tensor<Complex<f64>, DeviceBLAS, Ix2> = rt::asarray((&a[..], [2, 3].c(), &device)).into_dim().into_owned();
    let lx: Tensor<Complex<f64>, DeviceBLAS, Ix1> = rt::asarray((&x[..], [2].c(), &device)).into_dim().into_owned();

    let got = ZGEMV::default().a(la.view()).x(lx.view()).transa(ConjTrans).build().unwrap().run().unwrap().into_owned();

    // y(j) = sum_i conj(a[i, j]) x(i)
    let want: Vec<Complex<f64>> = (0..3).map(|j| (0..2).map(|i| a[i * 3 + j].conj() * x[i]).sum()).collect();
    assert!(close_c(&got.to_vec(), &want), "got {:?} want {:?}", got.to_vec(), want);
}

#[test]
fn gemv_f32() {
    let device = device();
    let a: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let x = [1.0f32, -1.0, 2.0, 0.5];
    let la: Tensor<f32, DeviceBLAS, Ix2> = rt::asarray((&a[..], [3, 4].c(), &device)).into_dim().into_owned();
    let lx: Tensor<f32, DeviceBLAS, Ix1> = rt::asarray((&x[..], [4].c(), &device)).into_dim().into_owned();

    let got = SGEMV::default().a(la.view()).x(lx.view()).build().unwrap().run().unwrap().into_owned();
    let want: Vec<f32> = (0..3).map(|i| (0..4).map(|j| a[i * 4 + j] * x[j]).sum()).collect();
    let got = got.to_vec();
    assert_eq!(got.len(), want.len());
    assert!(got.iter().zip(&want).all(|(g, w)| (g - w).abs() < 1e-5), "got {:?} want {:?}", got, want);
}

#[test]
fn gemv_conj_no_trans_rejected() {
    let device = device();
    let la: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros(([2, 3].c(), &device)).into_dim();
    let lx: Tensor<f64, DeviceBLAS, Ix1> = rt::zeros(([3].c(), &device)).into_dim();
    let res = DGEMV::default().a(la.view()).x(lx.view()).transa(FlagTrans::CN).build().unwrap().run();
    assert!(res.is_err(), "CN must be rejected like GEMM's run()");
}
