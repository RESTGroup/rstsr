#![cfg(feature = "faer")]

use rstsr::prelude::*;
use rstsr_core::prelude_dev::fingerprint;
use rstsr_test_manifest::get_vec;

#[allow(non_camel_case_types)]
type c64 = num::Complex<f64>;

macro_rules! c64 {
    ($real:expr, $imag:expr) => {
        c64::new($real, $imag)
    };
    ($real:expr) => {
        c64::new($real, 0.0)
    };
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_cholesky() {
        let device = DeviceFaer::default();
        let b = rt::asarray((get_vec::<c64>('b'), [1024, 1024].c(), &device));

        // default
        let c = rt::linalg::cholesky(b.view());
        assert!((fingerprint(&c) - c64!(62.89494065393874, -73.47055443374522)).norm() < 1e-8);

        // upper
        let c = rt::linalg::cholesky((b.view(), Upper));
        assert!((fingerprint(&c) - c64!(13.720509103165073, -1.8066465348490963)).norm() < 1e-8);
    }

    #[test]
    fn test_det() {
        let device = DeviceFaer::default();
        let a_vec = get_vec::<c64>('a')[..5 * 5].to_vec();
        let a = rt::asarray((a_vec, [5, 5].c(), &device));

        let det = rt::linalg::det(a.view());
        assert!((det - c64!(-24.808965756481086, 11.800248863799464)).norm() < 1e-8);
    }

    #[test]
    fn test_eigh() {
        let device = DeviceFaer::default();
        let mut a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device));

        // faer does not ensure the diagonal is real, so we need to do it ourselves
        for i in 0..1024 {
            a[[i, i]] = c64!(a[[i, i]].re, 0.0);
        }

        // default, a
        let (w, v) = rt::linalg::eigh((a.view(), Lower)).into();
        assert!((fingerprint(&w) - -100.79793355894122).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -7.450761195788254).abs() < 1e-8);

        // upper, a
        let (w, v) = rt::linalg::eigh((a.view(), Upper)).into();
        assert!((fingerprint(&w) - -103.99103522434956).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -12.184946930165328).abs() < 1e-8);
    }

    #[test]
    fn test_eigvalsh() {
        let device = DeviceFaer::default();
        let mut a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device));

        // faer does not ensure the diagonal is real, so we need to do it ourselves
        for i in 0..1024 {
            a[[i, i]] = c64!(a[[i, i]].re, 0.0);
        }

        // default, a
        let w = rt::linalg::eigvalsh(a.view());
        assert!((fingerprint(&w) - -100.79793355894122).abs() < 1e-8);

        // upper, a
        let w = rt::linalg::eigvalsh((a.view(), Upper));
        assert!((fingerprint(&w) - -103.99103522434956).abs() < 1e-8);
    }

    #[test]
    fn test_inv() {
        let device = DeviceFaer::default();
        let a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device));

        // immutable
        let a_inv = rt::linalg::inv(a.view());
        assert!((fingerprint(&a_inv) - c64!(-11.836382515156183, 8.250167298349842)).norm() < 1e-8);
    }

    #[test]
    fn test_pinv() {
        let device = DeviceFaer::default();

        // 1024 x 512
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        let (a_pinv, rank) = rt::linalg::pinv((a.view(), 20.0, 0.3)).into();
        println!("rank: {rank}, a_pinv: {:?}", fingerprint(&a_pinv));
        assert!((fingerprint(&a_pinv) - c64!(-0.03454885412959018, -0.023651876085623254)).norm() < 1e-8);
        assert_eq!(rank, 240);

        // 512 x 1024
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();

        let (a_pinv, rank) = rt::linalg::pinv((a.view(), 20.0, 0.3)).into();
        assert!((fingerprint(&a_pinv) - c64!(-0.2814806469687325, -0.15198888300458474)).norm() < 1e-8);
        assert_eq!(rank, 240);
    }

    #[test]
    fn test_solve_general() {
        let device = DeviceFaer::default();
        let mut a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
        let b_vec = get_vec::<c64>('b')[..1024 * 512].to_vec();
        let mut b = rt::asarray((b_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // default
        let x = rt::linalg::solve_general((a.view(), b.view()));
        assert!((fingerprint(&x) - c64!(404.1900761036138, -258.5602505551204)).norm() < 1e-8);

        // mutable changes itself
        rt::linalg::solve_general((a.view_mut(), b.view_mut()));
        assert!((fingerprint(&b) - c64!(404.1900761036138, -258.5602505551204)).norm() < 1e-8);
    }

    #[test]
    fn test_solve_general_for_vec() {
        let device = DeviceFaer::default();
        let mut a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
        let b_vec = get_vec::<c64>('b')[..1024].to_vec();
        let mut b = rt::asarray((b_vec, [1024].c(), &device)).into_dim::<Ix1>();

        // default
        let x = rt::linalg::solve_general((a.view(), b.view()));
        assert!((fingerprint(&x) - c64!(-15.070310793269726, -1.987917054716041)).norm() < 1e-8);

        // mutable changes itself
        rt::linalg::solve_general((a.view_mut(), b.view_mut()));
        assert!((fingerprint(&b) - c64!(-15.070310793269726, -1.987917054716041)).norm() < 1e-8);
    }

    #[test]
    fn test_solve_triangular() {
        let device = DeviceFaer::default();
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let mut a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((get_vec::<c64>('b'), [1024, 1024].c(), &device)).into_dim::<Ix2>();

        // default
        let x = rt::linalg::solve_triangular((b.view(), a.view()));
        assert!((fingerprint(&x) - c64!(-8.433708003916948, 20.578272827017052)).norm() < 1e-8);

        // upper, mutable changes a
        rt::linalg::solve_triangular((b.view(), a.view_mut(), Upper));
        assert!((fingerprint(&a) - c64!(0.1778922244846507, 11.42463765128442)).norm() < 1e-8);
    }

    #[test]
    fn test_svd() {
        let device = DeviceFaer::default();
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // default
        // note that full_matrices = true will give a different u compared to BLAS.
        let (_u, s, vt) = rt::linalg::svd(a.view()).into();
        assert!((fingerprint(&s) - 46.60343405921802).abs() < 1e-8);
        // assert!((fingerprint(&u.abs()) - -15.44133470545584).abs() < 1e-8);
        assert!((fingerprint(&vt.abs()) - 2.1605324161714172).abs() < 1e-8);

        // full_matrices = false
        let (u, s, vt) = rt::linalg::svd((a.view(), false)).into();
        assert!((fingerprint(&s) - 46.60343405921802).abs() < 1e-8);
        assert!((fingerprint(&u.abs()) - -1.9516528722381659).abs() < 1e-8);
        assert!((fingerprint(&vt.abs()) - 2.1605324161714172).abs() < 1e-8);

        // m < n, full_matrices = false
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [512, 1024].c(), &device));
        let (u, s, vt) = rt::linalg::svd((a.view(), false)).into();
        assert!((fingerprint(&s) - 47.599274835886646).abs() < 1e-8);
        assert!((fingerprint(&u.abs()) - 4.636614351700778).abs() < 1e-8);
        assert!((fingerprint(&vt.abs()) - 1.4497879458575658).abs() < 1e-8);
    }

    #[test]
    fn test_svdvals() {
        let device = DeviceFaer::default();
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // default
        let s = rt::linalg::svdvals(a.view());
        assert!((fingerprint(&s) - 46.60343405921802).abs() < 1e-8);

        // m < n
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();
        let s = rt::linalg::svdvals(a.view());
        assert!((fingerprint(&s) - 47.599274835886646).abs() < 1e-8);
    }
}
