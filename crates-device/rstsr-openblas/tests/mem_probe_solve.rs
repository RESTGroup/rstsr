//! One-off RSS probe: batched in-place solve must not duplicate `b`.
//!
//! Each test allocates `b` = 2 slices x (1024 x 262144) f64 = 4 GiB, snapshots
//! one slice for result verification, then solves in place and reports RSS /
//! peak (VmHWM) growth from `/proc/self/status`.
//!
//! Minimal expected results (64 GiB host, OpenBLAS):
//!
//! | test                        | layout   | peak growth | expected behavior                                          |
//! |-----------------------------|----------|-------------|------------------------------------------------------------|
//! | ..._solve_triangular        | row-major | ~60 MiB    | fully in place (TRSM, level-3 BLAS, no copies)             |
//! | ..._solve_general           | row-major | ~1 slice   | in place, but driver_gesv RowMajor transposes b into b_t   |
//! | ..._solve_general_col_major | col-major | < 1 GiB    | no b_t copy; residual growth is OpenBLAS-internal workspace |
//! | ..._solve_general_2d        | col-major | -          | control: same growth without any batching, for comparison  |
//!
//! Peak growth is a report, not an assert: thresholds are unreliable because
//! the backend may allocate workspace of unpredictable size.
//!
//! Run: `RSTSR_DEV=1 cargo test -p rstsr-openblas --test mem_probe_solve -- --ignored --nocapture
//! --test-threads=1`
#![allow(clippy::items_after_test_module)]

use rstsr::prelude::*;
use std::fs;

fn status_kib(tag: &str) -> i64 {
    let status = fs::read_to_string("/proc/self/status").unwrap();
    let field = format!("{tag}:");
    for line in status.lines() {
        if let Some(v) = line.strip_prefix(&field) {
            return v.split_whitespace().next().unwrap().parse().unwrap();
        }
    }
    panic!("{tag} not found");
}

fn gib(kib: f64) -> f64 {
    kib / (1 << 20) as f64
}

fn report(msg: &str) {
    let rss = status_kib("VmRSS") as f64;
    let hwm = status_kib("VmHWM") as f64;
    println!("{msg:<58} RSS = {:7.3} GiB | peak = {:7.3} GiB", gib(rss), gib(hwm));
}

const M: usize = 1024;
const K_RHS: usize = 262144; // columns of RHS per slice; slice = (M, K) = 2 GiB
const BATCH: usize = 2;
// one slice of b: N_RHS * M * 8 bytes = 2 GiB
const SLACK_KIB: f64 = 256.0 * 1024.0; // 256 MiB allowance for workspace/allocator noise

fn fill_b(b: &mut Tensor<f64, DeviceOpenBLAS, IxD>) {
    for (i, v) in b.raw_mut().iter_mut().enumerate() {
        *v = ((i * 2654435761) % 1000) as f64 / 1000.0 + 1.0;
    }
}

#[test]
#[ignore = "RSS probe: needs 8 GiB headroom and explicit run; see module docstring"]
fn probe_rss_inplace_solve_triangular() {
    let mut device = DeviceOpenBLAS::default();
    device.set_default_order(RowMajor);

    // lower-triangular well-conditioned A
    let mut a_data = vec![0.0f64; M * M];
    for i in 0..M {
        a_data[i * M + i] = 4.0 + (i % 7) as f64 * 0.1;
        for j in 0..i {
            a_data[i * M + j] = ((i * 31 + j) % 17) as f64 / 100.0;
        }
    }
    let a = rt::asarray((a_data, [1, M, M].c(), &device));

    report("start");
    let mut b = rt::zeros(([BATCH, M, K_RHS].c(), &device));
    fill_b(&mut b);
    report(&format!("after b ({BATCH} x {K_RHS} x {M} f64 = 4 GiB)"));

    // keep slice 0 for result verification (a 2 GiB copy)
    let keep = b.view().i((0, .., ..)).to_dim::<Ix2>().into_owned();
    let baseline = status_kib("VmRSS");
    report("after verification snapshot (baseline)");

    // in-place batched triangular solve; drop the returned mut handle
    let handle = rt::linalg::solve_triangular((a.view(), b.view_mut(), Lower));
    drop(handle);
    let after = status_kib("VmRSS");
    report("after in-place batched solve_triangular");

    let growth = (after - baseline) as f64;
    assert!(growth < SLACK_KIB, "in-place solve_triangular duplicated b: RSS grew by {:.3} GiB", gib(growth));
    assert!((status_kib("VmHWM") as f64) - (baseline as f64) < SLACK_KIB, "peak memory grew during solve");

    // correctness on slice 0 against the un-batched in-place solve
    let mut expect = keep;
    let a2 = a.i((0, .., ..)).into_dim::<Ix2>();
    drop(rt::linalg::solve_triangular((a2.view(), expect.view_mut(), Lower)));
    let got = b.view().i((0, .., ..)).to_dim::<Ix2>().into_owned();
    assert!(rt::allclose(&expect, &got, Some(1e-10)), "slice 0 result mismatch");
    println!("slice 0 correctness: OK");
    let _ = device;
}

#[test]
#[ignore = "RSS probe: needs 12 GiB headroom and explicit run; see module docstring"]
fn probe_rss_inplace_solve_general() {
    let mut device = DeviceOpenBLAS::default();
    device.set_default_order(RowMajor);

    let mut a_data = vec![0.0f64; M * M];
    for i in 0..M {
        for j in 0..M {
            a_data[i * M + j] = ((i * 7 + j * 13) % 29) as f64 / 50.0;
        }
        a_data[i * M + i] += 8.0;
    }
    let a = rt::asarray((a_data, [1, M, M].c(), &device));

    report("start");
    let mut b = rt::zeros(([BATCH, M, K_RHS].c(), &device));
    fill_b(&mut b);
    let keep = b.view().i((0, .., ..)).to_dim::<Ix2>().into_owned();
    let baseline = status_kib("VmRSS");
    report("after b + verification snapshot (baseline)");

    let handle = rt::linalg::solve_general((a.view(), b.view_mut()));
    drop(handle);
    let after = status_kib("VmRSS");
    report("after in-place batched solve");

    let growth = (after - baseline) as f64;
    // known: LAPACK row-major driver transposes b into a column-major copy
    // (one transient full slice); report instead of assert
    println!("[general] RSS growth: {:.3} GiB (transient copy in driver_gesv RowMajor branch)", gib(growth));

    let mut expect = keep;
    let a2 = a.i((0, .., ..)).into_dim::<Ix2>();
    drop(rt::linalg::solve_general((a2.view(), expect.view_mut())));
    let got = b.view().i((0, .., ..)).to_dim::<Ix2>().into_owned();
    assert!(rt::allclose(&expect, &got, Some(1e-8)), "slice 0 result mismatch");
    println!("slice 0 correctness: OK");
    let _ = device;
}

#[test]
#[ignore = "RSS probe: needs 12 GiB headroom and explicit run; see module docstring"]
fn probe_rss_inplace_solve_general_col_major() {
    let mut device = DeviceOpenBLAS::default();
    device.set_default_order(ColMajor);

    let mut a_data = vec![0.0f64; M * M];
    for i in 0..M {
        for j in 0..M {
            a_data[i * M + j] = ((i * 7 + j * 13) % 29) as f64 / 50.0;
        }
        a_data[i * M + i] += 8.0;
    }
    // col-major convention: inner (matrix) axes are the first two
    let a = rt::asarray((a_data, [M, M, 1].f(), &device));

    report("[colmajor] start");
    let mut b = rt::zeros(([M, K_RHS, BATCH].f(), &device));
    // fill by index math; f-layout: element (i, j, k) at i + j*M + k*M*K
    {
        let raw = b.raw_mut();
        for i in 0..M {
            for j in 0..K_RHS {
                for k in 0..BATCH {
                    raw[i + j * M + k * M * K_RHS] = ((i + j * 7 + k * 13) % 1000) as f64 / 1000.0 + 1.0;
                }
            }
        }
    }
    report("[colmajor] after b (4 GiB)");
    // snapshot slice 0, shape (M, K), f-prefer
    let keep = b.view().i((.., .., 0)).to_dim::<Ix2>().into_owned();
    let baseline = status_kib("VmRSS");
    report("[colmajor] after verification snapshot (baseline)");

    let handle = rt::linalg::solve_general((a.view(), b.view_mut()));
    drop(handle);
    let after = status_kib("VmRSS");
    report("[colmajor] after in-place batched solve_general (col-major b)");

    // peak == RSS here (no row-major b_t copy); residual growth attributed to
    // OpenBLAS-internal workspace (see notes on GESV fallback paths)
    let growth = (after - baseline) as f64;
    println!("[colmajor] RSS growth: {:.3} GiB", gib(growth));

    let mut expect = keep;
    let a2 = a.i((.., .., 0)).into_dim::<Ix2>();
    drop(rt::linalg::solve_general((a2.view(), expect.view_mut())));
    let got = b.view().i((.., .., 0)).to_dim::<Ix2>().into_owned();
    assert!(rt::allclose(&expect, &got, Some(1e-8)), "[colmajor] slice 0 result mismatch");
    println!("[colmajor] slice 0 correctness: OK");
}

#[test]
#[ignore = "RSS probe: needs 4 GiB headroom and explicit run; see module docstring"]
fn probe_rss_solve_general_2d() {
    let device = DeviceOpenBLAS::default();
    let k = K_RHS / 4; // 0.5 GiB b
    let mut a_data = vec![0.0f64; M * M];
    for i in 0..M {
        for j in 0..M {
            a_data[i * M + j] = ((i * 7 + j * 13) % 29) as f64 / 50.0;
        }
        a_data[i * M + i] += 8.0;
    }
    let a = rt::asarray((a_data, [M, M].f(), &device));
    let mut b = rt::zeros(([M, k].f(), &device));
    {
        let raw = b.raw_mut();
        for (i, v) in raw.iter_mut().enumerate().take(M * k) {
            *v = ((i * 7) % 1000) as f64 / 1000.0 + 1.0;
        }
    }
    report("[2d] baseline (b = 2 GiB, f-layout)");
    drop(rt::linalg::solve_general((a.view(), b.view_mut())));
    report("[2d] after in-place 2-D solve_general");
    println!("[2d] fingerprint = {:?}", b.sum());
}
