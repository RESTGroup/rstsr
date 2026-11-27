use super::*;
use rstsr::prelude::linalg::LdlResult;

// the result types follow the dims of a 2-D input
type LdlOutC = LdlResult<Tensor<c64, DeviceBLAS, Ix2>, Tensor<c64, DeviceBLAS, Ix2>, Tensor<usize, DeviceBLAS, Ix1>>;
type LdlTupleC = (Tensor<c64, DeviceBLAS, Ix2>, Tensor<c64, DeviceBLAS, Ix2>, Tensor<usize, DeviceBLAS, Ix1>);

/// Hermitian matrix (`n`, `n`) from `a`: `a + aᴴ`.
fn hermitize(a: TensorView<'_, c64, DeviceBLAS, Ix2>) -> Tensor<c64, DeviceBLAS, Ix2> {
    let n = a.shape()[0];
    let mut h = a.view().to_owned();
    for i in 0..n {
        for j in 0..n {
            h[[i, j]] = a[[i, j]] + a[[j, i]].conj();
        }
    }
    h
}

/// Symmetric matrix (`n`, `n`) from `a`: `a + aᵀ`.
fn symmetrize(a: TensorView<'_, c64, DeviceBLAS, Ix2>) -> Tensor<c64, DeviceBLAS, Ix2> {
    let n = a.shape()[0];
    let mut s = a.view().to_owned();
    for i in 0..n {
        for j in 0..n {
            s[[i, j]] = a[[i, j]] + a[[j, i]];
        }
    }
    s
}

/// Self-validation of one Hermitian result: `lu[perm, :]` unit triangular,
/// `d` block diagonal and Hermitian (or symmetric for `hermitian = false`),
/// `A = lu @ d @ luᴴ` (or `lu @ d @ luᵀ`).
fn check_ldl(
    a: &Tensor<c64, DeviceBLAS, Ix2>,
    lu: &Tensor<c64, DeviceBLAS, Ix2>,
    d: &Tensor<c64, DeviceBLAS, Ix2>,
    perm: &Tensor<usize, DeviceBLAS, Ix1>,
    lower: bool,
    hermitian: bool,
    at: &str,
) {
    let n = a.shape()[0];
    assert_eq!(lu.shape(), &[n, n], "lu shape, {at}");
    assert_eq!(d.shape(), &[n, n], "d shape, {at}");
    assert_eq!(perm.shape(), &[n], "perm shape, {at}");
    let mut sorted = perm.raw().to_vec();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..n).collect::<Vec<_>>(), "perm is a permutation, {at}");
    for row in 0..n {
        let src = perm[[row]];
        for col in 0..n {
            let v = lu[[src, col]];
            if col == row {
                assert!((v - c64!(1.0, 0.0)).norm() < 1e-13, "unit diagonal of lu[perm] ({src}, {col}), {at}");
            } else if (lower && col > row) || (!lower && col < row) {
                assert_eq!(v, c64!(0.0, 0.0), "zero outside the triangle of lu[perm] ({src}, {col}), {at}");
            }
        }
    }
    for i in 0..n {
        for j in 0..n {
            if (i as isize - j as isize).abs() >= 2 {
                assert_eq!(d[[i, j]], c64!(0.0, 0.0), "d band ({i}, {j}), {at}");
            }
        }
    }
    let mut blocks = 0;
    let mut k = 0;
    while k + 1 < n {
        let off = if lower { d[[k + 1, k]] } else { d[[k, k + 1]] };
        if off != c64!(0.0, 0.0) {
            blocks += 1;
            k += 2;
        } else {
            k += 1;
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let expected = if hermitian { d[[j, i]].conj() } else { d[[j, i]] };
            assert_eq!(d[[i, j]], expected, "d Hermitian/symmetric ({i}, {j}), {at}");
        }
    }
    let prod = rt::matmul(rt::matmul(lu.view(), d.view()), conj_transpose_for(lu, hermitian).view());
    assert!(rt::allclose(prod.view(), a.view(), Some(1e-12)), "A = lu @ d @ luᴴ/luᵀ ({blocks} 2x2 blocks), {at}");
}

/// `luᴴ` for the Hermitian form, `luᵀ` for the symmetric one.
fn conj_transpose_for(a: &Tensor<c64, DeviceBLAS, Ix2>, hermitian: bool) -> Tensor<c64, DeviceBLAS, Ix2> {
    let n = a.shape()[0];
    let mut t = a.view().to_owned();
    for i in 0..n {
        for j in 0..n {
            t[[i, j]] = if hermitian { a[[j, i]].conj() } else { a[[j, i]] };
        }
    }
    t
}

#[test]
fn test_ldl() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a');
    let base = rt::asarray((&a_vec[..1024 * 1024], [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let h = hermitize(base.view());

    // Hermitian lower (the default)
    let result: LdlOutC = rt::linalg::ldl(h.view());
    assert_eq!(result.lu.shape(), &[1024, 1024], "lu shape");
    assert_eq!(result.perm.shape(), &[1024], "perm shape");
    assert!((fingerprint(&result.lu.abs()) - -40.081477868486445).abs() < 1e-8, "fp(lu)");
    assert!((fingerprint(&result.d.abs()) - -755.7217236720919).abs() < 1e-8, "fp(d)");
    assert!((fingerprint(&result.perm.map(|&v| v as f64)) - -5295.552240978868).abs() < 1e-8, "fp(perm)");
    check_ldl(&h, &result.lu, &result.d, &result.perm, true, true, "hermitian lower");

    // Hermitian upper
    let result: LdlOutC = rt::linalg::ldl((h.view(), Upper));
    assert!((fingerprint(&result.lu.abs()) - 51.80872092720819).abs() < 1e-8, "fp(lu), upper");
    assert!((fingerprint(&result.d.abs()) - -180.08681799670805).abs() < 1e-8, "fp(d), upper");
    assert!((fingerprint(&result.perm.map(|&v| v as f64)) - 2583.421176397793).abs() < 1e-8, "fp(perm), upper");
    check_ldl(&h, &result.lu, &result.d, &result.perm, false, true, "hermitian upper");

    // complex symmetric (hermitian = false), lower
    let s = symmetrize(base.view());
    let result: LdlOutC = rt::linalg::ldl((s.view(), false));
    assert!((fingerprint(&result.lu.abs()) - 33.91460680449423).abs() < 1e-8, "fp(lu), symmetric");
    assert!((fingerprint(&result.d.abs()) - 184.86531997112618).abs() < 1e-8, "fp(d), symmetric");
    assert!((fingerprint(&result.perm.map(|&v| v as f64)) - 1695.0944808739519).abs() < 1e-8, "fp(perm), symmetric");
    check_ldl(&s, &result.lu, &result.d, &result.perm, true, false, "symmetric lower");
}

#[test]
fn test_ldl_relations() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a');
    for n in [1usize, 2, 3, 5, 6] {
        let base = rt::asarray((&a_vec[..n * n], [n, n].c(), &device)).into_dim::<Ix2>();
        let h = hermitize(base.view());
        let s = symmetrize(base.view());
        for lower in [true, false] {
            let uplo = if lower { Lower } else { Upper };
            let at = format!("{n}, {}", if lower { "lower" } else { "upper" });
            let result: LdlOutC = rt::linalg::ldl((h.view(), uplo));
            check_ldl(&h, &result.lu, &result.d, &result.perm, lower, true, &format!("hermitian {at}"));
            let result: LdlOutC = rt::linalg::ldl((s.view(), false, uplo));
            check_ldl(&s, &result.lu, &result.d, &result.perm, lower, false, &format!("symmetric {at}"));
        }
    }
}

/// Constructed 2x2-block cases and the Hermitian diagonal handling.
#[test]
fn test_ldl_2x2_blocks() {
    let device = DeviceBLAS::default();
    // zero-diagonal Hermitian 2x2: the block is [[0, 1 + 1j], [1 - 1j, 0]]
    let a =
        rt::asarray((vec![c64!(0.0, 0.0), c64!(1.0, 1.0), c64!(1.0, -1.0), c64!(0.0, 0.0)], [2usize, 2].c(), &device))
            .into_dim::<Ix2>();
    for lower in [true, false] {
        let uplo = if lower { Lower } else { Upper };
        let at = format!("hermitian 2x2, {}", if lower { "lower" } else { "upper" });
        let result: LdlOutC = rt::linalg::ldl((a.view(), uplo));
        assert_eq!(result.perm.raw().to_vec(), vec![0, 1], "perm, {at}");
        assert_eq!(result.d[[0, 0]], c64!(0.0, 0.0), "d[0, 0], {at}");
        assert_eq!(result.d[[1, 1]], c64!(0.0, 0.0), "d[1, 1], {at}");
        assert_eq!(result.d[[0, 1]], c64!(1.0, 1.0), "d[0, 1], {at}");
        assert_eq!(result.d[[1, 0]], c64!(1.0, -1.0), "d[1, 0], {at}");
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { c64!(1.0, 0.0) } else { c64!(0.0, 0.0) };
                assert_eq!(result.lu[[i, j]], expected, "lu = I, {at} ({i}, {j})");
            }
        }
    }
    // the same input under the symmetric form: no conjugation
    let result: LdlOutC = rt::linalg::ldl((a.view(), false, Lower));
    assert_eq!(result.d[[0, 1]], c64!(1.0, -1.0), "d[0, 1], symmetric 2x2 lower");
    assert_eq!(result.d[[1, 0]], c64!(1.0, -1.0), "d[1, 0], symmetric 2x2 lower");
    let result: LdlOutC = rt::linalg::ldl((a.view(), false, Upper));
    assert_eq!(result.d[[0, 1]], c64!(1.0, 1.0), "d[0, 1], symmetric 2x2 upper");
    assert_eq!(result.d[[1, 0]], c64!(1.0, 1.0), "d[1, 0], symmetric 2x2 upper");

    // a complex diagonal is ignored for the Hermitian form: the result equals
    // the factorization of the real-diagonal matrix
    let complex =
        rt::asarray((vec![c64!(1.0, 5.0), c64!(2.0, 1.0), c64!(2.0, -1.0), c64!(3.0, 7.0)], [2usize, 2].c(), &device))
            .into_dim::<Ix2>();
    let real_diag =
        rt::asarray((vec![c64!(1.0, 0.0), c64!(2.0, 1.0), c64!(2.0, -1.0), c64!(3.0, 0.0)], [2usize, 2].c(), &device))
            .into_dim::<Ix2>();
    let result: LdlOutC = rt::linalg::ldl(complex.view());
    let reference: LdlOutC = rt::linalg::ldl(real_diag.view());
    assert!(rt::allclose(result.lu.view(), reference.lu.view(), Some(1e-14)), "lu, complex diagonal ignored");
    assert!(rt::allclose(result.d.view(), reference.d.view(), Some(1e-14)), "d, complex diagonal ignored");
    assert_eq!(result.perm.raw().to_vec(), reference.perm.raw().to_vec(), "perm, complex diagonal ignored");
    // the diagonal of `d` is stored real
    for k in 0..2 {
        assert_eq!(result.d[[k, k]].im, 0.0, "real diagonal of d, k = {k}");
    }
}

#[test]
fn test_ldl_singular() {
    let device = DeviceBLAS::default();
    let a = rt::zeros_f(([3usize, 3], &device)).unwrap().into_dim::<Ix2>();
    let result: LdlOutC = rt::linalg::ldl(a.view());
    assert_eq!(result.perm.raw().to_vec(), vec![0, 1, 2], "perm, zeros");
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(
                result.lu[[i, j]],
                if i == j { c64!(1.0, 0.0) } else { c64!(0.0, 0.0) },
                "lu = I, zeros ({i}, {j})"
            );
            assert_eq!(result.d[[i, j]], c64!(0.0, 0.0), "d = 0, zeros ({i}, {j})");
        }
    }
}

/// Hermitianize every matrix slot of a stack in place (slot layout follows
/// the device order `order`: matrix axes last under `RowMajor`).
fn hermitize_stack(mut a: Tensor<c64, DeviceBLAS, IxD>, order: FlagOrder) -> Tensor<c64, DeviceBLAS, IxD> {
    for i in 0..2 {
        for j in 0..3 {
            let (slice, mut slot) = match order {
                RowMajor => (a.i((i, j)).into_owned().into_dim::<Ix2>(), a.i_mut((i, j))),
                ColMajor => (a.i((.., .., i, j)).into_owned().into_dim::<Ix2>(), a.i_mut((.., .., i, j))),
            };
            let h = hermitize(slice.view());
            slot.assign_f(h.view()).unwrap();
        }
    }
    a
}

#[test]
fn test_ldl_batch() {
    let a_vec = get_vec::<c64>('a');
    let n = 5;
    for order in [RowMajor, ColMajor] {
        for stored in [RowMajor, ColMajor] {
            let mut device = DeviceBLAS::default();
            device.set_default_order(order);
            let tag = format!("{order:?} device, {stored:?} stack");
            // Hermitianize every slot of the stack
            let make = || {
                hermitize_stack(
                    stack_tensor_c64(&a_vec[..2 * 3 * n * n], &[2, 3], &[n, n], order, stored, &device),
                    order,
                )
            };
            let a = make();
            let result = rt::linalg::ldl(a.view());
            assert_eq!(result.lu.shape(), a.shape(), "lu shape, {tag}");
            for i in 0..2 {
                for j in 0..3 {
                    let at = format!("slot ({i}, {j}), {tag}");
                    let (a_ij, lu_ij, perm_ij) = match order {
                        RowMajor => (
                            a.i((i, j, .., ..)).into_owned().into_dim::<Ix2>(),
                            result.lu.i((i, j, .., ..)),
                            result.perm.i((i, j, ..)),
                        ),
                        ColMajor => (
                            a.i((.., .., i, j)).into_owned().into_dim::<Ix2>(),
                            result.lu.i((.., .., i, j)),
                            result.perm.i((.., i, j)),
                        ),
                    };
                    let single: LdlOutC = rt::linalg::ldl(a_ij.view());
                    check_ldl(&a_ij, &single.lu, &single.d, &single.perm, true, true, &at);
                    assert!(rt::allclose(lu_ij, single.lu.view(), Some(1e-14)), "lu, {at}");
                    assert_eq!(perm_ij.to_vec(), single.perm.raw().to_vec(), "perm, {at}");
                }
            }
            // owned and reference inputs give the same result as the view call
            for (at, candidate) in [
                ("owned", rt::linalg::ldl(make())),
                ("reference", rt::linalg::ldl(&make())),
                ("view_mut", rt::linalg::ldl(make().view_mut())),
            ] {
                assert!(rt::allclose(candidate.lu.view(), result.lu.view(), Some(1e-14)), "lu, {at}, {tag}");
                assert_eq!(candidate.perm.raw().to_vec(), result.perm.raw().to_vec(), "perm, {at}, {tag}");
            }
        }
    }
}

/// [`stack_tensor`](super::stack_tensor) for complex data.
fn stack_tensor_c64(
    v: &[c64],
    batch: &[usize],
    inner: &[usize],
    order: FlagOrder,
    stored: FlagOrder,
    device: &DeviceBLAS,
) -> Tensor<c64, DeviceBLAS, IxD> {
    let stacked = match order {
        RowMajor => [batch, inner].concat(),
        ColMajor => [inner, batch].concat(),
    };
    let stack = rt::asarray((v.to_vec(), stacked.c(), device)).into_owned().into_dyn();
    match stored {
        RowMajor => stack,
        ColMajor => stack.to_contig_f(ColMajor).unwrap().into_owned().into_dyn(),
    }
}

#[test]
fn test_ldl_inplace() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a');
    let n = 5;
    let base = rt::asarray((&a_vec[..n * n], [n, n].c(), &device)).into_dim::<Ix2>();
    let h = hermitize(base.view());
    let reference: LdlOutC = rt::linalg::ldl(h.view());

    let result: LdlOutC = rt::linalg::ldl(h.clone());
    assert!(rt::allclose(result.lu.view(), reference.lu.view(), Some(1e-14)), "lu, owned");
    let mut m = h.clone();
    let ptr = m.raw().as_ptr();
    let result: LdlOutC = rt::linalg::ldl(m.view_mut());
    assert!(rt::allclose(result.lu.view(), reference.lu.view(), Some(1e-14)), "lu, view_mut");
    assert_eq!(m.raw().as_ptr(), ptr, "the input buffer is retained");
    assert!(m.raw().iter().zip(h.raw().iter()).any(|(x, y)| x != y), "the input content is destroyed");
    // off-order mutable input untouched (an f-contig copy of the same values)
    let buf_f = h.view().to_contig_f(ColMajor).unwrap().into_owned().into_dim::<Ix2>();
    let mut m = buf_f.clone();
    let result: LdlOutC = rt::linalg::ldl(m.view_mut());
    assert!(rt::allclose(result.lu.view(), reference.lu.view(), Some(1e-14)), "lu, off-order view_mut");
    assert_eq!(m.raw().to_vec(), buf_f.raw().to_vec(), "the off-order input is untouched");
}

#[test]
fn test_ldl_overloads() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a');
    let n = 4;
    let base = rt::asarray((&a_vec[..n * n], [n, n].c(), &device)).into_dim::<Ix2>();
    let h = hermitize(base.view());
    let make_owned = || h.clone();

    for lower in [true, false] {
        let uplo = if lower { Lower } else { Upper };
        let reference: LdlOutC = rt::linalg::ldl((h.view(), uplo));
        let reference_default: LdlOutC = rt::linalg::ldl(h.view());
        let forms: Vec<(LdlOutC, bool)> = vec![
            (rt::linalg::ldl((&make_owned(), true, uplo)), true),
            (rt::linalg::ldl((make_owned().view(), true, uplo)), true),
            (rt::linalg::ldl((make_owned(), true, uplo)), true),
            (rt::linalg::ldl((make_owned().view_mut(), true, uplo)), true),
            (rt::linalg::ldl((&make_owned(), true)), false),
            (rt::linalg::ldl((make_owned().view(), true)), false),
            (rt::linalg::ldl((make_owned(), true)), false),
            (rt::linalg::ldl((make_owned().view_mut(), true)), false),
            (rt::linalg::ldl((&make_owned(), uplo)), true),
            (rt::linalg::ldl((make_owned().view(), uplo)), true),
            (rt::linalg::ldl((make_owned(), uplo)), true),
            (rt::linalg::ldl((make_owned().view_mut(), uplo)), true),
            (rt::linalg::ldl(&make_owned()), false),
            (rt::linalg::ldl(make_owned().view()), false),
            (rt::linalg::ldl(make_owned()), false),
            (rt::linalg::ldl(make_owned().view_mut()), false),
        ];
        for (k, (candidate, explicit_uplo)) in forms.into_iter().enumerate() {
            let at = format!("lower {lower}, form {k}");
            let reference = if explicit_uplo { &reference } else { &reference_default };
            assert!(rt::allclose(candidate.lu.view(), reference.lu.view(), Some(1e-14)), "lu, {at}");
            assert!(rt::allclose(candidate.d.view(), reference.d.view(), Some(1e-14)), "d, {at}");
            assert_eq!(candidate.perm.raw().to_vec(), reference.perm.raw().to_vec(), "perm, {at}");
            let (lu, d, perm): LdlTupleC = candidate.into();
            assert!(rt::allclose(lu.view(), reference.lu.view(), Some(1e-14)), "tuple lu, {at}");
            assert!(rt::allclose(d.view(), reference.d.view(), Some(1e-14)), "tuple d, {at}");
            assert_eq!(perm.raw().to_vec(), reference.perm.raw().to_vec(), "tuple perm, {at}");
        }
    }
}

#[test]
fn test_ldl_zero_size() {
    let device = DeviceBLAS::default();
    let a: Tensor<c64, DeviceBLAS, Ix2> = rt::zeros_f(([0usize, 0], &device)).unwrap().into_dim::<Ix2>();
    let result = rt::linalg::ldl(a.view());
    assert_eq!(result.lu.shape(), &[0, 0], "lu, (0, 0)");
    assert_eq!(result.perm.shape(), &[0], "perm, (0, 0)");
    let a: Tensor<c64, DeviceBLAS, Ix3> = rt::zeros_f(([0usize, 3, 3], &device)).unwrap().into_dim::<Ix3>();
    let result = rt::linalg::ldl(a.view());
    assert_eq!(result.lu.shape(), &[0, 3, 3], "lu, empty batch");
}

#[test]
fn test_ldl_layouts() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a');
    let n = 6;
    let base = rt::asarray((&a_vec[..n * n], [n, n].c(), &device)).into_dim::<Ix2>();
    let h = hermitize(base.view());
    let reference: LdlOutC = rt::linalg::ldl(h.view());

    // f-contig input of the same matrix
    let buf_f = h.view().to_contig_f(ColMajor).unwrap().into_owned().into_dim::<Ix2>();
    let result: LdlOutC = rt::linalg::ldl(buf_f.view());
    assert_eq!(result.lu.raw().to_vec(), reference.lu.raw().to_vec(), "lu, f-contig");
    assert_eq!(result.perm.raw().to_vec(), reference.perm.raw().to_vec(), "perm, f-contig");

    // padded buffer (rows), the visible block is the matrix
    let mut padded = rt::zeros_f(([n + 4, n + 4], &device)).unwrap().into_dim::<Ix2>();
    padded.i_mut((0..n, 0..n)).assign_f(h.view()).unwrap();
    let result: LdlOutC = rt::linalg::ldl(padded.i((0..n, 0..n)).into_dim::<Ix2>());
    assert_eq!(result.lu.raw().to_vec(), reference.lu.raw().to_vec(), "lu, padded");

    // strided view: every second row and column of a double-size buffer
    let mut wide = rt::zeros_f(([2 * n, 2 * n], &device)).unwrap().into_dim::<Ix2>();
    for i in 0..n {
        for j in 0..n {
            wide[[2 * i, 2 * j]] = h[[i, j]];
        }
    }
    let result: LdlOutC = rt::linalg::ldl(wide.i((slice!(0, 2 * n, 2), slice!(0, 2 * n, 2))).into_dim::<Ix2>());
    assert_eq!(result.lu.raw().to_vec(), reference.lu.raw().to_vec(), "lu, strided");
    assert_eq!(result.perm.raw().to_vec(), reference.perm.raw().to_vec(), "perm, strided");
}

/// Only the selected triangle is read: garbage (NaN) in the other triangle
/// changes nothing.
#[test]
fn test_ldl_unreferenced_triangle() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a');
    let n = 5;
    let base = rt::asarray((&a_vec[..n * n], [n, n].c(), &device)).into_dim::<Ix2>();
    let h = hermitize(base.view());
    let reference: LdlOutC = rt::linalg::ldl((h.view(), Lower));
    // NaN in the strictly upper triangle (the diagonal is read in both)
    let mut garbage = h.clone();
    for i in 0..n {
        for j in (i + 1)..n {
            garbage[[i, j]] = c64!(f64::NAN, f64::NAN);
        }
    }
    let result: LdlOutC = rt::linalg::ldl((garbage.view(), Lower));
    assert_eq!(result.lu.raw().to_vec(), reference.lu.raw().to_vec(), "lu, NaN upper triangle");
    assert_eq!(result.d.raw().to_vec(), reference.d.raw().to_vec(), "d, NaN upper triangle");
    assert_eq!(result.perm.raw().to_vec(), reference.perm.raw().to_vec(), "perm, NaN upper triangle");
}

#[test]
fn test_ldl_ndim_error() {
    let device = DeviceBLAS::default();
    let v = rt::asarray((get_vec::<c64>('a')[..4].to_vec(), [4usize], &device)).into_dim::<Ix1>();
    let msg = format!("{}", rt::linalg::ldl_f((v.view(),)).err().expect("1-D a must fail"));
    assert!(msg.contains("InvalidLayout"), "1-D error message: {msg}");
    let a = rt::asarray((get_vec::<c64>('a')[..12].to_vec(), [3usize, 4].c(), &device)).into_dim::<Ix2>();
    let msg = format!("{}", rt::linalg::ldl_f((a.view(),)).err().expect("non-square a must fail"));
    assert!(msg.contains("InvalidLayout") && msg.contains("square"), "non-square error message: {msg}");
}
