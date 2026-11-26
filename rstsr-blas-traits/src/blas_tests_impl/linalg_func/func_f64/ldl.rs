use super::*;
use rstsr::prelude::linalg::LdlResult;

// the result types follow the dims of a 2-D input
type LdlOut = LdlResult<Tensor<f64, DeviceBLAS, Ix2>, Tensor<f64, DeviceBLAS, Ix2>, Tensor<usize, DeviceBLAS, Ix1>>;
type LdlTuple = (Tensor<f64, DeviceBLAS, Ix2>, Tensor<f64, DeviceBLAS, Ix2>, Tensor<usize, DeviceBLAS, Ix1>);

/// Symmetric matrix (`n`, `n`) from `a`: `a + aᵀ`.
fn symmetrize(a: TensorView<'_, f64, DeviceBLAS, Ix2>) -> Tensor<f64, DeviceBLAS, Ix2> {
    let n = a.shape()[0];
    let mut sym = a.view().to_owned();
    for i in 0..n {
        for j in 0..n {
            sym[[i, j]] = a[[i, j]] + a[[j, i]];
        }
    }
    sym
}

/// Self-validation of one result against the input slice: `lu[perm, :]` unit
/// triangular, `d` block diagonal and symmetric, `A = lu @ d @ luᵀ`.
fn check_ldl(
    a: &Tensor<f64, DeviceBLAS, Ix2>,
    lu: &Tensor<f64, DeviceBLAS, Ix2>,
    d: &Tensor<f64, DeviceBLAS, Ix2>,
    perm: &Tensor<usize, DeviceBLAS, Ix1>,
    lower: bool,
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
                assert!((v - 1.0).abs() < 1e-13, "unit diagonal of lu[perm] ({src}, {col}), {at}");
            } else if (lower && col > row) || (!lower && col < row) {
                assert_eq!(v, 0.0, "zero outside the triangle of lu[perm] ({src}, {col}), {at}");
            }
        }
    }
    for i in 0..n {
        for j in 0..n {
            if (i as isize - j as isize).abs() >= 2 {
                assert_eq!(d[[i, j]], 0.0, "d band ({i}, {j}), {at}");
            }
        }
    }
    let mut blocks = 0;
    let mut k = 0;
    while k + 1 < n {
        if d[[k + 1, k]] != 0.0 {
            blocks += 1;
            k += 2;
        } else {
            assert_eq!(d[[k, k + 1]], 0.0, "stray off-diagonal at {k}, {at}");
            k += 1;
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            assert_eq!(d[[i, j]], d[[j, i]], "d symmetric ({i}, {j}), {at}");
        }
    }
    let prod = rt::matmul(rt::matmul(lu.view(), d.view()), lu.view().into_reverse_axes());
    assert!(rt::allclose(prod.view(), a.view(), Some(1e-12)), "A = lu @ d @ luᵀ ({blocks} 2x2 blocks), {at}");
}

#[test]
fn test_ldl() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a');
    let base = rt::asarray((&a_vec[..1024 * 1024], [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let sym = symmetrize(base.view());

    // lower (the default)
    let result: LdlOut = rt::linalg::ldl(sym.view());
    assert_eq!(result.lu.shape(), &[1024, 1024], "lu shape");
    assert_eq!(result.d.shape(), &[1024, 1024], "d shape");
    assert_eq!(result.perm.shape(), &[1024], "perm shape");
    assert!((fingerprint(&result.lu) - 278.03503138268053).abs() < 1e-8, "fp(lu)");
    assert!((fingerprint(&result.d) - -1206.9319026825895).abs() < 1e-8, "fp(d)");
    assert!((fingerprint(&result.perm.map(|&v| v as f64)) - -868.5569955216156).abs() < 1e-8, "fp(perm)");
    check_ldl(&sym, &result.lu, &result.d, &result.perm, true, "lower");

    // upper
    let result: LdlOut = rt::linalg::ldl((sym.view(), Upper));
    assert!((fingerprint(&result.lu) - -40.67475056339296).abs() < 1e-8, "fp(lu), upper");
    assert!((fingerprint(&result.d) - 1613.3072712879875).abs() < 1e-8, "fp(d), upper");
    assert!((fingerprint(&result.perm.map(|&v| v as f64)) - -1842.9223859667964).abs() < 1e-8, "fp(perm), upper");
    check_ldl(&sym, &result.lu, &result.d, &result.perm, false, "upper");
}

#[test]
fn test_ldl_relations() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a');
    for n in [1usize, 2, 3, 5, 6] {
        let base = rt::asarray((&a_vec[..n * n], [n, n].c(), &device)).into_dim::<Ix2>();
        let sym = symmetrize(base.view());
        for lower in [true, false] {
            let uplo = if lower { Lower } else { Upper };
            let at = format!("{n}, {}", if lower { "lower" } else { "upper" });
            let result: LdlOut = rt::linalg::ldl((sym.view(), uplo));
            check_ldl(&sym, &result.lu, &result.d, &result.perm, lower, &at);
            // `hermitian = false` equals the default on real input
            let symm: LdlOut = rt::linalg::ldl((sym.view(), false, uplo));
            assert_eq!(symm.perm.raw().to_vec(), result.perm.raw().to_vec(), "perm, hermitian = false, {at}");
            assert!(rt::allclose(symm.lu.view(), result.lu.view(), Some(1e-14)), "lu, hermitian = false, {at}");
            assert!(rt::allclose(symm.d.view(), result.d.view(), Some(1e-14)), "d, hermitian = false, {at}");
        }
    }
}

/// Constructed 2x2-block cases and the SciPy docstring examples.
#[test]
fn test_ldl_2x2_blocks() {
    let device = DeviceBLAS::default();
    // zero diagonal: the off-diagonal pair must form a 2x2 block
    let a =
        rt::asarray((vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0], [3usize, 3].c(), &device)).into_dim::<Ix2>();
    let result: LdlOut = rt::linalg::ldl((a.view(), Lower));
    assert_eq!(result.perm.raw().to_vec(), vec![0, 1, 2], "perm, zero diagonal");
    for i in 0..3 {
        assert!((result.lu[[i, i]] - 1.0).abs() < 1e-14, "unit diagonal, zero diagonal lower");
        for j in 0..3 {
            if i != j {
                assert_eq!(result.lu[[i, j]], 0.0, "lu = I, zero diagonal lower ({i}, {j})");
            }
        }
    }
    assert_eq!(result.d[[1, 0]], 1.0, "d[1, 0], zero diagonal lower");
    assert_eq!(result.d[[0, 1]], 1.0, "d[0, 1], zero diagonal lower");
    let result: LdlOut = rt::linalg::ldl((a.view(), Upper));
    assert_eq!(result.d[[0, 1]], 1.0, "d[0, 1], zero diagonal upper");

    // SciPy docstring example (upper): A = [[2, -1, 3], [0, 2, 0], [0, 0, 1]]
    let a =
        rt::asarray((vec![2.0, -1.0, 3.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0], [3usize, 3].c(), &device)).into_dim::<Ix2>();
    let result: LdlOut = rt::linalg::ldl((a.view(), Upper));
    assert_eq!(result.perm.raw().to_vec(), vec![2, 1, 0], "perm, doc example");
    let lu_expected = [[0.0, 0.0, 1.0], [0.0, 1.0, -0.5], [1.0, 1.0, 1.5]];
    for i in 0..3 {
        for j in 0..3 {
            assert!((result.lu[[i, j]] - lu_expected[i][j]).abs() < 1e-14, "lu doc example ({i}, {j})");
        }
    }
    let d_expected = [[-5.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 2.0]];
    for i in 0..3 {
        for j in 0..3 {
            assert!((result.d[[i, j]] - d_expected[i][j]).abs() < 1e-14, "d doc example ({i}, {j})");
        }
    }

    // the symmetrized doc matrix: both forms, exact expectations from SciPy
    let s = symmetrize(a.view());
    let result: LdlOut = rt::linalg::ldl((s.view(), Lower));
    let lu_lower = [[1.0, 0.0, 0.0], [-0.25, 1.0, 0.0], [0.75, 0.2, 1.0]];
    let d_lower = [[4.0, 0.0, 0.0], [0.0, 3.75, 0.0], [0.0, 0.0, -0.4]];
    for i in 0..3 {
        for j in 0..3 {
            assert!((result.lu[[i, j]] - lu_lower[i][j]).abs() < 1e-14, "lu sym lower ({i}, {j})");
            assert!((result.d[[i, j]] - d_lower[i][j]).abs() < 1e-14, "d sym lower ({i}, {j})");
        }
    }
    let result: LdlOut = rt::linalg::ldl((s.view(), Upper));
    let lu_upper = [[1.0, -0.25, 1.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let d_upper = [[-0.75, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 2.0]];
    for i in 0..3 {
        for j in 0..3 {
            assert!((result.lu[[i, j]] - lu_upper[i][j]).abs() < 1e-14, "lu sym upper ({i}, {j})");
            assert!((result.d[[i, j]] - d_upper[i][j]).abs() < 1e-14, "d sym upper ({i}, {j})");
        }
    }
}

#[test]
fn test_ldl_singular() {
    let device = DeviceBLAS::default();
    // all-zero input: D(i,i) exactly zero on the first pivot, lu = I
    let a = rt::zeros_f(([3usize, 3], &device)).unwrap().into_dim::<Ix2>();
    let result: LdlOut = rt::linalg::ldl(a.view());
    assert_eq!(result.perm.raw().to_vec(), vec![0, 1, 2], "perm, zeros");
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(result.lu[[i, j]], if i == j { 1.0 } else { 0.0 }, "lu = I, zeros ({i}, {j})");
            assert_eq!(result.d[[i, j]], 0.0, "d = 0, zeros ({i}, {j})");
        }
    }
    // rank-2 symmetric: complete factors with a singular d
    let a =
        rt::asarray((vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0], [3usize, 3].c(), &device)).into_dim::<Ix2>();
    let sym = symmetrize(a.view());
    let result: LdlOut = rt::linalg::ldl(sym.view());
    assert!(result.lu.raw().iter().all(|v| v.is_finite()), "lu finite");
    assert!(result.d.raw().iter().all(|v| v.is_finite()), "d finite");
    assert!((0..3).any(|k| result.d[[k, k]] == 0.0), "d has an exactly-zero pivot");
    check_ldl(&sym, &result.lu, &result.d, &result.perm, true, "rank-2");
}

/// Symmetrize every matrix slot of a stack in place (slot layout follows the
/// device order `order`: matrix axes last under `RowMajor`).
fn symmetrize_stack(mut a: Tensor<f64, DeviceBLAS, IxD>, order: FlagOrder) -> Tensor<f64, DeviceBLAS, IxD> {
    for i in 0..2 {
        for j in 0..3 {
            let (slice, mut slot) = match order {
                RowMajor => (a.i((i, j)).into_owned().into_dim::<Ix2>(), a.i_mut((i, j))),
                ColMajor => (a.i((.., .., i, j)).into_owned().into_dim::<Ix2>(), a.i_mut((.., .., i, j))),
            };
            let sym = symmetrize(slice.view());
            slot.assign_f(sym.view()).unwrap();
        }
    }
    a
}

#[test]
fn test_ldl_batch() {
    let a_vec = get_vec::<f64>('a');
    let n = 5;
    for order in [RowMajor, ColMajor] {
        for stored in [RowMajor, ColMajor] {
            let mut device = DeviceBLAS::default();
            device.set_default_order(order);
            let tag = format!("{order:?} device, {stored:?} stack");
            // symmetrize every slot of the stack, so the owned / reference /
            // view calls and the rebuilt stacks all factor the same matrix
            let make = || {
                symmetrize_stack(stack_tensor(&a_vec[..2 * 3 * n * n], &[2, 3], &[n, n], order, stored, &device), order)
            };
            let a = make();
            let result = rt::linalg::ldl(a.view());
            // owned and reference inputs give the same result as the view call
            assert_eq!(result.lu.shape(), a.shape(), "lu shape, {tag}");
            for i in 0..2 {
                for j in 0..3 {
                    let at = format!("slot ({i}, {j}), {tag}");
                    let (a_ij, lu_ij, d_ij, perm_ij) = match order {
                        RowMajor => (
                            a.i((i, j, .., ..)).into_owned().into_dim::<Ix2>(),
                            result.lu.i((i, j, .., ..)),
                            result.d.i((i, j, .., ..)),
                            result.perm.i((i, j, ..)),
                        ),
                        ColMajor => (
                            a.i((.., .., i, j)).into_owned().into_dim::<Ix2>(),
                            result.lu.i((.., .., i, j)),
                            result.d.i((.., .., i, j)),
                            result.perm.i((.., i, j)),
                        ),
                    };
                    let single: LdlOut = rt::linalg::ldl(a_ij.view());
                    check_ldl(&a_ij, &single.lu, &single.d, &single.perm, true, &at);
                    assert!(rt::allclose(lu_ij, single.lu.view(), Some(1e-14)), "lu, {at}");
                    assert!(rt::allclose(d_ij, single.d.view(), Some(1e-14)), "d, {at}");
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
                assert!(rt::allclose(candidate.d.view(), result.d.view(), Some(1e-14)), "d, {at}, {tag}");
                assert_eq!(candidate.perm.raw().to_vec(), result.perm.raw().to_vec(), "perm, {at}, {tag}");
            }
        }
    }
}

#[test]
fn test_ldl_layouts() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a');
    let n = 6;
    let base = rt::asarray((&a_vec[..n * n], [n, n].c(), &device)).into_dim::<Ix2>();
    let sym = symmetrize(base.view());
    let reference: LdlOut = rt::linalg::ldl(sym.view());

    // f-contig input of the same matrix
    let buf_f = sym.view().to_contig_f(ColMajor).unwrap().into_owned().into_dim::<Ix2>();
    let result: LdlOut = rt::linalg::ldl(buf_f.view());
    assert_eq!(result.lu.raw().to_vec(), reference.lu.raw().to_vec(), "lu, f-contig");

    // padded buffer (rows), the visible block is the matrix
    let mut padded = rt::zeros_f(([n + 4, n + 4], &device)).unwrap().into_dim::<Ix2>();
    padded.i_mut((0..n, 0..n)).assign_f(sym.view()).unwrap();
    let result: LdlOut = rt::linalg::ldl(padded.i((0..n, 0..n)).into_dim::<Ix2>());
    assert_eq!(result.lu.raw().to_vec(), reference.lu.raw().to_vec(), "lu, padded");

    // strided view: every second row and column of a double-size buffer
    let mut wide = rt::zeros_f(([2 * n, 2 * n], &device)).unwrap().into_dim::<Ix2>();
    for i in 0..n {
        for j in 0..n {
            wide[[2 * i, 2 * j]] = sym[[i, j]];
        }
    }
    let result: LdlOut = rt::linalg::ldl(wide.i((slice!(0, 2 * n, 2), slice!(0, 2 * n, 2))).into_dim::<Ix2>());
    assert_eq!(result.lu.raw().to_vec(), reference.lu.raw().to_vec(), "lu, strided");
    assert_eq!(result.perm.raw().to_vec(), reference.perm.raw().to_vec(), "perm, strided");
}

#[test]
fn test_ldl_inplace() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a');
    let n = 5;
    let base = rt::asarray((&a_vec[..n * n], [n, n].c(), &device)).into_dim::<Ix2>();
    let sym = symmetrize(base.view());
    let reference: LdlOut = rt::linalg::ldl(sym.view());

    // an owned input is consumed and destroyed (the result is fresh anyway)
    let result: LdlOut = rt::linalg::ldl(sym.clone());
    assert!(rt::allclose(result.lu.view(), reference.lu.view(), Some(1e-14)), "lu, owned");
    assert_eq!(result.perm.raw().to_vec(), reference.perm.raw().to_vec(), "perm, owned");

    // an in-order mutable input is factored in place: the buffer is retained
    // and its content destroyed by the factorization
    let mut m = sym.clone();
    let ptr = m.raw().as_ptr();
    let result: LdlOut = rt::linalg::ldl(m.view_mut());
    assert!(rt::allclose(result.lu.view(), reference.lu.view(), Some(1e-14)), "lu, view_mut");
    assert_eq!(result.perm.raw().to_vec(), reference.perm.raw().to_vec(), "perm, view_mut");
    assert_eq!(m.raw().as_ptr(), ptr, "the input buffer is retained");
    assert!(m.raw().iter().zip(sym.raw().iter()).any(|(x, y)| x != y), "the input content is destroyed");

    // an off-order mutable input is only read (f-contig under a row-major device)
    let buf_f = sym.view().to_contig_f(ColMajor).unwrap().into_owned().into_dim::<Ix2>();
    let mut m = buf_f.clone();
    let result: LdlOut = rt::linalg::ldl(m.view_mut());
    assert!(rt::allclose(result.lu.view(), reference.lu.view(), Some(1e-14)), "lu, off-order view_mut");
    assert_eq!(m.raw().to_vec(), buf_f.raw().to_vec(), "the off-order input is untouched");
}

/// Only the selected triangle is read: garbage (NaN) in the other triangle
/// changes nothing.
#[test]
fn test_ldl_unreferenced_triangle() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a');
    let n = 5;
    let base = rt::asarray((&a_vec[..n * n], [n, n].c(), &device)).into_dim::<Ix2>();
    let sym = symmetrize(base.view());
    let reference: LdlOut = rt::linalg::ldl((sym.view(), Lower));
    // NaN in the strictly upper triangle (the diagonal is read in both)
    let mut garbage = sym.clone();
    for i in 0..n {
        for j in (i + 1)..n {
            garbage[[i, j]] = f64::NAN;
        }
    }
    let result: LdlOut = rt::linalg::ldl((garbage.view(), Lower));
    assert_eq!(result.lu.raw().to_vec(), reference.lu.raw().to_vec(), "lu, NaN upper triangle");
    assert_eq!(result.d.raw().to_vec(), reference.d.raw().to_vec(), "d, NaN upper triangle");
    assert_eq!(result.perm.raw().to_vec(), reference.perm.raw().to_vec(), "perm, NaN upper triangle");
}

#[test]
fn test_ldl_overloads() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a');
    let n = 4;
    let base = rt::asarray((&a_vec[..n * n], [n, n].c(), &device)).into_dim::<Ix2>();
    let sym = symmetrize(base.view());
    let make_owned = || sym.clone();

    for lower in [true, false] {
        let uplo = if lower { Lower } else { Upper };
        for hermitian in [true, false] {
            let at = format!("lower {lower}, hermitian {hermitian}");
            // expectations: the explicit-uplo forms follow `uplo`, the
            // implicit-uplo forms are always `Lower`
            let reference: LdlOut = rt::linalg::ldl((sym.view(), hermitian, uplo));
            let reference_default: LdlOut = rt::linalg::ldl((sym.view(), hermitian));
            // the four forms, each with the four input kinds; the bool marks
            // whether the form carries the explicit `uplo`
            let forms: Vec<(LdlOut, bool)> = vec![
                (rt::linalg::ldl((&make_owned(), hermitian, uplo)), true),
                (rt::linalg::ldl((make_owned().view(), hermitian, uplo)), true),
                (rt::linalg::ldl((make_owned(), hermitian, uplo)), true),
                (rt::linalg::ldl((make_owned().view_mut(), hermitian, uplo)), true),
                (rt::linalg::ldl((&make_owned(), hermitian)), false),
                (rt::linalg::ldl((make_owned().view(), hermitian)), false),
                (rt::linalg::ldl((make_owned(), hermitian)), false),
                (rt::linalg::ldl((make_owned().view_mut(), hermitian)), false),
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
                let at = format!("{at}, form {k}");
                let reference = if explicit_uplo { &reference } else { &reference_default };
                assert!(rt::allclose(candidate.lu.view(), reference.lu.view(), Some(1e-14)), "lu, {at}");
                assert!(rt::allclose(candidate.d.view(), reference.d.view(), Some(1e-14)), "d, {at}");
                assert_eq!(candidate.perm.raw().to_vec(), reference.perm.raw().to_vec(), "perm, {at}");
                // the tuple conversion destructures all three
                let (lu, d, perm): LdlTuple = candidate.into();
                assert!(rt::allclose(lu.view(), reference.lu.view(), Some(1e-14)), "tuple lu, {at}");
                assert!(rt::allclose(d.view(), reference.d.view(), Some(1e-14)), "tuple d, {at}");
                assert_eq!(perm.raw().to_vec(), reference.perm.raw().to_vec(), "tuple perm, {at}");
            }
        }
    }
}

#[test]
fn test_ldl_zero_size() {
    let device = DeviceBLAS::default();
    // (0, 0)
    let a: Tensor<f64, DeviceBLAS, Ix2> = rt::zeros_f(([0usize, 0], &device)).unwrap().into_dim::<Ix2>();
    let result = rt::linalg::ldl(a.view());
    assert_eq!(result.lu.shape(), &[0, 0], "lu, (0, 0)");
    assert_eq!(result.d.shape(), &[0, 0], "d, (0, 0)");
    assert_eq!(result.perm.shape(), &[0], "perm, (0, 0)");
    // batch of empties
    let a: Tensor<f64, DeviceBLAS, Ix3> = rt::zeros_f(([2usize, 0, 0], &device)).unwrap().into_dim::<Ix3>();
    let result = rt::linalg::ldl(a.view());
    assert_eq!(result.lu.shape(), &[2, 0, 0], "lu, (2, 0, 0)");
    assert_eq!(result.perm.shape(), &[2, 0], "perm, (2, 0, 0)");
    // empty batch
    let a: Tensor<f64, DeviceBLAS, Ix3> = rt::zeros_f(([0usize, 3, 3], &device)).unwrap().into_dim::<Ix3>();
    let result = rt::linalg::ldl(a.view());
    assert_eq!(result.lu.shape(), &[0, 3, 3], "lu, empty batch");
    assert_eq!(result.perm.shape(), &[0, 3], "perm, empty batch");
}

#[test]
fn test_ldl_ndim_error() {
    let device = DeviceBLAS::default();
    let v = rt::asarray((get_vec::<f64>('a')[..4].to_vec(), [4usize], &device)).into_dim::<Ix1>();
    let msg = format!("{}", rt::linalg::ldl_f((v.view(),)).err().expect("1-D a must fail"));
    assert!(msg.contains("InvalidLayout"), "1-D error message: {msg}");
    let a = rt::asarray((get_vec::<f64>('a')[..12].to_vec(), [3usize, 4].c(), &device)).into_dim::<Ix2>();
    let msg = format!("{}", rt::linalg::ldl_f((a.view(),)).err().expect("non-square a must fail"));
    assert!(msg.contains("InvalidLayout") && msg.contains("square"), "non-square error message: {msg}");
}
