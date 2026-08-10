#!/usr/bin/env python3
"""Extract line numbers and source hashes for tracked NumPy test functions.

Backs the `core-numpy-sync` skill steps 1-4: it does NOT edit tests - it produces
a machine-readable table (numpy_path, class, method, line, sha256, + metadata)
that feeds `numpy_coverage.csv`. Run on a version bump to detect line drift and
content drift (changed sha256) on transferred tests.

Usage:
    python3 sync_numpy.py <numpy_checkout_root>            # print CSV of tracked surface
    python3 sync_numpy.py <numpy_checkout_root> --hashes    # print path::class::method line sha256 only

The tracked surface (manipulation) is embedded in SURFACE below. To extend
coverage to another rstsr category, add (relpath, class, method) tuples and
their METADATA.
"""

import argparse
import ast
import hashlib
import sys
from pathlib import Path

PINNED_VERSION = "v2.5.2"  # keep in sync with the CSV/diff file headers

# ---------------------------------------------------------------------------
# Tracked surface: (relpath, class, method). class == "" => module-level fn.
# This is the NumPy manipulation test surface that rstsr-core tracks.
# ---------------------------------------------------------------------------
SURFACE = [
    # --- test_multiarray.py ---
    ("_core/tests/test_multiarray.py", "TestMethods", "test_reshape"),
    ("_core/tests/test_multiarray.py", "TestMethods", "test_transpose"),
    ("_core/tests/test_multiarray.py", "TestMethods", "test_swapaxes"),
    ("_core/tests/test_multiarray.py", "TestMethods", "test_squeeze"),
    ("_core/tests/test_multiarray.py", "TestMethods", "test_ravel"),
    ("_core/tests/test_multiarray.py", "TestMethods", "test_flatten"),
    ("_core/tests/test_multiarray.py", "TestMethods", "test_ravel_subclass"),
    ("_core/tests/test_multiarray.py", "TestCompress", "test_flatten"),
    # TestPickling::test_transposed_contiguous_array = pickle/OBB-buffer round-trip (N/A)
    ("_core/tests/test_multiarray.py", "TestPickling", "test_transposed_contiguous_array"),
    # TestArrayConstruction::test_array_cont = np.ascontiguousarray/asfortranarray analog (to_contig)
    ("_core/tests/test_multiarray.py", "TestArrayConstruction", "test_array_cont"),
    # --- test_numeric.py ---
    ("_core/tests/test_numeric.py", "TestNonarrayArgs", "test_reshape"),
    ("_core/tests/test_numeric.py", "TestNonarrayArgs", "test_reshape_shape_arg"),
    ("_core/tests/test_numeric.py", "TestNonarrayArgs", "test_reshape_copy_arg"),
    ("_core/tests/test_numeric.py", "TestNonarrayArgs", "test_ravel"),
    ("_core/tests/test_numeric.py", "TestNonarrayArgs", "test_squeeze"),
    ("_core/tests/test_numeric.py", "TestNonarrayArgs", "test_swapaxes"),
    ("_core/tests/test_numeric.py", "TestNonarrayArgs", "test_transpose"),
    ("_core/tests/test_numeric.py", "TestResize", "test_reshape_from_zero"),
    ("_core/tests/test_numeric.py", "TestMoveaxis", "test_move_to_end"),
    ("_core/tests/test_numeric.py", "TestMoveaxis", "test_move_new_position"),
    ("_core/tests/test_numeric.py", "TestMoveaxis", "test_preserve_order"),
    ("_core/tests/test_numeric.py", "TestMoveaxis", "test_move_multiples"),
    ("_core/tests/test_numeric.py", "TestMoveaxis", "test_errors"),
    ("_core/tests/test_numeric.py", "TestMoveaxis", "test_array_likes"),
    # NOTE: test_preserve_subtype is in TestRequire (np.require), not TestMoveaxis - out of scope.
    ("_core/tests/test_numeric.py", "TestRollaxis", "test_exceptions"),
    ("_core/tests/test_numeric.py", "TestRollaxis", "test_results"),
    ("_core/tests/test_numeric.py", "TestRoll", "test_roll1d"),
    ("_core/tests/test_numeric.py", "TestRoll", "test_roll2d"),
    ("_core/tests/test_numeric.py", "TestRoll", "test_roll_empty"),
    ("_core/tests/test_numeric.py", "TestRoll", "test_roll_unsigned_shift"),
    ("_core/tests/test_numeric.py", "TestRoll", "test_roll_big_int"),
    # --- test_regression.py ---
    ("_core/tests/test_regression.py", "TestRegression", "test_reshape_order"),
    ("_core/tests/test_regression.py", "TestRegression", "test_reshape_zero_strides"),
    ("_core/tests/test_regression.py", "TestRegression", "test_reshape_zero_size"),
    ("_core/tests/test_regression.py", "TestRegression", "test_reshape_trailing_ones_strides"),
    ("_core/tests/test_regression.py", "TestRegression", "test_reshape_size_overflow"),
    ("_core/tests/test_regression.py", "TestRegression", "test_ravel_with_order"),
    ("_core/tests/test_regression.py", "TestRegression", "test_arr_transpose"),
    ("_core/tests/test_regression.py", "TestRegression", "test_squeeze_type"),
    ("_core/tests/test_regression.py", "TestRegression", "test_squeeze_contiguous"),
    ("_core/tests/test_regression.py", "TestRegression", "test_squeeze_axis_handling"),
    ("_core/tests/test_regression.py", "TestRegression", "test_dtype_scalar_squeeze"),
    # --- _core/tests/test_shape_base.py (atleast_*; rstsr has no analog) ---
    ("_core/tests/test_shape_base.py", "TestAtleast1d", "test_0D_array"),
    ("_core/tests/test_shape_base.py", "TestAtleast1d", "test_1D_array"),
    ("_core/tests/test_shape_base.py", "TestAtleast1d", "test_2D_array"),
    ("_core/tests/test_shape_base.py", "TestAtleast1d", "test_3D_array"),
    ("_core/tests/test_shape_base.py", "TestAtleast1d", "test_r1array"),
    ("_core/tests/test_shape_base.py", "TestAtleast2d", "test_0D_array"),
    ("_core/tests/test_shape_base.py", "TestAtleast2d", "test_1D_array"),
    ("_core/tests/test_shape_base.py", "TestAtleast2d", "test_2D_array"),
    ("_core/tests/test_shape_base.py", "TestAtleast2d", "test_3D_array"),
    ("_core/tests/test_shape_base.py", "TestAtleast2d", "test_r2array"),
    ("_core/tests/test_shape_base.py", "TestAtleast3d", "test_0D_array"),
    ("_core/tests/test_shape_base.py", "TestAtleast3d", "test_1D_array"),
    ("_core/tests/test_shape_base.py", "TestAtleast3d", "test_2D_array"),
    ("_core/tests/test_shape_base.py", "TestAtleast3d", "test_3D_array"),
    # --- lib/tests/test_shape_base.py (expand_dims, squeeze) ---
    ("lib/tests/test_shape_base.py", "TestExpandDims", "test_functionality"),
    ("lib/tests/test_shape_base.py", "TestExpandDims", "test_axis_tuple"),
    ("lib/tests/test_shape_base.py", "TestExpandDims", "test_axis_out_of_range"),
    ("lib/tests/test_shape_base.py", "TestExpandDims", "test_repeated_axis"),
    ("lib/tests/test_shape_base.py", "TestExpandDims", "test_subclasses"),
    ("lib/tests/test_shape_base.py", "TestSqueeze", "test_basic"),
    # --- lib/tests/test_stride_tricks.py (module-level) ---
    ("lib/tests/test_stride_tricks.py", "", "test_broadcast_to_succeeds"),
    ("lib/tests/test_stride_tricks.py", "", "test_broadcast_to_raises"),
    ("lib/tests/test_stride_tricks.py", "", "test_broadcast_shape"),
    ("lib/tests/test_stride_tricks.py", "", "test_broadcast_shapes_succeeds"),
    ("lib/tests/test_stride_tricks.py", "", "test_broadcast_shapes_raises"),
    ("lib/tests/test_stride_tricks.py", "", "test_same"),
    ("lib/tests/test_stride_tricks.py", "", "test_broadcast_kwargs"),
    ("lib/tests/test_stride_tricks.py", "", "test_one_off"),
    ("lib/tests/test_stride_tricks.py", "", "test_same_input_shapes"),
    ("lib/tests/test_stride_tricks.py", "", "test_two_compatible_by_ones_input_shapes"),
    ("lib/tests/test_stride_tricks.py", "", "test_two_compatible_by_prepending_ones_input_shapes"),
    ("lib/tests/test_stride_tricks.py", "", "test_incompatible_shapes_raise_valueerror"),
    ("lib/tests/test_stride_tricks.py", "", "test_same_as_ufunc"),
    # --- lib/tests/test_function_base.py ---
    ("lib/tests/test_function_base.py", "TestFlip", "test_axes"),
    ("lib/tests/test_function_base.py", "TestFlip", "test_basic_lr"),
    ("lib/tests/test_function_base.py", "TestFlip", "test_basic_ud"),
    ("lib/tests/test_function_base.py", "TestFlip", "test_3d_swap_axis0"),
    ("lib/tests/test_function_base.py", "TestFlip", "test_3d_swap_axis1"),
    ("lib/tests/test_function_base.py", "TestFlip", "test_3d_swap_axis2"),
    ("lib/tests/test_function_base.py", "TestFlip", "test_4d"),
    ("lib/tests/test_function_base.py", "TestFlip", "test_default_axis"),
    ("lib/tests/test_function_base.py", "TestFlip", "test_multiple_axes"),
    # --- lib/tests/test_twodim_base.py ---
    ("lib/tests/test_twodim_base.py", "TestFliplr", "test_basic"),
    ("lib/tests/test_twodim_base.py", "TestFlipud", "test_basic"),
    # --- _core/tests/test_shape_base.py: joining (concat/stack/hstack/vstack) ---
    # rstsr `concat`/`concatenate`, `stack`, `hstack`, `vstack` live in
    # `creation_from_tensor.rs`; parity tests in core_func/creation_from_tensor/.
    ("_core/tests/test_shape_base.py", "TestConcatenate", "test_returns_copy"),
    ("_core/tests/test_shape_base.py", "TestConcatenate", "test_exceptions"),
    ("_core/tests/test_shape_base.py", "TestConcatenate", "test_huge_list_error"),
    ("_core/tests/test_shape_base.py", "TestConcatenate", "test_concatenate_axis_None"),
    ("_core/tests/test_shape_base.py", "TestConcatenate", "test_large_concatenate_axis_None"),
    ("_core/tests/test_shape_base.py", "TestConcatenate", "test_concatenate"),
    ("_core/tests/test_shape_base.py", "TestConcatenate", "test_concatenate_same_value"),
    ("_core/tests/test_shape_base.py", "TestConcatenate", "test_operator_concat"),
    ("_core/tests/test_shape_base.py", "TestConcatenate", "test_bad_out_shape"),
    ("_core/tests/test_shape_base.py", "TestConcatenate", "test_out_and_dtype"),
    ("_core/tests/test_shape_base.py", "TestConcatenate", "test_dtype_with_promotion"),
    ("_core/tests/test_shape_base.py", "TestConcatenate", "test_string_dtype_does_not_inspect"),
    ("_core/tests/test_shape_base.py", "TestConcatenate", "test_subarray_error"),
    ("_core/tests/test_shape_base.py", "TestHstack", "test_non_iterable"),
    ("_core/tests/test_shape_base.py", "TestHstack", "test_empty_input"),
    ("_core/tests/test_shape_base.py", "TestHstack", "test_0D_array"),
    ("_core/tests/test_shape_base.py", "TestHstack", "test_1D_array"),
    ("_core/tests/test_shape_base.py", "TestHstack", "test_2D_array"),
    ("_core/tests/test_shape_base.py", "TestHstack", "test_generator"),
    ("_core/tests/test_shape_base.py", "TestHstack", "test_casting_and_dtype"),
    ("_core/tests/test_shape_base.py", "TestHstack", "test_casting_and_dtype_type_error"),
    ("_core/tests/test_shape_base.py", "TestVstack", "test_non_iterable"),
    ("_core/tests/test_shape_base.py", "TestVstack", "test_empty_input"),
    ("_core/tests/test_shape_base.py", "TestVstack", "test_0D_array"),
    ("_core/tests/test_shape_base.py", "TestVstack", "test_1D_array"),
    ("_core/tests/test_shape_base.py", "TestVstack", "test_2D_array"),
    ("_core/tests/test_shape_base.py", "TestVstack", "test_2D_array2"),
    ("_core/tests/test_shape_base.py", "TestVstack", "test_generator"),
    ("_core/tests/test_shape_base.py", "TestVstack", "test_casting_and_dtype"),
    ("_core/tests/test_shape_base.py", "TestVstack", "test_casting_and_dtype_type_error"),
    ("_core/tests/test_shape_base.py", "", "test_stack"),
    ("_core/tests/test_shape_base.py", "", "test_unstack"),
    # --- lib/tests/test_twodim_base.py: diag ---
    ("lib/tests/test_twodim_base.py", "TestDiag", "test_vector"),
    ("lib/tests/test_twodim_base.py", "TestDiag", "test_matrix"),
    ("lib/tests/test_twodim_base.py", "TestDiag", "test_fortran_order"),
    ("lib/tests/test_twodim_base.py", "TestDiag", "test_diag_bounds"),
    ("lib/tests/test_twodim_base.py", "TestDiag", "test_failure"),
    # --- lib/tests/test_function_base.py: meshgrid ---
    ("lib/tests/test_function_base.py", "TestMeshgrid", "test_simple"),
    ("lib/tests/test_function_base.py", "TestMeshgrid", "test_single_input"),
    ("lib/tests/test_function_base.py", "TestMeshgrid", "test_no_input"),
    ("lib/tests/test_function_base.py", "TestMeshgrid", "test_indexing"),
    ("lib/tests/test_function_base.py", "TestMeshgrid", "test_sparse"),
    ("lib/tests/test_function_base.py", "TestMeshgrid", "test_always_tuple"),
    ("lib/tests/test_function_base.py", "TestMeshgrid", "test_invalid_arguments"),
    ("lib/tests/test_function_base.py", "TestMeshgrid", "test_return_type"),
    ("lib/tests/test_function_base.py", "TestMeshgrid", "test_writeback"),
    ("lib/tests/test_function_base.py", "TestMeshgrid", "test_nd_shape"),
    ("lib/tests/test_function_base.py", "TestMeshgrid", "test_nd_values"),
    ("lib/tests/test_function_base.py", "TestMeshgrid", "test_nd_indexing"),
    # --- reductions (sum/prod/mean/std/var/min/max/argmin/argmax/all/any/count_nonzero) ---
    ("_core/tests/test_numeric.py", "TestNonarrayArgs", "test_sum"),
    ("_core/tests/test_numeric.py", "TestNonarrayArgs", "test_prod"),
    ("_core/tests/test_numeric.py", "TestNonarrayArgs", "test_mean"),
    ("_core/tests/test_numeric.py", "TestNonarrayArgs", "test_std"),
    ("_core/tests/test_numeric.py", "TestNonarrayArgs", "test_var"),
    ("_core/tests/test_numeric.py", "TestNonarrayArgs", "test_count_nonzero"),
    ("lib/tests/test_function_base.py", "TestAmax", "test_basic"),
    ("lib/tests/test_function_base.py", "TestAmin", "test_basic"),
    ("_core/tests/test_multiarray.py", "TestArgmax", "test_combinations"),
    ("_core/tests/test_multiarray.py", "TestArgmin", "test_combinations"),
    ("_core/tests/test_regression.py", "TestRegression", "test_argmax"),
    ("lib/tests/test_function_base.py", "TestAll", "test_basic"),
    ("lib/tests/test_function_base.py", "TestAll", "test_nd"),
    ("lib/tests/test_function_base.py", "TestAny", "test_basic"),
    ("lib/tests/test_function_base.py", "TestAny", "test_nd"),
    # --- creation (arange/linspace/zeros/ones/full/eye/tril/triu) ---
    ("_core/tests/test_multiarray.py", "TestArange", "test_start_stop_kwarg"),
    ("_core/tests/test_multiarray.py", "TestArange", "test_zero_step"),
    ("_core/tests/test_function_base.py", "TestLinspace", "test_basic"),
    ("_core/tests/test_function_base.py", "TestLinspace", "test_corner"),
    ("_core/tests/test_numeric.py", "TestCreationFuncs", "test_zeros"),
    ("_core/tests/test_numeric.py", "TestCreationFuncs", "test_ones"),
    ("_core/tests/test_numeric.py", "TestCreationFuncs", "test_full"),
    ("lib/tests/test_twodim_base.py", "TestEye", "test_basic"),
    ("lib/tests/test_twodim_base.py", "", "test_tril_triu_ndim2"),
    ("lib/tests/test_twodim_base.py", "", "test_tril_triu_ndim3"),
    # --- linalg (matmul, matrix_transpose) ---
    ("_core/tests/test_multiarray.py", "MatmulCommon", "test_vector_vector_values"),
    ("_core/tests/test_multiarray.py", "MatmulCommon", "test_vector_matrix_values"),
    ("_core/tests/test_multiarray.py", "MatmulCommon", "test_matrix_vector_values"),
    ("_core/tests/test_multiarray.py", "MatmulCommon", "test_matrix_matrix_values"),
    ("_core/tests/test_arrayobject.py", "", "test_matrix_transpose_raises_error_for_1d"),
    ("_core/tests/test_arrayobject.py", "", "test_matrix_transpose_equals_transpose_2d"),
    ("_core/tests/test_arrayobject.py", "", "test_matrix_transpose_equals_swapaxes"),
    # --- linalg: vecdot ---
    ("_core/tests/test_ufunc.py", "TestUfunc", "test_vecdot"),
    ("_core/tests/test_ufunc.py", "TestUfunc", "test_broadcast"),
    ("_core/tests/test_ufunc.py", "TestUfunc", "test_vecdot_matvec_vecmat_complex"),
    # --- indexing (basic indexing via Tensor::i) ---
    ("_core/tests/test_indexing.py", "TestIndexing", "test_single_int_index"),
    ("_core/tests/test_indexing.py", "TestIndexing", "test_ellipsis_index"),
    # --- gap functions (rstsr has no analog): one representative method per NumPy
    #     test class, status `todo` in numpy_coverage.csv. The note cites the full
    #     class + method count; only the representative is hashed here. ---
    ("_core/tests/test_shape_base.py", "TestBlock", "test_block_simple_row_wise"),
    ("lib/tests/test_shape_base.py", "TestColumnStack", "test_2D_arrays"),
    ("lib/tests/test_shape_base.py", "TestDstack", "test_2D_array"),
    ("lib/tests/test_shape_base.py", "TestArraySplit", "test_integer_split"),
    ("lib/tests/test_shape_base.py", "TestSplit", "test_equal_split"),
    ("lib/tests/test_shape_base.py", "TestHsplit", "test_2D_array"),
    ("lib/tests/test_shape_base.py", "TestVsplit", "test_2D_array"),
    ("lib/tests/test_shape_base.py", "TestDsplit", "test_2D_array"),
    ("lib/tests/test_shape_base.py", "TestTile", "test_basic"),
]


def _index_file(tree, source):
    """Return {(class, name): (lineno, source_segment)} for a parsed module."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            cls = node.name
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    seg = ast.get_source_segment(source, item)
                    out[(cls, item.name)] = (item.lineno, seg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # module-level (only top-level; ast.walk also visits nested, filter later)
            pass
    # module-level fns
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seg = ast.get_source_segment(source, node)
            out[("", node.name)] = (node.lineno, seg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("numpy_root")
    ap.add_argument("--hashes", action="store_true",
                    help="print only path::class::method line sha256")
    args = ap.parse_args()
    root = Path(args.numpy_root)

    # group surface by file to parse each once
    by_file = {}
    for (rel, cls, method) in SURFACE:
        by_file.setdefault(rel, []).append((cls, method))

    rows = []
    for rel, items in sorted(by_file.items()):
        fpath = root / "numpy" / rel
        source = fpath.read_text()
        tree = ast.parse(source)
        idx = _index_file(tree, source)
        for (cls, method) in items:
            key = (cls, method)
            if key not in idx:
                rows.append((rel, cls, method, "MISSING", ""))
                continue
            lineno, seg = idx[key]
            h = hashlib.sha256(seg.encode()).hexdigest()[:12]
            rows.append((rel, cls, method, lineno, h))

    if args.hashes:
        for (rel, cls, method, lineno, h) in rows:
            ident = f"{rel}::{cls}::{method}" if cls else f"{rel}::{method}"
            print(f"{ident} {lineno} {h}")
        return

    # full CSV (line + hash only; status/rstsr_test/note merged by hand into
    # numpy_coverage.csv - see ADR-0003; this script is the line/hash oracle)
    print("numpy_path,class,method,line,numpy_source_hash")
    for (rel, cls, method, lineno, h) in rows:
        print(f"{rel},{cls},{method},{lineno},{h}")


if __name__ == "__main__":
    main()
