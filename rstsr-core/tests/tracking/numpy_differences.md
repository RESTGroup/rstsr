# NumPy Differences Report (rstsr-core)

Where rstsr diverges from NumPy. Distilled from the parity tests in `core_func/` and
the coverage checklist `numpy_coverage.csv`. See ADR-0003.

**Pinned NumPy:** v2.4.2 · **Checkout:** `../other-repos/numpy` (tag v2.4.2)

## Tags

- `intentional` - rstsr deliberately differs (ownership semantics, no `out=`, trait
  dispatch, col-major convention). Not a bug.
- `bug` - rstsr differs unintentionally. File an issue; link it here.
- `col-major-transfer` - difficulty mapping row-major NumPy behavior to rstsr's
  column-major convention. (Col-major tests live in `tests/col_major/`, deferred.)

## Format

One section per divergence. Cite the NumPy identifier and the rstsr test:

```
## <short title>

- **numpy:** _core/tests/test_multiarray.py::TestMethods::test_<x> (L<n>)
- **rstsr:** entry_row_cpu::core_func::<category>::test_<func>::numpy_<func>::<case>
- **tag:** intentional | bug | col-major-transfer
- **status:** open | fixed (issue #<n>) | wontfix

<what differs and why>
```

<!-- Entries below. Append new divergences here as parity tests are authored. -->
