# RSTSR Test Reformulation

The ubiquitous language for reformulating rstsr-core's test suite to track and
match NumPy. This context covers the *testing* domain; library-implementation
terms (tensor, device, layout) live in `.claude/` rules and source, not here.

## Language

### Faithfulness model

**Parity test**:
A test asserting rstsr's output matches NumPy's for inputs where both define
behavior. The NumPy source is kept as commented Python and acts as the
reference oracle. Lives in `core_func/`. Inherently row-major (NumPy is C-order).
_Avoid_: "NumPy-transferred test" (ambiguous with the coverage checklist)

**Coverage checklist**:
A machine-readable file listing every tracked NumPy test function with a
transfer status (`transferred` / `partial` / `not-applicable` / `skipped`).
A project-management artifact, not a test.
_Avoid_: "tracking file" (ambiguous - covers both checklist and report)

**Differences report**:
A human-readable file recording where rstsr diverges from NumPy, tagged
`intentional` / `bug` / `col-major-transfer`. Distilled from parity tests and
the checklist.

### Test structure

**Body** (shared body):
Device-agnostic test modules under `tests/` (`core_func/`, `doc_draft/`,
`test_issues/`) holding the test logic. `mod`-included by every entry binary.
Designed to be symlinked by device crates.

**Entry binary**:
A top-level `tests/entry_<order>_<device>.rs` that becomes one cargo test
binary. Defines `type DeviceType` and `static TESTCFG`, then `mod`-includes the
body. The order × device matrix is formed by entry binaries, not by duplicating
test code.

**doc_draft**:
A tree of tests exercising examples destined for API docstrings. Separated from
parity tests (`core_func/`).

**col_major body**:
A reserved tree of tests for rstsr's column-major convention. Not NumPy-parity
(NumPy is row-major); a separate, deferred track.

### Tracking

**numpy identifier**:
A pytest-nodeid-style key for a NumPy test function:
`<path>::<Class>::<method>`, e.g.
`_core/tests/test_multiarray.py::TestMethods::test_transpose`. Shared between
the inline provenance comment and the coverage-checklist row.

**specify_test!**:
The runtime test-gating macro. Checks a `[category, func, item]` path against
`TestCfg`'s skip/allow lists and returns early if skipped. Enables per-device
skips without recompilation.

**rstsr-test-manifest**:
A workspace crate holding NumPy-generated `.npy` golden arrays and loaders
(`get_vec::<T>(c)`), used by device crates for large reference data. **Out of scope
for rstsr-core**: core parity tests use `tensor_from_nested!` for small tensors and
`rt::asarray` with a Rust `Vec` otherwise. No `.npy`, no python regen step in core.

**rstest**:
The chosen Rust fixture/parametrization framework if a core test needs setup/teardown
or data reuse. Not adopted blanketly - only when a test genuinely needs it.

### Helpers & gotchas

**allclose broadcast gotcha**:
`rt::allclose` is broadcast-permissive - two tensors of *different* shapes can be
"allclose." `assert_equal` is safe (it `assert_eq!`s shapes first); raw `rt::allclose`
is not. Document wherever raw `allclose` is used.
