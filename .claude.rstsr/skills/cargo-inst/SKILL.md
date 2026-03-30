---
name: cargo-inst
description: Rust's cargo instructions for building, testing, API documentation for various cases and crates.
---

## Example

```bash
RUST_BACKTRACE=1 RSTSR_DEV=1 cargo test --package rstsr-core --test tests_core_row --features backtrace --no-default-features -- tests_core::manuplication::test_reshape::numpy_reshape::regression --exact --nocapture
```

## Testing rules

- Only test one crate at a time.

- Always add `RUST_BACKTRACE=1` (for backtrace) and `RSTSR_DEV=1` (for dynamic linking for development-only).

- Testing cases (`--test tests_core_row` in example above) depends on the real testing needs.
  - Doctest: `--doc`.
  - Integration test (in dir `tests`): `--test <test_name>`, where `<test_name>` is the name of the test `.rs` file under `tests` directory.
  - All integration tests: `--tests`.
  - Unittest (in dir `src`): `--lib`.

- Cargo feature selection (for crate `rstsr`):
  - **Only test usual situation, unless user explicitly specifies**.
  - Usual situation: `--features backtrace --no-default-features`; this will minimize building time and still perform correctly tests.
  - With parallel (using Faer device): ` ` (default features for users, but not for development).
  - With column-major: additionally add `col_major` after `--features` (for example, `--features "backtrace col_major"`).

- Cargo feature selection (for crates listed in `crates-device`):
  - **Only test usual situation, unless user explicitly specifies**.
  - **Additionally add `--test-threads=1` after `--exact --nocapture`** to disable parallel testing.
  - Usual situation: `--features "rstsr/backtrace"` with other default features.
    - Exception for crate `rstsr-openblas`'s usual case: `--features "rstsr/backtrace openmp"`
  - Currently integration test only allows row-major.

## API Document Building

```bash
RUSTDOCFLAGS="--html-in-header katex-header.html" cargo doc --no-deps
```

It is better to provide `katex-header.html`; however if that causes error, ignoring it is also fine.

Always create document at crate level, instead of workspace level.

## Linting and Formatting

```bash
cargo fmt  # Only run this before commit. This can be run at workspace level.
cargo clippy --all-targets --all-features -- -D warnings  # Run at crate level.
```
