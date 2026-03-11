# CLAUDE.md

## Overview

RSTSR is a rust, NumPy-like multi-dimensional tensor library.

RSTSR shares similar goals and design principles with other Rust tensor librarie `ndarray`, but different in that
- supports multiple backends (OpenBLAS, faer, MKL, etc) through trait-based design.
- supports both row/column-major.
- uses trait-based overloading for operations instead of macros.

```
rstsr/
├── rstsr-core/           # Core tensor types, storage, device traits, also naive and faer device ref impl
├── rstsr-common/         # Shared utilities, error handling, dimension types
├── rstsr-dtype-traits/   # Data type traits and promotions
├── rstsr-blas-traits/    # BLAS operation traits and ref impl
├── rstsr-linalg-traits/  # Linear algebra traits and ref impl
├── rstsr-sci-traits/     # Scientific computing traits (not utilized yet)
├── rstsr-native-impl/    # Native CPU with rayon parallel ref impl
├── crates-device/        # Device backends
│   ├── rstsr-openblas/   # OpenBLAS backend
│   ├── rstsr-mkl/        # Intel MKL backend
│   ├── rstsr-blis/       # BLIS/FLAME backend
|   ├── ...
└── crates-plugin/
    └── rstsr-tblis/      # TBLIS einsum plugin
```

## Notes

Please also see rules in directory `.claude/rules/` for more details on code style, workflow, etc.

The user may use other languages (Chinese, English, for example) to interact with AI agent. As AI agent, you should communicate in the same language as the user. However, the code/PR generation should always be in English.

@rules/agent-and-maintaince.md
