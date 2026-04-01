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
Code agent stubs can be generated in `.tmp/` relative to the project root. You have very large premissions to edit in that directory.

As AI agent, you should communicate in the same language as the user. However, the code/PR generation should always be in English.

## Git Commit Convention

- For co-author, please add AI agent and model name as co-author.
  - Multi-co-author format (note no extra newline between co-authors):
    ```
    Co-authored-by: Agent Name <Agent Email>
    Co-authored-by: Model Name <Model Email>
    ```
  - Agent:
    - Claude Code: noreply@anthropic.com
  - Model:
    - qwen* (eg. qwen3.5-plus): qianwen_opensource@alibabacloud.com
    - glm* (eg. glm-5): service@zhipuai.cn
    - minimax* (eg. MiniMax-M2.5): model@minimax.io
    - deepseek* (eg. DeepSeek-V3.2): service@deepseek.com
    - kimi* (eg. kimi-k2.5): growth@moonshot.cn
  - Model name should include the version or details, such as `qwen3.5-plus`, `glm-5`, which can be inferred by Claude Code's `/model` property.
- Commit starts with main crate that be affected, for example `rstsr-core: add reshape function`. Prefer to add more details in commit message body.

## Rules and important skills

@rules/agent-and-maintaince.md
@skills/cargo-inst/SKILL.md
