## AI Code Agent

You are a helper, not only generates code, but also suggested to ask the user for thoughts and suggestions.

Following has directory `../`, where the relative path is from working directory of this project.

- This project follows claude-code style; configurations/instructions are in directory `.claude/`. Useful skills (task-dependent instructions) at `./claude/skills/`.
- Documentation project may at `../rstsr-docs/`. 
- User may request code generation with help of other libraries such as numpy, array-api, scipy, reference-lapack. These repositories may symlinked at `../other-repos/`.
- If you found some temporary files generated is useful and may be kept for other sessions, generate them in `.claude/scratch/` and inform the user. You can take notes for yourself at `.claude/memory`.
- Always run formatter (`cargo fmt`) after code generation. Prefer `cargo clippy` for linting before commit-ready code generation. Only run tests if necessary, as they may take some time.
- You are suggested to use `rg` (ripgrep) instead of `grep` for searching code, if calling `rg` or `ripgrep` at bash success.

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

## Pull Request Convention

You should fill template `<...>` with proper information. If some sections/subsections are tagged optional, then you can remove them.

```md
# Summary

<...>

# Changes

## API breaking changes

<... (optional)>

## New features

<... (optional)>

## Feature improvements or changes

<... (optional)>

## Bug fixes

<... (optional)>

# Developer

## Code Style Update

<... (optional)>

---

PR summarized by
- Agent: <Agent Name>
- Model: <Model Name>
```
