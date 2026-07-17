# Developer Notes on Claude Code Usage

This folder will be at `.claude.rstsr` directory, which will be tracked by git. You can symlink it to `.claude` or `.agents` if you want.

This project excludes `.local`-like paths, and you can add local configurations to it, without polluting the main repository.

This folder can be regarded as a temporary developer documentation.

`AGENTS.md` is symlinked to `CLAUDE.md`, so Claude Code, Codex, and other AGENTS.md-aware tools read the same rules. The local symlinks (`.claude`, `.agents`) and tool-local state (`.codex/`, `.aider*`, `.cursor/`, `.continue/`, `.roo/`) are gitignored; see `.gitignore`.
