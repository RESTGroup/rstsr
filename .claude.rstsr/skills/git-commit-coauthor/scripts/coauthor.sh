#!/usr/bin/env bash
# Generate git Co-authored-by trailers for AI code agents (RSTSR convention).
#
# Usage:
#   coauthor.sh --agent "Codex" --model "glm-5.2"
#   coauthor.sh --agent "Claude Code" --model "qwen3.5-plus"
#
# This script is the authoritative email registry. Keep the table in
# ../SKILL.md in sync with it. Matching is case-insensitive by family prefix
# (e.g. glm*, qwen*). The agent name is canonicalized on output; the model
# name is passed through verbatim (it carries the version). Exits non-zero on
# unknown agent/model.
set -euo pipefail

agent=""
model=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent) agent="${2:-}"; shift 2 ;;
    --model) model="${2:-}"; shift 2 ;;
    -h|--help)
      echo "Usage: coauthor.sh --agent \"<Agent Name>\" --model \"<Model Name>\""
      echo "Generates Co-authored-by trailers (agent + model) per RSTSR convention."
      exit 0 ;;
    *) echo "coauthor.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done

agent_canonical() {
  local a
  a="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$a" in
    "claude code") echo "Claude Code" ;;
    "codex")       echo "Codex" ;;
    *)             echo "$1" ;;
  esac
}

agent_email() {
  local a
  a="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$a" in
    "claude code") echo "noreply@anthropic.com" ;;
    "codex")       echo "noreply@openai.com" ;;
    *)             echo "" ;;
  esac
}

model_email() {
  local m
  m="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$m" in
    glm*)       echo "service@zhipuai.cn" ;;
    qwen*)      echo "qianwen_opensource@alibabacloud.com" ;;
    minimax*)   echo "model@minimax.io" ;;
    deepseek*)  echo "service@deepseek.com" ;;
    kimi*)      echo "growth@moonshot.cn" ;;
    claude*)    echo "noreply@anthropic.com" ;;
    gpt*)       echo "noreply@openai.com" ;;
    *)          echo "" ;;
  esac
}

if [[ -z "$agent" && -z "$model" ]]; then
  echo "coauthor.sh: provide --agent and/or --model" >&2
  exit 2
fi

ac=""
ae=""
me=""
[[ -n "$agent" ]] && { ac="$(agent_canonical "$agent")"; ae="$(agent_email "$agent")"; }
[[ -n "$model" ]] && me="$(model_email "$model")"

if [[ -n "$agent" && -z "$ae" ]]; then
  echo "coauthor.sh: unknown agent '$agent' (no email in registry)" >&2
  exit 3
fi
if [[ -n "$model" && -z "$me" ]]; then
  echo "coauthor.sh: unknown model '$model' (no email in registry)" >&2
  exit 3
fi

# Emit co-authors with NO blank line between them.
if [[ -n "$ae" && -n "$me" ]]; then
  printf 'Co-authored-by: %s <%s>\nCo-authored-by: %s <%s>\n' "$ac" "$ae" "$model" "$me"
elif [[ -n "$ae" ]]; then
  printf 'Co-authored-by: %s <%s>\n' "$ac" "$ae"
else
  printf 'Co-authored-by: %s <%s>\n' "$model" "$me"
fi
