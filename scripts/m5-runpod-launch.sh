#!/bin/bash
set -euo pipefail

PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH
unset BASH_ENV ENV NODE_OPTIONS NODE_PATH PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONUSERBASE PYTHONBREAKPOINT
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for name in "${!GIT_@}"; do
  unset "$name"
done

safe_git() {
  HOME=/nonexistent/seroslop-m5-git \
  XDG_CONFIG_HOME=/nonexistent/seroslop-m5-git/xdg \
  GIT_CONFIG_NOSYSTEM=1 \
  GIT_CONFIG_GLOBAL=/dev/null \
  GIT_CONFIG_SYSTEM=/dev/null \
  GIT_NO_REPLACE_OBJECTS=1 \
  GIT_OPTIONAL_LOCKS=0 \
  GIT_TERMINAL_PROMPT=0 \
  GIT_ASKPASS=/bin/false \
  GIT_SSH_COMMAND=/bin/false \
  /usr/bin/git -c core.fsmonitor=false -c core.hooksPath=/dev/null -c core.pager=cat -c core.attributesFile=/dev/null "$@"
}

if [[ "$(/usr/bin/uname -s)" != "Linux" || "$(/usr/bin/uname -m)" != "x86_64" ]]; then
  echo "M5 RunPod launch requires pinned Linux x86_64" >&2
  exit 64
fi
if [[ "$(safe_git rev-parse --show-toplevel)" != "$PWD" ]]; then
  echo "M5 RunPod launch requires the repository root as the working directory" >&2
  exit 64
fi

readonly PYTHON=/opt/conda/bin/python
readonly NODE=/workspace/.seroslop/runtime/node-v24.18.1-linux-x64/bin/node

"$PYTHON" -I scripts/m5_node_bootstrap.py
exec /usr/bin/env -u NODE_OPTIONS -u NODE_PATH "$NODE" scripts/m5-python-launch.mjs "$@"
