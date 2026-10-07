#!/usr/bin/env bash
# Install the pinned external Kev runtime and public weights separately from MLX training.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
task_dir="${KEV_WORK_DIR:-$root/.kev}"
runtime="$task_dir/runtime"
revision=5e42a7a03f28134853dd3ff77461457e921e5ec1
mkdir -p "$task_dir"
if [[ ! -d "$runtime" ]]; then
    git clone --depth 1 --filter=blob:none --sparse https://github.com/jaredpalmer/kev.git "$runtime"
    git -C "$runtime" sparse-checkout set kev
    git -C "$runtime" fetch --depth 1 origin "$revision"
    git -C "$runtime" checkout --detach "$revision"
fi
[[ "$(git -C "$runtime" rev-parse HEAD)" == "$revision" ]] || {
    echo "Existing Kev runtime is a different revision; choose a new KEV_WORK_DIR." >&2
    exit 1
}
[[ -z "$(git -C "$runtime" status --porcelain)" ]] || {
    echo "Kev runtime contains local edits; refusing to use it." >&2
    exit 1
}
export UV_PROJECT_ENVIRONMENT="$task_dir/env"
uv sync --project "$runtime" --locked --no-dev --extra serve --python 3.12
export HF_HOME="$task_dir/hf"
"$task_dir/env/bin/hf" download jaredpalmer/kev-0.8b \
    adapter_config.json adapter_model.safetensors head.pt \
    --revision bf75a6a8848ea6960ff2ed108d9ed44c2941174f --local-dir "$task_dir/adapter"
"$task_dir/env/bin/hf" download Qwen/Qwen3.5-0.8B-Base \
    --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
    --include '*.json' --include '*.safetensors' --include '*.txt' --include '*.jinja'
echo "Kev prepared in $task_dir. Evaluation commands are in docs/kev_research.md."
