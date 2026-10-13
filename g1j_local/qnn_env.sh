#!/usr/bin/env bash
set -euo pipefail

QNN_WORK_ROOT=/home/halo/g1j-qnn
QNN_VENV="$QNN_WORK_ROOT/venv/qnn"
QAIRT_VERSION=2.42.0.251225
QNN_PYTHON_RUNTIME="$(dirname "$(dirname "$(readlink -f "$QNN_VENV/bin/python")")")"

export QNN_SDK_ROOT="$QNN_WORK_ROOT/sdk/qairt/$QAIRT_VERSION"
export HEXAGON_SDK_ROOT="$QNN_WORK_ROOT/toolchains/hexagon-sdks/hexagon-sdk-6.4.0"
export PATH="$QNN_VENV/bin:$QNN_SDK_ROOT/bin/x86_64-linux-clang:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_PYTHON_RUNTIME/lib:$QNN_WORK_ROOT/toolchains/local-libs/usr/lib/x86_64-linux-gnu:$QNN_SDK_ROOT/lib/x86_64-linux-clang"
export LIBRARY_PATH="$QNN_WORK_ROOT/toolchains/local-libs/usr/lib/x86_64-linux-gnu"
export CPATH="$QNN_WORK_ROOT/toolchains/local-libs/usr/include:$QNN_WORK_ROOT/toolchains/local-libs/usr/include/python3.10"
export TORCH_EXTENSIONS_DIR="$QNN_WORK_ROOT/cache/torch-extensions"
export HF_HOME="$QNN_WORK_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$QNN_WORK_ROOT/cache/huggingface/datasets"
export HF_HUB_DOWNLOAD_TIMEOUT=600
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
