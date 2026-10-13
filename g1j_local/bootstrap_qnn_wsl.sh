#!/usr/bin/env bash
set -euo pipefail

umask 077
QNN_WORK_ROOT=/home/halo/g1j-qnn
UV=/home/halo/.local/bin/uv
QNN_VENV="$QNN_WORK_ROOT/venv/qnn"
QAIRT_VERSION=2.42.0.251225
QAIRT_ZIP="$QNN_WORK_ROOT/downloads/v${QAIRT_VERSION}.zip"
QAIRT_URL="https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${QAIRT_VERSION}/v${QAIRT_VERSION}.zip"
HEXAGON_VERSION=6.4.0.2
HEXAGON_ZIP="$QNN_WORK_ROOT/downloads/Hexagon_SDK_lnx_${HEXAGON_VERSION}.zip"
HEXAGON_URL="https://softwarecenter.qualcomm.com/api/download/software/sdks/Hexagon_SDK/Linux/Debian/${HEXAGON_VERSION}/Hexagon_SDK_lnx.zip"
PYPI_INDEX=https://mirrors.aliyun.com/pypi/simple/
REPO=/mnt/c/repo/rwkv-qualcomm-g1j-work

mkdir -p "$QNN_WORK_ROOT"/{cache,downloads,logs,sdk,toolchains,venv}
exec > >(tee -a "$QNN_WORK_ROOT/logs/bootstrap-qnn.log") 2>&1

for tool in curl git cmake make unzip sha256sum; do
  command -v "$tool"
done
"$UV" --version

export UV_CACHE_DIR="$QNN_WORK_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$QNN_WORK_ROOT/toolchains/python"
"$UV" python install 3.10
if [[ ! -x "$QNN_VENV/bin/python" ]]; then
  "$UV" venv --python 3.10 --seed "$QNN_VENV"
fi
"$UV" pip install --python "$QNN_VENV/bin/python" \
  --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.1.2 torchvision==0.16.2
"$UV" pip install --python "$QNN_VENV/bin/python" --index-url "$PYPI_INDEX" \
  'Jinja2>=3.0.3' PyYAML 'bokeh<=3.6.3' cvxpy holoviews hvplot jsonschema \
  'matplotlib>=3' networkx 'numpy<2' fsspec==2023.6.0 onnx==1.16.1 onnx-graphsurgeon==0.5.8 \
  onnxruntime==1.15.1 onnxscript==0.2.6 osqp pandas psutil pyarrow==14.0.2 pybind11 \
  protobuf==5.29.3 safetensors==0.5.3 scikit-learn scipy tensorboard datasets \
  huggingface_hub==0.36.2 peft==0.20.0 transformers==4.52.1 accelerate==1.9.0 \
  lxml==5.2.1 setuptools==80.9.0 tqdm
"$UV" pip install --python "$QNN_VENV/bin/python" --no-deps --index-url "$PYPI_INDEX" aimet-torch==2.17.0

"$QNN_VENV/bin/python" -c 'import torch, aimet_torch, onnx, onnxruntime, onnxscript, datasets; print("python stack", torch.__version__, aimet_torch.__version__, onnx.__version__, onnxruntime.__version__, onnxscript.__version__, datasets.__version__, "cuda", torch.cuda.is_available())'

curl --http1.1 -L --fail --retry 20 --retry-all-errors --retry-delay 2 -C - -o "$QAIRT_ZIP" "$QAIRT_URL"
unzip -tq "$QAIRT_ZIP"
if [[ ! -d "$QNN_WORK_ROOT/sdk/qairt/$QAIRT_VERSION" ]]; then
  unzip -q "$QAIRT_ZIP" -d "$QNN_WORK_ROOT/sdk"
fi

curl --http1.1 -L --fail --retry 20 --retry-all-errors --retry-delay 2 -C - -o "$HEXAGON_ZIP" "$HEXAGON_URL"
unzip -tq "$HEXAGON_ZIP"
if [[ ! -d "$QNN_WORK_ROOT/toolchains/hexagon-sdk-package/Hexagon_SDK/$HEXAGON_VERSION" ]]; then
  mkdir -p "$QNN_WORK_ROOT/toolchains/hexagon-sdk-package"
  unzip -q "$HEXAGON_ZIP" -d "$QNN_WORK_ROOT/toolchains/hexagon-sdk-package"
fi
mkdir -p "$QNN_WORK_ROOT/toolchains/hexagon-sdks"
if [[ ! -e "$QNN_WORK_ROOT/toolchains/hexagon-sdks/hexagon-sdk-6.4.0" ]]; then
  ln -s "$QNN_WORK_ROOT/toolchains/hexagon-sdk-package/Hexagon_SDK/$HEXAGON_VERSION" \
    "$QNN_WORK_ROOT/toolchains/hexagon-sdks/hexagon-sdk-6.4.0"
fi

mkdir -p "$QNN_WORK_ROOT/toolchains/local-debs" "$QNN_WORK_ROOT/toolchains/local-libs"
(
  cd "$QNN_WORK_ROOT/toolchains/local-debs"
  apt-get download libc++1-18 libc++abi1-18 libunwind-18 clang-18 libclang-cpp18 \
    libllvm18 libclang-common-18-dev llvm-18-linker-tools libclang1-18 \
    libobjc-13-dev libc++-18-dev libc++abi-18-dev
  for package in ./*.deb; do
    dpkg-deb -x "$package" "$QNN_WORK_ROOT/toolchains/local-libs"
  done
)

sha256sum "$QAIRT_ZIP" "$HEXAGON_ZIP"
source "$REPO/g1j_local/qnn_env.sh"
qnn-context-binary-generator --version
test -f "$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04/Tools/libnative/lib/libnative.a"

cd "$REPO"
make -C hexagon/HTP/RwkvWkvOpPackage htp_x86 -j4 \
  X86_CXX="$QNN_WORK_ROOT/toolchains/local-libs/usr/bin/clang++-18"
python -c 'import torch.utils.cpp_extension; import rwkv_src.rwkv_v7_modules_conv; print("rwkv torch extension import ok")'

printf 'QNN bootstrap complete\n'
