# Re-point the dead /scratch365 paths that ~/.bashrc exports (see verl-sglang-env-constraints).
# Caches/temp go to node-local /tmp: the /groups/xzhang33 5 TB group quota is full.
G=/groups/xzhang33/tguo2taicheng
export HF_HOME=$G/workAmazon/hf HF_ROOT=$G/workAmazon/hf
export HF_DATASETS_CACHE=$HF_HOME/datasets HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers HF_MODULES_CACHE=$HF_HOME/modules
export XDG_CACHE_HOME=/tmp/tguo2-xdg TMPDIR=/tmp/tguo2-tmp PYARROW_TMPDIR=/tmp/tguo2-tmp
export UV_CACHE_DIR=/tmp/tguo2-uv-cache UV_PYTHON_INSTALL_DIR=$G/envs/uv-python
export PIP_CACHE_DIR=/tmp/tguo2-pip-cache
export UV_PYTHON_INSTALL_BIN=0 UV_PYTHON_DOWNLOADS=manual
mkdir -p $TMPDIR $XDG_CACHE_HOME
