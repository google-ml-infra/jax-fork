#!/bin/bash
# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Runs Pyest CPU tests. Requires all jaxlib, jax-cuda-plugin, and jax-cuda-pjrt
# wheels to be present inside $JAXCI_OUTPUT_DIR (../dist)
#
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exu -o history -o allexport

# Inherit default JAXCI environment variables.
source ci/envs/default.env

# Install jaxlib, jax-cuda-plugin, and jax-cuda-pjrt wheels on the system.
<<<<<<< HEAD
export JAXCI_INSTALL_WHEELS_LOCALLY=1
=======
echo "Installing wheels locally..."
source ./ci/utilities/install_wheels_locally.sh
>>>>>>> main

# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

<<<<<<< HEAD

# # Install cuda deps into the current python
# "$JAXCI_PYTHON" -m pip install "nvidia-cublas-cu12>=12.1.3.1"
# "$JAXCI_PYTHON" -m pip install "nvidia-cuda-cupti-cu12>=12.1.105"
# "$JAXCI_PYTHON" -m pip install "nvidia-cuda-nvcc-cu12>=12.1.105"
# "$JAXCI_PYTHON" -m pip install "nvidia-cuda-runtime-cu12>=12.1.105"
# "$JAXCI_PYTHON" -m pip install "nvidia-cudnn-cu12>=9.1,<10.0"
# "$JAXCI_PYTHON" -m pip install "nvidia-cufft-cu12>=11.0.2.54"
# "$JAXCI_PYTHON" -m pip install "nvidia-cusolver-cu12>=11.4.5.107"
# "$JAXCI_PYTHON" -m pip install "nvidia-cusparse-cu12>=12.1.0.106"
# "$JAXCI_PYTHON" -m pip install "nvidia-nccl-cu12>=2.18.1"
# "$JAXCI_PYTHON" -m pip install "nvidia-nvjitlink-cu12>=12.1.105"
# "$JAXCI_PYTHON" -m pip install "tensorrt"

# "$JAXCI_PYTHON" -m pip install "tensorrt-lean"
# "$JAXCI_PYTHON" -m pip install "tensorrt-dispatch"

=======
>>>>>>> main
export PY_COLORS=1
export JAX_SKIP_SLOW_TESTS=true

"$JAXCI_PYTHON" -c "import jax; print(jax.default_backend()); print(jax.devices()); print(len(jax.devices()))"

nvidia-smi
export NCCL_DEBUG=WARN
export TF_CPP_MIN_LOG_LEVEL=0

echo "Running GPU tests..."
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1
"$JAXCI_PYTHON" -m pytest -n 8 --tb=short --maxfail=20 \
tests examples \
--deselect=tests/multi_device_test.py::MultiDeviceTest::test_computation_follows_data \
--deselect=tests/xmap_test.py::XMapTest::testCollectivePermute2D \
--deselect=tests/multiprocess_gpu_test.py::MultiProcessGpuTest::test_distributed_jax_visible_devices \
--deselect=tests/compilation_cache_test.py::CompilationCacheTest::test_task_using_cache_metric