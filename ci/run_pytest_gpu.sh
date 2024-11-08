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
# Inherit default JAXCI environment variables.
source ci/envs/default.env

# Install jaxlib, jax-cuda-plugin, and jax-cuda-pjrt wheels on the system.
# Requires all 3 wheels to be present inside $JAXCI_OUTPUT_DIR (by default, ../dist)
export JAXCI_INSTALL_WHEELS_LOCALLY=1

# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

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