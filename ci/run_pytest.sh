#!/bin/bash
# Copyright 2024 JAX Authors. All Rights Reserved.
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
source "ci/utilities/setup.sh"

jaxrun "$JAXCI_PYTHON" -c "import jax; print(jax.default_backend()); print(jax.devices()); print(len(jax.devices()))"

if [[ $JAXCI_RUN_PYTEST_CPU == 1 ]]; then
  jaxrun "$JAXCI_PYTHON" -m pytest -n auto --tb=short --maxfail=20 tests examples
fi

if [[ $JAXCI_RUN_PYTEST_GPU == 1 ]]; then
  export XLA_PYTHON_CLIENT_ALLOCATOR=platform
  export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1
  jaxrun "$JAXCI_PYTHON" -m pytest -n 8 --tb=short --maxfail=20 \
  tests examples \
  --deselect=tests/multi_device_test.py::MultiDeviceTest::test_computation_follows_data \
  --deselect=tests/xmap_test.py::XMapTest::testCollectivePermute2D \
  --deselect=tests/multiprocess_gpu_test.py::MultiProcessGpuTest::test_distributed_jax_visible_devices \
  --deselect=tests/compilation_cache_test.py::CompilationCacheTest::test_task_using_cache_metric
fi