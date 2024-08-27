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

os=$(uname -s | awk '{print tolower($0)}')
arch=$(uname -m)

if [[ $JAXCI_BUILD_BAZEL_TEST_ENABLE == 1 ]]; then
      # Bazel build on RBE CPU. Used when RBE is not available for the platform. E.g
      # Linux Aarch64
      jaxrun bazel --bazelrc=ci/.bazelrc build --config=rbe_cross_compile_${os}_${arch} \
            --override_repository=xla="${KOKORO_ARTIFACTS_DIR}"/xla \
            --test_env=JAX_NUM_GENERATED_CASES=25 \
            //tests:cpu_tests //tests:backend_independent_tests
else
      # Bazel test on RBE CPU. Only Linux x86_64 can run tests on RBE at the moment.
      jaxrun bazel --bazelrc=ci/.bazelrc test --config=rbe_${os}_${arch} \
            --override_repository=xla="${KOKORO_ARTIFACTS_DIR}"/xla \
            --test_env=JAX_NUM_GENERATED_CASES=25 \
            //tests:cpu_tests //tests:backend_independent_tests
fi