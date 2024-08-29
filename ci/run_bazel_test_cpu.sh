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

os=$(uname -s | awk '{print tolower($0)}')
arch=$(uname -m)

# If running on Mac or Linux Aarch64, we only build the test targets. This is
# because these platforms do not have native RBE support. Instead, we
# cross-compile them on Linux x86 RBE pool. Since running the tests on a single
# machine can take a long time, we skip running them on these platforms.
if [[ $os == "darwin" ]] || ( [[ $os == "linux" ]] && [[ $arch == "aarch64" ]] ); then
      jaxrun bazel --bazelrc=ci/.bazelrc build --config=rbe_cross_compile_${os}_${arch} \
            --override_repository=xla="${JAXCI_XLA_GIT_DIR}" \
            --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
            --test_env=JAX_NUM_GENERATED_CASES=25 \
            //tests:cpu_tests //tests:backend_independent_tests
else
      jaxrun bazel --bazelrc=ci/.bazelrc test --config=rbe_${os}_${arch} \
            --override_repository=xla="${JAXCI_XLA_GIT_DIR}" \
            --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
            --test_env=JAX_NUM_GENERATED_CASES=25 \
            //tests:cpu_tests //tests:backend_independent_tests
fi