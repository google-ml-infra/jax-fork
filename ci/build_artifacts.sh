#!/bin/bash
# Copyright 2024 The JAX Authors.
##
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
# Source JAXCI environment variables.
source "ci/utilities/setup_jaxci_envs.sh" "$1"
# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

# Build the jax artifact
if [[ "$JAXCI_BUILD_JAX" == 1 ]]; then
  python3 -m build --outdir $JAXCI_OUTPUT_DIR
fi

# Build the jaxlib CPU artifact
if [[ "$JAXCI_BUILD_JAXLIB" == 1 ]]; then
  python3 build/build.py jaxlib --ci_mode --use_rbe --python_version=$JAXCI_HERMETIC_PYTHON_VERSION --verbose
fi

# Build the jax-cuda-plugin artifact
if [[ "$JAXCI_BUILD_PLUGIN" == 1 ]]; then
  python3 build/build.py jax-cuda-plugin --ci_mode --use_rbe --python_version=$JAXCI_HERMETIC_PYTHON_VERSION --verbose
fi

# Build the jax-cuda-pjrt artifact
if [[ "$JAXCI_BUILD_PJRT" == 1 ]]; then
  python3 build/build.py jax-cuda-pjrt --ci_mode --use_rbe --verbose
fi

# After building `jaxlib`, `jaxcuda-plugin`, and `jax-cuda-pjrt`, we run
# `auditwheel show` to ensure manylinux compliance.
if  [[ "$JAXCI_RUN_AUDITWHEEL" == 1 ]]; then
  ./ci/utilities/run_auditwheel.sh
fi
