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
# Runs Pyest CPU tests. Requires a jaxlib wheel to be present
# inside $JAXCI_OUTPUT_DIR (../dist)
#
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exu -o history -o allexport

# Inherit default JAXCI environment variables.
source ci/envs/default.env

# Install jaxlib wheel on the system. 
export JAXCI_INSTALL_WHEELS_LOCALLY=1

# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

export PY_COLORS=1
export JAX_SKIP_SLOW_TESTS=true

"$JAXCI_PYTHON" -c "import jax; print(jax.default_backend()); print(jax.devices()); print(len(jax.devices()))"

export TF_CPP_MIN_LOG_LEVEL=0
echo "Running CPU tests..."
"$JAXCI_PYTHON" -m pytest -n auto --tb=short --maxfail=20 tests examples