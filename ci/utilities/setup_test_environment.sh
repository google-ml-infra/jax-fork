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
#
# Set up the environment for JAX tests

if [[ $JAXCI_RUN_BAZEL_GPU_TEST_LOCAL == 1 ]]; then
  # Install the `jaxlib`, `jax-cuda-plugin` and `jax-pjrt` wheels.
  jaxrun bash -c "$JAXCI_PYTHON -m pip install $JAXCI_OUTPUT_DIR/*.whl"

  # Install JAX package at the current commit.
  # TODO(srnitin): Check if this is needed when running Bazel tests.
  jaxrun "$JAXCI_PYTHON" -m pip install -U -e .
fi
