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
# Common setup for all JAX builds.
#
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o pipefail: entire command fails if pipe fails. watch out for yes | ...
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -euo pipefail -o history -o allexport

if [[ -z "${ENV_FILE+dummy}" ]]; then
  echo "Setup script requires an ENV_FILE to be set."
  echo "If you are looking to build JAX artifacts, please set ENV_FILE to an"
  echo "env file in the ci/envs/build_artifacts directory."
  echo "If you are looking to run JAX tests, please set ENV_FILE to an"
  echo "env file in the ci/envs/run_tests directory."
  exit 1
fi
set -x
source "$ENV_FILE"

# Decide whether to use the release tag. JAX CI jobs build from the main
# branch by default. 
if [[ -n "$JAXCI_RELEASE_TAG" ]]; then
  git checkout tags/"$JAXCI_RELEASE_TAG"
fi

# Setup jaxrun, a helper function for executing steps that can either be run
# locally or run under Docker. setup_docker.sh, below, redefines it as "docker
# exec".
# Important: "jaxrun foo | bar" is "( jaxrun foo ) | bar", not "jaxrun (foo | bar)".
# Therefore, "jaxrun" commands cannot include pipes -- which is
# probably for the better. If a pipe is necessary for something, it is probably
# complex. Write a well-documented script under utilities/ to encapsulate the
# functionality instead.
jaxrun() { "$@"; }

# All builds except for Mac run under Docker.
# GitHub actions do not need to invoke this script. It always runs in a Docker
# container. The image and the runner type are set in the workflow file.
if [[ "$JAXCI_USE_DOCKER" == 1 ]] || [[ "$(uname -s)" != "Darwin" ]]; then
  source ./ci/utilities/setup_docker.sh
fi

# Set Bazel configs. Temporary; in the final version of the CL, after the build
# CLI has been reworked, these will be removed. The build CLI will handle
# setting the Bazel configs.
if [[ "$JAXCI_RUN_TESTS" != 1 ]]; then
  if [[ "$(uname -s)" == "Linux" && $(uname -m) == "x86_64" ]]; then
    if [[ "$JAXCI_BUILD_JAXLIB_ENABLE" == 1 ]]; then
      export BAZEL_CONFIG_CPU=rbe_linux_x86_64
    else
      export BAZEL_CONFIG_CUDA=rbe_linux_x86_64_cuda
    fi
  elif [[ "$(uname -s)" == "Linux" && $(uname -m) == "aarch64" ]]; then
    if [[ "$JAXCI_BUILD_JAXLIB_ENABLE" == 1 ]]; then
      export BAZEL_CONFIG_CPU=ci_linux_aarch64_cpu
    else
      export BAZEL_CONFIG_CUDA=ci_linux_aarch64_cuda
    fi
  elif [[ "$(uname -s)" =~ "MSYS_NT" && $(uname -m) == "x86_64" ]]; then
    export BAZEL_CONFIG_CPU=rbe_windows_x86_64
  elif [[ "$(uname -s)" == "Darwin" && $(uname -m) == "x86_64" ]]; then
    export BAZEL_CONFIG_CPU=ci_darwin_x86_64
  elif [[ "$(uname -s)" == "Darwin" && $(uname -m) == "arm64" ]]; then
    export BAZEL_CONFIG_CPU=ci_darwin_arm64
  else
    echo "Unsupported platform: $(uname -s) $(uname -m)"
    exit 1
  fi
fi

if [[ "$JAXCI_RUN_TESTS" == 1 ]]; then
  jaxrun bash -c "$JAXCI_PYTHON -m pip install $JAXCI_OUTPUT_DIR/*.whl"
fi

# TODO: cleanup steps