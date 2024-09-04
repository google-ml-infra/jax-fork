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

# If we are building artifacts and the user has passed in the name of the
# artifact to build, set the ENV_FILE to the corresponding env file.
if [[ -n "$1" ]] && [[ $0 =~ "build_artifacts.sh" ]]; then
  export ENV_FILE="ci/envs/build_artifacts/$1"
fi

# If the user has not passed in an ENV_FILE nor has called "build_artifacts.sh"
# with an artifact name, exit.
if [[ -z "${ENV_FILE}" ]]; then
    echo "ENV_FILE is not set."
    echo "Setup script requires an ENV_FILE to be set."
    echo "If you are looking to build JAX artifacts, please set ENV_FILE to an"
    echo "env file in the ci/envs/build_artifacts directory."
    echo "If you are looking to run JAX tests, please set ENV_FILE to an"
    echo "env file in the ci/envs/run_tests directory."
    exit 1
fi

# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o pipefail: entire command fails if pipe fails. watch out for yes | ...
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exuo pipefail -o history -o allexport
source "$ENV_FILE"

# Pre-emptively mark the git directory as safe. This is necessary for JAX CI
# jobs running on GitHub Actions. Without this, git complains that the directory
# has dubious ownership and refuses to run any commands.
git config --global --add safe.directory $JAXCI_JAX_GIT_DIR

# When building release artifacts, check out the release tag. JAX CI jobs build
# from the main branch by default. 
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

# When running tests, we need to check out XLA at HEAD.
if [[ -z ${JAXCI_XLA_GIT_DIR} ]] && [[ "$JAXCI_CLONE_MAIN_XLA" == 1 ]]; then
    if [[ ! -d $(pwd)/xla ]]; then
      echo "Checking out XLA..."
      jaxrun git clone --depth=1 https://github.com/openxla/xla.git $(pwd)/xla
      echo "Using XLA from $(pwd)/xla"
    fi
    export JAXCI_XLA_GIT_DIR=$(pwd)/xla
fi

# If a path to XLA is provided, use that to build JAX or run tests.
if [[ ! -z ${JAXCI_XLA_GIT_DIR} ]]; then
  echo "Using XLA from $JAXCI_XLA_GIT_DIR"
fi

# If a XLA commit is provided, check out XLA at that commit.
if [[ ! -z "$JAXCI_XLA_COMMIT" ]]; then
  jaxrun pushd "$JAXCI_XLA_GIT_DIR"

  jaxrun git fetch --depth=1 origin "$JAXCI_XLA_COMMIT"
  jaxrun git checkout "$JAXCI_XLA_COMMIT"
  jaxrun echo "XLA git hash: $(git rev-parse HEAD)"

  jaxrun popd
fi

# All CI builds except for Mac run under Docker.
# Jobs running on GitHub actions do not invoke this script. They define the
# Docker image via the `container` field in the workflow file.
if [[ "$JAXCI_SETUP_DOCKER" == 1 ]]; then
  source ./ci/utilities/setup_docker.sh
fi

# If we are running tests, set up the test environment.
if [[ "$JAXCI_INSTALL_WHEELS_LOCALLY" == 1 ]]; then
   source ./ci/utilities/install_wheels_locally.sh
fi

# TODO: cleanup steps