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
#
# Set up the build environment for JAX CI jobs. This script depends on the
# environment variables set in `setup_envs.sh`.
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o pipefail: entire command fails if pipe fails. watch out for yes | ...
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exuo pipefail -o history -o allexport

# Pre-emptively mark the JAX git directory as safe. This is necessary for JAX CI
# jobs running on Linux runners in GitHub Actions. Without this, git complains
# that the directory has dubious ownership and refuses to run any commands.
# Avoid running on Windows runners as git runs into issues with not being able
# to lock the config file. Other git commands seem to work on the Windows
# runners so we can skip this step for Windows.
# TODO(b/375073267): Remove this once we understand why git repositories are
# being marked as unsafe inside the self-hosted runners.
if [[ ! $(uname -s) =~ "MSYS_NT" ]]; then
  git config --global --add safe.directory $JAXCI_JAX_GIT_DIR
fi

function clone_main_xla() {
  echo "Cloning XLA at HEAD to $(pwd)/xla"
  git clone --depth=1 https://github.com/openxla/xla.git $(pwd)/xla
  export JAXCI_XLA_GIT_DIR=$(pwd)/xla
}

# Clone XLA at HEAD if required.
if [[ "$JAXCI_CLONE_MAIN_XLA" == 1 ]]; then
  # Clone only if $(pwd)/xla does not exist to avoid failure on re-runs.
  if [[ ! -d $(pwd)/xla ]]; then
    clone_main_xla
  fi
fi

# If a XLA commit is provided, check out XLA at that commit.
if [[ ! -z "$JAXCI_XLA_COMMIT" ]]; then
  # Clone XLA at HEAD if a path to local XLA is not provided.
  if [[ -z "$JAXCI_XLA_GIT_DIR" ]]; then
    clone_main_xla
  fi
  pushd "$JAXCI_XLA_GIT_DIR"

  git fetch --depth=1 origin "$JAXCI_XLA_COMMIT"
  echo "JAXCI_XLA_COMMIT is set. Checking out XLA at $JAXCI_XLA_COMMIT"
  git checkout "$JAXCI_XLA_COMMIT"

  popd
fi

if [[ ! -z ${JAXCI_XLA_GIT_DIR} ]]; then
  echo "INFO: Overriding XLA to be read from $JAXCI_XLA_GIT_DIR instead of the"
  echo "pinned version in the WORKSPACE."
  echo "If you would like to revert this behavior, unset JAXCI_CLONE_MAIN_XLA"
  echo "and JAXCI_XLA_COMMIT in your environment. Note that the Bazel RBE test"
  echo "commands overrides the XLA repository and thus require a local copy of"
  echo "XLA to run."
fi

# For Windows, convert MSYS Linux-like paths to Windows paths.
if [[ $(uname -s) =~ "MSYS_NT" ]]; then
  echo 'Converting MSYS Linux-like paths to Windows paths (for Docker, Python, etc.)'
  # Convert all "_DIR" variables to Windows paths.
  source <(python3 ./ci/utilities/convert_msys_paths_to_win_paths.py)
fi

# When running Pytests, we need to install the wheels locally.
if [[ "$JAXCI_INSTALL_WHEELS_LOCALLY" == 1 ]]; then
   echo "Installing wheels locally..."
   source ./ci/utilities/install_wheels_locally.sh
fi