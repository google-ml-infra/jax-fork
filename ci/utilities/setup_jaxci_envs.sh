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
# Source "JAXCI_" environment variables.

# If a JAX CI env file has not been passed, exit.
if [[ -z "$1" ]]; then
    echo "ERROR: No JAX CI env file passed."
    echo "source_jaxci_envs.sh requires that a path to a JAX CI env file to be"
    echo "passed as an argument when invoking the build scripts."
    echo "Pass in a corresponding env file from the ci/envs/run_tests"
    echo "directory to continue."
    exit 1
fi

# -e: abort script if one command fails
# -x: log all commands
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exu -o history -o allexport
source "$1"