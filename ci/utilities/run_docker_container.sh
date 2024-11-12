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
# Set up the Docker container and start it for JAX CI jobs.
#
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exu -o history -o allexport

source ./ci/envs/docker_images.env

# Keep the existing "jax" container if it's already present.
if ! docker container inspect jax >/dev/null 2>&1 ; then
  # Simple retry logic for docker-pull errors. Sleeps if a pull fails.
  # Pulling an already-pulled container image will finish instantly, so
  # repeating the command costs nothing.
  docker pull "$JAXCI_DOCKER_IMAGE" || sleep 15
  docker pull "$JAXCI_DOCKER_IMAGE"

  if [[ "$(uname -s)" =~ "MSYS_NT" ]]; then
    # Docker on Windows doesn't support the `host` networking mode, and so
    # port-forwarding is required for the container to detect it's running on GCE.
    export IP_ADDR=$(powershell -command "(Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias 'vEthernet (nat)').IPAddress")
    netsh interface portproxy add v4tov4 listenaddress=$IP_ADDR listenport=80 connectaddress=169.254.169.254 connectport=80
    JAXCI_DOCKER_ARGS="$JAXCI_DOCKER_ARGS -e GCE_METADATA_HOST=$IP_ADDR"
  else
    # The volume mapping flag below shares the user's gcloud credentials, if any,
    # with the container, in case the user has credentials stored there.
    # This would allow Bazel to authenticate for RBE.
    # Note: JAX's CI does not have any credentials stored there.
    JAXCI_DOCKER_ARGS="$JAXCI_DOCKER_ARGS -v $HOME/.config/gcloud:/root/.config/gcloud"
  fi

  # Start the container. `user_set_jaxci_envs` is read after `jax_ci_envs` to
  # allow the user to override any environment variables set by JAXCI_ENV_FILE.
  docker run $JAXCI_DOCKER_ARGS --name jax \
      -w /jax -itd --rm \
      -v "$JAXCI_JAX_GIT_DIR:/jax" \
      "$JAXCI_DOCKER_IMAGE" \
    bash

  if [[ "$(uname -s)" =~ "MSYS_NT" ]]; then
    # Allow requests from the container.
    CONTAINER_IP_ADDR=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' jax)
    netsh advfirewall firewall add rule name="Allow Metadata Proxy" dir=in action=allow protocol=TCP localport=80 remoteip="$CONTAINER_IP_ADDR"
  fi
fi