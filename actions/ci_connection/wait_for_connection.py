# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wait for an SSH connection from a user, if a wait was requested."""

import asyncio
import logging
import os
import sys
import time

from get_labels import retrieve_labels

# Check if debug logging should be enabled for the script:
# WAIT_FOR_CONNECTION_DEBUG is a custom variable.
# RUNNER_DEBUG and ACTIONS_RUNNER_DEBUG are GH env vars, which can be set
# in various ways, one of them - enabling debug logging from the UI, when
# triggering a run:
# https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/store-information-in-variables#default-environment-variables
# https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/troubleshooting-workflows/enabling-debug-logging#enabling-runner-diagnostic-logging
_SHOW_DEBUG = bool(
  os.getenv("WAIT_FOR_CONNECTION_DEBUG",
            os.getenv("RUNNER_DEBUG",
                      os.getenv("ACTIONS_RUNNER_DEBUG")))
)
logging.basicConfig(level=logging.INFO if not _SHOW_DEBUG else logging.DEBUG,
                    format="%(levelname)s: %(message)s", stream=sys.stderr)


def _is_true_like_env_var(var_name: str) -> bool:
  var_val = os.getenv(var_name, "").lower()
  negative_choices = {"0", "false", "n", "no", "none", "null", "n/a"}
  if var_val and var_val not in negative_choices:
    return True
  return False


def should_halt_for_connection() -> bool:
  """Check if the workflow should wait, due to inputs, vars, and labels."""

  logging.info("Checking if the workflow should be halted for a connection...")

  if not _is_true_like_env_var("INTERACTIVE_CI"):
    logging.info("INTERACTIVE_CI env var is not "
                 "set, or is set to a false-like value in the workflow")
    return False

  explicit_halt_requested = _is_true_like_env_var("HALT_DISPATCH_INPUT")
  if explicit_halt_requested:
    logging.info("Halt for connection requested via "
                 "explicit `halt-dispatch-input` input")
    return True

  # Check if any of the relevant labels are present
  labels = retrieve_labels(print_to_stdout=False)

  # Note: there's always a small possibility these labels may change on the
  # repo/org level, in which case, they'd need to be updated below as well.

  # TODO(belitskiy): Add the ability to halt on CI error.

  always_halt_label = "CI Connection Halt - Always"
  if always_halt_label in labels:
    logging.info(f"Halt for connection requested via presence "
                 f"of the {always_halt_label!r} label")
    return True

  attempt = int(os.getenv("GITHUB_RUN_ATTEMPT"))
  halt_on_retry_label = "CI Connection Halt - On Retry"
  if attempt > 1 and halt_on_retry_label in labels:
    logging.info(f"Halt for connection requested via presence "
                 f"of the {halt_on_retry_label!r} label, "
                 f"due to workflow run attempt being 2+ ({attempt})")
    return True

  return False


class WaitInfo:
  pre_connect_timeout = 10 * 60  # 10 minutes for initial connection
  # allow for reconnects, in case no 'closed' message is received
  re_connect_timeout = 15 * 60  # 15 minutes for reconnects
  # Dynamic, depending on whether a connection was established, or not
  timeout = pre_connect_timeout
  last_time = time.time()
  waiting_for_close = False
  stop_event = asyncio.Event()


async def process_message(reader, writer):
  data = await reader.read(1024)
  message = data.decode().strip()
  if message == "keep_alive":
    logging.info("Keep-alive received")
    WaitInfo.last_time = time.time()
  elif message == "connection_closed":
    WaitInfo.waiting_for_close = True
    WaitInfo.stop_event.set()
  elif message == "connection_established":
    WaitInfo.last_time = time.time()
    WaitInfo.timeout = WaitInfo.re_connect_timeout
    logging.info("SSH connection detected.")
  else:
    logging.warning(f"Unknown message received: {message!r}")
  writer.close()


async def wait_for_connection(host: str = 'localhost',
                              port: int = 12455):
  # Print out the data required to connect to this VM
  runner_name = os.getenv("HOSTNAME")
  cluster = os.getenv("CONNECTION_CLUSTER")
  location = os.getenv("CONNECTION_LOCATION")
  ns = os.getenv("CONNECTION_NS")
  actions_path = os.getenv("GITHUB_ACTION_PATH")

  logging.info("Googler connection only\n"
               "See go/ml-github-actions:ssh for details")
  logging.info(
    f"Connection string: ml-actions-connect "
    f"--runner={runner_name} "
    f"--ns={ns} "
    f"--loc={location} "
    f"--cluster={cluster} "
    f"--halt_directory={actions_path}"
  )

  server = await asyncio.start_server(process_message, host, port)
  addr = server.sockets[0].getsockname()
  terminate = False

  logging.info(f"Listening for connection notifications on {addr}...")
  async with server:
    while not WaitInfo.stop_event.is_set():
      await asyncio.wait([asyncio.create_task(WaitInfo.stop_event.wait())],
                         timeout=60,
                         return_when=asyncio.FIRST_COMPLETED)

      if WaitInfo.waiting_for_close:
        msg = "Connection was terminated."
        terminate = True
      elif elapsed > WaitInfo.timeout:
        terminate = True
        msg = f"No connection for {WaitInfo.timeout} seconds."

      if terminate:
        logging.info(f"{msg} Shutting down the waiting process...")
        server.close()
        await server.wait_closed()
        break

      elapsed = time.time() - WaitInfo.last_time
      logging.info(f"Time since last keep-alive: {int(elapsed)}s")

    logging.info("Waiting process terminated.")


if __name__ == "__main__":
  if not should_halt_for_connection():
    logging.info("No conditions for halting the workflow"
                 "for connection were met")
    exit()
  asyncio.run(wait_for_connection())
