#!/usr/bin/python
#
# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Helper script for building JAX's libjax easily.

import argparse
import asyncio
import logging
import os
import platform
import sys
import textwrap

from tools import command, utils


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BANNER = r"""
     _   _  __  __
    | | / \ \ \/ /
 _  | |/ _ \ \  /
| |_| / ___ \/  \
 \___/_/   \/_/\_\

"""

EPILOG = """
From the root directory of the JAX repository, run
  python build/build.py [jaxlib | jax-cuda-plugin | jax-cuda-pjrt | jax-rocm-plugin | jax-rocm-pjrt]

  to build one of: jaxlib, jax-cuda-plugin, jax-cuda-pjrt, jax-rocm-plugin, jax-rocm-pjrt
or
  python build/build.py requirements_update to update the requirements_lock.txt
"""

# Define the build target for each artifact.
ARTIFACT_BUILD_TARGET_DICT = {
    "jaxlib": "//jaxlib/tools:build_wheel",
    "jax-cuda-plugin": "//jaxlib/tools:build_gpu_kernels_wheel",
    "jax-cuda-pjrt": "//jaxlib/tools:build_gpu_plugin_wheel",
    "jax-rocm-plugin": "//jaxlib/tools:build_gpu_kernels_wheel",
    "jax-rocm-pjrt": "//jaxlib/tools:build_gpu_plugin_wheel",
}


def add_python_version_argument(parser: argparse.ArgumentParser):
  parser.add_argument(
      "--python_version",
      type=str,
      choices=["3.10", "3.11", "3.12", "3.13"],
      default=f"{sys.version_info.major}.{sys.version_info.minor}",
      help="Python version to use",
  )


def add_cuda_version_argument(parser: argparse.ArgumentParser):
  parser.add_argument(
      "--cuda_version",
      type=str,
      default="12.3.2",
      help="CUDA version to use",
  )


def add_cudnn_version_argument(parser: argparse.ArgumentParser):
  parser.add_argument(
      "--cudnn_version",
      type=str,
      default="9.1.1",
      help="cuDNN version to use",
  )


def add_disable_nccl_argument(parser: argparse.ArgumentParser):
  parser.add_argument(
      "--disable_nccl",
      action="store_true",
      help="Whether to disable NCCL for CUDA/ROCM builds.",
  )


def add_cuda_compute_capabilities_argument(parser: argparse.ArgumentParser):
  parser.add_argument(
      "--cuda_compute_capabilities",
      type=str,
      default=None,
      help="A comma-separated list of CUDA compute capabilities to support.",
  )


def add_build_cuda_with_clang_argument(parser: argparse.ArgumentParser):
  parser.add_argument(
      "--build_cuda_with_clang",
      action="store_true",
      help=(
          "Should we build CUDA code using Clang? The default value "
          "is to build CUDA with NVCC."
      ),
  )


def add_rocm_version_argument(parser: argparse.ArgumentParser):
  parser.add_argument(
      "--rocm_version",
      type=str,
      default="60",
      help="ROCm version to use",
  )


def add_rocm_amdgpu_targets_argument(parser: argparse.ArgumentParser):
  parser.add_argument(
      "--rocm_amdgpu_targets",
      type=str,
      default="gfx900,gfx906,gfx908,gfx90a,gfx1030",
      help="A comma-separated list of ROCm amdgpu targets to support.",
  )


def add_rocm_path_argument(parser: argparse.ArgumentParser):
  parser.add_argument(
      "--rocm_path",
      type=str,
      default="",
      help="Path to the ROCm toolkit.",
  )


def add_requirements_nightly_update_argument(parser: argparse.ArgumentParser):
  parser.add_argument(
      "--nightly_update",
      action="store_true",
      help=(
          "If true, updates requirements_lock.txt for a corresponding version"
          " of Python and will consider dev, nightly and pre-release versions"
          " of packages."
      ),
  )


def add_global_arguments(parser: argparse.ArgumentParser):
  """Adds all the global arguments that applies to all the CLI subcommands."""
  parser.add_argument(
      "--bazel_path",
      type=str,
      default="",
      help="""
        Path to the Bazel binary to use. The default is to find bazel via the
        PATH; if none is found, downloads a fresh copy of Bazel from GitHub.
        """,
  )

  parser.add_argument(
      "--bazel_startup_options",
      type=str,
      default="",
      help=(
          "Space separated list of additional startup options to pass to Bazel."
          "E.g. --bazel_startup_options='--nobatch --noclient_debug'"
      ),
  )

  parser.add_argument(
      "--bazel_build_options",
      type=str,
      default="",
      help=(
          "Space separated list of additional build options to pass to"
          " Bazel.E.g. --bazel_build_options='--local_resources=HOST_CPUS"
          " --nosandbox_debug'"
      ),
  )

  parser.add_argument(
      "--dry_run",
      action="store_true",
      help="Prints the Bazel command that is going will be executed.",
  )

  parser.add_argument(
      "--verbose",
      action="store_true",
      help="Produce verbose output for debugging.",
  )


def add_artifact_subcommand_global_arguments(parser: argparse.ArgumentParser):
  """Adds all the global arguments that applies to the artifact subcommands."""
  parser.add_argument(
      "--ci_mode",
      action="store_true",
      help="""
          When set, the CLI will assume the build is being run in CI or CI
          like environment and will use the "ci_" configs in the .bazelrc.
          """,
  )

  parser.add_argument(
      "--request_rbe",
      action="store_true",
      help="""
        If set, the build will try to use RBE. Currently, only Linux x86
        and Windows support RBE. RBE also requires GCP authentication set
        up to work.
        """,
  )

  parser.add_argument(
      "--editable",
      action="store_true",
      help="Create an 'editable' build instead of a wheel.",
  )

  parser.add_argument(
      "--enable_mkl_dnn",
      action="store_true",
      help="Enables MKL-DNN.",
  )

  parser.add_argument(
      "--target_cpu",
      default=None,
      help="CPU platform to target. Default is the same as the host machine. ",
  )

  parser.add_argument(
      "--target_cpu_features",
      choices=["release", "native", "default"],
      default="release",
      help=(
          "What CPU features should we target? 'release' enables CPU "
          "features that should be enabled for a release build, which on "
          "x86-64 architectures enables AVX. 'native' enables "
          "-march=native, which generates code targeted to use all "
          "features of the current machine. 'default' means don't opt-in "
          "to any architectural features and use whatever the C compiler "
          "generates by default."
      ),
  )

  parser.add_argument(
      "--clang_path",
      type=str,
      default="",
      help="Path to the Clang binary to use.",
  )

  parser.add_argument(
      "--local_xla_path",
      type=str,
      default=os.environ.get("JAXCI_XLA_GIT_DIR", ""),
      help="""
      Path to local XLA repository to use. If not set, Bazel uses the XLA
      at the pinned version in workspace.bzl.
      """,
  )

  parser.add_argument(
      "--output_path",
      type=str,
      default=os.environ.get(
          "JAXCI_OUTPUT_DIR", os.path.join(os.getcwd(), "dist")
      ),
      help="Directory to which the JAX wheel packages should be written.",
  )


def parse_and_append_bazel_options(
    bazel_command: command.CommandBuilder, bazel_options: str
):
  """Parses the bazel options and appends them to the bazel command."""
  for option in bazel_options.split(" "):
    bazel_command.append(option)


async def main():
  parser = argparse.ArgumentParser(
      description=(
          "CLI for building one of the following packages from source: jaxlib, "
          "jax-cuda-plugin, jax-cuda-pjrt, jax-rocm-plugin, jax-rocm-pjrt."
          "and for updating the requirements_lock.txt files"
      ),
      epilog=EPILOG,
  )

  # Create subparsers for jax, jaxlib, plugin, pjrt and requirements_update
  subparsers = parser.add_subparsers(dest="command", required=True)

  # requirements_update subcommand
  requirements_update_parser = subparsers.add_parser(
      "requirements_update", help="Updates the requirements_lock.txt files"
  )
  add_python_version_argument(requirements_update_parser)
  add_requirements_nightly_update_argument(requirements_update_parser)
  add_global_arguments(requirements_update_parser)

  # jaxlib subcommand
  jaxlib_parser = subparsers.add_parser(
      "jaxlib", help="Builds the jaxlib package."
  )
  add_python_version_argument(jaxlib_parser)
  add_artifact_subcommand_global_arguments(jaxlib_parser)
  add_global_arguments(jaxlib_parser)

  # jax-cuda-plugin subcommand
  cuda_plugin_parser = subparsers.add_parser(
      "jax-cuda-plugin", help="Builds the jax-cuda-plugin package."
  )
  add_python_version_argument(cuda_plugin_parser)
  add_build_cuda_with_clang_argument(cuda_plugin_parser)
  add_cuda_version_argument(cuda_plugin_parser)
  add_cudnn_version_argument(cuda_plugin_parser)
  add_cuda_compute_capabilities_argument(cuda_plugin_parser)
  add_disable_nccl_argument(cuda_plugin_parser)
  add_artifact_subcommand_global_arguments(cuda_plugin_parser)
  add_global_arguments(cuda_plugin_parser)

  # jax-cuda-pjrt subcommand
  cuda_pjrt_parser = subparsers.add_parser(
      "jax-cuda-pjrt", help="Builds the jax-cuda-pjrt package."
  )
  add_build_cuda_with_clang_argument(cuda_pjrt_parser)
  add_cuda_version_argument(cuda_pjrt_parser)
  add_cudnn_version_argument(cuda_pjrt_parser)
  add_cuda_compute_capabilities_argument(cuda_pjrt_parser)
  add_disable_nccl_argument(cuda_pjrt_parser)
  add_artifact_subcommand_global_arguments(cuda_pjrt_parser)
  add_global_arguments(cuda_pjrt_parser)

  # jax-rocm-plugin subcommand
  rocm_plugin_parser = subparsers.add_parser(
      "jax-rocm-plugin", help="Builds the jax-rocm-plugin package."
  )
  add_python_version_argument(rocm_plugin_parser)
  add_rocm_version_argument(rocm_plugin_parser)
  add_rocm_amdgpu_targets_argument(rocm_plugin_parser)
  add_rocm_path_argument(rocm_plugin_parser)
  add_disable_nccl_argument(rocm_plugin_parser)
  add_artifact_subcommand_global_arguments(rocm_plugin_parser)
  add_global_arguments(rocm_plugin_parser)

  # jax-rocm-pjrt subcommand
  rocm_pjrt_parser = subparsers.add_parser(
      "jax-rocm-pjrt", help="Builds the jax-rocm-pjrt package."
  )
  add_rocm_version_argument(rocm_pjrt_parser)
  add_rocm_amdgpu_targets_argument(rocm_pjrt_parser)
  add_rocm_path_argument(rocm_pjrt_parser)
  add_disable_nccl_argument(rocm_pjrt_parser)
  add_artifact_subcommand_global_arguments(rocm_pjrt_parser)
  add_global_arguments(rocm_pjrt_parser)

  arch = platform.machine().lower()
  os_name = platform.system().lower()

  args = parser.parse_args()

  logger.info("%s", BANNER)

  if args.verbose:
    logging.getLogger().setLevel(logging.DEBUG)
    logger.info("Verbose logging enabled")

  logger.info(
      "Building %s for %s %s...",
      args.command,
      os_name,
      arch,
  )

  bazel_path, bazel_version = utils.get_bazel_path(args.bazel_path)

  logging.debug("Bazel path: %s", bazel_path)
  logging.debug("Bazel version: %s", bazel_version)

  executor = command.SubprocessExecutor()

  # Start constructing the Bazel command
  bazel_command = command.CommandBuilder(bazel_path)

  if args.bazel_startup_options:
    logging.debug(
        "Additional Bazel startup options: %s", args.bazel_startup_options
    )
    parse_and_append_bazel_options(bazel_command, args.bazel_startup_options)

  bazel_command.append("run")

  if hasattr(args, "python_version"):
    logging.debug("Hermetic Python version: %s", args.python_version)
    bazel_command.append(
        f"--repo_env=HERMETIC_PYTHON_VERSION={args.python_version}"
    )

  if args.command == "requirements_update":
    if args.bazel_build_options:
      logging.debug(
          "Using additional build options: %s", args.bazel_build_options
      )
      parse_and_append_bazel_options(bazel_command, args.bazel_build_options)

    if args.nightly_update:
      logging.debug(
          "--nightly_update is set. Bazel will run"
          " //build:requirements_nightly.update"
      )
      bazel_command.append("//build:requirements_nightly.update")
    else:
      bazel_command.append("//build:requirements.update")

    await executor.run(bazel_command.command, args.dry_run)
    sys.exit(0)

  wheel_cpus = {
      "darwin_arm64": "arm64",
      "darwin_x86_64": "x86_64",
      "ppc": "ppc64le",
      "aarch64": "aarch64",
  }
  target_cpu = (
      wheel_cpus[args.target_cpu] if args.target_cpu is not None else arch
  )

  if args.ci_mode:
    logging.debug(
        "Running in CI mode. Run the CLI with --help for more details on what"
        " this means."
    )
    bazelrc_config = utils.get_bazelrc_config(
        os_name, arch, args.command, args.request_rbe
    )
    logging.debug("Using --config=%s from .bazelrc", bazelrc_config)
    bazel_command.append(f"--config={bazelrc_config}")
  else:
    clang_path = args.clang_path or utils.get_clang_path_or_exit()
    logging.debug("Using Clang as the compiler, clang path: %s", clang_path)
    bazel_command.append(f"--action_env CLANG_COMPILER_PATH={clang_path}")
    bazel_command.append(f"--repo_env CC={clang_path}")
    bazel_command.append(f"--repo_env BAZEL_COMPILER={clang_path}")
    bazel_command.append("--config=clang")

    if args.target_cpu:
      logging.debug("Target CPU: %s", args.target_cpu)
      bazel_command.append(f"--cpu={args.target_cpu}")

    if args.enable_mkl_dnn:
      logging.debug("Enabling MKL DNN")
      bazel_command.append("--config=mkl_open_source_only")

    if hasattr(args, "disable_nccl") and args.disable_nccl:
      logging.debug("Disabling NCCL")
      bazel_command.append("--config=nonccl")

    if args.target_cpu_features == "release":
      logging.debug(
          "Using release cpu features: --config=avx_%s",
          "windows" if utils.is_windows() else "posix",
      )
      if arch == "x86_64":
        bazel_command.append(
            "--config=avx_windows"
            if utils.is_windows()
            else "--config=avx_posix"
        )
    elif args.target_cpu_features == "native":
      if utils.is_windows():
        logger.warning(
            "--target_cpu_features=native is not supported on Windows;"
            " ignoring."
        )
      else:
        logging.debug("Using native cpu features: --config=native_arch_posix")
        bazel_command.append("--config=native_arch_posix")
    else:
      logging.debug("Using default cpu features")

    if "cuda" in args.command:
      bazel_command.append("--config=cuda")

      if args.build_cuda_with_clang:
        logging.debug("Building CUDA with Clang")
        bazel_command.append("--config=build_cuda_with_clang")
        bazel_command.append(
            f"--action_env=CLANG_CUDA_COMPILER_PATH={clang_path}"
        )
      else:
        logging.debug("Building CUDA with NVCC")
        bazel_command.append("--config=build_cuda_with_nvcc")

      if args.cuda_version:
        logging.debug("Hermetic CUDA version: %s", args.cuda_version)
        bazel_command.append(
            f"--repo_env=HERMETIC_CUDA_VERSION={args.cuda_version}"
        )
      if args.cudnn_version:
        logging.debug("Hermetic cuDNN version: %s", args.cudnn_version)
        bazel_command.append(
            f"--repo_env=HERMETIC_CUDNN_VERSION={args.cudnn_version}"
        )
      if args.cuda_compute_capabilities:
        logging.debug(
            "Hermetic CUDA compute capabilities: %s",
            args.cuda_compute_capabilities,
        )
        bazel_command.append(
            "--repo_env"
            f" HERMETIC_CUDA_COMPUTE_CAPABILITIES={args.cuda_compute_capabilities}"
        )

    if "rocm" in args.command:
      bazel_command.append("--config=rocm")
      bazel_command.append("--action_env=CLANG_COMPILER_PATH={clang_path}")

      if args.rocm_path:
        logging.debug("ROCm tookit path: %s", args.rocm_path)
        bazel_command.append(f"--action_env ROCM_PATH='{args.rocm_path}'")
      if args.rocm_amdgpu_targets:
        logging.debug("ROCm AMD GPU targets: %s", args.rocm_amdgpu_targets)
        bazel_command.append(
            f"--action_env TF_ROCM_AMDGPU_TARGETS={args.rocm_amdgpu_targets}"
        )

  if args.local_xla_path:
    logging.debug("Local XLA path: %s", args.local_xla_path)
    bazel_command.append(f"--override_repository=xla={args.local_xla_path}")

  if args.bazel_build_options:
    logging.debug(
        "Additional Bazel build options: %s", args.bazel_build_options
    )
    parse_and_append_bazel_options(bazel_command, args.bazel_build_options)

  # Append the build target to the Bazel command.
  build_target = ARTIFACT_BUILD_TARGET_DICT[args.command]
  bazel_command.append(build_target)

  # Read output directory. Default is store the artifacts in the "dist/"
  # directory in JAX's GitHub repository root.
  output_path = args.output_path

  # If running on Windows, adjust the paths for compatibility.
  if os_name == "windows":
    output_path, target_cpu = utils.adjust_paths_for_windows(
        output_path, target_cpu
    )

  logger.debug("Artifacts output directory: %s", output_path)

  bazel_command.append("--")

  if args.editable:
    logger.debug("Building an editable build")
    output_path = os.path.join(output_path, args.command)
    bazel_command.append("--editable")

  bazel_command.append(f"--output_path={output_path}")
  bazel_command.append(f"--cpu={target_cpu}")

  if "cuda" in args.command:
    bazel_command.append("--enable-cuda=True")
    cuda_major_version = args.cuda_version.split(".")[0]
    bazel_command.append(f"--platform_version={cuda_major_version}")

  if "rocm" in args.command:
    bazel_command.append("--enable-rocm=True")
    bazel_command.append(f"--platform_version={args.rocm_version}")

  git_hash = utils.get_githash()
  bazel_command.append(f"--jaxlib_git_hash={git_hash}")

  await executor.run(bazel_command.command, args.dry_run)


if __name__ == "__main__":
  asyncio.run(main())
