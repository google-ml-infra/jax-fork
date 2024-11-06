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
      help=
        """
        Hermetic Python version to use. Default is to use the version of the
        Python binary that executed the CLI.
        """,
  )


def add_cuda_version_argument(parser: argparse.ArgumentParser):
  parser.add_argument(
      "--cuda_version",
      type=str,
      default=None,
      help=
        """
        Hermetic CUDA version to use. Default is to use the version specified
        in the .bazelrc.
        """,
  )


def add_cudnn_version_argument(parser: argparse.ArgumentParser):
  parser.add_argument(
      "--cudnn_version",
      type=str,
      default=None,
      help=
        """
        Hermetic cuDNN version to use. Default is to use the version specified
        in the .bazelrc.
        """,
  )


def add_disable_nccl_argument(parser: argparse.ArgumentParser):
  parser.add_argument(
      "--disable_nccl",
      action="store_true",
      help="Should NCCL be disabled?",
  )


def add_cuda_compute_capabilities_argument(parser: argparse.ArgumentParser):
  parser.add_argument(
      "--cuda_compute_capabilities",
      type=str,
      default=None,
      help=
        """
        A comma-separated list of CUDA compute capabilities to support. Default
        is to use the values specified in the .bazelrc.
        """,
  )


def add_build_cuda_with_clang_argument(parser: argparse.ArgumentParser):
  parser.add_argument(
      "--build_cuda_with_clang",
      action="store_true",
      help="""
        Should CUDA code be compiled using Clang? The default behavior is to
        compile CUDA with NVCC. Ignored if --use_ci_bazelrc_flags is set, CI
        builds always build CUDA with NVCC in CI builds.
        """,
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
      help="""
        If true, updates requirements_lock.txt for a corresponding version of
        Python and will consider dev, nightly and pre-release versions of
        packages.
        """,
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
      action="append",
      default=[],
      help="""
        Additional startup options to pass to Bazel, can be specified multiple
        times to pass multiple options.
        E.g. --bazel_startup_options='--nobatch'
        """,
  )

  parser.add_argument(
      "--bazel_build_options",
      action="append",
      default=[],
      help="""
        Additional build options to pass to Bazel, can be specified multiple
        times to pass multiple options.
        E.g. --bazel_build_options='--local_resources=HOST_CPUS'
        """,
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
      "--use_ci_bazelrc_flags",
      action="store_true",
      help="""
        When set, the CLI will assume the build is being run in CI or CI like
        environment and will use the "rbe_/ci_" configs in the .bazelrc. These
        configs apply release features and set a custom C++ Clang toolchain.
        Only supported for jaxlib and CUDA builds.
        """,
  )

  parser.add_argument(
      "--editable",
      action="store_true",
      help="Create an 'editable' build instead of a wheel.",
  )

  parser.add_argument(
      "--disable_mkl_dnn",
      action="store_true",
      help="""
        Disables MKL-DNN. Ignored if --use_ci_bazelrc_flags is set, CI bazelrc
        flags enable MKL-DNN as default.
        """,
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
      help="""
        What CPU features should we target? Release enables CPU features that
        should be enabled for a release build, which on x86-64 architectures
        enables AVX. Native enables -march=native, which generates code targeted
        to use all features of the current machine. Default means don't opt-in
        to any architectural features and use whatever the C compiler generates
        by default. Ignored if --use_ci_bazelrc_flags is set, CI bazelrc flags
        enable release CPU features as default.
        """,
  )

  parser.add_argument(
      "--clang_path",
      type=str,
      default="",
      help="""
        Path to the Clang binary to use. Ignored if --use_ci_bazelrc_flags, CI
        bazelrc flags set a custom Clang toolchain.
        """,
  )

  parser.add_argument(
      "--local_xla_path",
      type=str,
      default=os.environ.get("JAXCI_XLA_GIT_DIR", ""),
      help="""
        Path to local XLA repository to use. If not set, Bazel uses the XLA at
        the pinned version in workspace.bzl.
        """,
  )

  parser.add_argument(
      "--output_path",
      type=str,
      default=os.path.join(os.getcwd(), "dist"),
      help="Directory to which the JAX wheel packages should be written.",
  )

  parser.add_argument(
    "--configure_only",
    action="store_true",
    help="""
      If true, writes the Bazel options to the .jax_configure.bazelrc file but
      does not build the artifacts. Ignored if --use_ci_bazelrc_flags is set.
      """,
  )


async def main():
  parser = argparse.ArgumentParser(
      description=r"""
        CLI for building one of the following packages from source: jaxlib,
        jax-cuda-plugin, jax-cuda-pjrt, jax-rocm-plugin, jax-rocm-pjrt and for
        updating the requirements_lock.txt files
        """,
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

  arch = platform.machine()
  # Switch to lower case to match the case for the "ci_"/"rbe_" configs in the
  # .bazelrc.
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
    for option in args.bazel_startup_options:
      bazel_command.append(option)

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
      for option in args.bazel_build_options:
        bazel_command.append(option)

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

  # If running in CI, we use the "ci_"/"rbe_" configs in the .bazelrc.
  # These set a custom C++ Clang toolchain and the CUDA compiler to NVCC
  # When not running in CI, we detect the path to Clang binary and pass it
  # to Bazel to use as the C++ compiler. NVCC is used as the CUDA compiler
  # unless the user explicitly sets --config=build_cuda_with_clang.
  if args.use_ci_bazelrc_flags and "rocm" not in args.command:
    bazelrc_config = utils.get_ci_bazelrc_config(os_name, arch.lower(), args.command)
    logging.debug("--use_ci_bazelrc_flags is set, using --config=%s from .bazelrc", bazelrc_config)
    bazel_command.append(f"--config={bazelrc_config}")
    bazel_command.append("--color=yes")
  else:
    clang_path = args.clang_path or utils.get_clang_path_or_exit()
    logging.debug("Using Clang as the compiler, clang path: %s", clang_path)
    # Use double quotes around clang path to avoid path issues on Windows.
    bazel_command.append(f'--action_env=CLANG_COMPILER_PATH="{clang_path}"')
    bazel_command.append(f'--repo_env=CC="{clang_path}"')
    bazel_command.append(f'--repo_env=BAZEL_COMPILER="{clang_path}"')
    bazel_command.append("--config=clang")

    if not args.disable_mkl_dnn:
      logging.debug("Enabling MKL DNN")
      bazel_command.append("--config=mkl_open_source_only")

    if "cuda" in args.command:
      bazel_command.append("--config=cuda")
      bazel_command.append(
            f'--action_env=CLANG_CUDA_COMPILER_PATH="{clang_path}"'
        )
      if args.build_cuda_with_clang:
        logging.debug("Building CUDA with Clang")
        bazel_command.append("--config=build_cuda_with_clang")
      else:
        logging.debug("Building CUDA with NVCC")
        bazel_command.append("--config=build_cuda_with_nvcc")

    if args.target_cpu_features == "release":
      logging.debug(
          "Using release cpu features: --config=avx_%s",
          "windows" if os_name == "windows" else "posix",
      )
      if arch in ["x86_64", "AMD64"]:
        bazel_command.append(
            "--config=avx_windows"
            if os_name == "windows"
            else "--config=avx_posix"
        )
    elif args.target_cpu_features == "native":
      if os_name == "windows":
        logger.warning(
            "--target_cpu_features=native is not supported on Windows;"
            " ignoring."
        )
      else:
        logging.debug("Using native cpu features: --config=native_arch_posix")
        bazel_command.append("--config=native_arch_posix")
    else:
      logging.debug("Using default cpu features")

  if args.target_cpu:
    logging.debug("Target CPU: %s", args.target_cpu)
    bazel_command.append(f"--cpu={args.target_cpu}")

  if hasattr(args, "disable_nccl") and args.disable_nccl:
    logging.debug("Disabling NCCL")
    bazel_command.append("--config=nonccl")

  if "cuda" in args.command:
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
          f"--repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES={args.cuda_compute_capabilities}"
      )

  if "rocm" in args.command:
    bazel_command.append("--config=rocm")

    if args.rocm_path:
      logging.debug("ROCm tookit path: %s", args.rocm_path)
      bazel_command.append(f'--action_env=ROCM_PATH="{args.rocm_path}"')
    if args.rocm_amdgpu_targets:
      logging.debug("ROCm AMD GPU targets: %s", args.rocm_amdgpu_targets)
      bazel_command.append(
          f"--action_env=TF_ROCM_AMDGPU_TARGETS={args.rocm_amdgpu_targets}"
      )

  if args.local_xla_path:
    logging.debug("Local XLA path: %s", args.local_xla_path)
    bazel_command.append(f'--override_repository=xla="{args.local_xla_path}"')

  if args.bazel_build_options:
    logging.debug(
        "Additional Bazel build options: %s", args.bazel_build_options
    )
    for option in args.bazel_build_options:
      bazel_command.append(option)

  if not args.use_ci_bazelrc_flags:
    with open(".jax_configure.bazelrc", "w") as f:
      jax_configure_options = utils.get_jax_configure_bazel_options(bazel_command.command)
      f.write(jax_configure_options)
      logging.debug("Bazel options written to .jax_configure.bazelrc")
    if args.configure_only:
      logging.debug("--configure_only is set, exiting without running any Bazel commands.")
      sys.exit(0)

  # Append the build target to the Bazel command.
  build_target = ARTIFACT_BUILD_TARGET_DICT[args.command]
  bazel_command.append(build_target)

  bazel_command.append("--")

  output_path = args.output_path
  logger.debug("Artifacts output directory: %s", output_path)

  if args.editable:
    logger.debug("Building an editable build")
    output_path = os.path.join(output_path, args.command)
    bazel_command.append("--editable")

  bazel_command.append(f'--output_path="{output_path}"')
  bazel_command.append(f"--cpu={target_cpu}")

  if "cuda" in args.command:
    bazel_command.append("--enable-cuda=True")
    if args.cuda_version:
      cuda_major_version = args.cuda_version.split(".")[0]
    else:
      cuda_major_version = utils.get_cuda_major_version()
    bazel_command.append(f"--platform_version={cuda_major_version}")

  if "rocm" in args.command:
    bazel_command.append("--enable-rocm=True")
    bazel_command.append(f"--platform_version={args.rocm_version}")

  git_hash = utils.get_githash()
  bazel_command.append(f"--jaxlib_git_hash={git_hash}")

  await executor.run(bazel_command.command, args.dry_run)


if __name__ == "__main__":
  asyncio.run(main())
