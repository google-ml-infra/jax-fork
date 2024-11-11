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
# CLI for building jaxlib, jax-cuda-plugin, jax-cuda-pjrt, jax-rocm-plugin,
# jax-rocm-pjrt and for updating the requirements_lock.txt files.

import argparse
import asyncio
import logging
import os
import platform
import sys
import copy

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

  bazel_group = parser.add_argument_group('Bazel Options')
  bazel_group.add_argument(
      "--bazel_path",
      type=str,
      default="",
      help="""
        Path to the Bazel binary to use. The default is to find bazel via the
        PATH; if none is found, downloads a fresh copy of Bazel from GitHub.
        """,
  )

  bazel_group.add_argument(
      "--bazel_startup_options",
      action="append",
      default=[],
      help="""
        Additional startup options to pass to Bazel, can be specified multiple
        times to pass multiple options.
        E.g. --bazel_startup_options='--nobatch'
        """,
  )

  bazel_group.add_argument(
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
      help="Prints the Bazel command that is going to be executed.",
  )

  parser.add_argument(
      "--verbose",
      action="store_true",
      help="Produce verbose output for debugging.",
  )


def add_artifact_subcommand_global_arguments(parser: argparse.ArgumentParser):
  """Adds all the global arguments that applies to the artifact subcommands."""
  parser.add_argument(
      "--wheel_list",
      type=str,
      default="jaxlib",
      help=
        f"""
        A comma seprated list of JAX artifacts to build. E.g: --wheel_list="jaxlib",
        --wheel_list="jaxlib,jax-cuda-plugin", etc.
        Valid options are: {','.join(ARTIFACT_BUILD_TARGET_DICT.keys())}
        """,
  )

  parser.add_argument(
      "--editable",
      action="store_true",
      help="Create an 'editable' build instead of a wheel.",
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
      does not build the artifacts.
      """,
  )

  # CUDA Options
  cuda_group = parser.add_argument_group('CUDA Options')
  cuda_group.add_argument(
      "--cuda_version",
      type=str,
      default=None,
      help=
        """
        Hermetic CUDA version to use. Default is to use the version specified
        in the .bazelrc.
        """,
  )

  cuda_group.add_argument(
      "--cudnn_version",
      type=str,
      default=None,
      help=
        """
        Hermetic cuDNN version to use. Default is to use the version specified
        in the .bazelrc.
        """,
  )

  cuda_group.add_argument(
      "--disable_nccl",
      action="store_true",
      help="Should NCCL be disabled?",
  )

  cuda_group.add_argument(
      "--cuda_compute_capabilities",
      type=str,
      default=None,
      help=
        """
        A comma-separated list of CUDA compute capabilities to support. Default
        is to use the values specified in the .bazelrc.
        """,
  )

  cuda_group.add_argument(
      "--build_cuda_with_clang",
      action="store_true",
      help="""
        Should CUDA code be compiled using Clang? The default behavior is to
        compile CUDA with NVCC. Ignored if --use_ci_bazelrc_flags is set, CI
        builds always build CUDA with NVCC in CI builds.
        """,
  )

  # ROCm Options
  rocm_group = parser.add_argument_group('ROCm Options')
  rocm_group.add_argument(
      "--rocm_version",
      type=str,
      default="60",
      help="ROCm version to use",
  )

  rocm_group.add_argument(
      "--rocm_amdgpu_targets",
      type=str,
      default="gfx900,gfx906,gfx908,gfx90a,gfx940,gfx941,gfx942,gfx1030,gfx1100",
      help="A comma-separated list of ROCm amdgpu targets to support.",
  )

  rocm_group.add_argument(
      "--rocm_path",
      type=str,
      default="",
      help="Path to the ROCm toolkit.",
  )

  # Compile Options
  compile_group = parser.add_argument_group('Compile Options')
  compile_group.add_argument(
      "--use_ci_bazelrc_flags",
      action="store_true",
      help="""
        When set, the CLI will assume the build is being run in CI or CI like
        environment and will use the "rbe_/ci_" configs in the .bazelrc. These
        configs apply release features and set a custom C++ Clang toolchain.
        Only supported for jaxlib and CUDA builds.
        """,
  )

  compile_group.add_argument(
      "--clang_path",
      type=str,
      default="",
      help="""
        Path to the Clang binary to use. Ignored if --use_ci_bazelrc_flags, CI
        bazelrc flags set a custom Clang toolchain.
        """,
  )

  compile_group.add_argument(
      "--disable_mkl_dnn",
      action="store_true",
      help="""
        Disables MKL-DNN. Ignored if --use_ci_bazelrc_flags is set, CI bazelrc
        flags enable MKL-DNN as default.
        """,
  )

  compile_group.add_argument(
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

  compile_group.add_argument(
      "--target_cpu",
      default=None,
      help="CPU platform to target. Default is the same as the host machine. ",
  )

  compile_group.add_argument(
      "--local_xla_path",
      type=str,
      default=os.environ.get("JAXCI_XLA_GIT_DIR", ""),
      help="""
        Path to local XLA repository to use. If not set, Bazel uses the XLA at
        the pinned version in workspace.bzl.
        """,
  )

def apply_compile_flags_non_ci(bazel_command: command.CommandBuilder, wheel: str, clang_path: str, disable_mkl_dnn: bool, build_cuda_with_clang: bool,\
    target_cpu_features: str, os_name: str, arch: str):
    clang_path = clang_path or utils.get_clang_path_or_exit()
    logging.debug("Using Clang as the compiler, clang path: %s", clang_path)
    # Use double quotes around clang path to avoid path issues on Windows.
    bazel_command.append(f"--action_env=CLANG_COMPILER_PATH=\"{clang_path}\"")
    bazel_command.append(f"--repo_env=CC=\"{clang_path}\"")
    bazel_command.append(f"--repo_env=BAZEL_COMPILER=\"{clang_path}\"")
    # Do not apply --config=clang on Mac as these settings do not apply to
    # Apple Clang.
    if os_name != "darwin":
      bazel_command.append("--config=clang")

    if not disable_mkl_dnn:
      logging.debug("Enabling MKL DNN")
      bazel_command.append("--config=mkl_open_source_only")

    if "cuda" in wheel:
      bazel_command.append("--config=cuda")
      bazel_command.append(
            f"--action_env=CLANG_CUDA_COMPILER_PATH=\"{clang_path}\""
        )
      if build_cuda_with_clang:
        logging.debug("Building CUDA with Clang")
        bazel_command.append("--config=build_cuda_with_clang")
      else:
        logging.debug("Building CUDA with NVCC")
        bazel_command.append("--config=build_cuda_with_nvcc")

    if target_cpu_features == "release":
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
    elif target_cpu_features == "native":
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

async def main():
  parser = argparse.ArgumentParser(
      description=r"""
        CLI for building JAX wheel packages from source: jaxlib,
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
  add_requirements_nightly_update_argument(requirements_update_parser)
  add_global_arguments(requirements_update_parser)

  # Build Artifact subcommand
  build_artifact_parser = subparsers.add_parser(
      "build_artifacts", help="Builds the jaxlib, plugin, PJRT artifact"
  )
  add_artifact_subcommand_global_arguments(build_artifact_parser)
  add_global_arguments(build_artifact_parser)

  arch = platform.machine()
  # Switch to lower case to match the case for the "ci_"/"rbe_" configs in the
  # .bazelrc.
  os_name = platform.system().lower()

  args = parser.parse_args()

  logger.info("%s", BANNER)

  if args.verbose:
    logging.getLogger().setLevel(logging.DEBUG)
    logger.info("Verbose logging enabled")

  bazel_path, bazel_version = utils.get_bazel_path(args.bazel_path)

  logging.debug("Bazel path: %s", bazel_path)
  logging.debug("Bazel version: %s", bazel_version)

  executor = command.SubprocessExecutor()

  # Start constructing the Bazel command
  bazel_command_base = command.CommandBuilder(bazel_path)

  if args.bazel_startup_options:
    logging.debug(
        "Additional Bazel startup options: %s", args.bazel_startup_options
    )
    for option in args.bazel_startup_options:
      bazel_command_base.append(option)

  bazel_command_base.append("run")

  if args.python_version:
    logging.debug("Hermetic Python version: %s", args.python_version)
    bazel_command_base.append(
        f"--repo_env=HERMETIC_PYTHON_VERSION={args.python_version}"
    )

  # Enable color in the Bazel output and verbose failures.
  bazel_command_base.append("--color=yes")
  bazel_command_base.append("--verbose_failures=true")

  # Requirements update subcommand execution
  if args.command == "requirements_update":
    requirements_command = copy.deepcopy(bazel_command_base)
    if args.bazel_build_options:
      logging.debug(
          "Using additional build options: %s", args.bazel_build_options
      )
      for option in args.bazel_build_options:
        requirements_command.append(option)

    if args.nightly_update:
      logging.debug(
          "--nightly_update is set. Bazel will run"
          " //build:requirements_nightly.update"
      )
      requirements_command.append("//build:requirements_nightly.update")
    else:
      requirements_command.append("//build:requirements.update")

    await executor.run(requirements_command.get_command_as_string(), args.dry_run)
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

  if args.local_xla_path:
    logging.debug("Local XLA path: %s", args.local_xla_path)
    bazel_command_base.append(f"--override_repository=xla=\"{args.local_xla_path}\"")

  if args.bazel_build_options:
    logging.debug(
        "Additional Bazel build options: %s", args.bazel_build_options
    )
    for option in args.bazel_build_options:
      bazel_command_base.append(option)

  if args.target_cpu:
    logging.debug("Target CPU: %s", args.target_cpu)
    bazel_command_base.append(f"--cpu={args.target_cpu}")

  if args.disable_nccl:
    logging.debug("Disabling NCCL")
    bazel_command_base.append("--config=nonccl")

  # Wheel build command execution
  for wheel in args.wheel_list.split(","):
    if wheel not in ARTIFACT_BUILD_TARGET_DICT.keys():
      logging.error("Incorrect wheel name provided: %s, valid choices are: %s", wheel, ",".join(ARTIFACT_BUILD_TARGET_DICT.keys()))
      continue
    wheel_build_command = copy.deepcopy(bazel_command_base)
    print("\n")
    logger.info(
      "Building %s for %s %s...",
      wheel,
      os_name,
      arch,
    ) 
    # If running in CI, we use the "ci_"/"rbe_" configs in the .bazelrc.
    # These set a custom C++ Clang toolchain and the CUDA compiler to NVCC
    # When not running in CI, we detect the path to Clang binary and pass it
    # to Bazel to use as the C++ compiler. NVCC is used as the CUDA compiler
    # unless the user explicitly sets --config=build_cuda_with_clang.
    if args.use_ci_bazelrc_flags and "rocm" not in wheel:
      bazelrc_config = utils.get_ci_bazelrc_config(os_name, arch.lower(), wheel)
      logging.info("--use_ci_bazelrc_flags is set, using --config=%s from .bazelrc", bazelrc_config)
      wheel_build_command.append(f"--config={bazelrc_config}")
    else:
      apply_compile_flags_non_ci(
          wheel_build_command,
          wheel,
          args.clang_path,
          args.disable_mkl_dnn,
          args.build_cuda_with_clang,
          args.target_cpu_features,
          os_name,
          arch,
      )

    if "cuda" in wheel:
      if args.cuda_version:
        logging.debug("Hermetic CUDA version: %s", args.cuda_version)
        wheel_build_command.append(
            f"--repo_env=HERMETIC_CUDA_VERSION={args.cuda_version}"
        )
      if args.cudnn_version:
        logging.debug("Hermetic cuDNN version: %s", args.cudnn_version)
        wheel_build_command.append(
            f"--repo_env=HERMETIC_CUDNN_VERSION={args.cudnn_version}"
        )
      if args.cuda_compute_capabilities:
        logging.debug(
            "Hermetic CUDA compute capabilities: %s",
            args.cuda_compute_capabilities,
        )
        wheel_build_command.append(
            f"--repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES={args.cuda_compute_capabilities}"
        )

    if "rocm" in wheel:
      wheel_build_command.append("--config=rocm")
      if args.rocm_path:
        logging.debug("ROCm tookit path: %s", args.rocm_path)
        wheel_build_command.append(f"--action_env=ROCM_PATH=\"{args.rocm_path}\"")
      if args.rocm_amdgpu_targets:
        logging.debug("ROCm AMD GPU targets: %s", args.rocm_amdgpu_targets)
        wheel_build_command.append(
            f"--action_env=TF_ROCM_AMDGPU_TARGETS={args.rocm_amdgpu_targets}"
        )

    if args.configure_only:
      with open(".jax_configure.bazelrc", "w") as f:
        jax_configure_options = utils.get_jax_configure_bazel_options(wheel_build_command.get_command_as_list())
        if not jax_configure_options:
          logging.error("Error retrieving the Bazel options to be written to .jax_configure.bazelrc, exiting.")
          sys.exit(1)
        f.write(jax_configure_options)
        logging.info("Bazel options written to .jax_configure.bazelrc")
        logging.info("--configure_only is set so not running any Bazel commands.")
    else:
      # Append the build target to the Bazel command.
      build_target = ARTIFACT_BUILD_TARGET_DICT[wheel]
      wheel_build_command.append(build_target)

      wheel_build_command.append("--")

      output_path = args.output_path
      logger.debug("Artifacts output directory: %s", output_path)

      if args.editable:
        logger.debug("Building an editable build")
        output_path = os.path.join(output_path, wheel)
        wheel_build_command.append("--editable")

      wheel_build_command.append(f'--output_path="{output_path}"')
      wheel_build_command.append(f"--cpu={target_cpu}")

      if "cuda" in wheel:
        wheel_build_command.append("--enable-cuda=True")
        if args.cuda_version:
          cuda_major_version = args.cuda_version.split(".")[0]
        else:
          cuda_major_version = utils.get_cuda_major_version()
        wheel_build_command.append(f"--platform_version={cuda_major_version}")

      if "rocm" in wheel:
        wheel_build_command.append("--enable-rocm=True")
        wheel_build_command.append(f"--platform_version={args.rocm_version}")

      git_hash = utils.get_githash()
      wheel_build_command.append(f"--jaxlib_git_hash={git_hash}")

      await executor.run(wheel_build_command.get_command_as_string(), args.dry_run)


if __name__ == "__main__":
  asyncio.run(main())