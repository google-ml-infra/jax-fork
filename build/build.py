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
import logging
import os
import platform
import textwrap

from tools import utils


from tools import command, utils


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def write_bazelrc(*, remote_build,
                  cuda_version, cudnn_version, rocm_toolkit_path,
                  cpu, cuda_compute_capabilities,
                  rocm_amdgpu_targets, target_cpu_features,
                  wheel_cpu, enable_mkl_dnn, use_clang, clang_path,
                  clang_major_version, python_version,
                  enable_cuda, enable_nccl, enable_rocm,
                  use_cuda_nvcc):

  with open("../.jax_configure.bazelrc", "w") as f:
    if not remote_build:
      f.write(textwrap.dedent("""\
        build --strategy=Genrule=standalone
        """))

    if use_clang:
      f.write(f'build --action_env CLANG_COMPILER_PATH="{clang_path}"\n')
      f.write(f'build --repo_env CC="{clang_path}"\n')
      f.write(f'build --repo_env BAZEL_COMPILER="{clang_path}"\n')
      f.write('build --copt=-Wno-error=unused-command-line-argument\n')
      if clang_major_version in (16, 17, 18):
        # Necessary due to XLA's old version of upb. See:
        # https://github.com/openxla/xla/blob/c4277a076e249f5b97c8e45c8cb9d1f554089d76/.bazelrc#L505
        f.write("build --copt=-Wno-gnu-offsetof-extensions\n")

    if rocm_toolkit_path:
      f.write("build --action_env ROCM_PATH=\"{rocm_toolkit_path}\"\n"
              .format(rocm_toolkit_path=rocm_toolkit_path))
    if rocm_amdgpu_targets:
      f.write(
        f'build:rocm --action_env TF_ROCM_AMDGPU_TARGETS="{rocm_amdgpu_targets}"\n')
    if cpu is not None:
      f.write(f"build --cpu={cpu}\n")

    if target_cpu_features == "release":
      if wheel_cpu == "x86_64":
        f.write("build --config=avx_windows\n" if utils.is_windows()
                else "build --config=avx_posix\n")
    elif target_cpu_features == "native":
      if utils.is_windows():
        print("--target_cpu_features=native is not supported on Windows; ignoring.")
      else:
        f.write("build --config=native_arch_posix\n")

    if enable_mkl_dnn:
      f.write("build --config=mkl_open_source_only\n")
    if enable_cuda:
      f.write("build --config=cuda\n")
      if use_cuda_nvcc:
        f.write("build --config=build_cuda_with_nvcc\n")
      else:
        f.write("build --config=build_cuda_with_clang\n")
      f.write(f"build --action_env=CLANG_CUDA_COMPILER_PATH={clang_path}\n")
      if not enable_nccl:
        f.write("build --config=nonccl\n")
      if cuda_version:
        f.write("build --repo_env HERMETIC_CUDA_VERSION=\"{cuda_version}\"\n"
                .format(cuda_version=cuda_version))
      if cudnn_version:
        f.write("build --repo_env HERMETIC_CUDNN_VERSION=\"{cudnn_version}\"\n"
                .format(cudnn_version=cudnn_version))
      if cuda_compute_capabilities:
        f.write(
          f'build:cuda --repo_env HERMETIC_CUDA_COMPUTE_CAPABILITIES="{cuda_compute_capabilities}"\n')
    if enable_rocm:
      f.write("build --config=rocm_base\n")
      if not enable_nccl:
        f.write("build --config=nonccl\n")
      if use_clang:
        f.write("build --config=rocm\n")
        f.write(f"build --action_env=CLANG_COMPILER_PATH={clang_path}\n")
    if python_version:
      f.write(
        "build --repo_env HERMETIC_PYTHON_VERSION=\"{python_version}\"".format(
            python_version=python_version))
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
      help="Hermetic Python version to use",
  )


def add_cuda_version_argument(parser: argparse.ArgumentParser):
  parser.add_argument(
      "--cuda_version",
      type=str,
      default=None,
      help="Hermetic CUDA version to use",
  )


def add_cudnn_version_argument(parser: argparse.ArgumentParser):
  parser.add_argument(
      "--cudnn_version",
      type=str,
      default=None,
      help="Hermetic cuDNN version to use",
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
      help="A comma-separated list of CUDA compute capabilities to support.",
  )


def add_build_cuda_with_clang_argument(parser: argparse.ArgumentParser):
  parser.add_argument(
      "--build_cuda_with_clang",
      action="store_true",
      help="""
        Should CUDA code be compiled using Clang? The default behavior is to
        compile CUDA with NVCC. Ignored if --use_ci_bazelrc_flags is set, we
        always build CUDA with NVCC in CI builds.
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
      type=str,
      default="",
      help="""
        Space separated list of additional startup options to pass to Bazel
        E.g. --bazel_startup_options='--nobatch --noclient_debug'
        """,
  )

  parser.add_argument(
      "--bazel_build_options",
      type=str,
      default="",
      help="""
        Space separated list of additional build options to pass to Bazel
        E.g. --bazel_build_options='--local_resources=HOST_CPUS --nosandbox_debug'
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
        environment and will use the "rbe_/ci_" configs in the .bazelrc.
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
      "--enable_release_cpu_features",
      action="store_true",
      help="""
        Enables CPU features that should be enabled for a release build, which
        on x86-64 architectures enables AVX.
        """,
  )

  parser.add_argument(
      "--enable_native_cpu_features",
      action="store_true",
      help="""
        Enables -march=native, which generates code targeted to use all features
        of the current machine.
        """,
  )

  parser.add_argument(
      "--clang_path",
      type=str,
      default="",
      help="""
        Path to the Clang binary to use. Ignored if --use_ci_bazelrc_flags is
        set as we use a custom Clang toolchain in that case.
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
      default=os.environ.get(
          "JAXCI_OUTPUT_DIR", os.path.join(os.getcwd(), "dist")
      ),
      help="Directory to which the JAX wheel packages should be written.",
  )


def parse_and_append_bazel_options(bazel_command: command.CommandBuilder, bazel_options: str):
  """Parses the bazel options and appends them to the bazel command."""
  for option in bazel_options.split(" "):
    bazel_command.append(option)


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

  # Find a working Bazel.
  bazel_path, bazel_version = utils.get_bazel_path(args.bazel_path)
  print(f"Bazel binary path: {bazel_path}")
  print(f"Bazel version: {bazel_version}")

  if args.python_version:
    python_version = args.python_version
  else:
    python_bin_path = utils.get_python_bin_path(args.python_bin_path)
    print(f"Python binary path: {python_bin_path}")
    python_version = utils.get_python_version(python_bin_path)
    print("Python version: {}".format(".".join(map(str, python_version))))
    utils.check_python_version(python_version)
    python_version = ".".join(map(str, python_version))

  print("Use clang: {}".format("yes" if args.use_clang else "no"))
  clang_path = args.clang_path
  clang_major_version = None
  if args.use_clang:
    if not clang_path:
      clang_path = utils.get_clang_path_or_exit()
    print(f"clang path: {clang_path}")
    clang_major_version = utils.get_clang_major_version(clang_path)

  if args.target_cpu:
    logging.debug("Target CPU: %s", args.target_cpu)
    bazel_command.append(f"--cpu={args.target_cpu}")

  if args.enable_mkl_dnn:
    logging.debug("Enabling MKL DNN")
    bazel_command.append("--config=mkl_open_source_only")

  if hasattr(args, "disable_nccl") and args.disable_nccl:
    logging.debug("Disabling NCCL")
    bazel_command.append("--config=nonccl")

  write_bazelrc(
      remote_build=args.remote_build,
      cuda_version=args.cuda_version,
      cudnn_version=args.cudnn_version,
      rocm_toolkit_path=rocm_toolkit_path,
      cpu=args.target_cpu,
      cuda_compute_capabilities=args.cuda_compute_capabilities,
      rocm_amdgpu_targets=args.rocm_amdgpu_targets,
      target_cpu_features=args.target_cpu_features,
      wheel_cpu=wheel_cpu,
      enable_mkl_dnn=args.enable_mkl_dnn,
      use_clang=args.use_clang,
      clang_path=clang_path,
      clang_major_version=clang_major_version,
      python_version=python_version,
      enable_cuda=args.enable_cuda,
      enable_nccl=args.enable_nccl,
      enable_rocm=args.enable_rocm,
      use_cuda_nvcc=args.use_cuda_nvcc,
  )

  if args.requirements_update or args.requirements_nightly_update:
    if args.requirements_update:
      task = "//build:requirements.update"
    else:  # args.requirements_nightly_update
      task = "//build:requirements_nightly.update"
    update_command = ([bazel_path] + args.bazel_startup_options +
      ["run", "--verbose_failures=true", task, *args.bazel_options])
    print(" ".join(update_command))
    utils.shell(update_command)
    return

  if args.configure_only:
    return

  print("\nBuilding XLA and installing it in the jaxlib source tree...")

  command_base = (
    bazel_path,
    *args.bazel_startup_options,
    "run",
    "--verbose_failures=true",
    *args.bazel_options,
  )

  if args.build_gpu_plugin and args.editable:
    output_path_jaxlib, output_path_jax_pjrt, output_path_jax_kernel = (
        _get_editable_output_paths(output_path)
    )
    if arch in ["x86_64", "AMD64"]:
      bazel_command.append(
          "--config=avx_windows"
          if os_name == "windows"
          else "--config=avx_posix"
      )

  if args.build_gpu_kernel_plugin == "" and not args.build_gpu_pjrt_plugin:
    build_cpu_wheel_command = [
        *command_base,
        "//jaxlib/tools:build_wheel",
        "--",
        f"--output_path={output_path_jaxlib}",
        f"--jaxlib_git_hash={utils.get_githash()}",
        f"--cpu={wheel_cpu}",
    ]
    if args.build_gpu_plugin:
      build_cpu_wheel_command.append("--skip_gpu_kernels")
    if args.editable:
      build_cpu_wheel_command.append("--editable")
    print(" ".join(build_cpu_wheel_command))
    utils.shell(build_cpu_wheel_command)

  if args.build_gpu_plugin or (args.build_gpu_kernel_plugin == "cuda") or \
      (args.build_gpu_kernel_plugin == "rocm"):
    build_gpu_kernels_command = [
        *command_base,
        "//jaxlib/tools:build_gpu_kernels_wheel",
        "--",
        f"--output_path={output_path_jax_kernel}",
        f"--jaxlib_git_hash={utils.get_githash()}",
        f"--cpu={wheel_cpu}",
    ]
    if args.enable_cuda:
      build_gpu_kernels_command.append(f"--enable-cuda={args.enable_cuda}")
      build_gpu_kernels_command.append(f"--platform_version={args.gpu_plugin_cuda_version}")
    elif args.enable_rocm:
      build_gpu_kernels_command.append(f"--enable-rocm={args.enable_rocm}")
      build_gpu_kernels_command.append(f"--platform_version={args.gpu_plugin_rocm_version}")
    else:
      raise ValueError("Unsupported GPU plugin backend. Choose either 'cuda' or 'rocm'.")
    if args.editable:
      build_gpu_kernels_command.append("--editable")
    print(" ".join(build_gpu_kernels_command))
    utils.shell(build_gpu_kernels_command)

  if args.build_gpu_plugin or args.build_gpu_pjrt_plugin:
    build_pjrt_plugin_command = [
        *command_base,
        "//jaxlib/tools:build_gpu_plugin_wheel",
        "--",
        f"--output_path={output_path_jax_pjrt}",
        f"--jaxlib_git_hash={utils.get_githash()}",
        f"--cpu={wheel_cpu}",
    ]
    if args.enable_cuda:
      build_pjrt_plugin_command.append(f"--enable-cuda={args.enable_cuda}")
      build_pjrt_plugin_command.append(f"--platform_version={args.gpu_plugin_cuda_version}")
    elif args.enable_rocm:
      build_pjrt_plugin_command.append(f"--enable-rocm={args.enable_rocm}")
      build_pjrt_plugin_command.append(f"--platform_version={args.gpu_plugin_rocm_version}")
    else:
      raise ValueError("Unsupported GPU plugin backend. Choose either 'cuda' or 'rocm'.")
    if args.editable:
      build_pjrt_plugin_command.append("--editable")
    print(" ".join(build_pjrt_plugin_command))
    utils.shell(build_pjrt_plugin_command)

  utils.shell([bazel_path] + args.bazel_startup_options + ["shutdown"])


if __name__ == "__main__":
  asyncio.run(main())
