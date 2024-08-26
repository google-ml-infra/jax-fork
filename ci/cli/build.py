#!/usr/bin/python
import argparse
import asyncio
import logging
import os
import platform
import collections
import sys
import subprocess
from helpers import command, tools

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ArtifactBuildSpec = collections.namedtuple(
    "ArtifactBuildSpec",
    ["bazel_build_target", "wheel_binary"],
)

ARTIFACT_BUILD_TARGET_DICT = {
    "jaxlib": ArtifactBuildSpec("//jaxlib/tools:build_wheel", "bazel-bin/jaxlib/tools/build_wheel"),
    "jax-cuda-plugin": ArtifactBuildSpec("//jaxlib/tools:build_gpu_kernels_wheel", "bazel-bin/jaxlib/tools/build_gpu_kernels_wheel"),
    "jax-cuda-pjrt": ArtifactBuildSpec("//jaxlib/tools:build_gpu_plugin_wheel", "bazel-bin/jaxlib/tools/build_gpu_plugin_wheel"),
}

def add_python_argument(parser: argparse.ArgumentParser):
  """Add Python version argument to the parser."""
  parser.add_argument(
      "--python_version",
      type=str,
      choices=["3.10", "3.11", "3.12"],
      default="3.12",
      help="Python version to use",
  )

# Target system is assumed to be the host sytem (auto-detected) unless
# specified otherwise, e.g. for cross-compile builds
# allow override to pass in custom flags for certain builds like the RBE
# jobs
def add_system_argument(parser: argparse.ArgumentParser):
  """Add Target System argument to the parser."""
  parser.add_argument(
      "--target_system",
      type=str,
      default="",
      choices=["linux_x86_64", "linux_aarch64", "darwin_x86_64", "darwin_arm64", "windows_x86_64"],
      help="Target system to build for",
  )

def add_cuda_argument(parser: argparse.ArgumentParser):
  """Add CUDA version argument to the parser."""
  parser.add_argument(
      "--cuda_version",
      type=str,
      default="12.3.2",
      help="CUDA version to use",
  )

def add_cudnn_argument(parser: argparse.ArgumentParser):
  """Add cuDNN version argument to the parser."""
  parser.add_argument(
      "--cudnn_version",
      type=str,
      default="9.1.1",
      help="cuDNN version to use",
  )

def get_bazelrc_config(os_name: str, arch: str, artifact: str, mode:str, use_rbe: bool):
  """Returns the bazelrc config for the given architecture, OS, and build type."""
  bazelrc_config = f"{os_name}_{arch}"

  # When the CLI is run by invoking ci/build_artifacts.sh, the CLI runs in CI
  # mode by default and will use one of the "ci_" configs in the .bazelrc. We
  # want to run certain CI builds with RBE and we also want to allow users the
  # flexibility to build JAX artifacts either by running the CLI or by running
  # ci/build_artifacts.sh. Because RBE requires permissions, we cannot enable it
  # by default in ci/build_artifacts.sh. Instead, we do not set `--use_rbe` in
  # build_artifacts.sh and have the CI builds set JAXCI_USE_RBE to 1 to enable
  # RBE.
  if os.environ.get("JAXCI_USE_RBE", "0") == "1":
    use_rbe = True

  # In CI, we want to use RBE where possible. At the moment, RBE is only
  # supported on Linux x86 and Windows. If an user is requesting RBE, the CLI
  # will use RBE if the host system supports it, otherwise it will use the
  # local config.
  if use_rbe and (os_name == "linux" or os_name == "windows") and arch == "x86_64":
    bazelrc_config = "rbe_" + bazelrc_config
  elif mode == "local":
    if use_rbe:
      logger.warning("RBE is not supported on %s_%s. Using Local config instead.", os_name, arch)
    if os_name == "linux" and arch == "aarch64" and artifact == "jaxlib":
      logger.info("Linux Aarch64 CPU builds do not have custom local config in JAX's root .bazelrc. Running with default configs.")
      bazelrc_config = ""
      return bazelrc_config
    bazelrc_config = "local_" + bazelrc_config
  else:
    if use_rbe:
      logger.warning("RBE is not supported on %s_%s. Using CI config instead.", os_name, arch)
    elif (os_name == "linux" or os_name == "windows") and arch == "x86_64":
      logger.info("RBE support is available for this platform. If you want to use RBE and have the required permissions, run the CLI with `--use_rbe` or set `JAXCI_USE_RBE=1`")
    bazelrc_config = "ci_" + bazelrc_config

  if artifact == "jax-cuda-plugin" or artifact == "jax-cuda-pjrt":
    bazelrc_config = bazelrc_config + "_cuda"

  return bazelrc_config

def get_jaxlib_git_hash():
  """Returns the git hash of the current repository."""
  res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
  return res.stdout

def check_whether_running_tests():
  """
  Returns True if running tests, False otherwise. When running tests, JAX
  artifacts are built with `JAX_ENABLE_X64=0` and the XLA repository is checked
  out at HEAD instead of the pinned version.
  """
  return os.environ.get("JAXCI_RUN_TESTS", "0") == "1"

async def main():
  parser = argparse.ArgumentParser(
      description=(
          "JAX CLI for building/testing jaxlib, jaxl-cuda-plugin, and jax-cuda-pjrt."
      ),
  )

  parser.add_argument(
      "--mode",
      type=str,
      choices=["ci", "local"],
      default="local",
      help=
        """
        Flags as requesting a CI or CI like build.  Setting this flag to CI
        will assume multiple settings expected in CI builds. These are set by
        the CI options in .bazelrc. To see best how this flag resolves you can
        run the artifact of choice with "--mode=[ci|local] --dry-run" to get the
        commands issued to Bazel for that artifact.
        """,
  )

  parser.add_argument(
      "--build_target_only",
      action="store_true",
      help="If set, the tool will only build the target and not the wheel.",
  )

  parser.add_argument(
      "--bazel_path",
      type=str,
      default="",
      help=
        """
        Path to the Bazel binary to use. The default is to find bazel via the
        PATH; if none is found, downloads a fresh copy of Bazelisk from 
        GitHub.
        """,
  )

  parser.add_argument(
      "--use_rbe",
      action="store_true",
      help=
        """
        If set, the build will use RBE where possible. Currently, only Linux x86
        and Windows builds can use RBE. On other platforms, setting this flag will
        be a no-op. RBE requires permissions to JAX's remote worker pool. Only
        Googlers and CI builds can use RBE.
        """,
  )

  parser.add_argument(
    "--use_clang",
    action="store_true",
    help=
      """
      If set, the build will use Clang as the C++ compiler. Requires Clang to
      be present on the PATH or a path is given with --clang_path. CI builds use
      Clang by default.
      """,
  )

  parser.add_argument(
    "--clang_path",
    type=str,
    default="",
    help=
      """
      Path to the Clang binary to use. If not set and --use_clang is set, the
      build will attempt to find Clang on the PATH.
      """,
  )

  parser.add_argument(
    "--local_xla_path",
    type=str,
    default=os.environ.get("JAXCI_XLA_GIT_DIR", ""),
    help=
      """
      Path to local XLA repository to use. If not set, Bazel uses the XLA
      at the pinned version in workspace.bzl.
      """,
  )
  
  parser.add_argument(
      "--dry_run",
      action="store_true",
      help="Prints the Bazel command that is going will be invoked.",
  )
  parser.add_argument("--verbose", action="store_true", help="Verbose output")

  global_args, remaining_args = parser.parse_known_args()

  # Create subparsers for jax, jaxlib, plugin, pjrt
  subparsers = parser.add_subparsers(
      dest="command", required=True, help="Artifact to build"
  )

  # Jaxlib subcommand
  jaxlib_parser = subparsers.add_parser("jaxlib", help="Builds the jaxlib package.")
  add_python_argument(jaxlib_parser)
  add_system_argument(jaxlib_parser)

  # jax-cuda-plugin subcommand
  plugin_parser = subparsers.add_parser("jax-cuda-plugin", help="Builds the jax-cuda-plugin package.")
  add_python_argument(plugin_parser)
  add_cuda_argument(plugin_parser)
  add_cudnn_argument(plugin_parser)
  add_system_argument(plugin_parser)

  # jax-cuda-pjrt subcommand
  pjrt_parser = subparsers.add_parser("jax-cuda-pjrt", help="Builds the jax-cuda-pjrt package.")
  add_cuda_argument(pjrt_parser)
  add_cudnn_argument(pjrt_parser)
  add_system_argument(pjrt_parser)

  # Get the host systems architecture
  arch = platform.machine()
  # On Windows, this returns "amd64" instead of "x86_64. However, they both
  # are essentially the same.
  if arch.lower() == "amd64":
    arch = "x86_64"

  # Get the host system OS
  os_name = platform.system().lower()

  args = parser.parse_args(remaining_args)

  for key, value in vars(global_args).items():
    setattr(args, key, value)

  logger.info(
      "Building %s for %s %s...",
      args.command,
      os_name,
      arch,
  )

  # Only jaxlib and jax-cuda-plugin are built for a specific python version
  if args.command == "jaxlib" or args.command == "jax-cuda-plugin":
    logger.info("Using Python version %s", args.python_version)

  if args.command == "jax-cuda-plugin" or args.command == "jax-cuda-pjrt":
    logger.info("Using CUDA version %s", args.cuda_version)
    logger.info("Using cuDNN version %s", args.cudnn_version)

  # Find the path to Bazel
  bazel_path = tools.get_bazel_path(args.bazel_path)

  executor = command.SubprocessExecutor()

  bazel_command = command.CommandBuilder(bazel_path)
  # Temporary; when we make the new scripts as the default we can remove this.
  bazel_command.append("--bazelrc=ci/.bazelrc")

  bazel_command.append("build")

  if args.use_clang or args.clang_path:
      # Find the path to Clang
    clang_path = tools.get_clang_path(args.clang_path)
    if clang_path:
      bazel_command.append(f"--action_env CLANG_COMPILER_PATH='{clang_path}'")
      bazel_command.append(f"--repo_env CC='{clang_path}'")
      bazel_command.append(f"--repo_env BAZEL_COMPILER='{clang_path}'")
      bazel_command.append("--config=clang")

  if args.mode == "ci":
    logging.info("Running in CI mode. Run the CLI with --help for more details on what this means.")

  # JAX's .bazelrc has custom configs for each build type, architecture, and
  # OS. Fetch the appropriate config and pass it to Bazel. A special case is
  # when building for Linux Aarch64, which does not have a custom local config
  # in JAX's .bazelrc. In this case, we build with the default configs.
  bazelrc_config = get_bazelrc_config(os_name, arch, args.command, args.mode, args.use_rbe)
  if bazelrc_config:
    bazel_command.append(f"--config={bazelrc_config}")

  # Check if we are running tests or if a local XLA path is set.
  # When running tests, JAX arifacts and tests are run with XLA at head.
  if check_whether_running_tests() or args.local_xla_path:
    bazel_command.append(f"--override_repository=xla='{args.local_xla_path}'")

  if hasattr(args, "python_version"):
    bazel_command.append(f"--repo_env=HERMETIC_PYTHON_VERSION={args.python_version}")

  # Set the CUDA and cuDNN versions if they are not the default.
  if hasattr(args, "cuda_version") and args.cuda_version != "12.3.2":
    bazel_command.append(f"--repo_env=HERMETIC_CUDA_VERSION={args.cuda_version}")

  if hasattr(args, "cudnn_version") and args.cudnn_version != "9.1.1":
    bazel_command.append(f"--repo_env=HERMETIC_CUDNN_VERSION={args.cudnn_version}")

  build_target, wheel_binary = ARTIFACT_BUILD_TARGET_DICT[args.command]
  bazel_command.append(build_target)

  logger.info("%s\n", bazel_command.command)

  if args.dry_run:
    logger.info("CLI is in dry run mode. Exiting without invoking Bazel.")
    sys.exit(0)

  await executor.run(bazel_command.command)

  if not args.build_target_only:
    logger.info("Building wheel...")
    run_wheel_binary = command.CommandBuilder(wheel_binary)

    # Read output directory from environment variable. If not set, set it to
    # dist/ in the current working directory.
    output_dir = os.getenv("JAXCI_OUTPUT_DIR", os.path.join(os.getcwd(), "dist"))
    run_wheel_binary.append(f"--output_path={output_dir}")

    run_wheel_binary.append(f"--cpu={arch}")

    if args.command == "jax-cuda-plugin" or args.command == "jax-cuda-pjrt":
      run_wheel_binary.append("--enable-cuda=True")
      major_cuda_version = args.cuda_version.split(".")[0]
      run_wheel_binary.append(f"--platform_version={major_cuda_version}")

    jaxlib_git_hash = get_jaxlib_git_hash()
    run_wheel_binary.append(f"--jaxlib_git_hash={jaxlib_git_hash}")

    logger.info("%s\n", run_wheel_binary.command)
    await executor.run(run_wheel_binary.command)

if __name__ == "__main__":
  asyncio.run(main())
  