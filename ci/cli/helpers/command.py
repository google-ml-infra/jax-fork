import asyncio
import dataclasses
import datetime
import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class CommandBuilder:
  def __init__(self, base_command: str):
    self.command = base_command

  def append(self, parameter: str):
    self.command += " {}".format(parameter)
    return self

@dataclasses.dataclass
class CommandResult:
  """
  Represents the result of executing a subprocess command.
  """

  command: str
  return_code: int = 2  # Defaults to not successful
  logs: str = ""
  start_time: datetime.datetime = dataclasses.field(
    default_factory=datetime.datetime.now
  )
  end_time: Optional[datetime.datetime] = None

class SubprocessExecutor:
  """
  Manages execution of subprocess commands with reusable environment and logging.
  """

  def __init__(self, environment: Dict[str, str] = None):
    """

    Args:
      environment:
    """
    self.environment = environment or dict(os.environ)

  async def run(self, cmd: str, dry_run: bool = False) -> CommandResult:
    """
    Executes a subprocess command.

    Args:
        cmd: The command to execute.
        dry_run: If True, prints the command instead of executing it.

    Returns:
        A CommandResult instance.
    """
    result = CommandResult(command=cmd)
    if dry_run:
      logger.info("[DRY RUN] %s", cmd)
      result.return_code = 0  # Dry run is a success
      return result

    logger.debug("Executing: %s", cmd)

    process = await asyncio.create_subprocess_shell(
      cmd,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
      env=self.environment,
    )

    async def log_stream(stream, result: CommandResult):
      while True:
        line_bytes = await stream.readline()
        if not line_bytes:
          break
        line = line_bytes.decode().rstrip()
        result.logs += line
        logger.info("%s", line)

    await asyncio.gather(
      log_stream(process.stdout, result), log_stream(process.stderr, result)
    )

    result.return_code = await process.wait()
    result.end_time = datetime.datetime.now()
    logger.debug("Command finished with return code %s", result.return_code)
    return result
