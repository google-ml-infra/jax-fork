import collections
import datetime
import pathlib
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

from absl import app
from absl import flags
import immutabledict

from google3.pyglib import gfile

ROOT_BIGSTORE_RELEASES_PATH = pathlib.Path('/bigstore/jax-releases')

def find_wheels(
    patterns: Sequence[str], is_nightly: bool, bucket: str
) -> List[str]:
  """Finds the wheels that match any of the input patterns."""
  wheels = []
  if is_nightly:
    wheel_version_keyword = datetime.date.today().strftime('%Y%m%d')
  else:
    wheel_version_keyword = f'-{jax.jaxlib.version._version}-'  # pylint: disable=protected-access

  bucket_path = ROOT_BIGSTORE_RELEASES_PATH / bucket
  files = gfile.ListDir(bucket_path)
  print("files:", files)
#   for file in files:
#     if wheel_version_keyword not in file:
#       continue
#     for pattern in patterns:
#       if re.fullmatch(pattern, file):
#         wheels.append(str(bucket_path / file))
#         break

  return wheels

def _move_cuda_plugin_wheels_to_release_bucket():
    """Move cuda plugin wheels from temp_wheels bucket to release bucket."""
    wheels = find_wheels(
        patterns=['.*jax_cuda.*[pjrt|plugin].*'], is_nightly=False,
        bucket='temp_wheels'
    )
    print("wheels:", wheels)
    # for wheel_path in wheels:
    # # TODO(jieying): make the bucket renaming here more flexible for future
    # # cuda versions.
    #     if 'cuda12' in wheel_path:
    #         release_bucket = 'cuda12_plugin'
    #         release_file_path = wheel_path.replace('temp_wheels', release_bucket)
    #         gfile.Rename(wheel_path, release_file_path, overwrite=True)
    #     else:
    #         print(f'Unexpected wheel found: {wheel_path}')


def main(argv: Sequence[str]) -> None:
  _move_cuda_plugin_wheels_to_release_bucket()


if __name__ == '__main__':
  app.run(main)
