"""Retrieve labels for the PR from context, if any.

While these labels are also available via GH context, and the event payload
file, they may be stale:
https://github.com/orgs/community/discussions/39062

As such, the API is used as the main source, with the event payload file
being the fallback.

The script is only geared towards use within a GH Action run.
"""

import json
import logging
import os
import re
import sys
import time
import urllib.request


# GET_LABELS_DEBUG is a variable specifically for this script.
# RUNNER_DEBUG and ACTIONS_RUNNER_DEBUG are GH env vars, which can be set
# in various ways, one of them - enabling debug logging from the UI, when
# triggering a run
# https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/store-information-in-variables#default-environment-variables
# https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/troubleshooting-workflows/enabling-debug-logging#enabling-runner-diagnostic-logging
_SHOW_DEBUG = bool(
  os.getenv('GET_LABELS_DEBUG',
            os.getenv('RUNNER_DEBUG',
                      os.getenv('ACTIONS_RUNNER_DEBUG')))
)
logging.basicConfig(level=logging.INFO if not _SHOW_DEBUG else logging.DEBUG,
                    format='%(levelname)s: %(message)s', stream=sys.stderr)


_GITHUB_REF = os.environ.get('GITHUB_REF')
# Outside a PR context - no labels to be found
if not _GITHUB_REF.startswith('refs/pull/'):
  logging.debug('Not a PR run')
  print([])
  raise SystemExit

# Since passing the previous check confirms this is a PR, there's no need
# to safeguard this regex
GH_ISSUE = re.search(r'\d+', _GITHUB_REF).group()
GH_REPO = os.environ.get('GITHUB_REPOSITORY')

logging.debug(f'{GH_ISSUE=!r}\n'
              f'{GH_REPO=!r}')

URL = f'https://api.github.com/repos/{GH_REPO}/issues/{GH_ISSUE}/labels'

WAIT_TIME = 3
ATTEMPTS = 3

data = None
cur_attempt = 1

while cur_attempt <= ATTEMPTS:
  request = urllib.request.Request(
    URL,
    headers={'Accept': 'application/vnd.github+json',
             'X-GitHub-Api-Version': '2022-11-28'}
  )
  logging.info(f'Retrieving PR labels via API - attempt {cur_attempt}...')
  response = urllib.request.urlopen(request)

  if response.status == 200:
      data = response.read().decode('utf-8')
      logging.debug('API labels data: \n'
                    f'{data}')
      break
  else:
      logging.error(f'Request failed with status code: {response.status}')
      cur_attempt += 1
      if cur_attempt <= ATTEMPTS:
        logging.info(f'Trying again in {WAIT_TIME} seconds')
        time.sleep(WAIT_TIME)


# The null check is probably unnecessary, but rather be safe
if data and data != 'null':
  data_json = json.loads(data)
else:
  # Fall back on labels from the event's payload
  event_payload_path = os.environ.get('GITHUB_EVENT_PATH')
  with open(event_payload_path, 'r', encoding='utf-8') as event_payload:
    data_json = json.load(event_payload).get('pull_request',
                                             {}).get('labels', [])
    logging.info('Using fallback labels')
    logging.info(f'Fallback labels: \n'
                 f'{data_json}')


labels = [label['name'] for label in data_json]
logging.debug(f'Final labels: \n'
              f'{labels}')
print(labels)
