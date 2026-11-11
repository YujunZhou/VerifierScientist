"""Bound API concurrency across reward processes sharing one budget ledger.

Use ``with bounded_api_slot(api_config):`` around the judge/API call. Every
participant must use the same budget_file and concurrency. This gate does not
reserve money, change the budget cap, retry an API call, or inspect credentials.
"""
from contextlib import ExitStack, contextmanager
import errno
import fcntl
import math
from pathlib import Path
import time


@contextmanager
def bounded_api_slot(config):
    """Acquire one OS-owned slot, waiting at most slot_wait_timeout seconds.

    Default wait is 1800 seconds; request timeout remains an independent client
    setting. Slots live in ``<budget_file>.slots``. Lock files are persistent:
    unlinking them could allow different processes to lock different inodes.
    Normal exceptions release in finally; process death releases the kernel lock.
    """
    limit = config.get('concurrency', 4)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError('API concurrency must be a positive integer')
    wait = float(config.get('slot_wait_timeout', 1800))
    if not math.isfinite(wait) or wait < 0:
        raise ValueError('API slot wait timeout must be finite and nonnegative')
    budget_file = Path(config['budget_file']).resolve()
    directory = Path(str(budget_file) + '.slots')
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as handles:
        files = [handles.enter_context((directory / f'{index:04d}.lock').open('a'))
                 for index in range(limit)]
        deadline = time.monotonic() + wait
        while True:
            for file in files:
                try:
                    fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EAGAIN):
                        raise
                    continue
                try:
                    yield
                finally:
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Timed out waiting for an API concurrency slot')
            time.sleep(min(.1, remaining))
