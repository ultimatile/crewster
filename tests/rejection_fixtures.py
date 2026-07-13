"""Shared job-id rejection fixtures.

Verbatim rejection stderr recorded on real clusters
(https://github.com/ultimatile/crewster/issues/50): slurm 23.02.6 on
HOKUSAI BW2 and PJM on Fugaku. Single source for every suite that pins
the classifier or replays the rejection, so a signature update cannot
leave a stale copy silently certifying a match the production matcher
no longer performs.

A plain helper module (not conftest.py) so the constants stay importable
at collection time — parametrize needs them — without importing the
pytest plugin module.
"""

from crewster.ssh import SSHError

SACCT_ID_REJECTION_STDERR = "sacct: fatal: Bad job/step specified: r1\n"
PJSTAT_ID_REJECTION_STDERR = "[ERR.] PJM 0211 pjstat Invalid jobid: r1.\n"


def make_ssh_failure(stderr: str, cmd: str = "sacct -j r1") -> SSHError:
    """Build the SSHError that run_command raises for a non-zero remote exit,
    shaped like the real raise site (message + structured stderr)."""
    return SSHError(f"SSH command failed (exit 1): {cmd}", stderr=stderr)
