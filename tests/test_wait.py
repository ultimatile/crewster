"""Job wait tests"""

from unittest.mock import MagicMock, patch

import pytest

from crewster.job import JobManager, JobStatus
from crewster.ssh import SSHError, SSHManager
from crewster.config import (
    HpcConfig,
    ClusterConfig,
    EnvConfig,
    PjmConfig,
    SlurmConfig,
)


@pytest.fixture
def mock_ssh_manager():
    return MagicMock(spec=SSHManager)


@pytest.fixture
def sample_config():
    return HpcConfig(
        cluster=ClusterConfig(host="myhpc", workdir="/scratch/user/proj"),
        env=EnvConfig(),
        slurm=SlurmConfig(partition="gpu", time="02:00:00", mem="32G"),
    )


@pytest.fixture
def pjm_config():
    return HpcConfig(
        cluster=ClusterConfig(
            host="myhpc", workdir="/scratch/user/proj", scheduler="pjm"
        ),
        env=EnvConfig(),
        pjm=PjmConfig(options=[["-L", "node=1"]]),
    )


class TestJobManagerWait:
    def test_wait_returns_final_status(self, mock_ssh_manager, sample_config):
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        mock_ssh_manager.run_command.return_value = MagicMock(stdout="COMPLETED\n")

        status = manager.wait_for_job("12345678", interval=0.01)
        assert status == JobStatus.COMPLETED

    def test_wait_polls_until_complete(self, mock_ssh_manager, sample_config):
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        mock_ssh_manager.run_command.side_effect = [
            MagicMock(stdout="PENDING\n"),
            MagicMock(stdout="RUNNING\n"),
            MagicMock(stdout="COMPLETED\n"),
        ]

        status = manager.wait_for_job("12345678", interval=0.01)
        assert status == JobStatus.COMPLETED
        assert mock_ssh_manager.run_command.call_count == 3

    def test_wait_returns_on_failed(self, mock_ssh_manager, sample_config):
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        mock_ssh_manager.run_command.side_effect = [
            MagicMock(stdout="RUNNING\n"),
            MagicMock(stdout="FAILED\n"),
        ]

        status = manager.wait_for_job("12345678", interval=0.01)
        assert status == JobStatus.FAILED

    def test_wait_keeps_polling_on_empty_status_output(
        self, mock_ssh_manager, sample_config
    ):
        # Regression guard for https://github.com/ultimatile/crewster/issues/13:
        # a freshly-submitted job whose sacct row hasn't been indexed yet
        # returns empty stdout. The old parse_status mapped that to FAILED
        # and wait_for_job exited prematurely; SchedulerError (subclass
        # of SSHError) now routes through the retry path so polling
        # continues.
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        mock_ssh_manager.run_command.side_effect = [
            MagicMock(stdout=""),
            MagicMock(stdout="COMPLETED\n"),
        ]

        status = manager.wait_for_job("12345678", interval=0.01)
        assert status == JobStatus.COMPLETED
        assert mock_ssh_manager.run_command.call_count == 2

    def test_wait_adaptive_interval(self, mock_ssh_manager, sample_config):
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        mock_ssh_manager.run_command.side_effect = [
            MagicMock(stdout="PENDING\n"),
            MagicMock(stdout="PENDING\n"),
            MagicMock(stdout="COMPLETED\n"),
        ]

        with patch("time.sleep") as mock_sleep:
            manager.wait_for_job("12345678", interval=10, adaptive=True)
            # Interval should increase
            intervals = [c[0][0] for c in mock_sleep.call_args_list]
            assert intervals[1] >= intervals[0]

    def test_wait_gives_up_after_max_missing_polls(
        self, mock_ssh_manager, sample_config
    ):
        # Issue #24: persistently-empty status output (never-submitted job,
        # or a PJM job aged out of every view) must terminate the wait
        # instead of looping forever. The bounded retry surfaces it as
        # JobStatus.UNKNOWN once the budget is exhausted.
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        mock_ssh_manager.run_command.return_value = MagicMock(stdout="")

        status = manager.wait_for_job("12345678", interval=0.01, max_missing_polls=3)
        assert status == JobStatus.UNKNOWN
        assert mock_ssh_manager.run_command.call_count == 3

    def test_wait_missing_budget_resets_on_successful_read(
        self, mock_ssh_manager, sample_config
    ):
        # The budget counts only *consecutive* empty polls: a successful
        # read in between clears the streak, so transient accounting lag
        # interleaved with real status never trips the limit.
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        mock_ssh_manager.run_command.side_effect = [
            MagicMock(stdout=""),  # missing 1
            MagicMock(stdout=""),  # missing 2 (budget is 3, not yet hit)
            MagicMock(stdout="RUNNING\n"),  # success -> reset to 0
            MagicMock(stdout=""),  # missing 1
            MagicMock(stdout=""),  # missing 2
            MagicMock(stdout="COMPLETED\n"),  # terminal
        ]

        status = manager.wait_for_job("12345678", interval=0.01, max_missing_polls=3)
        assert status == JobStatus.COMPLETED
        assert mock_ssh_manager.run_command.call_count == 6

    def test_wait_generic_ssh_error_does_not_count_toward_budget(
        self, mock_ssh_manager, sample_config
    ):
        # Only SchedulerError (no data row) is budgeted. A generic SSHError
        # is a transport hiccup (e.g. bastion rate limiting) and is retried
        # without bound, so repeated transport errors beyond the budget
        # must not synthesize UNKNOWN.
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        mock_ssh_manager.run_command.side_effect = [
            SSHError("connection reset"),
            SSHError("connection reset"),
            SSHError("connection reset"),
            SSHError("connection reset"),
            SSHError("connection reset"),
            MagicMock(stdout="COMPLETED\n"),
        ]

        status = manager.wait_for_job("12345678", interval=0.01, max_missing_polls=3)
        assert status == JobStatus.COMPLETED
        assert mock_ssh_manager.run_command.call_count == 6

    def test_wait_terminates_when_pjm_fallback_command_keeps_failing(
        self, mock_ssh_manager, pjm_config
    ):
        # End-to-end termination contract: when the PJM active view is
        # persistently empty and the history fallback command itself keeps
        # exiting non-zero (an unknown/expired job), the wait must still
        # terminate as UNKNOWN. The fallback's generic SSHError is folded
        # back into SchedulerError by get_job_status so it stays inside the
        # budget instead of being retried without bound. Each poll issues
        # two run_command calls (active view + history fallback).
        manager = JobManager(ssh_manager=mock_ssh_manager, config=pjm_config)
        mock_ssh_manager.run_command.side_effect = [
            MagicMock(stdout=""),  # poll 1: active view empty
            SSHError("pjstat -H exited non-zero"),  # poll 1: fallback errors
            MagicMock(stdout=""),  # poll 2
            SSHError("pjstat -H exited non-zero"),
            MagicMock(stdout=""),  # poll 3
            SSHError("pjstat -H exited non-zero"),
        ]

        status = manager.wait_for_job("12345678", interval=0.01, max_missing_polls=3)
        assert status == JobStatus.UNKNOWN
        assert mock_ssh_manager.run_command.call_count == 6
