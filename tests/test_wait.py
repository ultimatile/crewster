"""Job wait tests"""

from unittest.mock import MagicMock, patch

import pytest

from hpc.job import JobManager, JobStatus
from hpc.ssh import SSHManager
from hpc.config import HpcConfig, ClusterConfig, EnvConfig, SlurmConfig


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
        # Regression guard for https://github.com/ultimatile/hpc/issues/13:
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
