"""Git sync tests"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from crewster.sync import SyncManager
from crewster.ssh import SSHManager
from crewster.config import HpcConfig, ClusterConfig, EnvConfig, SlurmConfig


@pytest.fixture
def mock_ssh_manager():
    return MagicMock(spec=SSHManager)


@pytest.fixture
def sample_config():
    return HpcConfig(
        cluster=ClusterConfig(host="myhpc", workdir="/scratch/user/proj"),
        env=EnvConfig(),
        slurm=SlurmConfig(
            options={"partition": "gpu", "time": "02:00:00", "mem": "32G"}
        ),
    )


class TestGitStatus:
    def test_get_git_commit(self, mock_ssh_manager, sample_config):
        manager = SyncManager(ssh_manager=mock_ssh_manager, config=sample_config)
        with (
            patch("crewster.sync.subprocess.run") as mock_run,
            patch.object(Path, "exists", return_value=True),
        ):
            mock_run.return_value = MagicMock(
                returncode=0, stdout="abc1234567890abcdef1234567890abcdef1234\n"
            )
            commit = manager.get_git_commit(Path("/fake/repo"))
            assert commit == "abc1234567890abcdef1234567890abcdef1234"

    def test_get_git_commit_short(self, mock_ssh_manager, sample_config):
        manager = SyncManager(ssh_manager=mock_ssh_manager, config=sample_config)
        with (
            patch("crewster.sync.subprocess.run") as mock_run,
            patch.object(Path, "exists", return_value=True),
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="abc1234\n")
            commit = manager.get_git_commit(Path("/fake/repo"), short=True)
            assert commit == "abc1234"

    def test_has_uncommitted_changes_clean(self, mock_ssh_manager, sample_config):
        manager = SyncManager(ssh_manager=mock_ssh_manager, config=sample_config)
        with (
            patch("crewster.sync.subprocess.run") as mock_run,
            patch.object(Path, "exists", return_value=True),
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="")
            assert manager.has_uncommitted_changes(Path("/fake/repo")) is False

    def test_has_uncommitted_changes_dirty(self, mock_ssh_manager, sample_config):
        manager = SyncManager(ssh_manager=mock_ssh_manager, config=sample_config)
        with (
            patch("crewster.sync.subprocess.run") as mock_run,
            patch.object(Path, "exists", return_value=True),
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout=" M test.py\n")
            assert manager.has_uncommitted_changes(Path("/fake/repo")) is True

    def test_not_git_repo(self, mock_ssh_manager, sample_config):
        manager = SyncManager(ssh_manager=mock_ssh_manager, config=sample_config)
        with patch.object(Path, "exists", return_value=False):
            commit = manager.get_git_commit(Path("/not/a/repo"))
            assert commit is None
