"""Sync manager tests"""

from unittest.mock import patch, MagicMock

import pytest

from crewster.sync import SyncManager, SyncResult
from crewster.ssh import SSHManager
from crewster.config import HpcConfig, ClusterConfig, EnvConfig, SlurmConfig, SyncConfig


@pytest.fixture
def mock_ssh_manager():
    mock = MagicMock(spec=SSHManager)
    mock.use_control_master = True
    mock._control_path = "/tmp/crewster_ssh_myhpc_99999"
    return mock


@pytest.fixture
def sample_config():
    return HpcConfig(
        cluster=ClusterConfig(host="myhpc", workdir="/scratch/user/proj"),
        env=EnvConfig(),
        sync=SyncConfig(ignore=[".git", "__pycache__"]),
        slurm=SlurmConfig(partition="gpu", time="02:00:00", mem="32G"),
    )


class TestSyncResult:
    def test_sync_result_fields(self):
        result = SyncResult(success=True, dry_run=False)
        assert result.success is True
        assert result.dry_run is False


class TestSyncManagerInit:
    def test_init_with_ssh_and_config(self, mock_ssh_manager, sample_config):
        manager = SyncManager(ssh_manager=mock_ssh_manager, config=sample_config)
        assert manager.ssh_manager == mock_ssh_manager
        assert manager.config == sample_config


class TestSyncManagerSyncInputs:
    def test_sync_inputs_dry_run(self, mock_ssh_manager, sample_config, temp_dir):
        manager = SyncManager(ssh_manager=mock_ssh_manager, config=sample_config)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = manager.sync_inputs(local_path=temp_dir, dry_run=True)
            assert result.dry_run is True
            call_args = mock_run.call_args[0][0]
            assert "--dry-run" in call_args

    def test_sync_inputs_apply(self, mock_ssh_manager, sample_config, temp_dir):
        manager = SyncManager(ssh_manager=mock_ssh_manager, config=sample_config)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = manager.sync_inputs(local_path=temp_dir, dry_run=False)
            assert result.dry_run is False
            call_args = mock_run.call_args[0][0]
            assert "--dry-run" not in call_args

    def test_sync_inputs_uses_rsync(self, mock_ssh_manager, sample_config, temp_dir):
        manager = SyncManager(ssh_manager=mock_ssh_manager, config=sample_config)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            manager.sync_inputs(local_path=temp_dir, dry_run=True)
            call_args = mock_run.call_args[0][0]
            assert call_args[0] == "rsync"

    def test_sync_inputs_target_path(self, mock_ssh_manager, sample_config, temp_dir):
        manager = SyncManager(ssh_manager=mock_ssh_manager, config=sample_config)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            manager.sync_inputs(local_path=temp_dir, dry_run=True)
            call_args = mock_run.call_args[0][0]
            args_str = " ".join(call_args)
            assert "myhpc:/scratch/user/proj" in args_str

    def test_sync_inputs_excludes_patterns(
        self, mock_ssh_manager, sample_config, temp_dir
    ):
        manager = SyncManager(ssh_manager=mock_ssh_manager, config=sample_config)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            manager.sync_inputs(local_path=temp_dir, dry_run=True)
            call_args = mock_run.call_args[0][0]
            assert "--exclude" in call_args


class TestBuildRsyncControlMaster:
    def test_rsync_includes_control_master_when_enabled(
        self, mock_ssh_manager, sample_config, temp_dir
    ):
        mock_ssh_manager.use_control_master = True
        mock_ssh_manager._control_path = "/tmp/crewster_ssh_myhpc_12345"
        manager = SyncManager(ssh_manager=mock_ssh_manager, config=sample_config)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            manager.sync_push(local_path=temp_dir, dry_run=True)
            call_args = mock_run.call_args[0][0]
            e_index = call_args.index("-e")
            ssh_opts = call_args[e_index + 1]
            assert "ControlMaster=auto" in ssh_opts
            assert "ControlPath=/tmp/crewster_ssh_myhpc_12345" in ssh_opts
            assert "ControlPersist=10m" in ssh_opts

    def test_rsync_excludes_control_master_when_disabled(
        self, mock_ssh_manager, sample_config, temp_dir
    ):
        mock_ssh_manager.use_control_master = False
        manager = SyncManager(ssh_manager=mock_ssh_manager, config=sample_config)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            manager.sync_push(local_path=temp_dir, dry_run=True)
            call_args = mock_run.call_args[0][0]
            e_index = call_args.index("-e")
            ssh_opts = call_args[e_index + 1]
            assert "ControlMaster" not in ssh_opts
            assert ssh_opts == "ssh -o LogLevel=ERROR"


class TestRemoteDir:
    def test_ensure_remote_dir(self, mock_ssh_manager, sample_config):
        manager = SyncManager(ssh_manager=mock_ssh_manager, config=sample_config)
        manager.ensure_remote_dir()
        mock_ssh_manager.run_command.assert_called_once_with(
            "mkdir", ["-p", "/scratch/user/proj"]
        )

    def test_ensure_remote_dir_tilde(self, mock_ssh_manager, sample_config):
        sample_config.cluster.workdir = "~/proj"
        manager = SyncManager(ssh_manager=mock_ssh_manager, config=sample_config)
        mock_ssh_manager.run_command.side_effect = [
            MagicMock(stdout="/home/user\n"),  # printenv HOME
            MagicMock(),  # mkdir
        ]
        manager.ensure_remote_dir()
        mock_ssh_manager.run_command.assert_any_call("printenv", ["HOME"])
        mock_ssh_manager.run_command.assert_any_call("mkdir", ["-p", "/home/user/proj"])

    def test_remote_dir_exists_uses_resolve(self, mock_ssh_manager, sample_config):
        manager = SyncManager(ssh_manager=mock_ssh_manager, config=sample_config)
        assert manager.remote_dir_exists() is True
        mock_ssh_manager.run_command.assert_called_once_with(
            "test", ["-d", "/scratch/user/proj"]
        )
