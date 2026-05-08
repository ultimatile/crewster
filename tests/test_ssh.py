"""SSH manager tests"""

from unittest.mock import patch, MagicMock

import pytest

from hpc.ssh import SSHManager, SSHError


class TestSSHManagerInit:
    def test_init_with_host(self):
        manager = SSHManager(host="myhpc")
        assert manager.host == "myhpc"

    def test_init_with_user(self):
        manager = SSHManager(host="myhpc", user="testuser")
        assert manager.user == "testuser"

    def test_init_rejects_empty_host(self):
        with pytest.raises(ValueError):
            SSHManager(host="")

    def test_init_rejects_host_starting_with_dash(self):
        with pytest.raises(ValueError):
            SSHManager(host="-oProxyCommand=bad")

    def test_init_rejects_host_with_whitespace(self):
        with pytest.raises(ValueError):
            SSHManager(host="bad host")

    def test_init_rejects_user_starting_with_dash(self):
        with pytest.raises(ValueError):
            SSHManager(host="myhpc", user="-oProxyCommand=bad")

    def test_init_rejects_user_with_whitespace(self):
        with pytest.raises(ValueError):
            SSHManager(host="myhpc", user="bad user")


class TestSSHManagerConnection:
    def test_test_connection_success(self):
        manager = SSHManager(host="myhpc")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert manager.test_connection() is True

    def test_test_connection_failure(self):
        manager = SSHManager(host="myhpc")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert manager.test_connection() is False


class TestSSHManagerRunCommand:
    def test_run_command_success(self):
        manager = SSHManager(host="myhpc")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="output",
                stderr="",
            )
            result = manager.run_command("echo", ["hello"])
            assert result.returncode == 0
            assert result.stdout == "output"

    def test_run_command_with_quiet_option(self):
        manager = SSHManager(host="myhpc")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            manager.run_command("echo", ["hello"])
            call_args = mock_run.call_args[0][0]
            assert "-q" in call_args

    def test_run_command_failure_raises(self):
        manager = SSHManager(host="myhpc")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="Connection refused",
            )
            with pytest.raises(SSHError):
                manager.run_command("echo", ["hello"])

    def test_run_command_captures_stderr(self):
        manager = SSHManager(host="myhpc")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="output",
                stderr="warning message",
            )
            result = manager.run_command("some_cmd")
            assert result.stderr == "warning message"


class TestSSHManagerRunScript:
    def test_run_script_uses_bash_stdin(self):
        manager = SSHManager(host="myhpc")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            manager.run_script("echo hello")
            call_args = mock_run.call_args
            assert call_args[0][0][-1] == "bash -s"
            assert call_args[1]["input"] == "echo hello"
            assert call_args[1]["text"] is True

    def test_run_script_does_not_capture_output(self):
        manager = SSHManager(host="myhpc")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            manager.run_script("echo hello")
            call_kwargs = mock_run.call_args[1]
            assert "capture_output" not in call_kwargs

    def test_run_script_returns_exit_code(self):
        manager = SSHManager(host="myhpc")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=42)
            assert manager.run_script("exit 42") == 42


class TestSSHManagerControlMaster:
    def test_control_master_options_included(self):
        manager = SSHManager(host="myhpc", use_control_master=True)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            manager.run_command("echo", ["hello"])
            call_args = mock_run.call_args[0][0]
            args_str = " ".join(call_args)
            assert "ControlMaster" in args_str
            assert "ControlPath" in args_str


class TestSSHManagerRunStreaming:
    def test_run_streaming_does_not_capture_output(self):
        manager = SSHManager(host="myhpc")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            manager.run_streaming("tail", ["-F", "/some/file"])
            call_kwargs = mock_run.call_args[1]
            assert "capture_output" not in call_kwargs

    def test_run_streaming_returns_exit_code(self):
        manager = SSHManager(host="myhpc")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=130)
            assert manager.run_streaming("tail", ["-F", "/path"]) == 130

    def test_run_streaming_validates_command(self):
        manager = SSHManager(host="myhpc")
        with pytest.raises(ValueError):
            manager.run_streaming("tail; rm -rf /", ["-F", "/path"])

    def test_run_streaming_quotes_args(self):
        manager = SSHManager(host="myhpc")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            manager.run_streaming("tail", ["-F", "/path with space/file"])
            remote_cmd = mock_run.call_args[0][0][-1]
            assert "'/path with space/file'" in remote_cmd

    def test_run_streaming_includes_control_master(self):
        manager = SSHManager(host="myhpc", use_control_master=True)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            manager.run_streaming("tail", ["-F", "/path"])
            args_str = " ".join(mock_run.call_args[0][0])
            assert "ControlMaster" in args_str
            assert "ControlPath" in args_str

    def test_run_streaming_builds_tail_command(self):
        manager = SSHManager(host="myhpc")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            manager.run_streaming("tail", ["-F", "/path/file"])
            remote_cmd = mock_run.call_args[0][0][-1]
            assert remote_cmd == "tail -F /path/file"

    def test_run_streaming_no_args(self):
        manager = SSHManager(host="myhpc")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            manager.run_streaming("hostname")
            remote_cmd = mock_run.call_args[0][0][-1]
            assert remote_cmd == "hostname"
