"""SSH manager tests"""

from unittest.mock import patch, MagicMock

import pytest

from crewster.ssh import SSHManager, SSHError

from rejection_fixtures import SACCT_ID_REJECTION_STDERR


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

    def test_run_command_failure_includes_exit_code_and_command(self):
        # The message always contains the exit code and the executed
        # remote command so callers can locate which invocation failed
        # without re-running with extra logging.
        manager = SSHManager(host="myhpc")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=42, stdout="", stderr="boom")
            with pytest.raises(SSHError) as exc_info:
                manager.run_command("echo", ["hello"])
            msg = str(exc_info.value)
            assert msg.startswith("SSH command failed (exit 42):")
            assert "echo hello" in msg

    def test_run_command_failure_with_only_stderr_omits_stdout_section(self):
        # Typical SSH-transport / cat-missing-file failure: only stderr
        # carries the diagnostic. The empty stdout section must be
        # omitted so the message stays readable.
        manager = SSHManager(host="myhpc")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="cat: /missing: No such file or directory\n",
            )
            with pytest.raises(SSHError) as exc_info:
                manager.run_command("cat", ["/missing"])
            msg = str(exc_info.value)
            assert "stderr:\ncat: /missing: No such file or directory" in msg
            assert "stdout:" not in msg

    def test_run_command_failure_with_only_stdout_surfaces_it(self):
        # pjsub rejections and other tools that emit error text on stdout
        # used to vanish behind an empty trailing colon. The enriched
        # message must include the stdout section so the diagnostic is
        # visible to the caller.
        manager = SSHManager(host="myhpc")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="[ERR.] PJM 0000 pjsub Invalid option specified\n",
                stderr="",
            )
            with pytest.raises(SSHError) as exc_info:
                manager.run_command("pjsub", ["bad.sh"])
            msg = str(exc_info.value)
            assert "stdout:\n[ERR.] PJM 0000 pjsub Invalid option specified" in msg
            assert "stderr:" not in msg

    def test_run_command_failure_with_both_streams_includes_both(self):
        manager = SSHManager(host="myhpc")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=2, stdout="partial output\n", stderr="warning then error\n"
            )
            with pytest.raises(SSHError) as exc_info:
                manager.run_command("some_cmd")
            msg = str(exc_info.value)
            assert "stdout:\npartial output" in msg
            assert "stderr:\nwarning then error" in msg

    def test_run_command_failure_carries_structured_stderr(self):
        # The raised SSHError exposes the remote stderr as a field so
        # callers (JobManager's id-rejection classification) can match
        # scheduler signatures without parsing the display message.
        manager = SSHManager(host="myhpc")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr=SACCT_ID_REJECTION_STDERR,
            )
            with pytest.raises(SSHError) as exc_info:
                manager.run_command("sacct", ["-j", "r1"])
            assert exc_info.value.stderr == SACCT_ID_REJECTION_STDERR

    def test_ssherror_stderr_defaults_to_none(self):
        # Raise sites that predate stderr capture (or have none) must
        # still construct with the message alone.
        assert SSHError("boom").stderr is None

    def test_run_command_failure_with_empty_streams_keeps_command_and_code(self):
        # Boundary case: command failed but produced no output. The
        # message must still be informative — exit code and command name
        # remain — even though there is nothing else to surface.
        manager = SSHManager(host="myhpc")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=255, stdout="", stderr="")
            with pytest.raises(SSHError) as exc_info:
                manager.run_command("hostname")
            msg = str(exc_info.value)
            assert msg == "SSH command failed (exit 255): hostname"

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

    def test_run_streaming_returns_130_on_keyboard_interrupt(self):
        """Ctrl-C during streaming: subprocess.run re-raises KeyboardInterrupt;
        we convert to the conventional SIGINT exit code so Typer does not abort.
        """
        manager = SSHManager(host="myhpc")
        with patch("subprocess.run", side_effect=KeyboardInterrupt):
            assert manager.run_streaming("tail", ["-F", "/path"]) == 130
