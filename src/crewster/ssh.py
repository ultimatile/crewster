"""SSH connection management"""

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import Optional


class SSHError(Exception):
    """SSH operation error.

    ``stderr`` carries the remote command's captured stderr separately from
    the human-readable message so callers can classify the failure (e.g. a
    scheduler's job-id rejection signature) without parsing the message
    string, whose format is a display concern. ``None`` when the failure
    had no captured stderr (or the raise site predates capture).
    """

    def __init__(self, message: str, *, stderr: str | None = None):
        super().__init__(message)
        self.stderr = stderr


@dataclass
class CommandResult:
    """Result of SSH command execution"""

    returncode: int
    stdout: str
    stderr: str


class SSHManager:
    """SSH connection and command execution manager"""

    def __init__(
        self,
        host: str,
        user: Optional[str] = None,
        use_control_master: bool = True,
    ):
        self._validate_target_component("host", host)
        if user is not None:
            self._validate_target_component("user", user)
        self.host = host
        self.user = user
        self.use_control_master = use_control_master
        self._control_path = f"/tmp/crewster_ssh_{host}_{os.getpid()}"

    def _validate_target_component(self, label: str, value: str) -> None:
        """Reject values that could be treated as ssh options"""
        if not value:
            raise ValueError(f"{label} must not be empty")
        if value.startswith("-"):
            raise ValueError(f"{label} must not start with '-'")
        if re.search(r"\s", value):
            raise ValueError(f"{label} must not contain whitespace")

    def _build_ssh_command(self, cmd: str) -> list[str]:
        """Build SSH command with options"""
        ssh_cmd = ["ssh", "-q"]

        if self.use_control_master:
            ssh_cmd.extend(
                [
                    "-o",
                    "ControlMaster=auto",
                    "-o",
                    f"ControlPath={self._control_path}",
                    "-o",
                    "ControlPersist=10m",
                ]
            )

        target = f"{self.user}@{self.host}" if self.user else self.host
        ssh_cmd.append(target)
        ssh_cmd.append(cmd)

        return ssh_cmd

    def _validate_command_name(self, cmd: str) -> None:
        """Validate command name for safe execution"""
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", cmd):
            raise ValueError(f"Invalid command name: {cmd!r}")

    def test_connection(self) -> bool:
        """Test SSH connection"""
        try:
            result = subprocess.run(
                self._build_ssh_command("exit 0"),
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception:
            return False

    def run_script(self, script: str) -> int:
        """Execute a shell script on remote host, streaming output to terminal.

        Pipes script to 'bash -s' via SSH. Returns the process exit code.
        """
        result = subprocess.run(
            self._build_ssh_command("bash -s"),
            input=script,
            text=True,
        )
        return result.returncode

    def run_command(
        self,
        cmd: str,
        args: Optional[list[str]] = None,
        input_text: Optional[str] = None,
    ) -> CommandResult:
        """Execute command on remote host"""
        if args is None:
            args = []
        self._validate_command_name(cmd)
        quoted_parts = [shlex.quote(cmd), *[shlex.quote(arg) for arg in args]]
        command = " ".join(quoted_parts)

        result = subprocess.run(
            self._build_ssh_command(command),
            capture_output=True,
            text=True,
            input=input_text,
        )

        if result.returncode != 0:
            # Include exit code, the executed remote command, and both
            # streams so tools that emit diagnostics on stdout (e.g. pjsub
            # rejections, anything printing `[ERR.] ...` on stdout) are
            # not silently swallowed. Empty sections are omitted to keep
            # the common stderr-only case readable.
            parts = [f"SSH command failed (exit {result.returncode}): {command}"]
            if result.stdout:
                parts.append(f"stdout:\n{result.stdout.rstrip()}")
            if result.stderr:
                parts.append(f"stderr:\n{result.stderr.rstrip()}")
            raise SSHError("\n".join(parts), stderr=result.stderr)

        return CommandResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def run_streaming(
        self,
        cmd: str,
        args: Optional[list[str]] = None,
    ) -> int:
        """Execute command on remote host, streaming stdout/stderr to terminal.

        Mirrors run_command's argument shape and shell-injection defenses, but
        does not capture output — the remote process writes directly to the
        local terminal. Use for long-running, output-producing commands such
        as `tail -F`. Returns the SSH process exit code; does not raise on
        non-zero exit.

        On Ctrl-C, both Python and the SSH client receive SIGINT (foreground
        process group). subprocess.run kills + reaps the child and re-raises
        KeyboardInterrupt; we convert that to the conventional shell exit
        code 130 so callers can propagate cleanly without aborting.
        """
        if args is None:
            args = []
        self._validate_command_name(cmd)
        quoted_parts = [shlex.quote(cmd), *[shlex.quote(arg) for arg in args]]
        command = " ".join(quoted_parts)

        try:
            result = subprocess.run(self._build_ssh_command(command))
        except KeyboardInterrupt:
            return 130
        return result.returncode
