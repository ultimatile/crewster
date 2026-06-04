"""File synchronization management"""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import HpcConfig
from .ssh import SSHManager


@dataclass
class SyncResult:
    """Result of sync operation"""

    success: bool
    dry_run: bool
    returncode: int = 0


class SyncManager:
    """rsync-based file synchronization manager"""

    def __init__(self, ssh_manager: SSHManager, config: HpcConfig):
        self.ssh_manager = ssh_manager
        self.config = config

    def get_git_commit(self, path: Path, short: bool = False) -> Optional[str]:
        """Get current git commit hash"""
        try:
            # Check if path itself is a git repo
            git_dir = path / ".git"
            if not git_dir.exists():
                return None

            cmd = ["git", "rev-parse"]
            if short:
                cmd.append("--short")
            cmd.append("HEAD")
            result = subprocess.run(cmd, cwd=path, capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except Exception:
            return None

    def has_uncommitted_changes(self, path: Path) -> bool:
        """Check if there are uncommitted changes"""
        try:
            git_dir = path / ".git"
            if not git_dir.exists():
                return False

            result = subprocess.run(
                [
                    "git",
                    "--git-dir",
                    str(git_dir),
                    "--work-tree",
                    str(path),
                    "status",
                    "--porcelain",
                ],
                capture_output=True,
                text=True,
            )
            return len(result.stdout.strip()) > 0
        except Exception:
            return False

    def _build_rsync_command(
        self,
        local_path: Path,
        dry_run: bool,
        reverse: bool = False,
        extra_excludes: list[str] | None = None,
        use_checksum: bool = True,
    ) -> list[str]:
        """Build rsync command with options"""
        ssh_opts = "ssh -o LogLevel=ERROR"
        if self.ssh_manager.use_control_master:
            ssh_opts += (
                f" -o ControlMaster=auto"
                f" -o ControlPath={self.ssh_manager._control_path}"
                f" -o ControlPersist=10m"
            )
        cmd = ["rsync", "-avz", "-e", ssh_opts]

        if use_checksum:
            cmd.append("--checksum")

        if dry_run:
            cmd.append("--dry-run")

        # Common ignore patterns
        for pattern in self.config.sync.ignore:
            cmd.extend(["--exclude", pattern])

        # Direction-specific ignore patterns
        if reverse:
            for pattern in self.config.sync.ignore_pull:
                cmd.extend(["--exclude", pattern])
        else:
            for pattern in self.config.sync.ignore_push:
                cmd.extend(["--exclude", pattern])

        # Extra excludes (e.g., push targets excluded from pull)
        if extra_excludes:
            for pattern in extra_excludes:
                cmd.extend(["--exclude", pattern])

        remote = f"{self.config.cluster.host}:{self._resolve_remote_workdir()}"
        local = str(local_path) + "/"

        if reverse:
            cmd.extend([remote + "/", local])
        else:
            cmd.extend([local, remote])

        return cmd

    def _get_push_targets(
        self, local_path: Path, use_checksum: bool = True
    ) -> list[str]:
        """Get list of files/dirs that would be pushed (dry-run)"""
        cmd = self._build_rsync_command(
            local_path, dry_run=True, reverse=False, use_checksum=use_checksum
        )
        cmd.append("--itemize-changes")
        result = subprocess.run(cmd, capture_output=True, text=True)
        targets = []
        for line in result.stdout.splitlines():
            if not line:
                continue
            # <f = file sent, cd = directory created, .d = directory updated
            if line[0] == "<" or line.startswith("cd") or line.startswith(".d"):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    targets.append(parts[1])
        return targets

    def _resolve_remote_workdir(self) -> str:
        """Resolve remote workdir, expanding ~ to actual home path"""
        workdir = self.config.cluster.workdir
        if workdir.startswith("~/") or workdir == "~":
            result = self.ssh_manager.run_command("printenv", ["HOME"])
            home_dir = result.stdout.strip()
            workdir = workdir.replace("~", home_dir, 1) if workdir != "~" else home_dir
        return workdir

    def remote_dir_exists(self) -> bool:
        """Check if remote workdir exists"""
        try:
            workdir = self._resolve_remote_workdir()
            self.ssh_manager.run_command("test", ["-d", workdir])
            return True
        except Exception:
            return False

    def ensure_remote_dir(self) -> None:
        """Create remote workdir if it does not exist"""
        workdir = self._resolve_remote_workdir()
        self.ssh_manager.run_command("mkdir", ["-p", workdir])

    def sync_push(
        self, local_path: Path, dry_run: bool = True, use_checksum: bool = True
    ) -> SyncResult:
        """Sync local files to remote HPC cluster"""
        cmd = self._build_rsync_command(
            local_path, dry_run, reverse=False, use_checksum=use_checksum
        )
        result = subprocess.run(cmd)
        return SyncResult(
            success=result.returncode == 0,
            dry_run=dry_run,
            returncode=result.returncode,
        )

    def sync_pull(
        self,
        local_path: Path,
        dry_run: bool = True,
        exclude_push_targets: bool = False,
        use_checksum: bool = True,
        pull_dir: Path | None = None,
    ) -> SyncResult:
        """Sync remote files to local (or pull_dir if specified)"""
        extra_excludes = (
            self._get_push_targets(local_path, use_checksum)
            if exclude_push_targets
            else None
        )
        dest = pull_dir if pull_dir is not None else local_path
        cmd = self._build_rsync_command(
            dest,
            dry_run,
            reverse=True,
            extra_excludes=extra_excludes,
            use_checksum=use_checksum,
        )
        result = subprocess.run(cmd)
        return SyncResult(
            success=result.returncode == 0,
            dry_run=dry_run,
            returncode=result.returncode,
        )

    def sync_inputs(
        self, local_path: Path, dry_run: bool = True, use_checksum: bool = True
    ) -> SyncResult:
        """Sync local files to remote HPC cluster (alias for sync_push)"""
        return self.sync_push(local_path, dry_run, use_checksum)
