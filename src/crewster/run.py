"""Run management - tracking job executions"""

import tomllib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import tomli_w

from .config import HpcConfig


@dataclass
class RunConfig:
    """Configuration and metadata for a single run"""

    run_id: str
    cmd: str
    status: str
    job_id: Optional[str] = None
    git_commit: Optional[str] = None
    created_at: Optional[str] = None


class RunManager:
    """Manages run lifecycle and metadata.

    ``runs_dir`` is created lazily by the write paths (``create_run`` /
    ``save_run_meta``), never at construction time: read-only lookups must not
    leave an empty ``.crewster/runs`` tree behind when invoked against a
    project root that has no runs (e.g. a wrong ``--project-dir``).
    """

    def __init__(self, config: HpcConfig, runs_dir: Path):
        self.config = config
        self.runs_dir = runs_dir

    def _generate_run_id(self) -> str:
        """Generate unique run ID: YYYY-MM-DD_HHMMSS_<hash>"""
        import uuid

        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H%M%S")
        short_hash = uuid.uuid4().hex[:6]
        return f"{timestamp}_{short_hash}"

    def create_run(self, cmd: str, git_commit: Optional[str] = None) -> RunConfig:
        """Create a new run with generated ID"""
        run_id = self._generate_run_id()
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        return RunConfig(
            run_id=run_id,
            cmd=cmd,
            status="pending",
            git_commit=git_commit,
            created_at=datetime.now().isoformat(),
        )

    def save_run_meta(self, run: RunConfig) -> None:
        """Save run metadata to TOML file"""
        run_dir = self.runs_dir / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_path = run_dir / "meta.toml"

        data = {k: v for k, v in asdict(run).items() if v is not None}
        with open(meta_path, "wb") as f:
            tomli_w.dump(data, f)

    def load_run_meta(self, run_id: str) -> RunConfig:
        """Load run metadata from TOML file"""
        meta_path = self.runs_dir / run_id / "meta.toml"
        with open(meta_path, "rb") as f:
            data = tomllib.load(f)
        # Only use fields that are defined in RunConfig
        from dataclasses import fields

        valid_fields = {f.name for f in fields(RunConfig)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return RunConfig(**filtered_data)

    def list_runs(self) -> list[RunConfig]:
        """List all runs; a missing ``runs_dir`` reads as "no runs".

        EAFP instead of an ``is_dir()`` pre-check: no window between check
        and iteration, and a ``runs_dir`` that exists as a regular file
        still fails loudly instead of masquerading as an empty run list.
        """
        try:
            run_dirs = list(self.runs_dir.iterdir())
        except FileNotFoundError:
            return []
        runs = []
        for run_dir in run_dirs:
            if run_dir.is_dir():
                meta_path = run_dir / "meta.toml"
                if meta_path.exists():
                    runs.append(self.load_run_meta(run_dir.name))
        return runs

    def find_run_by_job_id(self, job_id: str) -> Optional[RunConfig]:
        """Find run by job ID"""
        for run in self.list_runs():
            if run.job_id == job_id:
                return run
        return None
