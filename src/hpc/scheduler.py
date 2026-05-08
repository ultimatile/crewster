"""Scheduler abstraction for Slurm and PJM"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class JobStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"


@dataclass
class JobDetail:
    """Raw scheduler-side job accounting fields.

    Strings are stored verbatim from the scheduler so the user sees
    the exact value (e.g. ``OUT_OF_MEMORY``, ``CANCELLED+``) without
    any enum/unit normalization.
    """

    state: str
    exit_code: str
    elapsed: str
    max_rss: str
    req_mem: str


class Scheduler(ABC):
    @abstractmethod
    def directive_prefix(self) -> str: ...

    @abstractmethod
    def submit_cmd(self) -> list[str]: ...

    @abstractmethod
    def parse_job_id(self, output: str) -> str: ...

    @abstractmethod
    def status_cmd(self, job_id: str) -> list[str]: ...

    @abstractmethod
    def parse_status(self, output: str) -> JobStatus: ...

    def detail_cmd(self, job_id: str) -> list[str] | None:
        """Command that fetches detailed accounting info, or None if unsupported."""
        return None

    def parse_detail(self, output: str) -> JobDetail | None:
        """Parse detail-command output, or None if unsupported / unparseable."""
        return None


class Slurm(Scheduler):
    def directive_prefix(self) -> str:
        return "#SBATCH"

    def submit_cmd(self) -> list[str]:
        return ["sbatch", "--parsable"]

    def parse_job_id(self, output: str) -> str:
        return output.strip()

    def status_cmd(self, job_id: str) -> list[str]:
        # -X suppresses jobsteps so each row is one allocation; for an array
        # job this yields exactly one row per task, which the aggregation
        # below relies on.
        # State%30 widens the column past sacct's 10-char default so that
        # long state names (CONFIGURING, OUT_OF_MEMORY, ...) are not
        # truncated to "CONFIGURI+" / "OUT_OF_ME+" — the truncation marker
        # would survive rstrip("+") as a prefix that misses _STATUS_MAP and
        # falls back to FAILED, exiting wait_for_job prematurely.
        return [
            "sacct",
            "-j",
            job_id,
            "--format=State%30",
            "--noheader",
            "-X",
        ]

    def parse_status(self, output: str) -> JobStatus:
        # Aggregate over all rows so an array job is reported as terminal
        # only once every task is terminal. A single non-terminal task in
        # the set must keep the aggregate non-terminal to prevent
        # wait_for_job from exiting prematurely.
        # Normalize each row to its first whitespace-separated token so
        # extended forms such as "CANCELLED by 12345" (emitted at the
        # widened State%30 column) reduce to "CANCELLED".
        lines = []
        for ln in output.strip().splitlines():
            tokens = ln.split()
            if tokens:
                lines.append(tokens[0].rstrip("+"))
        if not lines:
            return JobStatus.FAILED
        statuses = [_STATUS_MAP.get(s, JobStatus.FAILED) for s in lines]
        for status in (
            JobStatus.RUNNING,
            JobStatus.PENDING,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMEOUT,
        ):
            if status in statuses:
                return status
        return JobStatus.COMPLETED

    def detail_cmd(self, job_id: str) -> list[str] | None:
        return [
            "sacct",
            "-j",
            job_id,
            "--format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem",
            "--noheader",
            "-P",
        ]

    def parse_detail(self, output: str) -> JobDetail | None:
        # sacct -P emits one row per accounting record (the parent step plus
        # any sub-steps such as `.batch` / `.extern`). Columns appear in the
        # order requested via --format. -P (parsable2) does not add a trailing
        # '|', but we tolerate it so callers can pass -p output as well.
        rows: list[list[str]] = []
        for raw in output.splitlines():
            line = raw.strip()
            if not line:
                continue
            fields = line.split("|")
            if fields and fields[-1] == "":
                fields.pop()
            if len(fields) < 6:
                continue
            rows.append(fields[:6])

        if not rows:
            return None

        # Parent step rows have a JobID without a '.' separator; sub-steps
        # (`.batch`, `.extern`, `.0`, ...) always contain a dot.
        parent = next((r for r in rows if "." not in r[0]), None)
        if parent is None:
            return None

        # MaxRSS is per-step; the parent row typically reports an empty
        # MaxRSS, while `.batch` carries the script's peak RSS. Prefer the
        # `.batch` row's value, otherwise take the first non-empty across rows.
        batch = next((r for r in rows if r[0].endswith(".batch")), None)
        if batch is not None and batch[4]:
            max_rss = batch[4]
        else:
            max_rss = next((r[4] for r in rows if r[4]), "")

        state = parent[1]
        if not state:
            return None

        return JobDetail(
            state=state,
            exit_code=parent[2],
            elapsed=parent[3],
            max_rss=max_rss,
            req_mem=parent[5],
        )


class PJM(Scheduler):
    _STATUS_MAP = {
        "ACC": JobStatus.PENDING,
        "QUE": JobStatus.PENDING,
        "RNA": JobStatus.PENDING,
        "RNP": JobStatus.RUNNING,
        "RUN": JobStatus.RUNNING,
        "RNE": JobStatus.RUNNING,
        "RNO": JobStatus.RUNNING,
        "EXT": JobStatus.COMPLETED,
        "CCL": JobStatus.CANCELLED,
        "ERR": JobStatus.FAILED,
        "HLD": JobStatus.PENDING,
        "RJT": JobStatus.FAILED,
    }

    def directive_prefix(self) -> str:
        return "#PJM -L"

    def submit_cmd(self) -> list[str]:
        return ["pjsub"]

    def parse_job_id(self, output: str) -> str:
        # pjsub output example: "[INFO] PJM 0000 pjsub Job XXXXXXXX submitted."
        match = re.search(r"\bJob\s+(\d+)\b", output, flags=re.IGNORECASE)
        if match:
            return match.group(1)

        # Fallback: use the last numeric token to avoid picking status code "0000".
        numbers = re.findall(r"\d+", output)
        if numbers:
            return numbers[-1]

        return output.strip()

    def status_cmd(self, job_id: str) -> list[str]:
        return ["pjstat", "--choose", "st", job_id]

    def parse_status(self, output: str) -> JobStatus:
        lines = output.strip().splitlines()
        status_str = lines[1].strip() if len(lines) >= 2 else ""
        return self._STATUS_MAP.get(status_str, JobStatus.FAILED)


# Slurm job state strings as reported by `sacct`. Unknown states fall back
# to FAILED at the call site so that wait_for_job remains conservative;
# any state that should be treated as non-terminal must be listed here.
_STATUS_MAP = {
    # Non-terminal — pending/queued
    "PENDING": JobStatus.PENDING,
    "CONFIGURING": JobStatus.PENDING,
    "REQUEUED": JobStatus.PENDING,
    # Non-terminal — running/active
    "RUNNING": JobStatus.RUNNING,
    "RESIZING": JobStatus.RUNNING,
    "SUSPENDED": JobStatus.RUNNING,
    "COMPLETING": JobStatus.RUNNING,
    # Terminal — success
    "COMPLETED": JobStatus.COMPLETED,
    # Terminal — failure
    "FAILED": JobStatus.FAILED,
    "BOOT_FAIL": JobStatus.FAILED,
    "NODE_FAIL": JobStatus.FAILED,
    "OUT_OF_MEMORY": JobStatus.FAILED,
    # Terminal — cancelled
    "CANCELLED": JobStatus.CANCELLED,
    "PREEMPTED": JobStatus.CANCELLED,
    "REVOKED": JobStatus.CANCELLED,
    # Terminal — timeout
    "TIMEOUT": JobStatus.TIMEOUT,
    "DEADLINE": JobStatus.TIMEOUT,
}


def get_scheduler(name: str) -> Scheduler:
    schedulers = {"slurm": Slurm, "pjm": PJM}
    if name not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}")
    return schedulers[name]()
