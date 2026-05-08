"""Scheduler abstraction for Slurm and PJM"""

import re
from abc import ABC, abstractmethod
from enum import Enum


class JobStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"


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
        lines = [
            ln.strip().rstrip("+") for ln in output.strip().splitlines() if ln.strip()
        ]
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
