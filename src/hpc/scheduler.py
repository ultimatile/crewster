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


_MAX_RSS_UNITS = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def _max_rss_to_bytes(value: str) -> int:
    """Parse a Slurm MaxRSS field (e.g. ``1024K``, ``2.5G``, ``0``) to bytes.

    Unparseable inputs sort as 0 so they lose any max-comparison.
    """
    if not value:
        return 0
    suffix = value[-1].upper()
    if suffix in _MAX_RSS_UNITS:
        try:
            return int(float(value[:-1]) * _MAX_RSS_UNITS[suffix])
        except ValueError:
            return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


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

    @abstractmethod
    def output_directives(self, run_dir: str) -> list[str]:
        """Bookkeeping directives that route the job's stdout/stderr under
        ``run_dir``. Each scheduler's emitted path must be the one
        ``output_path`` returns; the two methods share one source of truth
        per scheduler so the directive write target and the JobManager read
        target stay in sync."""
        ...

    @abstractmethod
    def output_path(self, run_dir: str, job_id: str, error: bool = False) -> str:
        """The on-disk path produced by ``output_directives``.

        Mirrors directive semantics so ``JobManager.get_job_output`` /
        ``tail_job_output`` can read the file back. Schedulers whose
        directives are job-id-independent (PJM with fixed names) ignore
        the ``job_id`` argument."""
        ...

    def detail_cmd(self, job_id: str) -> list[str] | None:
        """Command that fetches detailed accounting info, or None if unsupported."""
        return None

    def parse_detail(self, output: str) -> JobDetail | None:
        """Parse detail-command output, or None if unsupported / unparseable."""
        return None


class Slurm(Scheduler):
    def directive_prefix(self) -> str:
        return "#SBATCH"

    def output_directives(self, run_dir: str) -> list[str]:
        return [
            f"#SBATCH --output={run_dir}/job-%j.out",
            f"#SBATCH --error={run_dir}/job-%j.err",
        ]

    def output_path(self, run_dir: str, job_id: str, error: bool = False) -> str:
        ext = "err" if error else "out"
        return f"{run_dir}/job-{job_id}.{ext}"

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
        # MaxRSS. For plain-cmd jobs the script's peak RSS lives on `.batch`,
        # but jobs that dispatch the workload via `srun` get the real RSS
        # on numbered step rows (`.0`, `.1`, ...) while `.batch` only
        # reflects the launcher. Pick the maximum across all non-empty
        # step rows so both shapes report correctly.
        non_empty = [r[4] for r in rows if r[4]]
        max_rss = max(non_empty, key=_max_rss_to_bytes, default="")

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
    # ``ST`` is only a coarse classification of the scheduler-side state.
    # ``EXT`` in particular means "exited", which covers both successful and
    # abnormal completions; ``parse_status`` consults ``EC`` and ``SN``
    # before applying the mapping for that case. The ``"EXT"`` entry is
    # retained here to document the bare-``ST`` semantics and so that any
    # future caller that reads the map directly still classifies it as a
    # terminal state, but normal status resolution goes through
    # ``parse_status``.
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

    def output_directives(self, run_dir: str) -> list[str]:
        # pjsub's ``-o`` / ``-e`` use the path argument literally; no
        # ``%j`` / ``%J`` substitution comparable to Slurm is part of the
        # documented behavior, so fixed filenames are used and the caller
        # is responsible for choosing a ``run_dir`` that disambiguates
        # across submissions.
        return [
            f"#PJM -o {run_dir}/job.out",
            f"#PJM -e {run_dir}/job.err",
        ]

    def output_path(self, run_dir: str, job_id: str, error: bool = False) -> str:
        # ``job_id`` is not part of the path because the directive uses a
        # literal fixed name. Accepted to match the Scheduler signature.
        ext = "err" if error else "out"
        return f"{run_dir}/job.{ext}"

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
        # Request ``ST`` together with ``EC`` (exit code) and ``SN``
        # (terminating signal). With ``ST=EXT`` alone, a successful exit
        # (``EC=0, SN=0``) and an abnormal one (``EC!=0`` or ``SN!=0``)
        # are indistinguishable, so ``parse_status`` cannot surface
        # script failures without these extra columns. ``-H`` is
        # deliberately omitted: the default non-history view briefly
        # retains ``EXT`` after completion, so a single query covers
        # both active and just-completed jobs, whereas ``-H`` would hide
        # currently-active jobs.
        return ["pjstat", "-v", "--choose", "jid,st,ec,sn", job_id]

    def parse_status(self, output: str) -> JobStatus:
        # Expected output shape (header row + one data row per job):
        #     JOB_ID     ST  EC  SN
        #     48969021   EXT 0   0
        # The first parseable row wins; multi-row aggregation for array
        # / step jobs is tracked separately (Issue #12) and out of scope
        # here. An empty / header-only response falls back to ``FAILED``,
        # the same fallback the pre-EC/SN parser used (Issue #8 owns
        # the principled fix).
        for raw in output.splitlines():
            tokens = raw.split()
            if len(tokens) < 4 or tokens[0] == "JOB_ID":
                continue
            st, ec, sn = tokens[1], tokens[2], tokens[3]
            if st == "EXT":
                return (
                    JobStatus.COMPLETED if ec == "0" and sn == "0" else JobStatus.FAILED
                )
            return self._STATUS_MAP.get(st, JobStatus.FAILED)
        return JobStatus.FAILED


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
