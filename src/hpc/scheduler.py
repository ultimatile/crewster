"""Scheduler abstraction for Slurm and PJM"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from .ssh import SSHError


class SchedulerError(SSHError):
    """Scheduler returned no usable response.

    Raised by ``parse_status`` when the scheduler's status command exits
    cleanly but its output contains no data row from which a ``JobStatus``
    can be derived — typically a freshly-submitted job that has not yet
    been indexed by the accounting database (Slurm's ``sacct``) or a job
    that aged out of the active view (PJM's ``pjstat`` post-``EXT``
    window). Distinguishing this from a real ``JobStatus.FAILED`` lets
    callers retry / fall through instead of treating the absence of
    information as a terminal failure.

    Inherits from ``SSHError`` so the transient-catch sites in
    ``JobManager`` cover it. ``get_job_status`` catches it to try the
    scheduler's ``status_fallback_cmd`` (e.g. PJM's history view) before
    surfacing the absence. ``wait_for_job`` catches it to apply a bounded
    retry budget — persistent absence terminates as ``JobStatus.UNKNOWN``
    rather than looping forever, while transient absence keeps polling.
    ``tail_job_output``'s pre-tail status probe falls through to
    ``tail -F``, and the inner status probe inside ``get_job_output``
    swallows it so the original "No such file" diagnostic re-raises. CLI
    surfaces that want to discriminate "no data yet" from a real SSH /
    command failure catch ``SchedulerError`` specifically before the
    generic ``SSHError``.
    """


class JobStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    # Synthesized only by ``JobManager.wait_for_job`` when the scheduler
    # produces no usable status output for a bounded number of consecutive
    # polls (a never-indexed / never-submitted job, or a PJM job that aged
    # out of every view). It is never returned by ``parse_status`` /
    # ``get_job_status``, which raise ``SchedulerError`` for that absence
    # instead; UNKNOWN is the terminal interpretation the polling loop
    # applies once the retry budget is exhausted.
    UNKNOWN = "UNKNOWN"


@dataclass
class JobDetail:
    """Raw scheduler-side job accounting fields for one job / task / component.

    Strings are stored verbatim from the scheduler so the user sees
    the exact value (e.g. ``OUT_OF_MEMORY``, ``CANCELLED+``) without
    any enum/unit normalization.

    ``job_id`` is the canonical scheduler JobID of this specific row
    (e.g. ``12345`` for a plain job, ``12345_0`` for an array task,
    ``12345+0`` for a heterogeneous-job component) so a multi-task
    result can be labeled and grouped per task.
    """

    job_id: str
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
    def parse_status(self, output: str, job_id: str | None = None) -> JobStatus:
        """Map status-command output to a ``JobStatus``.

        ``job_id`` is optional and, when given, lets a scheduler whose
        output carries a job-id column reject rows that do not belong to
        the requested job — a fail-closed guard for the fallback path
        (see ``status_fallback_cmd``). Schedulers whose output has no
        job-id column ignore it. Raise ``SchedulerError`` when no usable
        data row is present."""
        ...

    def status_fallback_cmd(self, job_id: str) -> list[str] | None:
        """Secondary status command, tried only when the primary
        ``status_cmd`` output yields no data row (``SchedulerError``).

        Returns ``None`` when the scheduler has no fallback source, which
        leaves the primary ``SchedulerError`` to propagate unchanged. The
        fallback output is parsed with the requested ``job_id`` so a
        differing column layout cannot be misread as a real status."""
        return None

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

    def parse_detail(self, output: str) -> list[JobDetail]:
        """Parse detail-command output into one ``JobDetail`` per job / task /
        component, in scheduler order. Returns ``[]`` when the scheduler has
        no detail support or the output has no usable row. Concrete base
        (not abstract) so schedulers without a detail source inherit ``[]``."""
        return []


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

    def parse_status(self, output: str, job_id: str | None = None) -> JobStatus:
        # ``sacct --format=State`` output has no job-id column, so the
        # ``job_id`` argument (accepted for the Scheduler contract) is
        # unused here.
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
            # No parseable row — sacct has not yet indexed the job (fresh
            # submission, accounting lag). Surface as a transient absence
            # rather than synthesizing a terminal FAILED.
            raise SchedulerError(
                "sacct returned no data row; job may not yet be indexed"
            )
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

    def parse_detail(self, output: str) -> list[JobDetail]:
        # sacct -P emits one row per accounting record: a parent row per job /
        # array task / het component (JobID without a '.') plus that row's
        # sub-steps (`.batch`, `.extern`, `.0`, ... — always containing a '.').
        # Columns appear in the --format order. -P (parsable2) omits the
        # trailing '|', but we tolerate it so -p output also parses.
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

        # A bracket-range JobID (e.g. `12345_[2-99]`) is sacct's compressed
        # summary of un-launched array elements. Its presence means the array
        # is only partially launched, so a per-task accounting view would be
        # incomplete: the range can neither be counted as one task (it stands
        # for many) nor silently dropped (that would hide pending tasks and
        # misreport, e.g., a half-pending array as COMPLETED). Defer the whole
        # job to the caller's aggregate-status fallback by returning no detail
        # rows; complete per-task detail returns once every element has
        # launched (no range row remains).
        if any("[" in r[0] for r in rows):
            return []

        # One JobDetail per parent row, in sacct order. Array tasks
        # (`12345_0`, `12345_1`) and het components (`12345+0`, `12345+1`)
        # each appear as their own parent row, so every task/component is
        # represented rather than only the first.
        details: list[JobDetail] = []
        for parent in rows:
            job_id = parent[0]
            if "." in job_id:
                continue  # sub-step row; attributed to its parent below
            state = parent[1]
            if not state:
                continue  # parent not yet recorded (no usable state)
            # MaxRSS is per-step and the parent row is usually empty; for a
            # plain job the peak RSS lives on `.batch`, while srun-dispatched
            # work reports it on numbered steps (`.0`, ...). Aggregate the max
            # over this task's own rows only — the parent plus its sub-steps
            # (`<job_id>.<step>`) — so per-task RSS never bleeds across tasks.
            group = [parent] + [r for r in rows if r[0].startswith(job_id + ".")]
            non_empty = [r[4] for r in group if r[4]]
            max_rss = max(non_empty, key=_max_rss_to_bytes, default="")
            details.append(
                JobDetail(
                    job_id=job_id,
                    state=state,
                    exit_code=parent[2],
                    elapsed=parent[3],
                    max_rss=max_rss,
                    req_mem=parent[5],
                )
            )
        return details


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

    def status_fallback_cmd(self, job_id: str) -> list[str] | None:
        # When the active view (``status_cmd``) has no row — a job that
        # aged out of it after ``EXT`` — the history view (``-H``) may
        # still hold the record. Tried only on the ``SchedulerError``
        # path, so the extra round-trip never burdens the normal query.
        # Same ``--choose`` columns as the primary command, so the same
        # ``parse_status`` applies; the caller passes ``job_id`` so a
        # differing ``-H`` layout fails closed rather than misreporting.
        return ["pjstat", "-H", "-v", "--choose", "jid,st,ec,sn", job_id]

    def parse_status(self, output: str, job_id: str | None = None) -> JobStatus:
        # Expected output shape (header row + one data row per job):
        #     JOB_ID     ST  EC  SN
        #     48969021   EXT 0   0
        # The first parseable row wins; multi-row aggregation for array
        # / step jobs is tracked separately (Issue #12) and out of scope
        # here. An empty / header-only / short-row-only response raises
        # ``SchedulerError`` so callers can distinguish "no data yet"
        # from a real terminal failure.
        #
        # When ``job_id`` is given (the fallback path), a row is trusted
        # only if its first column — the ``jid`` selected via ``--choose``
        # — equals the requested id. This fails closed: a history view
        # whose columns drift yields no matching row and raises
        # ``SchedulerError`` instead of mapping an unrelated column to a
        # bogus terminal status.
        for raw in output.splitlines():
            tokens = raw.split()
            if len(tokens) < 4 or tokens[0] == "JOB_ID":
                continue
            if job_id is not None and tokens[0] != job_id:
                continue
            st, ec, sn = tokens[1], tokens[2], tokens[3]
            if st == "EXT":
                return (
                    JobStatus.COMPLETED if ec == "0" and sn == "0" else JobStatus.FAILED
                )
            return self._STATUS_MAP.get(st, JobStatus.FAILED)
        raise SchedulerError(
            "pjstat returned no data row; job may not yet be indexed"
            " or has aged out of the active view"
        )


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
