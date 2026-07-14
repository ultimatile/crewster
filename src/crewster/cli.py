"""CLI command definitions"""

import os
import sys
import tomllib
from enum import Enum
from pathlib import Path
from typing import Optional

import tomli_w
import typer
from typing_extensions import Annotated

from .main import app
from .config import ConfigManager, HpcConfig, find_config
from .ssh import SSHManager
from .sync import SyncManager
from .job import JobManager, JobStatus
from .run import RunManager
from .scheduler import JobDetail, JobIdRejectedError, SchedulerError


class SchedulerChoice(str, Enum):
    """CLI choices for ``crewster init --scheduler``.

    ``str`` mix-in lets typer expose the enum as constrained string choices
    and lets ``.value`` be passed straight into the TOML payload.
    """

    slurm = "slurm"
    pjm = "pjm"


class DetailMode(str, Enum):
    """CLI choices for ``crewster status --detail``.

    ``summary`` (default) shows a single aggregate line for multi-task jobs;
    ``tasks`` adds one accounting block per array task / het component. The
    symmetric enum form leaves room for a future value (e.g. ``json``)
    without a breaking flag rename.
    """

    summary = "summary"
    tasks = "tasks"


# Type alias for config option
ConfigOption = Annotated[
    Optional[Path],
    typer.Option(
        "--config",
        "-c",
        help=(
            "Config file path. Its directory is the default project "
            "root (the local sync root)."
        ),
    ),
]
WorkdirOption = Annotated[
    Optional[str], typer.Option("--workdir", "-w", help="Override remote workdir")
]
ProjectDirOption = Annotated[
    Optional[Path],
    typer.Option(
        "--project-dir",
        help=(
            "Override the project root (default: the config file's directory). "
            "The project root is the local sync root, the base for the remote "
            "working-directory offset, and where .crewster/runs is stored."
        ),
    ),
]


def _warn_legacy(message: str) -> None:
    """Print a yellow deprecation warning to stderr.

    Matches the unknown-section warning style in ``config.py``. Used only for
    the implicit legacy config paths (``$HPC_CONFIG`` and ``hpc.toml``
    discovery); the read-only fallback they support is removed by v1.0.
    """
    print(f"\033[33mWarning: {message}\033[0m", file=sys.stderr)


def _resolve_config_path(
    config_path: Optional[Path],
    walk_up: bool = True,
    allow_legacy: bool = True,
) -> tuple[Path, bool]:
    """Resolve the config path, returning ``(path, explicit)``.

    Order: ``--config`` > ``$CREWSTER_CONFIG`` > ``$HPC_CONFIG`` > walk-up
    discovery (``crewster.toml`` then legacy ``hpc.toml``) > CWD
    ``crewster.toml``. The legacy env var and legacy filename are read-only
    fallbacks that emit a deprecation warning and are removed by v1.0.

    ``explicit`` is True when the user named the config directly (flag or an
    env var, the deprecated ``$HPC_CONFIG`` included) and False when it was
    discovered (walk-up or the CWD fallback). Provenance is reported here —
    the one place that owns the precedence order — so consumers such as the
    ``--project-dir`` guard cannot drift from the resolution branches.

    ``allow_legacy=False`` (used by ``init``) drops every legacy branch, so the
    command never resolves to ``$HPC_CONFIG`` nor to an ``hpc.toml`` name and
    can therefore never create the legacy file. An explicit ``--config`` is an
    arbitrary user path and never triggers a deprecation warning regardless of
    its filename.
    """
    # Explicit sources are expanded here so a tilde the shell did not expand
    # (quoted flag value, env var set outside a shell) resolves the same way
    # ``--project-dir`` and ``sync.pull_dir`` do. Discovered paths cannot
    # contain a tilde, so the walk-up / CWD branches need no expansion.
    if config_path:
        return _expand_user_path(config_path, "--config"), True
    if env_config := os.environ.get("CREWSTER_CONFIG"):
        return _expand_user_path(Path(env_config), "$CREWSTER_CONFIG"), True
    if allow_legacy and (env_config := os.environ.get("HPC_CONFIG")):
        _warn_legacy(
            "$HPC_CONFIG is deprecated; use $CREWSTER_CONFIG (removed in v1.0)"
        )
        return _expand_user_path(Path(env_config), "$HPC_CONFIG"), True
    if walk_up:
        names = ("crewster.toml", "hpc.toml") if allow_legacy else ("crewster.toml",)
        found = find_config(names)
        if found is not None:
            path, name = found
            if name == "hpc.toml":
                _warn_legacy(
                    f"{path} uses the legacy name 'hpc.toml'; "
                    "rename to 'crewster.toml' (removed in v1.0)"
                )
            return path, False
    return Path("crewster.toml"), False


def _expand_user_path(path: Path, label: str) -> Path:
    """``expanduser`` with the CLI's error convention.

    A tilde the shell did not expand (quoted, or injected from a script) must
    still resolve, but ``Path.expanduser`` raises ``RuntimeError`` for an
    unknown user (``~nosuchuser/...``) or an undeterminable home directory —
    turn that into the clean exit-1 error the surrounding validation promises
    instead of a traceback.
    """
    try:
        return path.expanduser()
    except RuntimeError:
        print(f"Cannot expand user in {label}: {path}")
        raise typer.Exit(1) from None


def _load_config(
    config_path: Optional[Path], project_dir: Optional[Path]
) -> tuple[Path, HpcConfig]:
    """Resolve and load config, returning the project root and the config.

    The project root defaults to the config file's parent directory, which
    keeps the config an in-tree anchor. ``project_dir`` overrides it for
    layouts where the config lives outside the working tree; the override is
    an explicit path only — never discovered, never implied by CWD. It is
    validated here so every consumer receives an existing, resolved directory
    (``relative_to`` comparisons downstream require an absolute path).
    ``project_dir`` is deliberately non-defaulted so every command states its
    choice — a command cannot accept ``--project-dir`` yet forget to forward it.

    ``project_dir`` additionally requires the config itself to be explicit
    (``--config`` or a config env var): pairing an explicit root with a
    walk-up-discovered config would let a stray ancestor ``crewster.toml``
    silently supply another project's cluster settings for the override tree.
    """
    path, explicit = _resolve_config_path(config_path)
    if project_dir is not None and not explicit:
        print(
            "--project-dir requires an explicit config: "
            "pass --config or set $CREWSTER_CONFIG"
        )
        raise typer.Exit(1)
    if not path.exists():
        print(f"Config file not found: {path}")
        raise typer.Exit(1)
    manager = ConfigManager()
    if project_dir is not None:
        # ``is_dir()`` is False for both a missing path and an existing
        # non-directory, so one check rejects both misuses.
        project_dir = _expand_user_path(project_dir, "--project-dir")
        if not project_dir.is_dir():
            print(f"Project directory not found or not a directory: {project_dir}")
            raise typer.Exit(1)
        project_root = project_dir.resolve()
    else:
        project_root = path.resolve().parent
    return project_root, manager.load_config(path)


def _print_run_not_found(id: str, run_manager: RunManager) -> None:
    """Report a failed run lookup with the directory it searched.

    A miss with no runs recorded at all is compatible with two causes — a
    project-root mismatch (the command was invoked with a different
    ``--project-dir`` / ``--config`` than at submit time, issue #44's
    scenario) or simply nothing submitted from this root yet — so the hint
    states the fact (no runs here) and phrases the remedy conditionally
    rather than presuming the mismatch. With runs present, the id itself is
    the likely problem and the hint would only misdirect. Emptiness is
    judged by ``list_runs`` — the manager's own definition of a recorded
    run — so stray entries (``.DS_Store``, a metadata-less dir left by an
    aborted submit) cannot suppress the hint, and the missing-directory
    case stays handled in one place.
    """
    print(f"Run not found: {id}")
    if not run_manager.list_runs():
        print(
            f"(no runs recorded under {run_manager.runs_dir} — if you "
            "submitted from a different project root, invoke with the same "
            "--project-dir/--config as at submit time)"
        )


def _get_user_config_path() -> Path:
    """Get user config path from XDG_CONFIG_HOME/crewster/config.toml"""
    xdg_config = os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
    return Path(xdg_config) / "crewster" / "config.toml"


def _apply_xdg_with_filter(src: Path, dst: Path, scheduler: SchedulerChoice) -> None:
    """Project XDG user-config onto the active scheduler and write to ``dst``.

    Drops the inactive scheduler's section so the resulting ``crewster.toml`` is
    consistent with ``--scheduler``, and forces ``cluster.scheduler`` to the
    active value. The source file itself is not modified.

    The destination inherits the source's permission bits so a restrictive
    XDG mode (e.g. ``0o600`` for a file carrying ``env`` export secrets) is
    not silently relaxed to the umask default — the previous ``shutil.copy``
    path preserved mode bits, and the rewrite must too.
    """
    with open(src, "rb") as f:
        data = tomllib.load(f)
    inactive = "pjm" if scheduler is SchedulerChoice.slurm else "slurm"
    data.pop(inactive, None)
    data.setdefault("cluster", {})["scheduler"] = scheduler.value
    src_mode = os.stat(src).st_mode & 0o777
    # ``os.open`` with the source mode avoids the exposure window where
    # ``open(dst, "wb")`` would briefly create ``dst`` with the umask-masked
    # default (often ``0o644``) before a subsequent ``chmod`` tightens it.
    # ``umask`` may still mask bits off the create, so chmod after to land
    # exactly on ``src_mode``; that direction only widens, never relaxes
    # past the source.
    fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, src_mode)
    with os.fdopen(fd, "wb") as f:
        tomli_w.dump(data, f)
    os.chmod(dst, src_mode)


@app.command()
def init(
    scheduler: Annotated[
        SchedulerChoice,
        typer.Option(
            "--scheduler",
            help=(
                "Target scheduler for the generated template. "
                "When a user-level XDG config exists, its inactive scheduler "
                "section is filtered out and cluster.scheduler is forced to "
                "this value."
            ),
        ),
    ] = SchedulerChoice.slurm,
    config: ConfigOption = None,
):
    """Initialize HPC project configuration"""
    # ``allow_legacy=False`` keeps init from ever resolving to $HPC_CONFIG or an
    # hpc.toml name, so it writes crewster.toml only and never the legacy file.
    config_path, _ = _resolve_config_path(config, walk_up=False, allow_legacy=False)
    if config_path.exists():
        print(f"Config file already exists: {config_path}")
        return

    user_config = _get_user_config_path()
    if user_config.exists():
        _apply_xdg_with_filter(user_config, config_path, scheduler)
        print(f"Applied {user_config} ({scheduler.value}): {config_path}")
    else:
        manager = ConfigManager()
        manager.generate_template(config_path, scheduler=scheduler.value)
        print(f"Created config file ({scheduler.value}): {config_path}")


@app.command()
def sync(
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Show what would be synced without executing"
    ),
    push: bool = typer.Option(False, "--push", help="Only push local to remote"),
    pull: bool = typer.Option(False, "--pull", help="Only pull remote to local"),
    workdir: WorkdirOption = None,
    config: ConfigOption = None,
    project_dir: ProjectDirOption = None,
):
    """Sync files bidirectionally with remote HPC cluster (push then pull)"""
    project_root, hpc_config = _load_config(config, project_dir)
    if workdir:
        hpc_config.cluster.workdir = workdir

    if push and pull:
        print("Error: cannot use --push and --pull together")
        raise typer.Exit(1)

    ssh = SSHManager(host=hpc_config.cluster.host)
    sync_manager = SyncManager(ssh_manager=ssh, config=hpc_config)
    local_path = project_root
    use_checksum = hpc_config.sync.compare == "checksum"

    # Default: bidirectional (push then pull)
    do_push = not pull
    do_pull = not push

    remote_exists = sync_manager.remote_dir_exists()

    # Create remote directory automatically when applying
    if not dry_run and do_push and not remote_exists:
        print("Creating remote directory...")
        sync_manager.ensure_remote_dir()
        remote_exists = True

    # Skip pull if remote dir doesn't exist
    if do_pull and not remote_exists:
        print("Remote directory does not exist, skipping pull")
        do_pull = False

    results = []

    if do_push:
        print("==> Push (local → remote)")
        results.append(
            sync_manager.sync_push(
                local_path=local_path, dry_run=dry_run, use_checksum=use_checksum
            )
        )
    if do_pull:
        pull_dir = None
        if hpc_config.sync.pull_dir:
            pull_dir = _expand_user_path(
                Path(hpc_config.sync.pull_dir), "sync.pull_dir"
            ).resolve()
            if not dry_run:
                pull_dir.mkdir(parents=True, exist_ok=True)
            print(f"==> Pull (remote → {pull_dir})")
        else:
            print("==> Pull (remote → local)")
        results.append(
            sync_manager.sync_pull(
                local_path=local_path,
                dry_run=dry_run,
                exclude_push_targets=dry_run and do_push,
                use_checksum=use_checksum,
                pull_dir=pull_dir,
            )
        )

    failed = [r for r in results if not r.success]
    if failed:
        for r in failed:
            if r.returncode == 255:
                print(
                    "Error: SSH connection failed. "
                    "Please check your SSH config and run 'crewster sync' again."
                )
            else:
                print(f"Error: rsync failed with exit code {r.returncode}")
        raise typer.Exit(1)

    if dry_run:
        print("Dry run completed.")
    else:
        print("Sync completed.")


@app.command(name="exec")
def exec_cmd(
    cmd: str = typer.Argument(None),
    script: Path = typer.Option(
        None, "--script", "-s", help="Shell script file to execute"
    ),
    workdir: Annotated[
        Optional[str], typer.Option("--workdir", help="Override remote workdir")
    ] = None,
    config: ConfigOption = None,
    project_dir: ProjectDirOption = None,
):
    """Execute a command on the login node (not via scheduler)"""
    project_root, hpc_config = _load_config(config, project_dir)
    if workdir:
        hpc_config.cluster.workdir = workdir

    if not cmd and not script:
        print("Error: provide a command or --script")
        raise typer.Exit(1)

    if script:
        if not script.exists():
            print(f"Script not found: {script}")
            raise typer.Exit(1)
        cmd = script.read_text()

    try:
        cwd_relative = Path.cwd().resolve().relative_to(project_root)
    except ValueError:
        cwd_relative = Path(".")

    ssh = SSHManager(host=hpc_config.cluster.host)

    # Resolve ~ in workdir
    remote_workdir = hpc_config.cluster.workdir
    if remote_workdir.startswith("~/") or remote_workdir == "~":
        result = ssh.run_command("printenv", ["HOME"])
        home = result.stdout.strip()
        remote_workdir = (
            remote_workdir.replace("~", home, 1) if remote_workdir != "~" else home
        )
    job_workdir = str(Path(remote_workdir) / cwd_relative)

    # Build script: set -e → cd → env setup → user command
    import shlex

    lines = [
        "set -e",
        f"cd {shlex.quote(job_workdir)}",
        *hpc_config.env.get_setup_commands(),
        cmd,
    ]
    script_text = "\n".join(lines)

    returncode = ssh.run_script(script_text)
    if returncode != 0:
        raise typer.Exit(returncode)


@app.command()
def submit(
    cmd: str = typer.Argument(None),
    script: Path = typer.Option(
        None, "--script", "-s", help="Shell script file to submit"
    ),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for job completion"),
    workdir: Annotated[
        Optional[str], typer.Option("--workdir", help="Override remote workdir")
    ] = None,
    config: ConfigOption = None,
    project_dir: ProjectDirOption = None,
):
    """Submit a job to the scheduler"""
    project_root, hpc_config = _load_config(config, project_dir)
    if workdir:
        hpc_config.cluster.workdir = workdir

    if not cmd and not script:
        print("Error: provide a command or --script")
        raise typer.Exit(1)

    if script:
        if not script.exists():
            print(f"Script not found: {script}")
            raise typer.Exit(1)
        cmd = script.read_text()

    # Get git commit if in a git repo
    ssh = SSHManager(host=hpc_config.cluster.host)
    sync_manager = SyncManager(ssh_manager=ssh, config=hpc_config)
    git_commit = sync_manager.get_git_commit(project_root, short=True)

    if sync_manager.has_uncommitted_changes(project_root):
        print("Warning: uncommitted changes detected")

    # Compute CWD offset from project root for job working directory
    try:
        cwd_relative = Path.cwd().resolve().relative_to(project_root)
    except ValueError:
        cwd_relative = Path(".")

    runs_dir = project_root / ".crewster" / "runs"
    run_manager = RunManager(config=hpc_config, runs_dir=runs_dir)
    run = run_manager.create_run(cmd, git_commit=git_commit)

    job_manager = JobManager(ssh_manager=ssh, config=hpc_config)

    job_id = job_manager.submit_run(run, cwd_relative=cwd_relative)
    run.job_id = job_id
    run.status = "submitted"
    run_manager.save_run_meta(run)

    print(f"Submitted run: {run.run_id}")
    print(f"Job ID: {job_id}")
    if git_commit:
        print(f"Git commit: {git_commit}")

    if wait:
        print("Waiting for job completion...")
        status = job_manager.wait_for_job(job_id, adaptive=True)
        run.status = status.value.lower()
        run_manager.save_run_meta(run)
        if status == JobStatus.UNKNOWN:
            print(
                f"Job {job_id}: final state unknown (scheduler stopped returning data)"
            )
        else:
            print(f"Job finished: {status.value}")
        if status != JobStatus.COMPLETED:
            raise typer.Exit(1)


# Slurm State values that mean the job has finished and the accounting
# fields (ExitCode / Elapsed / MaxRSS / ReqMem) are meaningful to display.
_TERMINAL_SACCT_STATES = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "BOOT_FAIL",
        "NODE_FAIL",
        "PREEMPTED",
        "DEADLINE",
        "REVOKED",
        "SPECIAL_EXIT",
    }
)


def _normalize_sacct_state(state: str) -> str:
    """Strip Slurm State decorations like ``CANCELLED+`` / ``CANCELLED by 12345``."""
    head = state.split()[0] if state else ""
    return head.rstrip("+")


def _print_detail_fields(d: JobDetail, indent: str) -> None:
    """Print the four accounting fields, but only for terminal states where
    they are meaningful. Non-terminal states (RUNNING / PENDING) print just
    the state line at the call site."""
    if _normalize_sacct_state(d.state) in _TERMINAL_SACCT_STATES:
        print(f"{indent}ExitCode: {d.exit_code or '-'}")
        print(f"{indent}Elapsed:  {d.elapsed or '-'}")
        print(f"{indent}MaxRSS:   {d.max_rss or '-'}")
        print(f"{indent}ReqMem:   {d.req_mem or '-'}")


@app.command()
def status(
    id: str = typer.Argument(None),
    detail: DetailMode = typer.Option(
        DetailMode.summary,
        "--detail",
        help="For array / het jobs: 'summary' (aggregate line) or 'tasks' (per-task blocks)",
    ),
    config: ConfigOption = None,
    project_dir: ProjectDirOption = None,
):
    """Check job status (accepts run_id or job_id)"""
    project_root, hpc_config = _load_config(config, project_dir)

    if not id:
        print("Please specify a run_id or job_id")
        raise typer.Exit(1)

    runs_dir = project_root / ".crewster" / "runs"
    run_manager = RunManager(config=hpc_config, runs_dir=runs_dir)

    # Try as run_id first, then as job_id
    try:
        run = run_manager.load_run_meta(id)
        job_id = run.job_id
    except FileNotFoundError:
        run = run_manager.find_run_by_job_id(id)
        job_id = id

    if run and not job_id:
        print(f"Run {run.run_id} has no job ID")
        raise typer.Exit(1)

    ssh = SSHManager(host=hpc_config.cluster.host)
    job_manager = JobManager(ssh_manager=ssh, config=hpc_config)

    # One try covers both scheduler queries so a SchedulerError — parse-side
    # absence or affirmative id rejection, whichever query it comes from —
    # is handled at its first observation, without re-querying the scheduler.
    # Other SSHError (real transport/command failure) propagates so
    # legitimate failures stay visible to the user.
    try:
        details = job_manager.get_job_detail(job_id)
        if not details:
            # Scheduler does not support detail (PJM -> None), or sacct has
            # not yet recorded this job ([] -> no usable row). Fall back to
            # the single-line display.
            job_status = job_manager.get_job_status(job_id)
            print(f"Job {job_id}: {job_status.value}")
            return
    except SchedulerError as e:
        if run is None:
            # The id failed both lookups: no local run metadata and no
            # scheduler data (or the scheduler rejected the id outright).
            # That is indistinguishable from accounting lag on a raw
            # just-submitted job id, but run metadata lives under the
            # project root used at submit time, so a wrong
            # --project-dir/--config produces exactly this double miss.
            # Report the metadata miss the way job-output/wait do rather
            # than implying the scheduler knows the job.
            _print_run_not_found(id, run_manager)
            raise typer.Exit(1)
        if isinstance(e, JobIdRejectedError):
            # Run metadata exists but the scheduler rejects its job id —
            # deterministic, so "not ready yet" would misdirect: the id
            # will never become queryable (stale or corrupt metadata).
            print(
                f"Job {job_id}: the scheduler rejected this job id"
                f" (run {run.run_id} metadata may be stale or corrupt)"
            )
            raise typer.Exit(1)
        print(f"Job {job_id}: status unavailable yet (scheduler accounting not ready)")
        return

    if len(details) == 1:
        # Single job: unchanged from the pre-array display.
        d = details[0]
        print(f"Job {job_id}: {d.state}")
        _print_detail_fields(d, indent="  ")
        return

    # Array / het job: always surface mixed outcomes via an aggregate line so
    # a failed task is never hidden behind the first task's state. Counts are
    # grouped by normalized state in first-appearance order, so Slurm
    # decorations (`CANCELLED+`, `CANCELLED by <uid>`) collapse into one bucket
    # instead of fragmenting the breakdown; per-task blocks below keep the raw
    # state.
    counts: dict[str, int] = {}
    for d in details:
        key = _normalize_sacct_state(d.state)
        counts[key] = counts.get(key, 0) + 1
    unit = "components" if any("+" in d.job_id for d in details) else "tasks"
    breakdown = ", ".join(f"{n} {state}" for state, n in counts.items())
    print(f"Job {job_id}: {len(details)} {unit} ({breakdown})")

    if detail == DetailMode.tasks:
        for d in details:
            print(f"  {d.job_id}: {d.state}")
            _print_detail_fields(d, indent="    ")


@app.command(name="list")
def list_runs(config: ConfigOption = None, project_dir: ProjectDirOption = None):
    """List all runs"""
    project_root, hpc_config = _load_config(config, project_dir)

    runs_dir = project_root / ".crewster" / "runs"
    run_manager = RunManager(config=hpc_config, runs_dir=runs_dir)
    runs = run_manager.list_runs()

    if not runs:
        print("No runs found.")
        return

    for run in runs:
        job_info = f" (job: {run.job_id})" if run.job_id else ""
        print(f"{run.run_id}: {run.status}{job_info} - {run.cmd}")


@app.command(name="job-output")
def job_output(
    id: str,
    error: bool = typer.Option(
        False, "--error", "-e", help="Show stderr instead of stdout"
    ),
    follow: bool = typer.Option(
        False,
        "--follow",
        "-f",
        help="Stream output as the job runs (tail -F equivalent)",
    ),
    config: ConfigOption = None,
    project_dir: ProjectDirOption = None,
):
    """Show job output (accepts run_id or job_id)"""
    project_root, hpc_config = _load_config(config, project_dir)

    runs_dir = project_root / ".crewster" / "runs"
    run_manager = RunManager(config=hpc_config, runs_dir=runs_dir)

    # Try as run_id first, then as job_id
    try:
        run = run_manager.load_run_meta(id)
    except FileNotFoundError:
        run = run_manager.find_run_by_job_id(id)

    if not run:
        _print_run_not_found(id, run_manager)
        raise typer.Exit(1)

    if not run.job_id:
        print(f"Run {run.run_id} has no job ID")
        raise typer.Exit(1)

    ssh = SSHManager(host=hpc_config.cluster.host)
    job_manager = JobManager(ssh_manager=ssh, config=hpc_config)

    if follow:
        rc = job_manager.tail_job_output(run.run_id, run.job_id, error=error)
        if rc != 0:
            raise typer.Exit(rc)
        return

    output = job_manager.get_job_output(run.run_id, run.job_id, error=error)
    print(output, end="")


@app.command()
def wait(id: str, config: ConfigOption = None, project_dir: ProjectDirOption = None):
    """Wait for a run to complete (accepts run_id or job_id)"""
    project_root, hpc_config = _load_config(config, project_dir)

    runs_dir = project_root / ".crewster" / "runs"
    run_manager = RunManager(config=hpc_config, runs_dir=runs_dir)

    # Try as run_id first, then as job_id
    try:
        run = run_manager.load_run_meta(id)
    except FileNotFoundError:
        run = run_manager.find_run_by_job_id(id)

    if not run:
        _print_run_not_found(id, run_manager)
        raise typer.Exit(1)

    if not run.job_id:
        print(f"Run {run.run_id} has no job ID")
        raise typer.Exit(1)

    ssh = SSHManager(host=hpc_config.cluster.host)
    job_manager = JobManager(ssh_manager=ssh, config=hpc_config)

    print(f"Waiting for job {run.job_id}...")
    status = job_manager.wait_for_job(run.job_id, adaptive=True)
    run.status = status.value.lower()
    run_manager.save_run_meta(run)
    if status == JobStatus.UNKNOWN:
        print(
            f"Job {run.job_id}: final state unknown (scheduler stopped returning data)"
        )
    else:
        print(f"Job finished: {status.value}")
    if status != JobStatus.COMPLETED:
        raise typer.Exit(1)
