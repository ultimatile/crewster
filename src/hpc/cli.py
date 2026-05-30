"""CLI command definitions"""

import os
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
from .scheduler import JobDetail, SchedulerError


class SchedulerChoice(str, Enum):
    """CLI choices for ``hpc init --scheduler``.

    ``str`` mix-in lets typer expose the enum as constrained string choices
    and lets ``.value`` be passed straight into the TOML payload.
    """

    slurm = "slurm"
    pjm = "pjm"


class DetailMode(str, Enum):
    """CLI choices for ``hpc status --detail``.

    ``summary`` (default) shows a single aggregate line for multi-task jobs;
    ``tasks`` adds one accounting block per array task / het component. The
    symmetric enum form leaves room for a future value (e.g. ``json``)
    without a breaking flag rename.
    """

    summary = "summary"
    tasks = "tasks"


# Type alias for config option
ConfigOption = Annotated[
    Optional[Path], typer.Option("--config", "-c", help="Config file path")
]
WorkdirOption = Annotated[
    Optional[str], typer.Option("--workdir", "-w", help="Override remote workdir")
]


def _resolve_config_path(config_path: Optional[Path], walk_up: bool = True) -> Path:
    """Resolve config path: --config > $HPC_CONFIG > walk-up discovery > CWD"""
    if config_path:
        return config_path
    if env_config := os.environ.get("HPC_CONFIG"):
        return Path(env_config)
    if walk_up:
        found = find_config("hpc.toml")
        if found is not None:
            return found
    return Path("hpc.toml")


def _load_config(config_path: Optional[Path]) -> tuple[Path, Path, HpcConfig]:
    """Resolve and load config, returning config path, project root, and config"""
    path = _resolve_config_path(config_path)
    if not path.exists():
        print(f"Config file not found: {path}")
        raise typer.Exit(1)
    manager = ConfigManager()
    project_root = path.resolve().parent
    return path, project_root, manager.load_config(path)


def _get_user_config_path() -> Path:
    """Get user config path from XDG_CONFIG_HOME/hpc/config.toml"""
    xdg_config = os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
    return Path(xdg_config) / "hpc" / "config.toml"


def _apply_xdg_with_filter(src: Path, dst: Path, scheduler: SchedulerChoice) -> None:
    """Project XDG user-config onto the active scheduler and write to ``dst``.

    Drops the inactive scheduler's section so the resulting ``hpc.toml`` is
    consistent with ``--scheduler``, and forces ``cluster.scheduler`` to the
    active value. The source file itself is not modified.

    The destination inherits the source's permission bits so a restrictive
    XDG mode (e.g. ``0o600`` for a file carrying ``env.exports`` secrets) is
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
    config_path = _resolve_config_path(config, walk_up=False)
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
):
    """Sync files bidirectionally with remote HPC cluster (push then pull)"""
    config_path, project_root, hpc_config = _load_config(config)
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
            pull_dir = Path(hpc_config.sync.pull_dir).expanduser().resolve()
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
                    "Please check your SSH config and run 'hpc sync' again."
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
):
    """Execute a command on the login node (not via scheduler)"""
    config_path, project_root, hpc_config = _load_config(config)
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
):
    """Submit a job to the scheduler"""
    config_path, project_root, hpc_config = _load_config(config)
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

    runs_dir = project_root / ".hpc" / "runs"
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
):
    """Check job status (accepts run_id or job_id)"""
    config_path, project_root, hpc_config = _load_config(config)

    if not id:
        print("Please specify a run_id or job_id")
        raise typer.Exit(1)

    runs_dir = project_root / ".hpc" / "runs"
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

    if not job_id:
        print(f"Run not found: {id}")
        raise typer.Exit(1)

    ssh = SSHManager(host=hpc_config.cluster.host)
    job_manager = JobManager(ssh_manager=ssh, config=hpc_config)

    details = job_manager.get_job_detail(job_id)
    if not details:
        # Scheduler does not support detail (PJM -> None), or sacct has not yet
        # recorded this job ([] -> no usable row). Fall back to the existing
        # single-line display. SchedulerError (parse-side absence) becomes a
        # friendly message; other SSHError (real transport/command failure)
        # propagates so legitimate failures stay visible to the user.
        try:
            job_status = job_manager.get_job_status(job_id)
        except SchedulerError:
            print(
                f"Job {job_id}: status unavailable yet (scheduler accounting not ready)"
            )
            return
        print(f"Job {job_id}: {job_status.value}")
        return

    if len(details) == 1:
        # Single job: unchanged from the pre-array display.
        d = details[0]
        print(f"Job {job_id}: {d.state}")
        _print_detail_fields(d, indent="  ")
        return

    # Array / het job: always surface mixed outcomes via an aggregate line so
    # a failed task is never hidden behind the first task's state. Counts are
    # grouped by state in first-appearance order.
    counts: dict[str, int] = {}
    for d in details:
        counts[d.state] = counts.get(d.state, 0) + 1
    unit = "components" if any("+" in d.job_id for d in details) else "tasks"
    breakdown = ", ".join(f"{n} {state}" for state, n in counts.items())
    print(f"Job {job_id}: {len(details)} {unit} ({breakdown})")

    if detail == DetailMode.tasks:
        for d in details:
            print(f"  {d.job_id}: {d.state}")
            _print_detail_fields(d, indent="    ")


@app.command(name="list")
def list_runs(config: ConfigOption = None):
    """List all runs"""
    config_path, project_root, hpc_config = _load_config(config)

    runs_dir = project_root / ".hpc" / "runs"
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
):
    """Show job output (accepts run_id or job_id)"""
    config_path, project_root, hpc_config = _load_config(config)

    runs_dir = project_root / ".hpc" / "runs"
    run_manager = RunManager(config=hpc_config, runs_dir=runs_dir)

    # Try as run_id first, then as job_id
    try:
        run = run_manager.load_run_meta(id)
    except FileNotFoundError:
        run = run_manager.find_run_by_job_id(id)

    if not run:
        print(f"Run not found: {id}")
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
def wait(id: str, config: ConfigOption = None):
    """Wait for a run to complete (accepts run_id or job_id)"""
    config_path, project_root, hpc_config = _load_config(config)

    runs_dir = project_root / ".hpc" / "runs"
    run_manager = RunManager(config=hpc_config, runs_dir=runs_dir)

    # Try as run_id first, then as job_id
    try:
        run = run_manager.load_run_meta(id)
    except FileNotFoundError:
        run = run_manager.find_run_by_job_id(id)

    if not run:
        print(f"Run not found: {id}")
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
