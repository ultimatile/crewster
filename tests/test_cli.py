"""CLI command tests"""

from pathlib import Path
from unittest.mock import patch, MagicMock

from crewster.main import app
from crewster import cli  # noqa: F401 - register commands


def test_help(cli_runner):
    result = cli_runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "HPC job execution support tool" in result.stdout


def test_init_command_exists(cli_runner):
    result = cli_runner.invoke(app, ["init", "--help"])
    assert result.exit_code == 0


def test_init_creates_config_file(cli_runner, temp_dir, monkeypatch):
    monkeypatch.chdir(temp_dir)
    result = cli_runner.invoke(app, ["init"])
    assert result.exit_code == 0
    config_path = temp_dir / "crewster.toml"
    assert config_path.exists()
    content = config_path.read_text()
    assert "[cluster]" in content


def test_init_scheduler_default_slurm(cli_runner, temp_dir, monkeypatch):
    """No ``--scheduler`` keeps the Slurm shape (regression guard for default)."""
    monkeypatch.chdir(temp_dir)
    result = cli_runner.invoke(app, ["init"])
    assert result.exit_code == 0
    content = (temp_dir / "crewster.toml").read_text()
    assert "[slurm.options]" in content
    assert "[pjm" not in content
    assert 'scheduler = "slurm"' in content


def test_init_scheduler_pjm_template(cli_runner, temp_dir, monkeypatch):
    """``--scheduler pjm`` emits a PJM-shaped template."""
    monkeypatch.chdir(temp_dir)
    result = cli_runner.invoke(app, ["init", "--scheduler", "pjm"])
    assert result.exit_code == 0
    content = (temp_dir / "crewster.toml").read_text()
    assert "[pjm]" in content
    assert 'scheduler = "pjm"' in content
    assert "[slurm" not in content


def test_init_scheduler_rejects_unknown_value(cli_runner, temp_dir, monkeypatch):
    """typer constrains ``--scheduler`` to known enum values."""
    monkeypatch.chdir(temp_dir)
    result = cli_runner.invoke(app, ["init", "--scheduler", "lsf"])
    assert result.exit_code != 0


def _write_xdg(temp_dir, monkeypatch, body: str) -> Path:
    """Place ``body`` at ``$XDG_CONFIG_HOME/crewster/config.toml`` and return it."""
    xdg = temp_dir / "xdg"
    (xdg / "crewster").mkdir(parents=True)
    user_cfg = xdg / "crewster" / "config.toml"
    user_cfg.write_text(body)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    return user_cfg


def test_init_xdg_filter_slurm(cli_runner, temp_dir, monkeypatch):
    """XDG carrying both scheduler sections drops [pjm] under --scheduler slurm."""
    monkeypatch.chdir(temp_dir)
    _write_xdg(
        temp_dir,
        monkeypatch,
        """
[cluster]
host = "myhpc"
workdir = "/scratch/me"

[slurm.options]
partition = "gpu"

[pjm]
options = [["-L", "node=1"]]
""",
    )
    result = cli_runner.invoke(app, ["init"])
    assert result.exit_code == 0
    content = (temp_dir / "crewster.toml").read_text()
    assert "[slurm.options]" in content
    assert "[pjm" not in content
    assert 'scheduler = "slurm"' in content


def test_init_xdg_filter_pjm(cli_runner, temp_dir, monkeypatch):
    """Same XDG drops [slurm.*] under --scheduler pjm."""
    monkeypatch.chdir(temp_dir)
    _write_xdg(
        temp_dir,
        monkeypatch,
        """
[cluster]
host = "myhpc"
workdir = "/scratch/me"

[slurm.options]
partition = "gpu"

[pjm]
options = [["-L", "node=1"]]
""",
    )
    result = cli_runner.invoke(app, ["init", "--scheduler", "pjm"])
    assert result.exit_code == 0
    content = (temp_dir / "crewster.toml").read_text()
    assert "[pjm]" in content
    assert "[slurm" not in content
    assert 'scheduler = "pjm"' in content


def test_init_xdg_overrides_cluster_scheduler(cli_runner, temp_dir, monkeypatch):
    """XDG's cluster.scheduler is rewritten to match --scheduler."""
    monkeypatch.chdir(temp_dir)
    _write_xdg(
        temp_dir,
        monkeypatch,
        """
[cluster]
host = "myhpc"
workdir = "/scratch/me"
scheduler = "slurm"

[pjm]
options = [["-L", "node=1"]]
""",
    )
    result = cli_runner.invoke(app, ["init", "--scheduler", "pjm"])
    assert result.exit_code == 0
    content = (temp_dir / "crewster.toml").read_text()
    assert 'scheduler = "pjm"' in content
    assert 'scheduler = "slurm"' not in content


def test_init_xdg_preserves_unknown_sections(cli_runner, temp_dir, monkeypatch):
    """Sections other than the inactive scheduler pass through unchanged.

    The filter only removes the inactive scheduler's top-level table; anything
    else in XDG (including sections not recognized by ``ConfigManager``) is
    preserved verbatim.
    """
    monkeypatch.chdir(temp_dir)
    _write_xdg(
        temp_dir,
        monkeypatch,
        """
[cluster]
host = "myhpc"
workdir = "/scratch/me"

[slurm.options]
partition = "gpu"

[custom]
note = "preserve me"
""",
    )
    result = cli_runner.invoke(app, ["init"])
    assert result.exit_code == 0
    content = (temp_dir / "crewster.toml").read_text()
    assert "[custom]" in content
    assert 'note = "preserve me"' in content


def test_init_xdg_only_one_section(cli_runner, temp_dir, monkeypatch):
    """XDG holding only [slurm.*] plus --scheduler pjm degenerates to a valid
    scheduler-less file; load path falls back to PJM defaults without error."""
    monkeypatch.chdir(temp_dir)
    _write_xdg(
        temp_dir,
        monkeypatch,
        """
[cluster]
host = "myhpc"
workdir = "/scratch/me"

[slurm.options]
partition = "gpu"
""",
    )
    result = cli_runner.invoke(app, ["init", "--scheduler", "pjm"])
    assert result.exit_code == 0
    content = (temp_dir / "crewster.toml").read_text()
    assert "[slurm" not in content
    assert "[pjm" not in content
    assert 'scheduler = "pjm"' in content

    from crewster.config import ConfigManager

    loaded = ConfigManager().load_config(temp_dir / "crewster.toml")
    assert loaded.cluster.scheduler == "pjm"
    assert loaded.pjm.options == []
    assert loaded.pjm.submit_options == []


def test_init_xdg_preserves_source_mode(cli_runner, temp_dir, monkeypatch):
    """Restrictive permissions on the XDG file carry over to ``crewster.toml``.

    Guards against silently widening a ``0o600`` user config to ``0o644``
    via the umask of the rewrite ``open(dst, "wb")``.
    """
    import os
    import stat

    monkeypatch.chdir(temp_dir)
    user_cfg = _write_xdg(
        temp_dir,
        monkeypatch,
        """
[cluster]
host = "myhpc"
workdir = "/scratch/me"

[slurm.options]
partition = "gpu"
""",
    )
    os.chmod(user_cfg, 0o600)
    result = cli_runner.invoke(app, ["init"])
    assert result.exit_code == 0
    dst_mode = stat.S_IMODE((temp_dir / "crewster.toml").stat().st_mode)
    assert dst_mode == 0o600


def test_init_xdg_source_not_modified(cli_runner, temp_dir, monkeypatch):
    """Filter merge reads XDG; the source file must remain byte-identical."""
    monkeypatch.chdir(temp_dir)
    body = """
[cluster]
host = "myhpc"
workdir = "/scratch/me"

[slurm.options]
partition = "gpu"

[pjm]
options = [["-L", "node=1"]]
"""
    user_cfg = _write_xdg(temp_dir, monkeypatch, body)
    before = user_cfg.read_text()
    result = cli_runner.invoke(app, ["init", "--scheduler", "pjm"])
    assert result.exit_code == 0
    assert user_cfg.read_text() == before


def test_sync_command_exists(cli_runner):
    result = cli_runner.invoke(app, ["sync", "--help"])
    assert result.exit_code == 0


def test_sync_requires_config(cli_runner, temp_dir, monkeypatch):
    monkeypatch.chdir(temp_dir)
    result = cli_runner.invoke(app, ["sync"])
    assert result.exit_code != 0
    assert "Config file not found" in result.stdout or "crewster.toml" in result.stdout


def test_submit_command_exists(cli_runner):
    result = cli_runner.invoke(app, ["submit", "--help"])
    assert result.exit_code == 0


def test_submit_requires_config(cli_runner, temp_dir, monkeypatch):
    monkeypatch.chdir(temp_dir)
    result = cli_runner.invoke(app, ["submit", "python train.py"])
    assert result.exit_code != 0


def test_submit_requires_cmd_or_script(cli_runner, temp_dir, monkeypatch):
    monkeypatch.chdir(temp_dir)
    (temp_dir / "crewster.toml").write_text(
        "[cluster]\nhost = 'test'\nworkdir = '/tmp'"
    )
    result = cli_runner.invoke(app, ["submit"])
    assert result.exit_code != 0
    assert "provide a command or --script" in result.stdout


def test_submit_script_not_found(cli_runner, temp_dir, monkeypatch):
    monkeypatch.chdir(temp_dir)
    (temp_dir / "crewster.toml").write_text(
        "[cluster]\nhost = 'test'\nworkdir = '/tmp'"
    )
    result = cli_runner.invoke(app, ["submit", "--script", "nonexistent.sh"])
    assert result.exit_code != 0
    assert "Script not found" in result.stdout


def test_status_command_exists(cli_runner):
    result = cli_runner.invoke(app, ["status", "--help"])
    assert result.exit_code == 0


def test_status_prints_sacct_fields_for_terminal_job(cli_runner, temp_dir, monkeypatch):
    """Terminal Slurm jobs show ExitCode/Elapsed/MaxRSS/ReqMem in addition to state."""
    from crewster.scheduler import JobDetail

    monkeypatch.chdir(temp_dir)
    cli_runner.invoke(app, ["init"])

    with patch("crewster.cli.JobManager") as MockJobManager:
        instance = MockJobManager.return_value
        instance.get_job_detail.return_value = [
            JobDetail(
                job_id="12345678",
                state="OUT_OF_MEMORY",
                exit_code="0:125",
                elapsed="00:01:23",
                max_rss="1024K",
                req_mem="16Gn",
            )
        ]
        result = cli_runner.invoke(app, ["status", "12345678"])

    assert result.exit_code == 0
    assert "Job 12345678: OUT_OF_MEMORY" in result.stdout
    assert "ExitCode: 0:125" in result.stdout
    assert "Elapsed:  00:01:23" in result.stdout
    assert "MaxRSS:   1024K" in result.stdout
    assert "ReqMem:   16Gn" in result.stdout


def test_status_omits_sacct_fields_for_running_job(cli_runner, temp_dir, monkeypatch):
    """Running jobs print only the state line; runtime fields are not yet meaningful."""
    from crewster.scheduler import JobDetail

    monkeypatch.chdir(temp_dir)
    cli_runner.invoke(app, ["init"])

    with patch("crewster.cli.JobManager") as MockJobManager:
        instance = MockJobManager.return_value
        instance.get_job_detail.return_value = [
            JobDetail(
                job_id="12345678",
                state="RUNNING",
                exit_code="0:0",
                elapsed="00:00:30",
                max_rss="",
                req_mem="16Gn",
            )
        ]
        result = cli_runner.invoke(app, ["status", "12345678"])

    assert result.exit_code == 0
    assert "Job 12345678: RUNNING" in result.stdout
    assert "ExitCode" not in result.stdout
    assert "Elapsed" not in result.stdout


def test_status_falls_back_when_detail_unavailable(cli_runner, temp_dir, monkeypatch):
    """PJM (and not-yet-recorded jobs) fall back to the single-line display."""
    from crewster.job import JobStatus

    monkeypatch.chdir(temp_dir)
    cli_runner.invoke(app, ["init"])

    with patch("crewster.cli.JobManager") as MockJobManager:
        instance = MockJobManager.return_value
        instance.get_job_detail.return_value = None
        instance.get_job_status.return_value = JobStatus.PENDING
        result = cli_runner.invoke(app, ["status", "12345678"])

    assert result.exit_code == 0
    assert "Job 12345678: PENDING" in result.stdout
    assert "ExitCode" not in result.stdout


def test_status_prints_friendly_message_when_status_unavailable(
    cli_runner, temp_dir, monkeypatch
):
    """SchedulerError on the fallback ``get_job_status`` becomes a friendly
    "status unavailable yet" line instead of an SSHError stack."""
    from crewster.scheduler import SchedulerError

    monkeypatch.chdir(temp_dir)
    cli_runner.invoke(app, ["init"])

    with patch("crewster.cli.JobManager") as MockJobManager:
        instance = MockJobManager.return_value
        instance.get_job_detail.return_value = None
        instance.get_job_status.side_effect = SchedulerError(
            "sacct returned no data row"
        )
        result = cli_runner.invoke(app, ["status", "12345678"])

    assert result.exit_code == 0
    assert "Job 12345678: status unavailable yet" in result.stdout
    assert "scheduler accounting not ready" in result.stdout


def test_status_normalizes_decorated_cancelled_state(cli_runner, temp_dir, monkeypatch):
    """``CANCELLED+`` and ``CANCELLED by 12345`` should still trigger detail rendering."""
    from crewster.scheduler import JobDetail

    monkeypatch.chdir(temp_dir)
    cli_runner.invoke(app, ["init"])

    with patch("crewster.cli.JobManager") as MockJobManager:
        instance = MockJobManager.return_value
        instance.get_job_detail.return_value = [
            JobDetail(
                job_id="12345678",
                state="CANCELLED by 12345",
                exit_code="0:15",
                elapsed="00:00:42",
                max_rss="",
                req_mem="8Gn",
            )
        ]
        result = cli_runner.invoke(app, ["status", "12345678"])

    assert result.exit_code == 0
    assert "Job 12345678: CANCELLED by 12345" in result.stdout
    assert "ExitCode: 0:15" in result.stdout
    assert "MaxRSS:   -" in result.stdout


def test_status_array_job_aggregate_line_by_default(cli_runner, temp_dir, monkeypatch):
    """Issue #16: a multi-task array surfaces mixed outcomes via an aggregate
    line by default, never collapsing to the first task's state. No per-task
    accounting fields without --detail tasks."""
    from crewster.scheduler import JobDetail

    monkeypatch.chdir(temp_dir)
    cli_runner.invoke(app, ["init"])

    with patch("crewster.cli.JobManager") as MockJobManager:
        instance = MockJobManager.return_value
        instance.get_job_detail.return_value = [
            JobDetail(
                job_id="12345_0",
                state="COMPLETED",
                exit_code="0:0",
                elapsed="00:01:00",
                max_rss="1024K",
                req_mem="16Gn",
            ),
            JobDetail(
                job_id="12345_1",
                state="COMPLETED",
                exit_code="0:0",
                elapsed="00:01:00",
                max_rss="1024K",
                req_mem="16Gn",
            ),
            JobDetail(
                job_id="12345_2",
                state="OUT_OF_MEMORY",
                exit_code="0:125",
                elapsed="00:00:30",
                max_rss="2048K",
                req_mem="16Gn",
            ),
        ]
        result = cli_runner.invoke(app, ["status", "12345"])

    assert result.exit_code == 0
    assert "Job 12345: 3 tasks (2 COMPLETED, 1 OUT_OF_MEMORY)" in result.stdout
    # No per-task drill-down in the default (summary) mode.
    assert "12345_2:" not in result.stdout
    assert "ExitCode" not in result.stdout


def test_status_array_job_per_task_blocks_with_detail_tasks(
    cli_runner, temp_dir, monkeypatch
):
    """--detail tasks adds one accounting block per task, labeled by the
    canonical JobID, with terminal fields shown per task."""
    from crewster.scheduler import JobDetail

    monkeypatch.chdir(temp_dir)
    cli_runner.invoke(app, ["init"])

    with patch("crewster.cli.JobManager") as MockJobManager:
        instance = MockJobManager.return_value
        instance.get_job_detail.return_value = [
            JobDetail(
                job_id="12345_0",
                state="COMPLETED",
                exit_code="0:0",
                elapsed="00:01:00",
                max_rss="1024K",
                req_mem="16Gn",
            ),
            JobDetail(
                job_id="12345_1",
                state="OUT_OF_MEMORY",
                exit_code="0:125",
                elapsed="00:00:30",
                max_rss="2048K",
                req_mem="16Gn",
            ),
        ]
        result = cli_runner.invoke(app, ["status", "12345", "--detail", "tasks"])

    assert result.exit_code == 0
    assert "Job 12345: 2 tasks (1 COMPLETED, 1 OUT_OF_MEMORY)" in result.stdout
    assert "12345_0: COMPLETED" in result.stdout
    assert "12345_1: OUT_OF_MEMORY" in result.stdout
    # Per-task accounting for each terminal task.
    assert "MaxRSS:   1024K" in result.stdout
    assert "MaxRSS:   2048K" in result.stdout


def test_status_array_aggregate_groups_normalized_states(
    cli_runner, temp_dir, monkeypatch
):
    """Decorated Slurm states (CANCELLED+ / CANCELLED by <uid>) must collapse
    into one aggregate bucket rather than fragmenting the breakdown."""
    from crewster.scheduler import JobDetail

    monkeypatch.chdir(temp_dir)
    cli_runner.invoke(app, ["init"])

    with patch("crewster.cli.JobManager") as MockJobManager:
        instance = MockJobManager.return_value
        instance.get_job_detail.return_value = [
            JobDetail(
                job_id="12345_0",
                state="CANCELLED+",
                exit_code="0:15",
                elapsed="00:00:03",
                max_rss="",
                req_mem="8Gn",
            ),
            JobDetail(
                job_id="12345_1",
                state="CANCELLED by 4011",
                exit_code="0:15",
                elapsed="00:00:03",
                max_rss="",
                req_mem="8Gn",
            ),
            JobDetail(
                job_id="12345_2",
                state="COMPLETED",
                exit_code="0:0",
                elapsed="00:01:00",
                max_rss="1024K",
                req_mem="8Gn",
            ),
        ]
        result = cli_runner.invoke(app, ["status", "12345"])

    assert result.exit_code == 0
    assert "Job 12345: 3 tasks (2 CANCELLED, 1 COMPLETED)" in result.stdout


def test_status_empty_detail_list_falls_back(cli_runner, temp_dir, monkeypatch):
    """An empty detail list (supported scheduler, no row yet) falls back to the
    single-line status display, same as the None (unsupported) case."""
    from crewster.job import JobStatus

    monkeypatch.chdir(temp_dir)
    cli_runner.invoke(app, ["init"])

    with patch("crewster.cli.JobManager") as MockJobManager:
        instance = MockJobManager.return_value
        instance.get_job_detail.return_value = []
        instance.get_job_status.return_value = JobStatus.PENDING
        result = cli_runner.invoke(app, ["status", "12345678"])

    assert result.exit_code == 0
    assert "Job 12345678: PENDING" in result.stdout


def test_wait_reports_unknown_state_and_exits_nonzero(
    cli_runner, temp_dir, monkeypatch
):
    """Issue #24: when wait_for_job exhausts its retry budget it returns
    JobStatus.UNKNOWN; the CLI prints an explicit unknown-state line,
    persists the run as ``unknown``, and exits non-zero."""
    from crewster.job import JobStatus

    monkeypatch.chdir(temp_dir)
    cli_runner.invoke(app, ["init"])

    run = MagicMock()
    run.run_id = "test_run"
    run.job_id = "12345678"

    with (
        patch("crewster.cli.JobManager") as MockJobManager,
        patch("crewster.cli.RunManager") as MockRunManager,
    ):
        MockRunManager.return_value.load_run_meta.return_value = run
        MockJobManager.return_value.wait_for_job.return_value = JobStatus.UNKNOWN
        result = cli_runner.invoke(app, ["wait", "test_run"])

    assert result.exit_code == 1
    assert "final state unknown" in result.stdout
    assert run.status == "unknown"


def test_submit_wait_reports_unknown_state_and_exits_nonzero(
    cli_runner, temp_dir, monkeypatch
):
    """The submit ``--wait`` path shares the same UNKNOWN handling as the
    standalone ``wait`` command."""
    from crewster.job import JobStatus

    monkeypatch.chdir(temp_dir)
    cli_runner.invoke(app, ["init"])

    with patch("crewster.cli.JobManager") as MockJobManager:
        instance = MockJobManager.return_value
        instance.submit_run.return_value = "12345678"
        instance.wait_for_job.return_value = JobStatus.UNKNOWN
        result = cli_runner.invoke(app, ["submit", "python train.py", "--wait"])

    assert result.exit_code == 1
    assert "final state unknown" in result.stdout


def test_config_option(cli_runner, temp_dir, monkeypatch):
    """Test --config option loads specified config file"""
    monkeypatch.chdir(temp_dir)
    custom_config = temp_dir / "custom.toml"
    custom_config.write_text("[cluster]\nhost = 'test'\nworkdir = '/tmp'")
    result = cli_runner.invoke(app, ["--config", str(custom_config), "sync"])
    # Should not fail with "Config file not found" since custom.toml exists
    assert "Config file not found" not in result.stdout


def test_config_env_var(cli_runner, temp_dir, monkeypatch):
    """Test CREWSTER_CONFIG environment variable"""
    monkeypatch.chdir(temp_dir)
    custom_config = temp_dir / "env.toml"
    custom_config.write_text("[cluster]\nhost = 'test'\nworkdir = '/tmp'")
    monkeypatch.setenv("CREWSTER_CONFIG", str(custom_config))
    result = cli_runner.invoke(app, ["sync"])
    assert "Config file not found" not in result.stdout


def test_config_option_overrides_env(cli_runner, temp_dir, monkeypatch):
    """Test --config takes precedence over CREWSTER_CONFIG"""
    monkeypatch.chdir(temp_dir)
    opt_config = temp_dir / "opt.toml"
    opt_config.write_text("[cluster]\nhost = 'test'\nworkdir = '/tmp'")
    monkeypatch.setenv("CREWSTER_CONFIG", "nonexistent.toml")
    result = cli_runner.invoke(app, ["--config", str(opt_config), "sync"])
    assert "Config file not found" not in result.stdout


# --- Config-resolution fallback contract (crewster.toml <- legacy hpc.toml) ---
# The deprecation warning must fire ONLY on the implicit legacy paths
# ($HPC_CONFIG, hpc.toml discovery); an explicit --config never warns.


def test_resolve_crewster_env_wins_over_legacy(temp_dir, monkeypatch, capsys):
    """$CREWSTER_CONFIG beats $HPC_CONFIG and emits no deprecation warning."""
    monkeypatch.chdir(temp_dir)
    monkeypatch.setenv("CREWSTER_CONFIG", "/new/crewster.toml")
    monkeypatch.setenv("HPC_CONFIG", "/old/hpc.toml")
    path = cli._resolve_config_path(None)
    assert path == Path("/new/crewster.toml")
    assert capsys.readouterr().err == ""


def test_resolve_legacy_env_warns(temp_dir, monkeypatch, capsys):
    """A lone $HPC_CONFIG is honored but warns on stderr."""
    monkeypatch.chdir(temp_dir)
    monkeypatch.setenv("HPC_CONFIG", "/old/hpc.toml")
    path = cli._resolve_config_path(None)
    assert path == Path("/old/hpc.toml")
    assert "HPC_CONFIG" in capsys.readouterr().err


def test_resolve_explicit_config_never_warns(temp_dir, monkeypatch, capsys):
    """An explicit --config path is used verbatim and never warns,
    even when its basename is the legacy hpc.toml."""
    monkeypatch.chdir(temp_dir)
    legacy = temp_dir / "hpc.toml"
    path = cli._resolve_config_path(legacy)
    assert path == legacy
    assert capsys.readouterr().err == ""


def test_resolve_walk_up_crewster_no_warn(temp_dir, monkeypatch, capsys):
    """Walk-up discovery of crewster.toml emits no warning."""
    (temp_dir / "crewster.toml").write_text("x")
    monkeypatch.chdir(temp_dir)
    path = cli._resolve_config_path(None)
    assert path == (temp_dir / "crewster.toml").resolve()
    assert capsys.readouterr().err == ""


def test_resolve_walk_up_legacy_warns(temp_dir, monkeypatch, capsys):
    """Walk-up discovery falling back to hpc.toml warns on stderr."""
    (temp_dir / "hpc.toml").write_text("x")
    monkeypatch.chdir(temp_dir)
    path = cli._resolve_config_path(None)
    assert path == (temp_dir / "hpc.toml").resolve()
    assert "hpc.toml" in capsys.readouterr().err


def test_resolve_crewster_wins_same_dir(temp_dir, monkeypatch, capsys):
    """When both names sit in one directory, crewster.toml wins silently."""
    (temp_dir / "crewster.toml").write_text("x")
    (temp_dir / "hpc.toml").write_text("x")
    monkeypatch.chdir(temp_dir)
    path = cli._resolve_config_path(None)
    assert path == (temp_dir / "crewster.toml").resolve()
    assert capsys.readouterr().err == ""


def test_resolve_nearest_legacy_beats_ancestor_crewster(temp_dir, monkeypatch, capsys):
    """A nearer hpc.toml is used even when an ancestor holds crewster.toml
    (nearest-config-wins regression guard)."""
    (temp_dir / "crewster.toml").write_text("x")
    child = temp_dir / "sub"
    child.mkdir()
    (child / "hpc.toml").write_text("x")
    monkeypatch.chdir(child)
    path = cli._resolve_config_path(None)
    assert path == (child / "hpc.toml").resolve()
    assert "hpc.toml" in capsys.readouterr().err


def test_init_with_legacy_env_writes_crewster(cli_runner, temp_dir, monkeypatch):
    """``crewster init`` writes crewster.toml in CWD even when $HPC_CONFIG is set,
    never resolving to or creating the legacy file."""
    monkeypatch.chdir(temp_dir)
    monkeypatch.setenv("HPC_CONFIG", str(temp_dir / "legacy" / "hpc.toml"))
    result = cli_runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert (temp_dir / "crewster.toml").exists()
    assert not (temp_dir / "legacy" / "hpc.toml").exists()


def test_walk_up_finds_config(cli_runner, temp_dir, monkeypatch):
    """Walk-up discovery finds crewster.toml in parent directory"""
    (temp_dir / "crewster.toml").write_text(
        "[cluster]\nhost = 'test'\nworkdir = '/tmp'"
    )
    child = temp_dir / "runs" / "bench1"
    child.mkdir(parents=True)
    monkeypatch.chdir(child)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = cli_runner.invoke(app, ["sync"])
        assert result.exit_code == 0
        assert "Config file not found" not in result.stdout


def test_sync_uses_project_root(cli_runner, temp_dir, monkeypatch):
    """sync uses project root (crewster.toml location) as local path, not CWD"""
    (temp_dir / "crewster.toml").write_text(
        "[cluster]\nhost = 'test'\nworkdir = '/tmp'"
    )
    child = temp_dir / "runs" / "bench1"
    child.mkdir(parents=True)
    monkeypatch.chdir(child)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        cli_runner.invoke(app, ["sync"])
        # rsync should use project root, not the child directory
        call_args = mock_run.call_args[0][0]
        local_arg = [a for a in call_args if str(temp_dir.resolve()) in str(a)]
        assert local_arg
        # Should NOT contain the child subpath as the source
        assert not any(str(child) in str(a) for a in call_args)


def test_init_does_not_walk_up(cli_runner, temp_dir, monkeypatch):
    """init creates crewster.toml in CWD, does not walk up"""
    (temp_dir / "crewster.toml").write_text(
        "[cluster]\nhost = 'test'\nworkdir = '/tmp'"
    )
    child = temp_dir / "subdir"
    child.mkdir()
    monkeypatch.chdir(child)
    result = cli_runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert (child / "crewster.toml").exists()


def test_job_output_follow_in_help(cli_runner):
    result = cli_runner.invoke(app, ["job-output", "--help"])
    assert result.exit_code == 0
    assert "--follow" in result.stdout
    assert "-f" in result.stdout


def _setup_run_meta(temp_dir, run_id="r1", job_id="12345678"):
    """Write crewster.toml + a fake run meta so job-output can resolve the run."""
    (temp_dir / "crewster.toml").write_text(
        "[cluster]\nhost = 'test'\nworkdir = '/tmp'\n"
    )
    runs_dir = temp_dir / ".crewster" / "runs" / run_id
    runs_dir.mkdir(parents=True)
    (runs_dir / "meta.toml").write_text(
        f'run_id = "{run_id}"\n'
        f'cmd = "echo hi"\n'
        f'status = "running"\n'
        f'job_id = "{job_id}"\n'
    )


def test_job_output_follow_passes_error_flag(cli_runner, temp_dir, monkeypatch):
    monkeypatch.chdir(temp_dir)
    _setup_run_meta(temp_dir)

    with patch("crewster.cli.JobManager") as mock_job_cls:
        mock_job = MagicMock()
        mock_job.tail_job_output.return_value = 0
        mock_job_cls.return_value = mock_job

        result = cli_runner.invoke(app, ["job-output", "-f", "-e", "r1"])
        assert result.exit_code == 0
        mock_job.tail_job_output.assert_called_once_with("r1", "12345678", error=True)
        mock_job.get_job_output.assert_not_called()


def test_job_output_follow_propagates_exit_code(cli_runner, temp_dir, monkeypatch):
    monkeypatch.chdir(temp_dir)
    _setup_run_meta(temp_dir)

    with patch("crewster.cli.JobManager") as mock_job_cls:
        mock_job = MagicMock()
        mock_job.tail_job_output.return_value = 130  # Ctrl-C
        mock_job_cls.return_value = mock_job

        result = cli_runner.invoke(app, ["job-output", "-f", "r1"])
        assert result.exit_code == 130


def test_job_output_without_follow_uses_get_job_output(
    cli_runner, temp_dir, monkeypatch
):
    """Regression guard: --follow=False must still use the cat-based path."""
    monkeypatch.chdir(temp_dir)
    _setup_run_meta(temp_dir)

    with patch("crewster.cli.JobManager") as mock_job_cls:
        mock_job = MagicMock()
        mock_job.get_job_output.return_value = "static output\n"
        mock_job_cls.return_value = mock_job

        result = cli_runner.invoke(app, ["job-output", "r1"])
        assert result.exit_code == 0
        mock_job.get_job_output.assert_called_once_with("r1", "12345678", error=False)
        mock_job.tail_job_output.assert_not_called()
        assert "static output" in result.stdout
