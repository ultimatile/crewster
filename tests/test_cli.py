"""CLI command tests"""

from unittest.mock import patch, MagicMock

from hpc.main import app
from hpc import cli  # noqa: F401 - register commands


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
    config_path = temp_dir / "hpc.toml"
    assert config_path.exists()
    content = config_path.read_text()
    assert "[cluster]" in content


def test_sync_command_exists(cli_runner):
    result = cli_runner.invoke(app, ["sync", "--help"])
    assert result.exit_code == 0


def test_sync_requires_config(cli_runner, temp_dir, monkeypatch):
    monkeypatch.chdir(temp_dir)
    result = cli_runner.invoke(app, ["sync"])
    assert result.exit_code != 0
    assert "Config file not found" in result.stdout or "hpc.toml" in result.stdout


def test_submit_command_exists(cli_runner):
    result = cli_runner.invoke(app, ["submit", "--help"])
    assert result.exit_code == 0


def test_submit_requires_config(cli_runner, temp_dir, monkeypatch):
    monkeypatch.chdir(temp_dir)
    result = cli_runner.invoke(app, ["submit", "python train.py"])
    assert result.exit_code != 0


def test_submit_requires_cmd_or_script(cli_runner, temp_dir, monkeypatch):
    monkeypatch.chdir(temp_dir)
    (temp_dir / "hpc.toml").write_text("[cluster]\nhost = 'test'\nworkdir = '/tmp'")
    result = cli_runner.invoke(app, ["submit"])
    assert result.exit_code != 0
    assert "provide a command or --script" in result.stdout


def test_submit_script_not_found(cli_runner, temp_dir, monkeypatch):
    monkeypatch.chdir(temp_dir)
    (temp_dir / "hpc.toml").write_text("[cluster]\nhost = 'test'\nworkdir = '/tmp'")
    result = cli_runner.invoke(app, ["submit", "--script", "nonexistent.sh"])
    assert result.exit_code != 0
    assert "Script not found" in result.stdout


def test_status_command_exists(cli_runner):
    result = cli_runner.invoke(app, ["status", "--help"])
    assert result.exit_code == 0


def test_status_prints_sacct_fields_for_terminal_job(cli_runner, temp_dir, monkeypatch):
    """Terminal Slurm jobs show ExitCode/Elapsed/MaxRSS/ReqMem in addition to state."""
    from hpc.scheduler import JobDetail

    monkeypatch.chdir(temp_dir)
    cli_runner.invoke(app, ["init"])

    with patch("hpc.cli.JobManager") as MockJobManager:
        instance = MockJobManager.return_value
        instance.get_job_detail.return_value = JobDetail(
            state="OUT_OF_MEMORY",
            exit_code="0:125",
            elapsed="00:01:23",
            max_rss="1024K",
            req_mem="16Gn",
        )
        result = cli_runner.invoke(app, ["status", "12345678"])

    assert result.exit_code == 0
    assert "Job 12345678: OUT_OF_MEMORY" in result.stdout
    assert "ExitCode: 0:125" in result.stdout
    assert "Elapsed:  00:01:23" in result.stdout
    assert "MaxRSS:   1024K" in result.stdout
    assert "ReqMem:   16Gn" in result.stdout


def test_status_omits_sacct_fields_for_running_job(cli_runner, temp_dir, monkeypatch):
    """Running jobs print only the state line; runtime fields are not yet meaningful."""
    from hpc.scheduler import JobDetail

    monkeypatch.chdir(temp_dir)
    cli_runner.invoke(app, ["init"])

    with patch("hpc.cli.JobManager") as MockJobManager:
        instance = MockJobManager.return_value
        instance.get_job_detail.return_value = JobDetail(
            state="RUNNING",
            exit_code="0:0",
            elapsed="00:00:30",
            max_rss="",
            req_mem="16Gn",
        )
        result = cli_runner.invoke(app, ["status", "12345678"])

    assert result.exit_code == 0
    assert "Job 12345678: RUNNING" in result.stdout
    assert "ExitCode" not in result.stdout
    assert "Elapsed" not in result.stdout


def test_status_falls_back_when_detail_unavailable(cli_runner, temp_dir, monkeypatch):
    """PJM (and not-yet-recorded jobs) fall back to the single-line display."""
    from hpc.job import JobStatus

    monkeypatch.chdir(temp_dir)
    cli_runner.invoke(app, ["init"])

    with patch("hpc.cli.JobManager") as MockJobManager:
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
    from hpc.scheduler import SchedulerError

    monkeypatch.chdir(temp_dir)
    cli_runner.invoke(app, ["init"])

    with patch("hpc.cli.JobManager") as MockJobManager:
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
    from hpc.scheduler import JobDetail

    monkeypatch.chdir(temp_dir)
    cli_runner.invoke(app, ["init"])

    with patch("hpc.cli.JobManager") as MockJobManager:
        instance = MockJobManager.return_value
        instance.get_job_detail.return_value = JobDetail(
            state="CANCELLED by 12345",
            exit_code="0:15",
            elapsed="00:00:42",
            max_rss="",
            req_mem="8Gn",
        )
        result = cli_runner.invoke(app, ["status", "12345678"])

    assert result.exit_code == 0
    assert "Job 12345678: CANCELLED by 12345" in result.stdout
    assert "ExitCode: 0:15" in result.stdout
    assert "MaxRSS:   -" in result.stdout


def test_config_option(cli_runner, temp_dir, monkeypatch):
    """Test --config option loads specified config file"""
    monkeypatch.chdir(temp_dir)
    custom_config = temp_dir / "custom.toml"
    custom_config.write_text("[cluster]\nhost = 'test'\nworkdir = '/tmp'")
    result = cli_runner.invoke(app, ["--config", str(custom_config), "sync"])
    # Should not fail with "Config file not found" since custom.toml exists
    assert "Config file not found" not in result.stdout


def test_config_env_var(cli_runner, temp_dir, monkeypatch):
    """Test HPC_CONFIG environment variable"""
    monkeypatch.chdir(temp_dir)
    custom_config = temp_dir / "env.toml"
    custom_config.write_text("[cluster]\nhost = 'test'\nworkdir = '/tmp'")
    monkeypatch.setenv("HPC_CONFIG", str(custom_config))
    result = cli_runner.invoke(app, ["sync"])
    assert "Config file not found" not in result.stdout


def test_config_option_overrides_env(cli_runner, temp_dir, monkeypatch):
    """Test --config takes precedence over HPC_CONFIG"""
    monkeypatch.chdir(temp_dir)
    opt_config = temp_dir / "opt.toml"
    opt_config.write_text("[cluster]\nhost = 'test'\nworkdir = '/tmp'")
    monkeypatch.setenv("HPC_CONFIG", "nonexistent.toml")
    result = cli_runner.invoke(app, ["--config", str(opt_config), "sync"])
    assert "Config file not found" not in result.stdout


def test_walk_up_finds_config(cli_runner, temp_dir, monkeypatch):
    """Walk-up discovery finds hpc.toml in parent directory"""
    (temp_dir / "hpc.toml").write_text("[cluster]\nhost = 'test'\nworkdir = '/tmp'")
    child = temp_dir / "runs" / "bench1"
    child.mkdir(parents=True)
    monkeypatch.chdir(child)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = cli_runner.invoke(app, ["sync"])
        assert result.exit_code == 0
        assert "Config file not found" not in result.stdout


def test_sync_uses_project_root(cli_runner, temp_dir, monkeypatch):
    """sync uses project root (hpc.toml location) as local path, not CWD"""
    (temp_dir / "hpc.toml").write_text("[cluster]\nhost = 'test'\nworkdir = '/tmp'")
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
    """init creates hpc.toml in CWD, does not walk up"""
    (temp_dir / "hpc.toml").write_text("[cluster]\nhost = 'test'\nworkdir = '/tmp'")
    child = temp_dir / "subdir"
    child.mkdir()
    monkeypatch.chdir(child)
    result = cli_runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert (child / "hpc.toml").exists()


def test_job_output_follow_in_help(cli_runner):
    result = cli_runner.invoke(app, ["job-output", "--help"])
    assert result.exit_code == 0
    assert "--follow" in result.stdout
    assert "-f" in result.stdout


def _setup_run_meta(temp_dir, run_id="r1", job_id="12345678"):
    """Write hpc.toml + a fake run meta so job-output can resolve the run."""
    (temp_dir / "hpc.toml").write_text("[cluster]\nhost = 'test'\nworkdir = '/tmp'\n")
    runs_dir = temp_dir / ".hpc" / "runs" / run_id
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

    with patch("hpc.cli.JobManager") as mock_job_cls:
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

    with patch("hpc.cli.JobManager") as mock_job_cls:
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

    with patch("hpc.cli.JobManager") as mock_job_cls:
        mock_job = MagicMock()
        mock_job.get_job_output.return_value = "static output\n"
        mock_job_cls.return_value = mock_job

        result = cli_runner.invoke(app, ["job-output", "r1"])
        assert result.exit_code == 0
        mock_job.get_job_output.assert_called_once_with("r1", "12345678", error=False)
        mock_job.tail_job_output.assert_not_called()
        assert "static output" in result.stdout
