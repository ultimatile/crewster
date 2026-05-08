"""Job manager tests"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hpc.job import JobManager, JobStatus
from hpc.scheduler import JobDetail
from hpc.ssh import SSHManager, SSHError
from hpc.config import HpcConfig, ClusterConfig, EnvConfig, SlurmConfig, PjmConfig


@pytest.fixture
def mock_ssh_manager():
    return MagicMock(spec=SSHManager)


@pytest.fixture
def sample_config():
    return HpcConfig(
        cluster=ClusterConfig(host="myhpc", workdir="/scratch/user/proj"),
        env=EnvConfig(modules=["gcc/12.2.0"]),
        slurm=SlurmConfig(
            options={"partition": "gpu", "time": "02:00:00", "mem": "32G", "gpus": 1}
        ),
    )


class TestJobStatus:
    def test_job_status_values(self):
        assert JobStatus.PENDING.value == "PENDING"
        assert JobStatus.RUNNING.value == "RUNNING"
        assert JobStatus.COMPLETED.value == "COMPLETED"
        assert JobStatus.FAILED.value == "FAILED"


class TestJobManagerInit:
    def test_init_with_ssh_and_config(self, mock_ssh_manager, sample_config):
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        assert manager.ssh_manager == mock_ssh_manager
        assert manager.config == sample_config


class TestJobManagerSubmit:
    def test_submit_job_returns_job_id(self, mock_ssh_manager, sample_config):
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        mock_ssh_manager.run_command.return_value = MagicMock(stdout="12345678\n")

        job_id = manager.submit_job("python train.py")
        assert job_id == "12345678"

    def test_submit_job_uses_sbatch_parsable(self, mock_ssh_manager, sample_config):
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        mock_ssh_manager.run_command.return_value = MagicMock(stdout="12345678\n")

        manager.submit_job("python train.py")
        call_args = mock_ssh_manager.run_command.call_args
        assert call_args.args[0] == "sbatch"
        assert "--parsable" in call_args.args[1]

    def test_submit_job_includes_pjm_submit_options(self, mock_ssh_manager):
        config = HpcConfig(
            cluster=ClusterConfig(
                host="myhpc", workdir="/scratch/user/proj", scheduler="pjm"
            ),
            env=EnvConfig(),
            pjm=PjmConfig(
                options=[["-L", "node=12"]],
                submit_options=["--no-check-directory"],
            ),
        )
        manager = JobManager(ssh_manager=mock_ssh_manager, config=config)
        mock_ssh_manager.run_command.return_value = MagicMock(
            stdout="[INFO] PJM 0000 pjsub Job 12345678 submitted.\n"
        )

        manager.submit_job("python train.py")
        call_args = mock_ssh_manager.run_command.call_args
        assert call_args.args[0] == "pjsub"
        assert "--no-check-directory" in call_args.args[1]

    def test_submit_job_includes_slurm_submit_options(self, mock_ssh_manager):
        config = HpcConfig(
            cluster=ClusterConfig(host="myhpc", workdir="/scratch/user/proj"),
            env=EnvConfig(),
            slurm=SlurmConfig(
                options={"partition": "gpu"},
                submit_options=["--export=ALL"],
            ),
        )
        manager = JobManager(ssh_manager=mock_ssh_manager, config=config)
        mock_ssh_manager.run_command.return_value = MagicMock(stdout="12345678\n")

        manager.submit_job("python train.py")
        call_args = mock_ssh_manager.run_command.call_args
        assert call_args.args[0] == "sbatch"
        assert "--parsable" in call_args.args[1]
        assert "--export=ALL" in call_args.args[1]

    def test_submit_run_includes_pjm_submit_options(self, mock_ssh_manager):
        config = HpcConfig(
            cluster=ClusterConfig(
                host="myhpc", workdir="/scratch/user/proj", scheduler="pjm"
            ),
            env=EnvConfig(),
            pjm=PjmConfig(
                options=[["-L", "node=12"]],
                submit_options=["--no-check-directory"],
            ),
        )
        manager = JobManager(ssh_manager=mock_ssh_manager, config=config)
        mock_ssh_manager.run_command.return_value = MagicMock(
            stdout="[INFO] PJM 0000 pjsub Job 12345678 submitted.\n"
        )
        from hpc.run import RunConfig

        run = RunConfig(run_id="test_run", cmd="python train.py", status="pending")
        manager.submit_run(run)

        # Last run_command call is the submit
        call_args = mock_ssh_manager.run_command.call_args
        assert call_args.args[0] == "pjsub"
        assert "--no-check-directory" in call_args.args[1]

    def test_submit_run_includes_slurm_submit_options(self, mock_ssh_manager):
        config = HpcConfig(
            cluster=ClusterConfig(host="myhpc", workdir="/scratch/user/proj"),
            env=EnvConfig(),
            slurm=SlurmConfig(
                options={"partition": "gpu"},
                submit_options=["--export=ALL"],
            ),
        )
        manager = JobManager(ssh_manager=mock_ssh_manager, config=config)
        mock_ssh_manager.run_command.return_value = MagicMock(stdout="12345678\n")
        from hpc.run import RunConfig

        run = RunConfig(run_id="test_run", cmd="python train.py", status="pending")
        manager.submit_run(run)

        call_args = mock_ssh_manager.run_command.call_args
        assert call_args.args[0] == "sbatch"
        assert "--parsable" in call_args.args[1]
        assert "--export=ALL" in call_args.args[1]


class TestJobManagerStatus:
    def test_get_job_status_completed(self, mock_ssh_manager, sample_config):
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        mock_ssh_manager.run_command.return_value = MagicMock(stdout="COMPLETED\n")

        status = manager.get_job_status("12345678")
        assert status == JobStatus.COMPLETED

    def test_get_job_status_running(self, mock_ssh_manager, sample_config):
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        mock_ssh_manager.run_command.return_value = MagicMock(stdout="RUNNING\n")

        status = manager.get_job_status("12345678")
        assert status == JobStatus.RUNNING

    def test_get_job_status_uses_sacct(self, mock_ssh_manager, sample_config):
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        mock_ssh_manager.run_command.return_value = MagicMock(stdout="COMPLETED\n")

        manager.get_job_status("12345678")
        call_args = mock_ssh_manager.run_command.call_args
        assert call_args.args[0] == "sacct"
        assert "12345678" in call_args.args[1]
        assert "--noheader" in call_args.args[1]


class TestJobManagerDetail:
    def test_get_job_detail_slurm_invokes_sacct_and_returns_detail(
        self, mock_ssh_manager, sample_config
    ):
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        mock_ssh_manager.run_command.return_value = MagicMock(
            stdout=(
                "12345|COMPLETED|0:0|00:01:23||16Gn\n"
                "12345.batch|COMPLETED|0:0|00:01:23|1024K|16Gn\n"
            )
        )

        detail = manager.get_job_detail("12345")
        assert detail == JobDetail(
            state="COMPLETED",
            exit_code="0:0",
            elapsed="00:01:23",
            max_rss="1024K",
            req_mem="16Gn",
        )
        call_args = mock_ssh_manager.run_command.call_args
        assert call_args.args[0] == "sacct"
        assert "-P" in call_args.args[1]

    def test_get_job_detail_pjm_returns_none_without_ssh_call(self, mock_ssh_manager):
        config = HpcConfig(
            cluster=ClusterConfig(
                host="myhpc", workdir="/scratch/user/proj", scheduler="pjm"
            ),
            env=EnvConfig(),
            pjm=PjmConfig(options=[["-L", "node=1"]]),
        )
        manager = JobManager(ssh_manager=mock_ssh_manager, config=config)

        assert manager.get_job_detail("12345") is None
        mock_ssh_manager.run_command.assert_not_called()


class TestJobManagerTemplate:
    def test_render_job_script(self, mock_ssh_manager, sample_config):
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        from hpc.run import RunConfig

        run = RunConfig(run_id="test_run", cmd="python train.py", status="pending")
        script = manager._render_job_script(run)

        assert "#!/bin/bash" in script
        assert "#SBATCH --partition=gpu" in script
        assert "#SBATCH --time=02:00:00" in script
        assert "#SBATCH --mem=32G" in script
        assert "#SBATCH --job-name=test_run" in script
        assert "python train.py" in script

    def test_render_job_script_default_cwd(self, mock_ssh_manager, sample_config):
        """Default cwd_relative=Path('.') uses workdir as job working directory"""
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        from hpc.run import RunConfig

        run = RunConfig(run_id="test_run", cmd="echo hi", status="pending")
        script = manager._render_job_script(run)
        assert "cd /scratch/user/proj" in script

    def test_render_job_script_with_subdirectory(self, mock_ssh_manager, sample_config):
        """cwd_relative appends subdirectory to workdir for job cd"""
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        from hpc.run import RunConfig

        run = RunConfig(run_id="test_run", cmd="echo hi", status="pending")
        script = manager._render_job_script(run, cwd_relative=Path("runs/bench1"))
        assert "cd /scratch/user/proj/runs/bench1" in script
        # Output paths still use base workdir
        assert "--output=/scratch/user/proj/.hpc/runs/test_run" in script


class TestJobManagerTailJobOutput:
    def _running_status(self):
        return MagicMock(stdout="RUNNING\n")

    def _completed_status(self):
        return MagicMock(stdout="COMPLETED\n")

    def test_tail_job_output_active_uses_tail_F(self, mock_ssh_manager, sample_config):
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        mock_ssh_manager.run_command.return_value = self._running_status()
        mock_ssh_manager.run_streaming.return_value = 0

        rc = manager.tail_job_output("run_id", "12345678")

        assert rc == 0
        mock_ssh_manager.run_streaming.assert_called_once_with(
            "tail",
            ["-F", "/scratch/user/proj/.hpc/runs/run_id/job-12345678.out"],
        )

    def test_tail_job_output_terminal_falls_back_to_get_job_output(
        self, mock_ssh_manager, sample_config, capsys
    ):
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        # First call: status=COMPLETED (terminal). Second call: cat output.
        mock_ssh_manager.run_command.side_effect = [
            self._completed_status(),
            MagicMock(stdout="final job output\n"),
        ]

        rc = manager.tail_job_output("run_id", "12345678")

        assert rc == 0
        mock_ssh_manager.run_streaming.assert_not_called()
        captured = capsys.readouterr()
        assert captured.out == "final job output\n"

    def test_tail_job_output_unknown_status_streams(
        self, mock_ssh_manager, sample_config
    ):
        """SSHError on get_job_status falls through to tail -F (safe-side default)."""
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        mock_ssh_manager.run_command.side_effect = SSHError("transient ssh failure")
        mock_ssh_manager.run_streaming.return_value = 0

        rc = manager.tail_job_output("run_id", "12345678")

        assert rc == 0
        mock_ssh_manager.run_streaming.assert_called_once()

    def test_tail_job_output_error_flag_uses_err_extension(
        self, mock_ssh_manager, sample_config
    ):
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        mock_ssh_manager.run_command.return_value = self._running_status()
        mock_ssh_manager.run_streaming.return_value = 0

        manager.tail_job_output("run_id", "12345678", error=True)

        call_args = mock_ssh_manager.run_streaming.call_args
        assert call_args.args[0] == "tail"
        assert call_args.args[1] == [
            "-F",
            "/scratch/user/proj/.hpc/runs/run_id/job-12345678.err",
        ]

    def test_tail_job_output_returns_streaming_exit_code(
        self, mock_ssh_manager, sample_config
    ):
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        mock_ssh_manager.run_command.return_value = self._running_status()
        mock_ssh_manager.run_streaming.return_value = 130  # Ctrl-C exit code

        rc = manager.tail_job_output("run_id", "12345678")
        assert rc == 130
