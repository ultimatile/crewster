"""Job manager tests"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hpc.job import JobManager, JobStatus, _extract_prologue_directives
from hpc.scheduler import JobDetail, SchedulerError
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

    def test_submit_job_creates_run_dir_before_submission(self, mock_ssh_manager):
        # The rendered script's ``-o`` / ``--output=`` directives point
        # under ``<workdir>/.hpc/runs/job/``; the scheduler must be able
        # to open those paths, so the directory has to exist when pjsub /
        # sbatch consumes the script.
        config = HpcConfig(
            cluster=ClusterConfig(
                host="myhpc", workdir="/scratch/user/proj", scheduler="pjm"
            ),
            env=EnvConfig(),
            pjm=PjmConfig(options=[["-L", "node=1"]]),
        )
        manager = JobManager(ssh_manager=mock_ssh_manager, config=config)
        mock_ssh_manager.run_command.return_value = MagicMock(
            stdout="[INFO] PJM 0000 pjsub Job 12345678 submitted.\n"
        )

        manager.submit_job("python train.py")

        calls = mock_ssh_manager.run_command.call_args_list
        mkdir_idx = next((i for i, c in enumerate(calls) if c.args[0] == "mkdir"), None)
        submit_idx = next(
            (i for i, c in enumerate(calls) if c.args[0] == "pjsub"), None
        )
        assert mkdir_idx is not None, "submit_job must mkdir -p its run_dir"
        assert submit_idx is not None, "submit_job must invoke pjsub"
        # Ordering matters: pjsub opens the directives' output paths,
        # so the directory must exist before pjsub runs.
        assert mkdir_idx < submit_idx
        assert calls[mkdir_idx].args[1] == [
            "-p",
            "/scratch/user/proj/.hpc/runs/job",
        ]

    def test_submit_job_creates_run_dir_before_submission_slurm(self, mock_ssh_manager):
        # Same pair-invariant as the PJM variant: the contract holds for
        # every scheduler the legacy path can target, not only PJM.
        config = HpcConfig(
            cluster=ClusterConfig(host="myhpc", workdir="/scratch/user/proj"),
            env=EnvConfig(),
            slurm=SlurmConfig(options={"partition": "gpu"}),
        )
        manager = JobManager(ssh_manager=mock_ssh_manager, config=config)
        mock_ssh_manager.run_command.return_value = MagicMock(stdout="12345678\n")

        manager.submit_job("python train.py")

        calls = mock_ssh_manager.run_command.call_args_list
        mkdir_idx = next((i for i, c in enumerate(calls) if c.args[0] == "mkdir"), None)
        submit_idx = next(
            (i for i, c in enumerate(calls) if c.args[0] == "sbatch"), None
        )
        assert mkdir_idx is not None and submit_idx is not None
        assert mkdir_idx < submit_idx
        assert calls[mkdir_idx].args[1] == [
            "-p",
            "/scratch/user/proj/.hpc/runs/job",
        ]

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

    def test_submit_run_creates_run_dir_before_submission(self, mock_ssh_manager):
        # Pair-invariant: every submission path's rendered script directs
        # the scheduler's stdout/stderr under ``run_dir``, so ``run_dir``
        # must exist before the scheduler is invoked. Asserting the mkdir
        # call protects the pair from regressing in either submission
        # path.
        config = HpcConfig(
            cluster=ClusterConfig(
                host="myhpc", workdir="/scratch/user/proj", scheduler="pjm"
            ),
            env=EnvConfig(),
            pjm=PjmConfig(options=[["-L", "node=1"]]),
        )
        manager = JobManager(ssh_manager=mock_ssh_manager, config=config)
        mock_ssh_manager.run_command.return_value = MagicMock(
            stdout="[INFO] PJM 0000 pjsub Job 12345678 submitted.\n"
        )
        from hpc.run import RunConfig

        run = RunConfig(run_id="test_run", cmd="echo hi", status="pending")
        manager.submit_run(run)

        calls = mock_ssh_manager.run_command.call_args_list
        mkdir_idx = next((i for i, c in enumerate(calls) if c.args[0] == "mkdir"), None)
        submit_idx = next(
            (i for i, c in enumerate(calls) if c.args[0] == "pjsub"), None
        )
        assert mkdir_idx is not None, "submit_run must mkdir -p its run_dir"
        assert submit_idx is not None, "submit_run must invoke pjsub"
        assert mkdir_idx < submit_idx
        assert calls[mkdir_idx].args[1] == [
            "-p",
            "/scratch/user/proj/.hpc/runs/test_run",
        ]

    def test_submit_run_creates_run_dir_before_submission_slurm(self, mock_ssh_manager):
        # Pair-invariant holds for Slurm submissions too.
        config = HpcConfig(
            cluster=ClusterConfig(host="myhpc", workdir="/scratch/user/proj"),
            env=EnvConfig(),
            slurm=SlurmConfig(options={"partition": "gpu"}),
        )
        manager = JobManager(ssh_manager=mock_ssh_manager, config=config)
        mock_ssh_manager.run_command.return_value = MagicMock(stdout="12345678\n")
        from hpc.run import RunConfig

        run = RunConfig(run_id="test_run", cmd="echo hi", status="pending")
        manager.submit_run(run)

        calls = mock_ssh_manager.run_command.call_args_list
        mkdir_idx = next((i for i, c in enumerate(calls) if c.args[0] == "mkdir"), None)
        submit_idx = next(
            (i for i, c in enumerate(calls) if c.args[0] == "sbatch"), None
        )
        assert mkdir_idx is not None and submit_idx is not None
        assert mkdir_idx < submit_idx
        assert calls[mkdir_idx].args[1] == [
            "-p",
            "/scratch/user/proj/.hpc/runs/test_run",
        ]

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

    def test_get_job_status_raises_scheduler_error_on_empty_output(
        self, mock_ssh_manager, sample_config
    ):
        # Boundary pin: the SchedulerError raised by parse_status reaches
        # JobManager's public API unchanged (no conversion to a different
        # type). Since SchedulerError is an SSHError subclass, the three
        # transient-handling callers still catch it via `except SSHError`.
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        mock_ssh_manager.run_command.return_value = MagicMock(stdout="")

        with pytest.raises(SchedulerError):
            manager.get_job_status("12345678")


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

    def test_render_job_script_pjm_emits_pjm_output_directives(self, mock_ssh_manager):
        """PJM jobs must emit ``#PJM -o`` / ``#PJM -e`` for the bookkeeping
        output paths, not the Slurm-shaped ``--output=`` / ``--error=``
        that pjsub does not honor."""
        config = HpcConfig(
            cluster=ClusterConfig(
                host="myhpc", workdir="/scratch/user/proj", scheduler="pjm"
            ),
            env=EnvConfig(),
            pjm=PjmConfig(options=[["-L", "node=12"]]),
        )
        manager = JobManager(ssh_manager=mock_ssh_manager, config=config)
        from hpc.run import RunConfig

        run = RunConfig(run_id="test_run", cmd="echo hi", status="pending")
        script = manager._render_job_script(run)

        assert "#PJM -o /scratch/user/proj/.hpc/runs/test_run/job.out" in script
        assert "#PJM -e /scratch/user/proj/.hpc/runs/test_run/job.err" in script
        # Disconfirming: the Slurm-shaped forms (the original bug) must
        # not leak into a PJM-rendered script.
        assert "--output=" not in script
        assert "--error=" not in script


class TestJobManagerGetJobOutput:
    """Path resolution in ``get_job_output`` routes through
    ``scheduler.output_path`` so PJM (fixed names) and Slurm
    (``job-<id>`` names) both work."""

    def test_pjm_uses_fixed_filename(self, mock_ssh_manager):
        config = HpcConfig(
            cluster=ClusterConfig(
                host="myhpc", workdir="/scratch/user/proj", scheduler="pjm"
            ),
            env=EnvConfig(),
            pjm=PjmConfig(options=[["-L", "node=1"]]),
        )
        manager = JobManager(ssh_manager=mock_ssh_manager, config=config)
        mock_ssh_manager.run_command.return_value = MagicMock(stdout="hello\n")

        out = manager.get_job_output("test_run", "12345678")

        assert out == "hello\n"
        call_args = mock_ssh_manager.run_command.call_args
        assert call_args.args[0] == "cat"
        assert call_args.args[1] == ["/scratch/user/proj/.hpc/runs/test_run/job.out"]

    def test_pjm_error_flag_uses_err_extension(self, mock_ssh_manager):
        config = HpcConfig(
            cluster=ClusterConfig(
                host="myhpc", workdir="/scratch/user/proj", scheduler="pjm"
            ),
            env=EnvConfig(),
            pjm=PjmConfig(options=[["-L", "node=1"]]),
        )
        manager = JobManager(ssh_manager=mock_ssh_manager, config=config)
        mock_ssh_manager.run_command.return_value = MagicMock(stdout="oops\n")

        manager.get_job_output("test_run", "12345678", error=True)

        call_args = mock_ssh_manager.run_command.call_args
        assert call_args.args[1] == ["/scratch/user/proj/.hpc/runs/test_run/job.err"]

    def test_inner_scheduler_error_does_not_mask_original_no_such_file(
        self, mock_ssh_manager, sample_config
    ):
        # Regression guard: when ``cat`` raises "No such file" and the
        # inner status probe raises SchedulerError (job not yet indexed),
        # the existing ``except SSHError: pass`` swallows the inner
        # SchedulerError (subclass of SSHError) and re-raises the original
        # cat error. The user must see the original missing-file
        # diagnostic, not the inner accounting-not-ready one.
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        mock_ssh_manager.run_command.side_effect = [
            SSHError(
                "SSH command failed (exit 1): cat /scratch/user/proj/.hpc/runs/test_run/job-12345678.out\n"
                "stderr:\ncat: No such file or directory"
            ),
            MagicMock(stdout=""),  # status probe; parse_status raises SchedulerError
        ]

        with pytest.raises(SSHError) as exc_info:
            manager.get_job_output("test_run", "12345678")
        assert "No such file" in str(exc_info.value)
        assert not isinstance(exc_info.value, SchedulerError)


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

    def test_tail_job_output_pjm_uses_fixed_filename(self, mock_ssh_manager):
        config = HpcConfig(
            cluster=ClusterConfig(
                host="myhpc", workdir="/scratch/user/proj", scheduler="pjm"
            ),
            env=EnvConfig(),
            pjm=PjmConfig(options=[["-L", "node=1"]]),
        )
        manager = JobManager(ssh_manager=mock_ssh_manager, config=config)
        # ``pjstat -v --choose jid,st,ec,sn`` emits a header row followed
        # by one data row per job. A non-terminal ST makes
        # ``tail_job_output`` take the streaming path rather than falling
        # back to ``get_job_output``.
        mock_ssh_manager.run_command.return_value = MagicMock(
            stdout="JOB_ID ST EC SN\n12345678 RUN 0 0\n"
        )
        mock_ssh_manager.run_streaming.return_value = 0

        rc = manager.tail_job_output("test_run", "12345678")

        assert rc == 0
        mock_ssh_manager.run_streaming.assert_called_once_with(
            "tail",
            ["-F", "/scratch/user/proj/.hpc/runs/test_run/job.out"],
        )


class TestExtractPrologueDirectives:
    """Helper that hoists scheduler directives from user-supplied script content."""

    def test_empty_content(self):
        directives, body = _extract_prologue_directives("", "#SBATCH")
        assert directives == []
        assert body == ""

    def test_shebang_only(self):
        directives, body = _extract_prologue_directives("#!/bin/bash\n", "#SBATCH")
        assert directives == []
        assert body == ""

    def test_shebang_dropped_with_body(self):
        content = "#!/bin/bash\necho hi\n"
        directives, body = _extract_prologue_directives(content, "#SBATCH")
        assert directives == []
        assert body == "echo hi\n"

    def test_no_shebang_no_directives_unchanged(self):
        directives, body = _extract_prologue_directives("echo hi\n", "#SBATCH")
        assert directives == []
        assert body == "echo hi\n"

    def test_extracts_slurm_directive_after_shebang(self):
        content = "#!/bin/bash\n#SBATCH --array=1-10%5\necho hi\n"
        directives, body = _extract_prologue_directives(content, "#SBATCH")
        assert directives == ["#SBATCH --array=1-10%5"]
        assert body == "echo hi\n"

    def test_extracts_pjm_directive(self):
        content = '#!/bin/bash\n#PJM -L "node=4"\n#PJM -j\necho pjm\n'
        directives, body = _extract_prologue_directives(content, "#PJM")
        assert directives == ['#PJM -L "node=4"', "#PJM -j"]
        assert body == "echo pjm\n"

    def test_pjm_prefix_does_not_match_sbatch_lines(self):
        content = "#PJM -L node=4\n#SBATCH --array=1-5\necho hi\n"
        directives, body = _extract_prologue_directives(content, "#PJM")
        assert directives == ["#PJM -L node=4"]
        # `#SBATCH ...` is a non-directive comment for the PJM matcher and
        # stays in the body. Prologue scan continues past it.
        assert body == "#SBATCH --array=1-5\necho hi\n"

    def test_directive_after_first_executable_not_hoisted(self):
        # Mirrors scheduler behavior: once an executable line is seen,
        # subsequent #SBATCH lines are ignored.
        content = "#!/bin/bash\necho before\n#SBATCH --array=1-5\necho after\n"
        directives, body = _extract_prologue_directives(content, "#SBATCH")
        assert directives == []
        assert body == "echo before\n#SBATCH --array=1-5\necho after\n"

    def test_heredoc_directive_lookalike_not_hoisted(self):
        content = "#!/bin/bash\ncat <<EOF\n#SBATCH --bogus=1\nEOF\necho done\n"
        directives, body = _extract_prologue_directives(content, "#SBATCH")
        assert directives == []
        # `cat <<EOF` is the first executable line; everything after it is
        # preserved verbatim, including the directive-look-alike heredoc body.
        assert body == "cat <<EOF\n#SBATCH --bogus=1\nEOF\necho done\n"

    def test_blank_lines_between_directives_preserved(self):
        content = (
            "#!/bin/bash\n"
            "\n"
            "#SBATCH --partition=gpu\n"
            "\n"
            "#SBATCH --time=01:00:00\n"
            "echo hi\n"
        )
        directives, body = _extract_prologue_directives(content, "#SBATCH")
        assert directives == ["#SBATCH --partition=gpu", "#SBATCH --time=01:00:00"]
        # Blank lines between hoisted directives are kept in body — they are
        # comments to the scheduler and do not terminate the prologue scan.
        assert body == "\n\necho hi\n"

    def test_non_directive_comment_preserved_in_prologue(self):
        # A `# regular comment` line does not terminate the prologue scan,
        # but it is also not a directive — it stays in the body.
        content = "#!/bin/bash\n# user explanation\n#SBATCH --partition=gpu\necho hi\n"
        directives, body = _extract_prologue_directives(content, "#SBATCH")
        assert directives == ["#SBATCH --partition=gpu"]
        assert body == "# user explanation\necho hi\n"

    def test_indented_directive_not_hoisted(self):
        # Schedulers require column-zero `#SBATCH`; an indented line is
        # not a directive, and being non-blank-non-column-zero-comment
        # it also terminates the prologue scan.
        content = "#!/bin/bash\n  #SBATCH --partition=gpu\necho hi\n"
        directives, body = _extract_prologue_directives(content, "#SBATCH")
        assert directives == []
        assert body == "  #SBATCH --partition=gpu\necho hi\n"

    def test_indented_comment_terminates_prologue(self):
        # An indented `# comment` is bash-style comment but not a
        # column-zero comment; the schedulers stop scanning at this
        # line, so a directive following it is not hoisted (matching
        # what `sbatch script.sh` standalone would do).
        content = "#!/bin/bash\n  # indented comment\n#SBATCH --array=1-5\necho hi\n"
        directives, body = _extract_prologue_directives(content, "#SBATCH")
        assert directives == []
        assert body == "  # indented comment\n#SBATCH --array=1-5\necho hi\n"

    def test_whitespace_only_line_terminates_prologue(self):
        # A line containing only whitespace (not a true `\n`-only blank)
        # is treated as scan-terminating, matching the schedulers'
        # column-sensitive prologue rule.
        content = "#!/bin/bash\n   \n#SBATCH --array=1-5\necho hi\n"
        directives, body = _extract_prologue_directives(content, "#SBATCH")
        assert directives == []
        assert body == "   \n#SBATCH --array=1-5\necho hi\n"

    def test_crlf_blank_line_kept_in_prologue(self):
        # Truly-blank lines (`\n` or `\r\n`) do not terminate the scan.
        content = "#!/bin/bash\r\n\r\n#SBATCH --array=1-5\r\necho hi\r\n"
        directives, body = _extract_prologue_directives(content, "#SBATCH")
        assert directives == ["#SBATCH --array=1-5"]
        assert body == "\r\necho hi\r\n"

    def test_no_trailing_newline_preserved(self):
        directives, body = _extract_prologue_directives("echo hi", "#SBATCH")
        assert directives == []
        assert body == "echo hi"


class TestJobManagerHoistsUserDirectives:
    def test_render_hoists_sbatch_directive_from_script_body(
        self, mock_ssh_manager, sample_config
    ):
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        from hpc.run import RunConfig

        cmd = "#!/bin/bash\n#SBATCH --array=1-10%5\necho hi\n"
        run = RunConfig(run_id="test_run", cmd=cmd, status="pending")
        script = manager._render_job_script(run)

        # User directive lands in the prologue, before `cd`.
        array_idx = script.index("#SBATCH --array=1-10%5")
        cd_idx = script.index("cd /scratch/user/proj")
        assert array_idx < cd_idx
        # And before the template's hardcoded --output= bookkeeping line, so
        # hpc's run-tracking output path always wins over any user override.
        output_idx = script.index("--output=/scratch/user/proj/.hpc/runs/test_run")
        assert array_idx < output_idx
        # User shebang is stripped (template emits its own).
        assert script.count("#!/bin/bash") == 1
        # Body retains the executable line.
        assert "echo hi" in script

    def test_render_hoists_multiple_pjm_directives(self, mock_ssh_manager):
        config = HpcConfig(
            cluster=ClusterConfig(
                host="myhpc", workdir="/scratch/user/proj", scheduler="pjm"
            ),
            env=EnvConfig(),
            pjm=PjmConfig(options=[["-L", "node=12"]]),
        )
        manager = JobManager(ssh_manager=mock_ssh_manager, config=config)
        from hpc.run import RunConfig

        cmd = '#!/bin/bash\n#PJM -L "rscgrp=small"\n#PJM -j\n#PJM -N myjob\necho pjm\n'
        run = RunConfig(run_id="test_run", cmd=cmd, status="pending")
        script = manager._render_job_script(run)

        cd_idx = script.index("cd /scratch/user/proj")
        for directive in ('#PJM -L "rscgrp=small"', "#PJM -j", "#PJM -N myjob"):
            assert directive in script
            assert script.index(directive) < cd_idx
        # Original order preserved.
        assert (
            script.index('#PJM -L "rscgrp=small"')
            < script.index("#PJM -j")
            < script.index("#PJM -N myjob")
        )

    def test_user_directive_emitted_after_config_directive_slurm(
        self, mock_ssh_manager, sample_config
    ):
        """User directive wins on conflict via scheduler last-occurrence-wins."""
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        from hpc.run import RunConfig

        cmd = "#!/bin/bash\n#SBATCH --partition=cpu\necho hi\n"
        run = RunConfig(run_id="test_run", cmd=cmd, status="pending")
        script = manager._render_job_script(run)

        config_partition = script.index("#SBATCH --partition=gpu")
        user_partition = script.index("#SBATCH --partition=cpu")
        assert config_partition < user_partition

    def test_user_directive_emitted_after_config_directive_pjm(self, mock_ssh_manager):
        config = HpcConfig(
            cluster=ClusterConfig(
                host="myhpc", workdir="/scratch/user/proj", scheduler="pjm"
            ),
            env=EnvConfig(),
            pjm=PjmConfig(options=[["-L", "node=12"]]),
        )
        manager = JobManager(ssh_manager=mock_ssh_manager, config=config)
        from hpc.run import RunConfig

        cmd = '#!/bin/bash\n#PJM -L "node=4"\necho pjm\n'
        run = RunConfig(run_id="test_run", cmd=cmd, status="pending")
        script = manager._render_job_script(run)

        config_node = script.index("#PJM -L node=12")
        user_node = script.index('#PJM -L "node=4"')
        assert config_node < user_node

    def test_render_no_user_directives_unchanged(self, mock_ssh_manager, sample_config):
        """Plain command without directives: rendered output matches the
        pre-hoist behavior — user_directives block emits nothing extra."""
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        from hpc.run import RunConfig

        run = RunConfig(run_id="test_run", cmd="python train.py", status="pending")
        script = manager._render_job_script(run)
        # Sanity: the command body is at the bottom, config directives at top.
        assert "python train.py" in script
        assert script.index("#SBATCH --partition=gpu") < script.index("python train.py")

    def test_legacy_submit_job_hoists_user_directives(
        self, mock_ssh_manager, sample_config
    ):
        """Legacy `submit_job(cmd)` path also hoists user directives so the
        two submission paths render consistent prologues."""
        manager = JobManager(ssh_manager=mock_ssh_manager, config=sample_config)
        mock_ssh_manager.run_command.return_value = MagicMock(stdout="12345678\n")

        cmd = "#!/bin/bash\n#SBATCH --array=1-5\necho hi\n"
        manager.submit_job(cmd)

        # The rendered script is piped to sbatch via input_text.
        call_args = mock_ssh_manager.run_command.call_args
        script = call_args.kwargs["input_text"]
        cd_idx = script.index("cd /scratch/user/proj")
        assert script.index("#SBATCH --array=1-5") < cd_idx
        assert script.count("#!/bin/bash") == 1
