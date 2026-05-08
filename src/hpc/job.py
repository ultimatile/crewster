"""Job management for HPC schedulers"""

from pathlib import Path

from jinja2 import Template

from .config import HpcConfig
from .ssh import SSHManager
from .run import RunConfig
from .scheduler import JobStatus, get_scheduler


def _resolve_home_path(ssh_manager, path: str) -> str:
    """Resolve ~ to actual home directory path via SSH"""
    if path.startswith("~/") or path == "~":
        result = ssh_manager.run_command("printenv", ["HOME"])
        home_dir = result.stdout.strip()
        if path == "~":
            return home_dir
        else:
            return path.replace("~", home_dir, 1)
    return path


JOB_TEMPLATE = """#!/bin/bash
{% for directive in directives %}
{{ directive }}
{% endfor %}
{{ scheduler.directive_prefix().split()[0] }} --output={{ workdir }}/.hpc/runs/{{ run_id }}/job-%j.out
{{ scheduler.directive_prefix().split()[0] }} --error={{ workdir }}/.hpc/runs/{{ run_id }}/job-%j.err

cd {{ job_workdir }}

{% for cmd in setup_commands %}
{{ cmd }}
{% endfor %}

{{ cmd }}
"""


class JobManager:
    """Job submission and monitoring"""

    def __init__(self, ssh_manager: SSHManager, config: HpcConfig):
        self.ssh_manager = ssh_manager
        self.config = config
        self.scheduler = get_scheduler(config.cluster.scheduler)

    def _get_submit_options(self) -> list[str]:
        """Get submit command options from config."""
        return (
            self.config.pjm.submit_options
            if self.config.cluster.scheduler == "pjm"
            else self.config.slurm.submit_options
        )

    def _build_directives(
        self, options: dict | list, job_name: str | None = None
    ) -> list[str]:
        """Build scheduler directives from options"""
        if isinstance(options, list):
            # PJM format: [["-L", "node=12"], ["-s"]]
            directives = []
            for opt in options:
                if not opt:
                    continue
                if len(opt) == 1:
                    directives.append(f"#PJM {opt[0]}")
                else:
                    directives.append(f"#PJM {opt[0]} {opt[1]}")
            return directives
        else:
            # Slurm format: {"partition": "gpu", ...}
            prefix = self.scheduler.directive_prefix()
            directives = []
            opts = options.copy()
            if job_name and "job_name" not in opts and "job-name" not in opts:
                opts["job_name"] = job_name
            for key, value in opts.items():
                directives.append(f"{prefix} --{key.replace('_', '-')}={value}")
            return directives

    def _render_job_script(self, run: RunConfig, cwd_relative: Path = Path(".")) -> str:
        """Render job script from template"""
        template = Template(JOB_TEMPLATE)
        workdir = _resolve_home_path(self.ssh_manager, self.config.cluster.workdir)
        job_workdir = str(Path(workdir) / cwd_relative)
        options = (
            self.config.pjm.options
            if self.config.cluster.scheduler == "pjm"
            else self.config.slurm.options
        )
        directives = self._build_directives(options, run.run_id)
        setup_commands = self.config.env.get_setup_commands()
        return template.render(
            run_id=run.run_id,
            directives=directives,
            scheduler=self.scheduler,
            workdir=workdir,
            job_workdir=job_workdir,
            setup_commands=setup_commands,
            cmd=run.cmd,
        )

    def submit_run(self, run: RunConfig, cwd_relative: Path = Path(".")) -> str:
        """Submit run and return job ID"""
        script = self._render_job_script(run, cwd_relative=cwd_relative)

        workdir = _resolve_home_path(self.ssh_manager, self.config.cluster.workdir)
        run_dir = f"{workdir}/.hpc/runs/{run.run_id}"
        self.ssh_manager.run_command("mkdir", ["-p", run_dir])

        script_path = f"{run_dir}/job.sh"
        self.ssh_manager.run_command("tee", [script_path], input_text=script)

        cmd = self.scheduler.submit_cmd()
        submit_options = self._get_submit_options()
        result = self.ssh_manager.run_command(
            cmd[0], cmd[1:] + submit_options + [script_path]
        )
        return self.scheduler.parse_job_id(result.stdout)

    def submit_job(self, cmd: str) -> str:
        """Legacy: Submit job without run tracking"""
        template = Template(JOB_TEMPLATE)
        workdir = _resolve_home_path(self.ssh_manager, self.config.cluster.workdir)
        options = (
            self.config.pjm.options
            if self.config.cluster.scheduler == "pjm"
            else self.config.slurm.options
        )
        directives = self._build_directives(options, "job")

        setup_commands = self.config.env.get_setup_commands()
        script = template.render(
            run_id="job",
            directives=directives,
            scheduler=self.scheduler,
            workdir=workdir,
            job_workdir=workdir,
            setup_commands=setup_commands,
            cmd=cmd,
        )
        submit_cmd = self.scheduler.submit_cmd()
        submit_options = self._get_submit_options()
        result = self.ssh_manager.run_command(
            submit_cmd[0], submit_cmd[1:] + submit_options, input_text=script
        )
        return self.scheduler.parse_job_id(result.stdout)

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get job status"""
        cmd = self.scheduler.status_cmd(job_id)
        result = self.ssh_manager.run_command(cmd[0], cmd[1:])
        return self.scheduler.parse_status(result.stdout)

    def get_job_output(self, run_id: str, job_id: str, error: bool = False) -> str:
        """Get job output file contents"""
        from .ssh import SSHError

        workdir = _resolve_home_path(self.ssh_manager, self.config.cluster.workdir)
        ext = "err" if error else "out"
        output_path = f"{workdir}/.hpc/runs/{run_id}/job-{job_id}.{ext}"

        try:
            result = self.ssh_manager.run_command("cat", [output_path])
            return result.stdout
        except SSHError as e:
            # Check if the file simply doesn't exist (job still running)
            if "No such file" in str(e):
                try:
                    status = self.get_job_status(job_id)
                except SSHError:
                    pass  # Can't check status either; re-raise original error
                else:
                    if status in (JobStatus.PENDING, JobStatus.RUNNING):
                        return f"Job {job_id} is {status.value}. Output file not yet available.\n"
            raise

    def tail_job_output(self, run_id: str, job_id: str, error: bool = False) -> int:
        """Stream job output via `tail -F` (equivalent to live tailing).

        For terminal-state jobs, falls back to `get_job_output` because no
        further output is coming and `tail -F` would otherwise spin forever
        on a missing file. For active or unknown-status jobs, runs `tail -F`
        which retries internally until the output file appears.
        """
        from .ssh import SSHError

        try:
            status = self.get_job_status(job_id)
        except SSHError:
            status = None  # status unknown; fall through to tail -F

        terminal_states = {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMEOUT,
        }
        if status in terminal_states:
            print(self.get_job_output(run_id, job_id, error=error), end="")
            return 0

        workdir = _resolve_home_path(self.ssh_manager, self.config.cluster.workdir)
        ext = "err" if error else "out"
        output_path = f"{workdir}/.hpc/runs/{run_id}/job-{job_id}.{ext}"
        return self.ssh_manager.run_streaming("tail", ["-F", output_path])

    def wait_for_job(
        self,
        job_id: str,
        interval: float = 60,
        adaptive: bool = False,
        max_interval: float = 300,
        growth_factor: float = 2.0,
    ) -> JobStatus:
        """Wait for job to complete, polling at interval

        Args:
            job_id: Job ID
            interval: Initial polling interval in seconds
            adaptive: If True, increase interval geometrically
            max_interval: Maximum polling interval (default 5 minutes)
            growth_factor: Multiplier for adaptive interval (default 2x)
        """
        import time

        from .ssh import SSHError

        current_interval = interval
        terminal_states = {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMEOUT,
        }

        while True:
            time.sleep(current_interval)

            try:
                status = self.get_job_status(job_id)
            except SSHError:
                # Transient SSH failures (e.g. bastion rate limiting); retry
                if adaptive:
                    current_interval = min(
                        current_interval * growth_factor, max_interval
                    )
                continue

            if status in terminal_states:
                return status

            if adaptive:
                current_interval = min(current_interval * growth_factor, max_interval)
