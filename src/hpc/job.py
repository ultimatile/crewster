"""Job management for HPC schedulers"""

import re
from pathlib import Path

from jinja2 import Template

from .config import HpcConfig
from .ssh import SSHManager
from .run import RunConfig
from .scheduler import JobDetail, JobStatus, get_scheduler


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
{% for directive in user_directives %}
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


def _extract_prologue_directives(content: str, prefix: str) -> tuple[list[str], str]:
    """Hoist scheduler directives from a user-supplied script's prologue.

    Mirrors the prologue-scan that sbatch and pjsub themselves perform:
    starting from the top of `content`, accept only shebang, blank,
    comment, and matching-prefix directive lines; stop at the first
    non-comment, non-blank executable line and leave everything from
    there onward untouched. Directive lines are matched at column zero
    with ``^<prefix>\\b`` (matching the schedulers' own column-zero
    rule) and removed from the body. A leading ``#!`` shebang line, if
    present, is dropped because the rendered job-script template
    injects its own.

    Without this hoist, directives written inside user scripts land
    after the template's ``cd workdir`` and env-setup lines and are
    silently ignored by the scheduler.

    Returns ``(directives_in_original_order, body_text)``. Directives
    have no trailing newline; body_text concatenates the kept lines
    and preserves original line endings.
    """
    if not content:
        return [], content

    lines = content.splitlines(keepends=True)
    directives: list[str] = []
    body_parts: list[str] = []

    # Strip a leading shebang only; the template injects its own.
    start = 1 if lines and lines[0].startswith("#!") else 0

    directive_re = re.compile(rf"^{re.escape(prefix)}\b")
    in_prologue = True

    for line in lines[start:]:
        if in_prologue:
            stripped = line.strip()
            if not stripped:
                body_parts.append(line)
            elif directive_re.match(line):
                directives.append(line.rstrip("\r\n"))
            elif stripped.startswith("#"):
                # Non-directive comment (any indent): bash treats it as
                # a comment and the scheduler's prologue scan skips it,
                # so we keep it in the body and continue scanning.
                body_parts.append(line)
            else:
                in_prologue = False
                body_parts.append(line)
        else:
            body_parts.append(line)

    return directives, "".join(body_parts)


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
        prefix = self.scheduler.directive_prefix().split()[0]
        user_directives, body = _extract_prologue_directives(run.cmd, prefix)
        return template.render(
            run_id=run.run_id,
            directives=directives,
            user_directives=user_directives,
            scheduler=self.scheduler,
            workdir=workdir,
            job_workdir=job_workdir,
            setup_commands=setup_commands,
            cmd=body,
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
        prefix = self.scheduler.directive_prefix().split()[0]
        user_directives, body = _extract_prologue_directives(cmd, prefix)
        script = template.render(
            run_id="job",
            directives=directives,
            user_directives=user_directives,
            scheduler=self.scheduler,
            workdir=workdir,
            job_workdir=workdir,
            setup_commands=setup_commands,
            cmd=body,
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

    def get_job_detail(self, job_id: str) -> JobDetail | None:
        """Get detailed accounting info, or None if the scheduler does not support it."""
        cmd = self.scheduler.detail_cmd(job_id)
        if cmd is None:
            return None
        result = self.ssh_manager.run_command(cmd[0], cmd[1:])
        return self.scheduler.parse_detail(result.stdout)

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
