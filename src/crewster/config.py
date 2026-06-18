"""Configuration management"""

import re
import shlex
import sys
import tomllib
from pathlib import Path
from typing import Literal

import tomli_w
from pydantic import BaseModel, ConfigDict, field_validator

# A setup item is a single-key table ``{command: args}``. ``args`` is a string
# or list of arguments, except for ``export`` whose args is a ``{KEY: value}``
# table. Bare strings are intentionally excluded: a raw, unvalidated command is
# arbitrary code execution, so arbitrary shell logic must go through a remote
# script invoked via ``{source = ["script.sh"]}`` instead.
SetupItem = dict[str, str | list[str] | dict[str, str]]

SHELL_SPECIAL = set(";|&`$<>\\'\"\n ")


def _validate_arg(arg: str, *, allow_space: bool = False) -> None:
    # ``allow_space`` is set for ``module`` / ``spack`` specs: a spec is one
    # logical unit that legitimately contains spaces (``boost@1.0 ~mpi
    # arch=...``), and each token is ``shlex.quote``d before reaching the
    # script, so the space cannot break out of its argument. The genuinely
    # dangerous metacharacters stay banned regardless.
    forbidden = SHELL_SPECIAL - {" "} if allow_space else SHELL_SPECIAL
    if bad := forbidden & set(arg):
        raise ValueError(f"Shell special characters not allowed: {bad}")


# Values rendered into a double-quoted shell context — ``export KEY="<value>"``
# in the job script and ``cd "<job_workdir>"`` — must still permit remote
# parameter expansion (``${USER}``, ``${SLURM_CPUS_PER_TASK}``) yet admit no
# path to command execution. Inside double quotes the only execution vectors
# are command substitution (``$(`` / backtick), arithmetic expansion (``$((``),
# prompt re-expansion (``${x@P}``), and indirect expansion (``${!x}``); bash
# does NOT re-scan the result of a parameter expansion, so a value that merely
# contains ``$(...)`` as text stays inert without an ``@P`` trigger. A literal
# ``"`` or backslash would close or escape out of the quotes.
#
# A blocklist of ``$(`` / backtick is therefore insufficient: ``${x@P}`` can
# re-evaluate a payload assembled from other allowed values containing none of
# the blocked substrings. Instead, strip the only forms we want (simple
# ``$VAR`` / ``${VAR}`` references) and reject any residual ``$`` (rejected
# conservatively: a leftover ``$`` may begin a dangerous ``$``-form, and a
# bare literal ``$`` — inert but unneeded here — is not worth distinguishing),
# backtick, ``"``, backslash, newline, or NUL.
_SAFE_VAR_REF = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*")
_DQ_FORBIDDEN = set('$`"\\\n\x00')


def _validate_dq_shell_value(value: str, label: str = "value") -> None:
    """Reject a double-quoted-context value that could execute a command.

    See the module comment above for why a whitelist (strip simple variable
    references, then forbid residual shell-active characters) is required
    rather than a blocklist of ``$(`` / backtick.
    """
    residual = _SAFE_VAR_REF.sub("", value)
    if bad := _DQ_FORBIDDEN & set(residual):
        raise ValueError(
            f"Unsafe characters in {label} (rendered in a double-quoted shell "
            f"context): {''.join(sorted(bad))!r} in {value!r}"
        )


def _validate_export_value(value: str) -> None:
    """Reject export values that could execute a command while still allowing
    simple variable references such as ``${SLURM_CPUS_PER_TASK}``."""
    _validate_dq_shell_value(value, label="export value")


def build_setup_commands(setup: list[SetupItem]) -> list[str]:
    """Build shell commands from an ordered list of setup items.

    Each item is a single-key table ``{command: args}`` rendered in list order,
    so the config author controls execution order directly. Item kinds:

    - ``{export = {KEY = "val"}}`` → ``export KEY="val"`` (rendered in a
      double-quoted context and validated with ``_validate_export_value``).
    - ``{module = ...}`` / ``{spack = ...}`` → ``module load`` / ``spack load``.
    - ``{<cmd> = [args...]}`` → ``<cmd> <args...>`` — the safe escape hatch.
      Every token is metacharacter-validated and ``shlex.quote``d, so no shell
      control operator can originate from config.

    A table value is accepted only for ``export``; for any other command it is a
    configuration error rather than a silently mishandled item.
    """
    cmds = []
    for item in setup:
        # A multi-key table would silently drop all but the first command; make
        # it an explicit error instead of guessing the author's intent.
        if len(item) != 1:
            raise ValueError(
                f"Each setup item must have exactly one command key: {item!r}"
            )
        cmd, args = next(iter(item.items()))
        _validate_arg(cmd)
        if cmd == "export":
            if not isinstance(args, dict):
                raise ValueError(
                    f"'export' requires a table of KEY = value pairs: {args!r}"
                )
            for key, value in args.items():
                _validate_arg(key)
                _validate_export_value(value)
                cmds.append(f'export {key}="{value}"')
            continue
        if isinstance(args, dict):
            raise ValueError(
                f"A table value is only allowed for 'export', not {cmd!r}: {args!r}"
            )
        allow_space = cmd in {"module", "spack"}
        args_list = [args] if isinstance(args, str) else args
        args_list = [a for a in args_list if a]
        for a in args_list:
            _validate_arg(a, allow_space=allow_space)
        if cmd in {"module", "spack"}:
            # ``module load`` / ``spack load`` need an operand; an empty spec
            # would emit a bare ``load`` that errors on the remote, so reject it
            # rather than ship a malformed command.
            if not args_list:
                raise ValueError(f"{cmd!r} requires a spec: {args!r}")
            quoted_args = " ".join(shlex.quote(a) for a in args_list)
            cmds.append(f"{cmd} load {quoted_args}")
        else:
            parts = [shlex.quote(cmd)] + [shlex.quote(a) for a in args_list]
            cmds.append(" ".join(parts))
    return cmds


class ClusterConfig(BaseModel):
    """Cluster connection configuration"""

    host: str
    workdir: str
    scheduler: str = "slurm"


class EnvConfig(BaseModel):
    """Environment configuration.

    Environment setup is a single ordered ``setup`` list; commands render in
    list order, which is the one place execution order is declared.
    ``extra="forbid"`` makes a removed bucket key (the former ``modules`` /
    ``spack`` / ``exports`` fields) fail loudly rather than be silently dropped
    — a silent drop would leave the environment unconfigured.
    """

    model_config = ConfigDict(extra="forbid")

    setup: list[SetupItem] = []

    def get_setup_commands(self) -> list[str]:
        """Render the ordered setup list to shell commands."""
        return build_setup_commands(self.setup)


class SyncConfig(BaseModel):
    """Sync configuration"""

    ignore: list[str] = []
    ignore_push: list[str] = []
    ignore_pull: list[str] = []
    compare: Literal["checksum", "timestamp"] = "checksum"
    pull_dir: str = ""


def _reject_directive_control_chars(value: str, label: str) -> None:
    """Reject newline / NUL — the only characters that break a scheduler
    directive (``#SBATCH`` / ``#PJM``) comment line into an executable
    script line. In-line shell metacharacters are inert on a ``#``-comment
    line, so unlike a double-quoted value (see ``_validate_dq_shell_value``)
    nothing else needs rejecting here."""
    if "\n" in value or "\x00" in value:
        raise ValueError(f"Newline and NUL characters are not allowed in {label}")


def _validate_submit_options(opts: list[str]) -> list[str]:
    """Reject only structurally unsafe characters in submit options."""
    for opt in opts:
        _reject_directive_control_chars(opt, "submit_options")
    return opts


class SlurmConfig(BaseModel):
    """Slurm job configuration"""

    options: dict[str, str | int] = {}
    submit_options: list[str] = []

    @field_validator("submit_options")
    @classmethod
    def check_submit_options(cls, v: list[str]) -> list[str]:
        return _validate_submit_options(v)

    @field_validator("options")
    @classmethod
    def check_options(cls, v: dict[str, str | int]) -> dict[str, str | int]:
        # ``options`` render into the job-script directive lines
        # (``#SBATCH --key=value``); a newline injects an executable script
        # line, exactly the hazard ``submit_options`` already guards against
        # (those are passed as submit-command argv, but share the same ban on
        # structural control chars). Both the key and a string value reach the
        # rendered line; an int value cannot carry control characters.
        for key, value in v.items():
            _reject_directive_control_chars(key, "slurm.options keys")
            if isinstance(value, str):
                _reject_directive_control_chars(value, "slurm.options values")
        return v


class PjmConfig(BaseModel):
    """PJM job configuration"""

    options: list[list[str]] = []
    submit_options: list[str] = []

    @field_validator("submit_options")
    @classmethod
    def check_submit_options(cls, v: list[str]) -> list[str]:
        return _validate_submit_options(v)

    @field_validator("options")
    @classmethod
    def check_options(cls, v: list[list[str]]) -> list[list[str]]:
        # Every inner-list element can reach a ``#PJM`` directive line; reject
        # the newline / NUL that would terminate the comment line. The renderer
        # currently emits only ``opt[0]`` / ``opt[1]``, but validating every
        # element matches ``submit_options``' policy and is conservative.
        for opt in v:
            for element in opt:
                _reject_directive_control_chars(element, "pjm.options elements")
        return v


class HpcConfig(BaseModel):
    """Combined HPC configuration"""

    cluster: ClusterConfig
    env: EnvConfig
    sync: SyncConfig = SyncConfig()
    slurm: SlurmConfig = SlurmConfig()
    pjm: PjmConfig = PjmConfig()


def find_config(
    filenames: tuple[str, ...] = ("crewster.toml", "hpc.toml"),
) -> tuple[Path, str] | None:
    """Walk up from CWD to find a config file, like git finds .git.

    ``filenames`` are checked in priority order *at each directory level*
    during a single upward walk, so the nearest directory always wins and a
    distant-ancestor ``crewster.toml`` never shadows a nearer legacy
    ``hpc.toml``. Returns the resolved path together with the matched filename
    (so the caller can warn only when the legacy name was hit), or ``None``.
    """
    current = Path.cwd().resolve()
    while True:
        for name in filenames:
            candidate = current / name
            if candidate.is_file():
                return candidate, name
        parent = current.parent
        if parent == current:
            return None
        current = parent


KNOWN_SECTIONS = {"cluster", "env", "sync", "slurm", "pjm"}


class ConfigManager:
    """TOML configuration file manager"""

    def load_config(self, path: Path) -> HpcConfig:
        """Load configuration from TOML file"""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)

        unknown = set(data.keys()) - KNOWN_SECTIONS
        for section in sorted(unknown):
            print(
                f"\033[33mWarning: unknown section [{section}] in {path}\033[0m",
                file=sys.stderr,
            )

        return HpcConfig(
            cluster=ClusterConfig(**data["cluster"]),
            env=EnvConfig(**data.get("env", {})),
            sync=SyncConfig(**data.get("sync", {})),
            slurm=SlurmConfig(
                options=data.get("slurm", {}).get("options", {}),
                submit_options=data.get("slurm", {}).get("submit_options", []),
            ),
            pjm=PjmConfig(
                options=data.get("pjm", {}).get("options", []),
                submit_options=data.get("pjm", {}).get("submit_options", []),
            ),
        )

    def generate_template(self, path: Path, scheduler: str = "slurm") -> None:
        """Generate template configuration file for the given scheduler.

        ``scheduler`` selects which scheduler-specific section is emitted.
        ``cluster.scheduler`` is written explicitly so the file is symmetric
        across schedulers even though ``ClusterConfig.scheduler`` defaults
        to ``"slurm"``.

        Unknown values raise ``ValueError`` rather than silently emitting a
        Slurm-shaped template, so library callers that bypass the CLI's
        ``SchedulerChoice`` constraint cannot slip through with a typo.
        """
        if scheduler not in {"slurm", "pjm"}:
            raise ValueError(f"Unknown scheduler: {scheduler!r}")
        if scheduler == "pjm":
            template = {
                "cluster": {
                    "host": "myhpc",
                    "workdir": "/scratch/${USER}/myproj",
                    "scheduler": "pjm",
                },
                "env": {
                    "setup": [
                        {"module": "gcc/12.2.0"},
                    ],
                },
                "sync": {
                    "ignore": ["crewster.toml", ".git"],
                    "ignore_push": [".crewster"],
                },
                "pjm": {
                    "submit_options": [],
                    "options": [
                        ["-L", "node=1"],
                        ["-L", "rscgrp=small"],
                        ["-L", "elapse=02:00:00"],
                        ["-g", "myaccount"],
                    ],
                },
            }
        else:
            template = {
                "cluster": {
                    "host": "myhpc",
                    "workdir": "/scratch/${USER}/myproj",
                    "scheduler": "slurm",
                },
                "env": {
                    "setup": [
                        {"module": "gcc/12.2.0"},
                        {"module": "cuda/12.2"},
                    ],
                },
                "sync": {
                    "ignore": ["crewster.toml", ".git"],
                    "ignore_push": [".crewster"],
                },
                "slurm": {
                    "submit_options": [],
                    "options": {
                        "partition": "gpu",
                        "time": "02:00:00",
                        "mem": "32G",
                        "gpus": 1,
                        "account": "myaccount",
                    },
                },
            }
        with open(path, "wb") as f:
            tomli_w.dump(template, f)
