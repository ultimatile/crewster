"""Configuration management"""

import shlex
import sys
import tomllib
from pathlib import Path
from typing import Literal

import tomli_w
from pydantic import BaseModel, field_validator

# str: command without args, dict: {cmd: args}
SetupItem = str | dict[str, str | list[str]]

SHELL_SPECIAL = set(";|&`$<>\\'\"\n ")


def _validate_arg(arg: str) -> None:
    if bad := SHELL_SPECIAL & set(arg):
        raise ValueError(f"Shell special characters not allowed: {bad}")


def _validate_export_value(value: str) -> None:
    """Reject command substitution in export values while allowing variable references."""
    if "$(" in value or "`" in value:
        raise ValueError(f"Command substitution not allowed in export value: {value!r}")


def build_setup_commands(setup: list[SetupItem]) -> list[str]:
    """Build shell commands from setup items"""
    cmds = []
    for item in setup:
        if isinstance(item, str):
            _validate_arg(item)
            cmds.append(shlex.quote(item))
        else:
            cmd, args = next(iter(item.items()))
            _validate_arg(cmd)
            args_list = [args] if isinstance(args, str) else args
            args_list = [a for a in args_list if a]
            for a in args_list:
                _validate_arg(a)
            quoted_args = " ".join(shlex.quote(a) for a in args_list)
            if cmd == "module":
                cmds.append(f"module load {quoted_args}")
            elif cmd == "spack":
                cmds.append(f"spack load {quoted_args}")
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
    """Environment configuration"""

    modules: list[str] = []
    spack: list[str] = []
    setup: list[SetupItem] = []
    exports: dict[str, str] = {}

    def get_setup_commands(self) -> list[str]:
        """Build commands: modules → spack → setup → exports"""
        items: list[SetupItem] = []
        for m in self.modules:
            items.append({"module": m})
        for s in self.spack:
            items.append({"spack": s})
        items.extend(self.setup)
        cmds = build_setup_commands(items)
        for key, value in self.exports.items():
            _validate_arg(key)
            _validate_export_value(value)
            cmds.append(f'export {key}="{value}"')
        return cmds


class SyncConfig(BaseModel):
    """Sync configuration"""

    ignore: list[str] = []
    ignore_push: list[str] = []
    ignore_pull: list[str] = []
    compare: Literal["checksum", "timestamp"] = "checksum"
    pull_dir: str = ""


def _validate_submit_options(opts: list[str]) -> list[str]:
    """Reject only structurally unsafe characters in submit options."""
    for opt in opts:
        if "\n" in opt or "\x00" in opt:
            raise ValueError(
                "Newline and NUL characters are not allowed in submit_options"
            )
    return opts


class SlurmConfig(BaseModel):
    """Slurm job configuration"""

    options: dict[str, str | int] = {}
    submit_options: list[str] = []

    @field_validator("submit_options")
    @classmethod
    def check_submit_options(cls, v: list[str]) -> list[str]:
        return _validate_submit_options(v)


class PjmConfig(BaseModel):
    """PJM job configuration"""

    options: list[list[str]] = []
    submit_options: list[str] = []

    @field_validator("submit_options")
    @classmethod
    def check_submit_options(cls, v: list[str]) -> list[str]:
        return _validate_submit_options(v)


class HpcConfig(BaseModel):
    """Combined HPC configuration"""

    cluster: ClusterConfig
    env: EnvConfig
    sync: SyncConfig = SyncConfig()
    slurm: SlurmConfig = SlurmConfig()
    pjm: PjmConfig = PjmConfig()


def find_config(filename: str = "hpc.toml") -> Path | None:
    """Walk up from CWD to find config file, like git finds .git"""
    current = Path.cwd().resolve()
    while True:
        candidate = current / filename
        if candidate.is_file():
            return candidate
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
                    "modules": ["gcc/12.2.0"],
                },
                "sync": {
                    "ignore": ["hpc.toml", ".git"],
                    "ignore_push": [".hpc"],
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
                    "modules": ["gcc/12.2.0", "cuda/12.2"],
                },
                "sync": {
                    "ignore": ["hpc.toml", ".git"],
                    "ignore_push": [".hpc"],
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
