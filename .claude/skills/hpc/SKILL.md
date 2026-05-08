---
name: hpc
description: HPC cluster workflow automation - sync files to remote cluster, submit scheduler jobs (Slurm/PJM), and monitor job status. Use when working with hpc.toml, submitting HPC jobs, syncing to clusters, or checking job status.
allowed-tools: Bash(hpc:*)
---

# hpc CLI

HPC workflow automation tool for file sync and scheduler job management (Slurm/PJM).

## CLI Reference

```
!`hpc --skill`
```

## Getting Started

If `hpc.toml` does not exist in the project, run `hpc init` to create it, then ask the user to edit it with their cluster settings (host, workdir, scheduler, etc.) before proceeding.

## Typical Workflow

1. `hpc sync` - Sync files to remote cluster (`--dry-run` to preview)
2. `hpc exec "command"` - Run setup on the login node (package installs, configure, etc.)
3. `hpc submit "command"` or `hpc submit -s script.sh` - Submit a job
4. `hpc status <id>` - Check job status (accepts run_id or job_id)
5. `hpc job-output <id>` - View stdout (`-e` for stderr)
6. `hpc wait <id>` - Wait for completion

## Key Concepts

- **Project root**: hpc walks up from CWD to find `hpc.toml` (like git finds `.git`). All commands work from any subdirectory.
- **run_id vs job_id**: `hpc submit` returns both. Either can be used with `status`, `job-output`, `wait`.
- **Default working directory**: `hpc submit` runs the job with its PWD set to the resolved `[cluster].workdir` from `hpc.toml`. Scripts passed via `-s` therefore do **not** need to `cd` to the project root — they start there. Adding a manual `cd` (or relying on `$SLURM_SUBMIT_DIR`, which may not be propagated to the job's environment) can move the job to the wrong directory.
- **Multi-setup runs**: Submit from subdirectories to set the job's remote working directory accordingly (e.g., `cd runs/setup-a && hpc submit "python main.py"` runs in `/remote/project/runs/setup-a`).
- **Config resolution**: `--config` / `-c` > `$HPC_CONFIG` > walk-up discovery > `./hpc.toml`
- **Sync scope**: `hpc sync` always syncs the entire project root, regardless of CWD.

## Writing scripts for `hpc submit -s`

- Start from the assumption that PWD is already `[cluster].workdir`. No `cd` needed for the common case.
- If you genuinely need a different working directory, prefer an explicit absolute path or a `WORKDIR` env-var override; do **not** rely on `$SLURM_SUBMIT_DIR` (the scheduler does not always propagate it into the job's shell env).
- The script's stdout/stderr land in the scheduler's job-output files, retrievable via `hpc job-output <id>`.
- `#SBATCH` / `#PJM` directives at the top of the script (column-zero, before the first executable line) are honored: hpc hoists them into the rendered job-script prologue. On conflict with `[slurm.options]` / `[pjm.options]`, the script's value wins.

## Common pitfalls

- **Adding `cd "$HOME/path/to/project"` to a submitted script**: The job already starts in workdir; the manual `cd` is redundant and silently incorrect if the path doesn't exactly match the deployed one.
- **Hardcoding deployment-specific paths in repo-tracked scripts**: Use a `WORKDIR` env override or script-relative resolution; deployment paths belong in `hpc.toml`, not script source.
