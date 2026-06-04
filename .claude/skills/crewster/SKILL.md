---
name: crewster
description: HPC cluster workflow automation - sync files to remote cluster, submit scheduler jobs (Slurm/PJM), and monitor job status. Use when working with crewster.toml, submitting HPC jobs, syncing to clusters, or checking job status.
allowed-tools: Bash(crewster:*)
---

# crewster CLI

HPC workflow automation tool for file sync and scheduler job management (Slurm/PJM).

## CLI Reference

```
!`crewster --skill`
```

## Getting Started

If `crewster.toml` does not exist in the project, run `crewster init` to create it, then ask the user to edit it with their cluster settings (host, workdir, scheduler, etc.) before proceeding.

## Typical Workflow

1. `crewster sync` - Sync files to remote cluster (`--dry-run` to preview)
2. `crewster exec "command"` - Run setup on the login node (package installs, configure, etc.)
3. `crewster submit "command"` or `crewster submit -s script.sh` - Submit a job
4. `crewster status <id>` - Check job status (accepts run_id or job_id)
5. `crewster job-output <id>` - View stdout (`-e` for stderr)
6. `crewster wait <id>` - Wait for completion

## Key Concepts

- **Project root**: crewster walks up from CWD to find `crewster.toml` (like git finds `.git`). All commands work from any subdirectory.
- **run_id vs job_id**: `crewster submit` returns both. Either can be used with `status`, `job-output`, `wait`.
- **Default working directory**: `crewster submit` runs the job with its PWD set to the resolved `[cluster].workdir` from `crewster.toml`. Scripts passed via `-s` therefore do **not** need to `cd` to the project root — they start there. Adding a manual `cd` (or relying on `$SLURM_SUBMIT_DIR`, which may not be propagated to the job's environment) can move the job to the wrong directory.
- **Multi-setup runs**: Submit from subdirectories to set the job's remote working directory accordingly (e.g., `cd runs/setup-a && crewster submit "python main.py"` runs in `/remote/project/runs/setup-a`).
- **Config resolution**: `--config` / `-c` > `$CREWSTER_CONFIG` > walk-up discovery > `./crewster.toml`
- **Sync scope**: `crewster sync` always syncs the entire project root, regardless of CWD.

## Writing scripts for `crewster submit -s`

- Start from the assumption that PWD is already `[cluster].workdir`. No `cd` needed for the common case.
- If you genuinely need a different working directory, prefer an explicit absolute path or a `WORKDIR` env-var override; do **not** rely on `$SLURM_SUBMIT_DIR` (the scheduler does not always propagate it into the job's shell env).
- The script's stdout/stderr land in the scheduler's job-output files, retrievable via `crewster job-output <id>`.
- `#SBATCH` / `#PJM` directives at the top of the script (column-zero, before the first executable line) are honored: crewster hoists them into the rendered job-script prologue. On conflict with the corresponding config entry (`[slurm.options]` for Slurm, the `pjm.options` array for PJM), the script's value wins.

## Common pitfalls

- **Adding `cd "$HOME/path/to/project"` to a submitted script**: The job already starts in workdir; the manual `cd` is redundant and silently incorrect if the path doesn't exactly match the deployed one.
- **Hardcoding deployment-specific paths in repo-tracked scripts**: Use a `WORKDIR` env override or script-relative resolution; deployment paths belong in `crewster.toml`, not script source.
