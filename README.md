# hpc

One-stop lightweight wrapper for seamless remote-HPC dev-loop for coding agents and humans: sync the working tree, run a quick verification job (Slurm/PJM), pull results.

## Why not Snakemake?

hpc is **not** a job orchestrator, and does not try to be one.

Its responsibility is the inner development loop against a remote environment: you are actively editing code — often *before it is committed* — and you want to push the current working tree to a remote HPC environment and run a quick test there, repeatedly and fast. [Snakemake](https://github.com/snakemake/snakemake), [Nextflow](https://github.com/nextflow-io/nextflow), and similar tools assume a defined, committed pipeline and manage its execution graph; they serve the *other* end of the lifecycle.

They are complementary, not competing:

- **hpc** — frequent pre-commit source sync, single test / verification runs, a tight edit → run → observe loop. Built for a coding agent iterating against a remote environment.
- **A workflow runner** — dependency graphs, multi-step pipelines, retries, production runs.

If you need to orchestrate a complex production run, use a real orchestrator — and you can still launch it *through* hpc (`hpc submit "snakemake ..."`): hpc handles the transport (sync up, run, pull results) and treats the workflow as opaque user code. hpc owns the transport and the dev loop, never the run graph.

## Installation

One-shot execution (no install):

```bash
uvx --from git+https://github.com/ultimatile/hpc hpc
```

Permanent install:

```bash
uv tool install git+https://github.com/ultimatile/hpc
```

## Quick Start

```bash
# 1. Initialize project
hpc init

# 2. Edit configuration
vim hpc.toml

# 3. Sync files to cluster
hpc sync
hpc sync --dry-run  # preview only

# 4. Submit job
hpc submit "python train.py"

# 5. Check status
hpc status 12345678

# 6. View job output
hpc job-output 12345678
```

## Commands

### `hpc init`

Creates `hpc.toml` configuration file in the current directory.

```bash
hpc init                      # Slurm template (default)
hpc init --scheduler pjm      # PJM-oriented template
```

`--scheduler` selects which scheduler-specific section is written. The default is `slurm`.

When `$XDG_CONFIG_HOME/hpc/config.toml` exists, it is used as the source instead of the built-in template. See [User-level XDG config](#user-level-xdg-config) for the filter-merge semantics.

### `hpc sync`

Syncs local files to the remote HPC cluster using rsync.
Always syncs the entire project root (where `hpc.toml` is located), regardless of which subdirectory you run from.

```bash
hpc sync                # sync files
hpc sync --dry-run      # preview without syncing (-n for short)
hpc sync --workdir /scratch/user/other   # override remote workdir
hpc sync --push         # push only (local → remote)
hpc sync --pull         # pull only (remote → local)
```

### `hpc exec`

Executes a command directly on the login node (not via scheduler). Useful for setup tasks that need internet access (package installs, dependency downloads).

```bash
hpc exec "julia -e 'using Pkg; Pkg.instantiate()'"
hpc exec --script setup.sh
hpc exec --workdir /scratch/user/other "cmake .."
```

Environment setup (`[env]` section) is applied automatically. The working directory follows the same CWD-relative logic as `hpc submit`.

### `hpc submit`

Submits a job to the configured scheduler.
Returns both run_id (e.g., `20260109_1234`, hpc's local tracking ID) and job_id (scheduler job ID, e.g., `12345678`).

The job's working directory is set based on your current position relative to the project root (see [Multi-Setup Runs](#multi-setup-runs)).

```bash
hpc submit "python train.py"
hpc submit --script run.sh
hpc submit -s run.sh --wait
hpc submit --workdir /scratch/user/other "python train.py"  # override remote workdir
```

`#SBATCH` (Slurm) and `#PJM` (PJM) directives written at the top of a script passed via `--script` are honored: hpc hoists them into the prologue of the rendered job script, so they are scanned by `sbatch` / `pjsub` instead of being silently treated as comments.

Only column-zero directive lines that appear before the first executable line in the user script are hoisted, matching the schedulers' own prologue-scan rule. Directives after an executable line, or inside heredocs, are left in the body as-is.

When the same option is set both via config (`[slurm.options]` for Slurm, the `pjm.options` array for PJM) and via a `#SBATCH` / `#PJM` line in the script, the script's value wins (the scheduler's last-occurrence-wins semantics for duplicate directives). The `submit_options` list is passed as command-line flags to `sbatch` / `pjsub` and, per scheduler specifications, overrides script directives unconditionally.

### `hpc status`

Checks the status of a submitted job.
Accepts either run_id or job_id.

```bash
hpc status 12345678
```

### `hpc job-output`

Shows the output of a submitted job.
Accepts either run_id or job_id.

```bash
hpc job-output 12345678
```

Pass `--follow` / `-f` to stream the output of a running job in real time
(equivalent to `tail -F` on the remote output file). Combine with `--error` /
`-e` to follow stderr instead of stdout. For terminal-state jobs the command
prints the final output and exits.

```bash
hpc job-output -f 12345678
hpc job-output -f -e 12345678
```

### `hpc wait`

Waits for a run to complete.
Accepts either run_id or job_id.

```bash
hpc wait 12345678
```

## Project Root and Config Discovery

hpc walks up from the current directory to find `hpc.toml`, similar to how git finds `.git`. This means you can run hpc commands from any subdirectory within your project.

Resolution order: `--config` / `-c` > `$HPC_CONFIG` > walk-up discovery > `./hpc.toml`.

The directory containing `hpc.toml` is the **project root**. This affects:

- **`hpc sync`**: always syncs the entire project root to `workdir`, regardless of CWD
- **`hpc submit`**: sets the job's `cd` to `workdir` + (CWD relative to project root)
- **`.hpc/runs/`**: run metadata is always stored at the project root

`hpc init` does not walk up — it always creates `hpc.toml` in the current directory.

## Multi-Setup Runs

When running multiple benchmarks or parameter sets from a single project, use subdirectories. hpc automatically maps your local directory structure to the remote.

```
myproject/
  hpc.toml              # workdir = "/remote/myproject"
  src/main.py
  runs/
    setup-a/
      input.dat
    setup-b/
      input.dat
```

```bash
# Sync the entire project (same result from any subdirectory)
hpc sync

# Submit from a subdirectory — job runs in the matching remote path
cd runs/setup-a
hpc submit "python src/main.py"
# → job cd's to /remote/myproject/runs/setup-a

cd ../setup-b
hpc submit "python src/main.py"
# → job cd's to /remote/myproject/runs/setup-b
```

Key points:

- **sync** is always project-wide. The remote mirrors your local project structure exactly.
- **submit** uses your CWD to determine the job's working directory on the remote.
- **`--workdir`** overrides `cluster.workdir` for one-off use without editing `hpc.toml`.
- Large artifacts that shouldn't be synced are managed via `[sync] ignore`.

## Configuration

Edit `hpc.toml`:

```toml
[cluster]
host = "myhpc"                    # SSH host (from ~/.ssh/config)
workdir = "/scratch/user/proj"    # Remote working directory; all codes and data will be synced here
scheduler = "slurm"                # "slurm" (default) or "pjm"

[env]
modules = ["gcc/12.2.0", "cuda/12.2"]  # Modules to load (shorthand for module load)
spack = ["python@3.11"]                # Spack packages to load (shorthand for spack load)
setup = [                              # Additional setup commands
    {source = "/path/to/venv/bin/activate"},
    {export = ["VAR=value"]},          # {command = [args...]} format
    "some_cmd",                        # String: command without args
]

[sync]
ignore = ["hpc.toml", ".git"]  # Patterns to exclude from sync
compare = "checksum"           # File comparison: "checksum" (content-based, default) or "timestamp"
pull_dir = "~/data/myproj"     # Pull destination (default: project root). Useful for keeping git repo clean

[slurm.options]
partition = "gpu"      # Example (Slurm): partition
time = "02:00:00"      # Example (Slurm): time limit
mem = "32G"            # Example (Slurm): memory
gpus = 1               # Example (Slurm): number of GPUs
```

### Environment Setup

Commands are executed in this order: `modules` → `spack` → `setup`.

`modules` and `spack` are shorthand syntax:

- `modules = ["gcc/12.2.0"]` expands to `module load gcc/12.2.0`
- `spack = ["python@3.11"]` expands to `spack load python@3.11`

`setup` accepts:

- String: command without args (e.g., `"some_cmd"`)
- Dict: `{command = args}` format (e.g., `{export = ["VAR=value"]}` → `export VAR=value`)
- Special commands `module` and `spack` in dict format expand to `module load` / `spack load`

If you need a different execution order, put everything in `setup`:

```toml
[env]
setup = [
    {spack = "python@3.11"},
    {module = "gcc/12.2.0"},
    {source = "/path/to/venv/bin/activate"},
]
```

Shell special characters (`` ;|&`$<>\'"\n `` and space) are prohibited in arguments for security.

### PJM Configuration

For PJM scheduler, use array format for options:

```toml
[cluster]
scheduler = "pjm"

[pjm]
options = [
    ["-L", "node=12"],
    ["-L", "rscgrp=small"],
    ["-L", "elapse=00:30:00"],
    ["--mpi", "max-proc-per-node=4"],
    ["-g", "laa4Hoo5"],
    ["-s"]
]
```

### User-level XDG config

`$XDG_CONFIG_HOME/hpc/config.toml` (default: `~/.config/hpc/config.toml`), when present, is used as the source for `hpc init` instead of the built-in template. The file is filter-merged onto the chosen scheduler:

- The inactive scheduler's top-level section (`[pjm]` under `--scheduler slurm`, `[slurm]` under `--scheduler pjm`) is dropped.
- `cluster.scheduler` is forced to match the `--scheduler` argument.
- All other sections (including unknown ones) carry over with their parsed TOML values intact.

The source XDG file is not modified. This lets the XDG file carry both `[slurm]` and `[pjm]` sections side by side so that `hpc init --scheduler {slurm,pjm}` projects out the active half. Because the file goes through `tomllib.load` and `tomli_w.dump`, comments and original formatting (e.g. inline-array layout) are not preserved in the generated `hpc.toml`; only the parsed data is.

## Requirements

- Python 3.11+
- SSH access to HPC cluster (key-based authentication recommended)
- rsync
- Slurm or PJM on the remote cluster

### rsync Note

rsync from <https://rsync.samba.org/> is recommended over macOS's built-in openrsync. When using checksum-based comparison (`compare = "checksum"`, default), openrsync has a bug where files with sizes that are exact multiples of 64 bytes are always detected as changed, even when identical. This is due to a protocol 29 checksum boundary issue. Confirmed with macOS 15.7's openrsync (protocol version 29, rsync version 2.6.9 compatible). If concerned, use `[sync] compare = "timestamp"` instead.

On macOS, install rsync via Homebrew:

```bash
brew install rsync
```

## Claude Code Integration

This project includes a [Claude Code skill](https://docs.anthropic.com/en/docs/claude-code/skills) (`.claude/skills/hpc/SKILL.md`) that teaches Claude how to use the hpc CLI. The CLI reference in the skill is dynamically generated via `hpc --skill` to stay in sync with the code.

## Development

```bash
make test      # run tests
make lint      # run linter
make check     # run all checks
```
