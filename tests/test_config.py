"""Config manager tests"""

import pytest
from pathlib import Path
from pydantic import ValidationError

from crewster.config import (
    ClusterConfig,
    EnvConfig,
    SlurmConfig,
    PjmConfig,
    HpcConfig,
    ConfigManager,
    find_config,
    _validate_dq_shell_value,
)


class TestDqShellValueValidator:
    """The shared validator guarding double-quoted shell contexts
    (`export KEY="..."`, `cd "..."`)."""

    @pytest.mark.parametrize(
        "value",
        [
            'a"; touch /tmp/pwned; echo "',  # quote breakout
            "$(touch /tmp/pwned)",  # command substitution
            "`touch /tmp/pwned`",  # backtick substitution
            "${BAR@P}",  # prompt re-expansion bypass
            "${!BAR}",  # indirect expansion
            "${BAR:-$(x)}",  # default-value form smuggling $(
            "$((1 + 1))",  # arithmetic expansion
            "a\nb",  # newline
            "a\x00b",  # NUL
            "a\\b",  # backslash escape
        ],
    )
    def test_rejects_unsafe(self, value):
        with pytest.raises(ValueError, match="Unsafe characters"):
            _validate_dq_shell_value(value)

    @pytest.mark.parametrize(
        "value",
        [
            "$HOME",
            "${USER}",
            "${SLURM_CPUS_PER_TASK}",
            "/scratch/${USER}/proj",
            "/scratch/user/proj",
            "02:00:00",
            "plain",
        ],
    )
    def test_allows_safe(self, value):
        _validate_dq_shell_value(value)  # must not raise


class TestClusterConfig:
    def test_cluster_config_required_fields(self):
        config = ClusterConfig(host="myhpc", workdir="/scratch/user/proj")
        assert config.host == "myhpc"
        assert config.workdir == "/scratch/user/proj"

    def test_cluster_config_missing_host_raises(self):
        with pytest.raises(Exception):
            ClusterConfig(workdir="/scratch/user/proj")


class TestEnvConfig:
    def test_env_config_with_setup(self):
        config = EnvConfig(setup=[{"module": "gcc/12.2.0"}, {"spack": "cuda@12"}])
        assert len(config.setup) == 2

    def test_env_config_defaults(self):
        config = EnvConfig()
        assert config.setup == []

    def test_module_and_spack_render_load_commands(self):
        config = EnvConfig(setup=[{"module": "gcc/12.2.0"}, {"spack": "cuda@12"}])
        assert config.get_setup_commands() == [
            "module load gcc/12.2.0",
            "spack load cuda@12",
        ]

    def test_setup_preserves_list_order(self):
        # The single ordered list is the one place execution order is declared;
        # commands must render in exactly the authored order.
        config = EnvConfig(
            setup=[
                {"export": {"SPACK_ROOT": "/opt/spack"}},
                {"source": ["/opt/spack/share/spack/setup-env.sh"]},
                {"spack": "boost@1.86.0"},
                {"module": "gcc/12.2.0"},
            ]
        )
        assert config.get_setup_commands() == [
            'export SPACK_ROOT="/opt/spack"',
            "source /opt/spack/share/spack/setup-env.sh",
            "spack load boost@1.86.0",
            "module load gcc/12.2.0",
        ]

    def test_spack_spec_allows_spaces(self):
        # A spack spec is one logical unit that legitimately contains spaces
        # (version + variant + arch). It renders to a single quoted shell word,
        # which spack parses back into one spec.
        config = EnvConfig(
            setup=[{"spack": "boost@1.86.0 ~mpi arch=linux-rhel8-a64fx"}]
        )
        assert config.get_setup_commands() == [
            "spack load 'boost@1.86.0 ~mpi arch=linux-rhel8-a64fx'"
        ]

    def test_module_spec_allows_spaces(self):
        config = EnvConfig(setup=[{"module": "gcc 12.2.0"}])
        assert config.get_setup_commands() == ["module load 'gcc 12.2.0'"]

    def test_spack_still_rejects_dangerous_metacharacters(self):
        # The space relaxation must not relax the genuinely dangerous chars.
        config = EnvConfig(setup=[{"spack": "boost; rm -rf ~"}])
        with pytest.raises(ValueError, match="Shell special characters"):
            config.get_setup_commands()

    def test_generic_command_is_safe_escape_hatch(self):
        config = EnvConfig(setup=[{"ulimit": ["-s", "unlimited"]}])
        assert config.get_setup_commands() == ["ulimit -s unlimited"]

    def test_generic_command_no_args(self):
        config = EnvConfig(setup=[{"nvidia-smi": []}])
        assert config.get_setup_commands() == ["nvidia-smi"]

    @pytest.mark.parametrize("spec", ["", []])
    def test_module_requires_spec(self, spec):
        # An empty spec would emit a bare `module load` that errors remotely.
        config = EnvConfig(setup=[{"module": spec}])
        with pytest.raises(ValueError, match="requires a spec"):
            config.get_setup_commands()

    @pytest.mark.parametrize("spec", ["", []])
    def test_spack_requires_spec(self, spec):
        config = EnvConfig(setup=[{"spack": spec}])
        with pytest.raises(ValueError, match="requires a spec"):
            config.get_setup_commands()

    def test_generic_command_rejects_metacharacters(self):
        # The escape hatch quotes each token, but still bans control operators
        # so a shell metacharacter can never originate from config.
        config = EnvConfig(setup=[{"ulimit": ["-s; rm -rf ~"]}])
        with pytest.raises(ValueError, match="Shell special characters"):
            config.get_setup_commands()

    # --- export item ---------------------------------------------------------

    def test_export_item_generates_export_commands(self):
        config = EnvConfig(
            setup=[
                {"export": {"OMP_NUM_THREADS": "${SLURM_CPUS_PER_TASK}", "FOO": "bar"}}
            ]
        )
        cmds = config.get_setup_commands()
        assert 'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"' in cmds
        assert 'export FOO="bar"' in cmds

    def test_export_allows_simple_variable_references(self):
        # Simple `$VAR` / `${VAR}` parameter references must survive — they are
        # the intended remote-expansion use case.
        config = EnvConfig(
            setup=[{"export": {"HOME_REF": "$HOME", "NESTED": "/scratch/${USER}/out"}}]
        )
        cmds = config.get_setup_commands()
        assert 'export HOME_REF="$HOME"' in cmds
        assert 'export NESTED="/scratch/${USER}/out"' in cmds

    def test_export_rejects_command_substitution_dollar(self):
        config = EnvConfig(setup=[{"export": {"FOO": "$(rm -rf /)"}}])
        with pytest.raises(ValueError, match="Unsafe characters"):
            config.get_setup_commands()

    def test_export_rejects_command_substitution_backtick(self):
        config = EnvConfig(setup=[{"export": {"FOO": "`rm -rf /`"}}])
        with pytest.raises(ValueError, match="Unsafe characters"):
            config.get_setup_commands()

    def test_export_rejects_double_quote_breakout(self):
        # The value is rendered inside `export KEY="<value>"`; a literal `"`
        # closes the quote and turns the rest into commands.
        config = EnvConfig(setup=[{"export": {"FOO": 'x"; touch /tmp/pwned; echo "'}}])
        with pytest.raises(ValueError, match="Unsafe characters"):
            config.get_setup_commands()

    def test_export_rejects_prompt_expansion(self):
        # `${VAR@P}` re-evaluates its value as a prompt string, performing
        # command substitution assembled from otherwise-allowed values.
        config = EnvConfig(setup=[{"export": {"FOO": "${BAR@P}"}}])
        with pytest.raises(ValueError, match="Unsafe characters"):
            config.get_setup_commands()

    def test_export_rejects_newline(self):
        config = EnvConfig(setup=[{"export": {"FOO": "a\nb"}}])
        with pytest.raises(ValueError, match="Unsafe characters"):
            config.get_setup_commands()

    def test_export_rejects_nul(self):
        config = EnvConfig(setup=[{"export": {"FOO": "a\x00b"}}])
        with pytest.raises(ValueError, match="Unsafe characters"):
            config.get_setup_commands()

    def test_export_rejects_invalid_key(self):
        config = EnvConfig(setup=[{"export": {"FOO;BAR": "value"}}])
        with pytest.raises(ValueError):
            config.get_setup_commands()

    # --- clean-break rejections ---------------------------------------------

    @pytest.mark.parametrize("field", ["modules", "spack", "exports"])
    def test_old_bucket_keys_rejected(self, field):
        # The removed top-level buckets must fail loudly (extra="forbid"),
        # not be silently dropped into an empty environment.
        with pytest.raises(ValidationError):
            EnvConfig(**{field: ["gcc/12.2.0"]})

    def test_bare_string_item_rejected(self):
        with pytest.raises(ValidationError):
            EnvConfig(setup=["my_setup"])

    def test_multi_key_item_rejected(self):
        config = EnvConfig(setup=[{"module": "gcc", "spack": "cuda"}])
        with pytest.raises(ValueError, match="exactly one command key"):
            config.get_setup_commands()

    def test_table_value_rejected_for_non_export(self):
        config = EnvConfig(setup=[{"module": {"foo": "bar"}}])
        with pytest.raises(ValueError, match="only allowed for 'export'"):
            config.get_setup_commands()

    def test_export_requires_table_value(self):
        config = EnvConfig(setup=[{"export": ["FOO=bar"]}])
        with pytest.raises(ValueError, match="requires a table"):
            config.get_setup_commands()


class TestSlurmConfig:
    def test_slurm_config_default_options(self):
        config = SlurmConfig()
        assert config.options == {}
        assert config.submit_options == []

    def test_slurm_config_with_options(self):
        config = SlurmConfig(
            options={"partition": "gpu", "time": "02:00:00", "gpus": 1}
        )
        assert config.options["partition"] == "gpu"
        assert config.options["gpus"] == 1

    def test_slurm_config_with_submit_options(self):
        config = SlurmConfig(submit_options=["--export=ALL"])
        assert config.submit_options == ["--export=ALL"]

    def test_slurm_config_rejects_newline_in_submit_options(self):
        with pytest.raises(ValueError, match="Newline and NUL"):
            SlurmConfig(submit_options=["--opt\nmalicious"])

    def test_slurm_config_rejects_nul_in_submit_options(self):
        with pytest.raises(ValueError, match="Newline and NUL"):
            SlurmConfig(submit_options=["--opt\x00malicious"])

    def test_slurm_config_rejects_newline_in_option_value(self):
        # options render into `#SBATCH --key=value` directive lines; a newline
        # in the value injects an executable script line.
        with pytest.raises(ValueError, match="Newline and NUL"):
            SlurmConfig(options={"partition": "gpu\ntouch /tmp/pwned"})

    def test_slurm_config_rejects_newline_in_option_key(self):
        with pytest.raises(ValueError, match="Newline and NUL"):
            SlurmConfig(options={"part\nition": "gpu"})

    def test_slurm_config_rejects_nul_in_option_value(self):
        with pytest.raises(ValueError, match="Newline and NUL"):
            SlurmConfig(options={"partition": "gpu\x00x"})

    def test_slurm_config_allows_int_option_value(self):
        # int values cannot carry control chars; the validator must skip them.
        config = SlurmConfig(options={"nodes": 1, "time": "02:00:00"})
        assert config.options["nodes"] == 1


class TestPjmConfig:
    def test_pjm_config_defaults(self):
        config = PjmConfig()
        assert config.options == []
        assert config.submit_options == []

    def test_pjm_config_with_submit_options(self):
        config = PjmConfig(submit_options=["--no-check-directory"])
        assert config.submit_options == ["--no-check-directory"]

    def test_pjm_config_rejects_newline_in_submit_options(self):
        with pytest.raises(ValueError, match="Newline and NUL"):
            PjmConfig(submit_options=["--opt\nmalicious"])

    def test_pjm_config_rejects_newline_in_option_element(self):
        # Each inner-list element can reach a `#PJM` directive line.
        with pytest.raises(ValueError, match="Newline and NUL"):
            PjmConfig(options=[["-L", "node=1\ntouch /tmp/pwned"]])

    def test_pjm_config_rejects_nul_in_option_element(self):
        with pytest.raises(ValueError, match="Newline and NUL"):
            PjmConfig(options=[["-L", "node=1\x00x"]])

    def test_pjm_config_allows_normal_options(self):
        config = PjmConfig(options=[["-L", "node=1"], ["-g", "myaccount"]])
        assert config.options[0] == ["-L", "node=1"]


class TestHpcConfig:
    def test_hpc_config_combines_all(self):
        config = HpcConfig(
            cluster=ClusterConfig(host="myhpc", workdir="/scratch/user/proj"),
            env=EnvConfig(setup=[{"module": "gcc/12.2.0"}]),
            slurm=SlurmConfig(options={"partition": "gpu"}),
        )
        assert config.cluster.host == "myhpc"
        assert config.slurm.options["partition"] == "gpu"


class TestConfigManager:
    def test_load_config_from_toml(self, temp_dir):
        config_path = temp_dir / "crewster.toml"
        config_path.write_text("""
[cluster]
host = "myhpc"
workdir = "/scratch/user/proj"

[env]
setup = [
    { module = "gcc/12.2.0" },
    { spack = "cuda@12" },
]

[slurm.options]
partition = "gpu"
time = "02:00:00"
mem = "32G"
gpus = 1
""")
        manager = ConfigManager()
        config = manager.load_config(config_path)

        assert config.cluster.host == "myhpc"
        assert config.cluster.workdir == "/scratch/user/proj"
        assert len(config.env.setup) == 2
        assert config.slurm.options["partition"] == "gpu"
        assert config.slurm.options["time"] == "02:00:00"
        assert config.slurm.options["mem"] == "32G"
        assert config.slurm.options["gpus"] == 1

    def test_load_pjm_config_with_submit_options(self, temp_dir):
        config_path = temp_dir / "crewster.toml"
        config_path.write_text("""
[cluster]
host = "myhpc"
workdir = "/scratch/user/proj"
scheduler = "pjm"

[pjm]
options = [["-L", "node=12"], ["-s"]]
submit_options = ["--no-check-directory"]
""")
        manager = ConfigManager()
        config = manager.load_config(config_path)

        assert config.pjm.submit_options == ["--no-check-directory"]
        assert config.pjm.options == [["-L", "node=12"], ["-s"]]

    def test_load_slurm_config_with_submit_options(self, temp_dir):
        config_path = temp_dir / "crewster.toml"
        config_path.write_text("""
[cluster]
host = "myhpc"
workdir = "/scratch/user/proj"

[slurm]
submit_options = ["--export=ALL"]

[slurm.options]
partition = "gpu"
""")
        manager = ConfigManager()
        config = manager.load_config(config_path)

        assert config.slurm.submit_options == ["--export=ALL"]
        assert config.slurm.options["partition"] == "gpu"

    def test_load_config_file_not_found(self):
        manager = ConfigManager()
        with pytest.raises(FileNotFoundError):
            manager.load_config(Path("/nonexistent/crewster.toml"))

    def test_load_config_invalid_toml(self, temp_dir):
        config_path = temp_dir / "crewster.toml"
        config_path.write_text("invalid toml [[[")
        manager = ConfigManager()
        with pytest.raises(Exception):
            manager.load_config(config_path)

    def test_generate_template(self, temp_dir):
        manager = ConfigManager()
        config_path = temp_dir / "crewster.toml"
        manager.generate_template(config_path)

        assert config_path.exists()
        content = config_path.read_text()
        assert "[cluster]" in content
        assert "[env]" in content
        assert "[slurm.options]" in content

    def test_generate_template_pjm(self, temp_dir):
        """``scheduler="pjm"`` emits a PJM-shaped template."""
        manager = ConfigManager()
        config_path = temp_dir / "crewster.toml"
        manager.generate_template(config_path, scheduler="pjm")

        assert config_path.exists()
        content = config_path.read_text()
        assert "[cluster]" in content
        assert 'scheduler = "pjm"' in content
        assert "[pjm]" in content
        assert "[slurm" not in content

        # Round-trip through load_config so the dispatch path is exercised.
        loaded = manager.load_config(config_path)
        assert loaded.cluster.scheduler == "pjm"
        assert loaded.pjm.options
        assert loaded.slurm.options == {}

    def test_generate_template_rejects_unknown_scheduler(self, temp_dir):
        """Library-level guard against bypassing the CLI's enum constraint."""
        manager = ConfigManager()
        config_path = temp_dir / "crewster.toml"
        with pytest.raises(ValueError, match="Unknown scheduler"):
            manager.generate_template(config_path, scheduler="lsf")
        assert not config_path.exists()

    def test_generate_template_slurm_includes_explicit_scheduler_field(self, temp_dir):
        """``scheduler="slurm"`` writes the field explicitly even though it is
        the ``ClusterConfig`` default, so the file is symmetric across
        schedulers."""
        manager = ConfigManager()
        config_path = temp_dir / "crewster.toml"
        manager.generate_template(config_path, scheduler="slurm")

        content = config_path.read_text()
        assert 'scheduler = "slurm"' in content


class TestFindConfig:
    def test_find_config_in_cwd(self, temp_dir, monkeypatch):
        (temp_dir / "crewster.toml").write_text("[cluster]\nhost='x'\nworkdir='/'")
        monkeypatch.chdir(temp_dir)
        result = find_config()
        assert result == ((temp_dir / "crewster.toml").resolve(), "crewster.toml")

    def test_find_config_in_parent(self, temp_dir, monkeypatch):
        (temp_dir / "crewster.toml").write_text("[cluster]\nhost='x'\nworkdir='/'")
        child = temp_dir / "runs" / "bench1"
        child.mkdir(parents=True)
        monkeypatch.chdir(child)
        result = find_config()
        assert result == ((temp_dir / "crewster.toml").resolve(), "crewster.toml")

    def test_find_config_returns_none(self, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)
        result = find_config()
        assert result is None

    def test_find_config_reports_legacy_name(self, temp_dir, monkeypatch):
        # The legacy name is still discovered, and the matched filename is
        # surfaced so the caller can emit a deprecation warning.
        (temp_dir / "hpc.toml").write_text("[cluster]\nhost='x'\nworkdir='/'")
        monkeypatch.chdir(temp_dir)
        result = find_config()
        assert result == ((temp_dir / "hpc.toml").resolve(), "hpc.toml")

    def test_find_config_prefers_crewster_in_same_dir(self, temp_dir, monkeypatch):
        (temp_dir / "crewster.toml").write_text("x")
        (temp_dir / "hpc.toml").write_text("x")
        monkeypatch.chdir(temp_dir)
        assert find_config()[1] == "crewster.toml"

    def test_find_config_nearest_dir_wins(self, temp_dir, monkeypatch):
        # A distant-ancestor crewster.toml must not shadow a nearer hpc.toml:
        # nearest directory wins, preserving the git-like discovery invariant.
        (temp_dir / "crewster.toml").write_text("x")
        child = temp_dir / "sub"
        child.mkdir()
        (child / "hpc.toml").write_text("x")
        monkeypatch.chdir(child)
        result = find_config()
        assert result == ((child / "hpc.toml").resolve(), "hpc.toml")
