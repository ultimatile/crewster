"""Config manager tests"""

import pytest
from pathlib import Path

from hpc.config import (
    ClusterConfig,
    EnvConfig,
    SlurmConfig,
    PjmConfig,
    HpcConfig,
    ConfigManager,
    find_config,
)


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

    def test_env_config_with_string_command(self):
        config = EnvConfig(setup=["my_setup"])
        assert config.setup == ["my_setup"]

    def test_env_config_defaults(self):
        config = EnvConfig()
        assert config.setup == []

    def test_env_config_rejects_shell_special(self):
        config = EnvConfig(setup=[{"module": "gcc; rm -rf ~"}])
        with pytest.raises(Exception):
            config.get_setup_commands()

    def test_exports_generates_export_commands(self):
        config = EnvConfig(
            exports={"OMP_NUM_THREADS": "${SLURM_CPUS_PER_TASK}", "FOO": "bar"}
        )
        cmds = config.get_setup_commands()
        assert 'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"' in cmds
        assert 'export FOO="bar"' in cmds

    def test_exports_rejects_command_substitution_dollar(self):
        config = EnvConfig(exports={"FOO": "$(rm -rf /)"})
        with pytest.raises(ValueError, match="Command substitution"):
            config.get_setup_commands()

    def test_exports_rejects_command_substitution_backtick(self):
        config = EnvConfig(exports={"FOO": "`rm -rf /`"})
        with pytest.raises(ValueError, match="Command substitution"):
            config.get_setup_commands()

    def test_exports_rejects_invalid_key(self):
        config = EnvConfig(exports={"FOO;BAR": "value"})
        with pytest.raises(ValueError):
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
        config_path = temp_dir / "hpc.toml"
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
        config_path = temp_dir / "hpc.toml"
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
        config_path = temp_dir / "hpc.toml"
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
            manager.load_config(Path("/nonexistent/hpc.toml"))

    def test_load_config_invalid_toml(self, temp_dir):
        config_path = temp_dir / "hpc.toml"
        config_path.write_text("invalid toml [[[")
        manager = ConfigManager()
        with pytest.raises(Exception):
            manager.load_config(config_path)

    def test_generate_template(self, temp_dir):
        manager = ConfigManager()
        config_path = temp_dir / "hpc.toml"
        manager.generate_template(config_path)

        assert config_path.exists()
        content = config_path.read_text()
        assert "[cluster]" in content
        assert "[env]" in content
        assert "[slurm.options]" in content

    def test_generate_template_pjm(self, temp_dir):
        """``scheduler="pjm"`` emits a PJM-shaped template."""
        manager = ConfigManager()
        config_path = temp_dir / "hpc.toml"
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
        config_path = temp_dir / "hpc.toml"
        with pytest.raises(ValueError, match="Unknown scheduler"):
            manager.generate_template(config_path, scheduler="lsf")
        assert not config_path.exists()

    def test_generate_template_slurm_includes_explicit_scheduler_field(self, temp_dir):
        """``scheduler="slurm"`` writes the field explicitly even though it is
        the ``ClusterConfig`` default, so the file is symmetric across
        schedulers."""
        manager = ConfigManager()
        config_path = temp_dir / "hpc.toml"
        manager.generate_template(config_path, scheduler="slurm")

        content = config_path.read_text()
        assert 'scheduler = "slurm"' in content


class TestFindConfig:
    def test_find_config_in_cwd(self, temp_dir, monkeypatch):
        (temp_dir / "hpc.toml").write_text("[cluster]\nhost='x'\nworkdir='/'")
        monkeypatch.chdir(temp_dir)
        result = find_config("hpc.toml")
        assert result == (temp_dir / "hpc.toml").resolve()

    def test_find_config_in_parent(self, temp_dir, monkeypatch):
        (temp_dir / "hpc.toml").write_text("[cluster]\nhost='x'\nworkdir='/'")
        child = temp_dir / "runs" / "bench1"
        child.mkdir(parents=True)
        monkeypatch.chdir(child)
        result = find_config("hpc.toml")
        assert result == (temp_dir / "hpc.toml").resolve()

    def test_find_config_returns_none(self, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)
        result = find_config("hpc.toml")
        assert result is None
