"""Run manager tests"""

from datetime import datetime
from unittest.mock import patch

import pytest

from crewster.run import RunManager, RunConfig
from crewster.config import HpcConfig, ClusterConfig, EnvConfig, SlurmConfig


@pytest.fixture
def sample_config():
    return HpcConfig(
        cluster=ClusterConfig(host="myhpc", workdir="/scratch/user/proj"),
        env=EnvConfig(modules=["gcc/12.2.0"]),
        slurm=SlurmConfig(partition="gpu", time="02:00:00", mem="32G", gpus=1),
    )


class TestRunConfig:
    def test_run_config_fields(self):
        config = RunConfig(
            run_id="2025-12-26_183012_ab12cd",
            cmd="python train.py",
            status="pending",
        )
        assert config.run_id == "2025-12-26_183012_ab12cd"
        assert config.cmd == "python train.py"
        assert config.status == "pending"

    def test_run_config_optional_fields(self):
        config = RunConfig(
            run_id="2025-12-26_183012_ab12cd",
            cmd="python train.py",
            status="pending",
            job_id="12345678",
            git_commit="ab12cd34",
        )
        assert config.job_id == "12345678"
        assert config.git_commit == "ab12cd34"


class TestRunManagerCreateRun:
    def test_create_run_generates_run_id(self, sample_config, temp_dir):
        manager = RunManager(config=sample_config, runs_dir=temp_dir)
        run = manager.create_run("python train.py")
        assert run.run_id is not None
        assert len(run.run_id) > 0

    def test_create_run_id_format(self, sample_config, temp_dir):
        manager = RunManager(config=sample_config, runs_dir=temp_dir)
        with patch("crewster.run.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2025, 12, 26, 18, 30, 12)
            run = manager.create_run("python train.py")
            assert run.run_id.startswith("2025-12-26_183012_")

    def test_create_run_stores_cmd(self, sample_config, temp_dir):
        manager = RunManager(config=sample_config, runs_dir=temp_dir)
        run = manager.create_run("python train.py --epochs 100")
        assert run.cmd == "python train.py --epochs 100"

    def test_create_run_initial_status(self, sample_config, temp_dir):
        manager = RunManager(config=sample_config, runs_dir=temp_dir)
        run = manager.create_run("python train.py")
        assert run.status == "pending"


class TestRunManagerSaveLoad:
    def test_save_run_meta(self, sample_config, temp_dir):
        manager = RunManager(config=sample_config, runs_dir=temp_dir)
        run = manager.create_run("python train.py")
        manager.save_run_meta(run)

        meta_path = temp_dir / run.run_id / "meta.toml"
        assert meta_path.exists()

    def test_load_run_meta(self, sample_config, temp_dir):
        manager = RunManager(config=sample_config, runs_dir=temp_dir)
        run = manager.create_run("python train.py")
        run.job_id = "12345678"
        manager.save_run_meta(run)

        loaded = manager.load_run_meta(run.run_id)
        assert loaded.run_id == run.run_id
        assert loaded.cmd == run.cmd
        assert loaded.job_id == "12345678"

    def test_update_run_meta(self, sample_config, temp_dir):
        manager = RunManager(config=sample_config, runs_dir=temp_dir)
        run = manager.create_run("python train.py")
        manager.save_run_meta(run)

        run.status = "running"
        run.job_id = "12345678"
        manager.save_run_meta(run)

        loaded = manager.load_run_meta(run.run_id)
        assert loaded.status == "running"
        assert loaded.job_id == "12345678"


class TestRunManagerList:
    def test_list_runs_empty(self, sample_config, temp_dir):
        manager = RunManager(config=sample_config, runs_dir=temp_dir)
        runs = manager.list_runs()
        assert runs == []

    def test_list_runs_returns_all(self, sample_config, temp_dir):
        manager = RunManager(config=sample_config, runs_dir=temp_dir)

        run1 = manager.create_run("python train.py")
        manager.save_run_meta(run1)

        run2 = manager.create_run("python test.py")
        manager.save_run_meta(run2)

        runs = manager.list_runs()
        assert len(runs) == 2
