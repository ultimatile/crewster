"""Integration tests for HPC CLI"""

from unittest.mock import patch, MagicMock


from crewster.main import app
from crewster import cli  # noqa: F401


class TestEndToEndWorkflow:
    """Test init -> sync -> submit -> status workflow"""

    def test_init_creates_valid_config(self, cli_runner, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)

        result = cli_runner.invoke(app, ["init"])
        assert result.exit_code == 0

        config_path = temp_dir / "crewster.toml"
        assert config_path.exists()

        # Verify config is loadable
        from crewster.config import ConfigManager

        manager = ConfigManager()
        config = manager.load_config(config_path)
        assert config.cluster.host == "myhpc"

    def test_sync_after_init(self, cli_runner, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)

        # Init first
        cli_runner.invoke(app, ["init"])

        # Sync with mocked rsync
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = cli_runner.invoke(app, ["sync"])
            assert result.exit_code == 0
            assert "Sync completed" in result.stdout

    def test_submit_after_init(self, cli_runner, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)

        # Init first
        cli_runner.invoke(app, ["init"])

        # Submit with mocked SSH
        with patch("crewster.ssh.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="12345678\n", stderr=""
            )
            result = cli_runner.invoke(app, ["submit", "python train.py"])
            assert result.exit_code == 0
            assert "12345678" in result.stdout

    def test_status_after_submit(self, cli_runner, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)

        # Init first
        cli_runner.invoke(app, ["init"])

        # Status with mocked SSH
        with patch("crewster.ssh.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="RUNNING\n", stderr=""
            )
            result = cli_runner.invoke(app, ["status", "12345678"])
            assert result.exit_code == 0
            assert "RUNNING" in result.stdout


class TestErrorHandling:
    """Test error handling scenarios"""

    def test_sync_without_config(self, cli_runner, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)
        result = cli_runner.invoke(app, ["sync"])
        assert result.exit_code != 0
        assert "Config file not found" in result.stdout

    def test_submit_without_config(self, cli_runner, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)
        result = cli_runner.invoke(app, ["submit", "python train.py"])
        assert result.exit_code != 0
        assert "Config file not found" in result.stdout

    def test_status_without_config(self, cli_runner, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)
        result = cli_runner.invoke(app, ["status", "12345678"])
        assert result.exit_code != 0

    def test_status_without_job_id(self, cli_runner, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)
        cli_runner.invoke(app, ["init"])
        result = cli_runner.invoke(app, ["status"])
        assert result.exit_code != 0

    def test_init_does_not_overwrite(self, cli_runner, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)

        # Create config
        cli_runner.invoke(app, ["init"])

        # Modify it
        config_path = temp_dir / "crewster.toml"
        original_content = config_path.read_text()

        # Try init again
        result = cli_runner.invoke(app, ["init"])
        assert "already exists" in result.stdout

        # Content unchanged
        assert config_path.read_text() == original_content
