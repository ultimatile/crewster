"""Pytest configuration and fixtures for hpc tests"""

import tempfile
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner


@pytest.fixture
def cli_runner():
    """Typer CLI runner for testing commands"""
    return CliRunner()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture(autouse=True)
def reset_env_config(monkeypatch, tmp_path):
    """Isolate tests from real environment.

    - ``HPC_CONFIG`` is cleared so tests don't accidentally consume a
      developer-set override.
    - ``XDG_CONFIG_HOME`` is pinned to a fresh empty temp dir so
      ``hpc init`` does not pick up the developer's real user-level
      ``~/.config/hpc/config.toml``. Tests that exercise the
      XDG filter-merge path set their own ``XDG_CONFIG_HOME``.
    """
    monkeypatch.delenv("HPC_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-empty"))
