"""Smoke tests for tools/composer/regen_fixtures.py CLI.

Per PR 4.5 plan Task 4: the CLI is a thin wrapper around
``COMPOSITIONS`` × ``snapshot`` + ``canvas.savefig``. It must exit 0
on a clean run, support ``--only NAME``, and exit 1 on ``--check``
when committed fixtures differ from live geometry.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CLI = REPO / "tools" / "composer" / "regen_fixtures.py"


def test_regen_fixtures_cli_exists():
    assert CLI.exists(), f"missing CLI at {CLI}"


def test_regen_fixtures_cli_runs_clean():
    """Running the CLI on an up-to-date checkout exits 0 with no diffs."""
    result = subprocess.run(
        [sys.executable, str(CLI)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"CLI exited {result.returncode}\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_regen_fixtures_cli_only_filters():
    """--only NAME limits the run to a single composition."""
    result = subprocess.run(
        [sys.executable, str(CLI), "--only", "cell-2col-simple"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    # Only cell-2col-simple should appear in stdout (no other names).
    assert "cell-2col-simple" in result.stdout
    assert "nature-2col-abc" not in result.stdout


def test_regen_fixtures_cli_only_unknown_name_exits_nonzero():
    """--only with a name not in COMPOSITIONS exits non-zero."""
    result = subprocess.run(
        [sys.executable, str(CLI), "--only", "does-not-exist"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "does-not-exist" in result.stderr or "does-not-exist" in result.stdout


def test_regen_fixtures_cli_check_mode_clean():
    """--check on an up-to-date checkout exits 0."""
    result = subprocess.run(
        [sys.executable, str(CLI), "--check"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"--check exited {result.returncode} on a clean checkout\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
