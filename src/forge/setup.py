# Copyright 2026. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forge environment setup — install required global tools."""

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SETUP_PATH = Path.home() / ".forge" / "setup.json"

REQUIRED_TOOLS = [
    {"name": "eslint", "install_cmd": "npm install -g eslint"},
    {"name": "prettier", "install_cmd": "npm install -g prettier"},
]


def check_setup() -> bool:
    """Return True if setup.json exists and all required tools are on PATH."""
    if not SETUP_PATH.exists():
        return False
    return all(shutil.which(t["name"]) for t in REQUIRED_TOOLS)


def run_setup(force: bool = False) -> None:
    """Install required global tools if missing. Write setup.json on success.

    Raises RuntimeError if tools cannot be made available.
    """
    missing = [t["name"] for t in REQUIRED_TOOLS if not shutil.which(t["name"])]

    if not missing and not force:
        _write_state()
        return

    if force:
        missing = [t["name"] for t in REQUIRED_TOOLS]

    if not shutil.which("npm"):
        raise RuntimeError(
            f"Required tools not found: {', '.join(missing)}. "
            "Cannot auto-install because npm is not available. "
            f"Install Node.js/npm first, then run: npm install -g {' '.join(missing)}"
        )

    try:
        subprocess.run(
            ["npm", "install", "-g", *missing],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to install {', '.join(missing)} via npm: {e.stderr.strip()}"
        ) from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Timed out installing {', '.join(missing)} via npm"
        ) from None

    still_missing = [name for name in missing if not shutil.which(name)]
    if still_missing:
        raise RuntimeError(
            f"Installation completed but tools still not found on PATH: "
            f"{', '.join(still_missing)}. Check your npm global prefix and PATH."
        )

    _write_state()


def run_preset_setup(preset_path: str, force: bool = False) -> None:
    """Run preset-specific setup commands declared in manifest.json.

    Each setup entry has: name, command, cwd, check (file to verify), description.
    Skips entries whose check file exists unless force=True.
    Raises RuntimeError if a command fails.
    """
    manifest = Path(preset_path) / "manifest.json"
    if not manifest.is_file():
        return
    try:
        data = json.loads(manifest.read_text())
    except (json.JSONDecodeError, OSError):
        return

    setup_entries = data.get("setup", [])
    if not setup_entries:
        return

    preset_dir = str(Path(preset_path).resolve())

    for entry in setup_entries:
        name = entry.get("name", "unknown")
        check_path = entry.get("check", "")
        if check_path:
            check_path = check_path.replace("${PRESET_DIR}", preset_dir)
        cwd = entry.get("cwd", preset_dir).replace("${PRESET_DIR}", preset_dir)
        command = entry.get("command", "")

        if not command:
            continue

        if not force and check_path and Path(check_path).exists():
            continue

        if not Path(cwd).is_dir():
            raise RuntimeError(
                f"Setup '{name}': working directory does not exist: {cwd}"
            )

        try:
            subprocess.run(
                command, shell=True, check=True,
                capture_output=True, text=True,
                cwd=cwd, timeout=300,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Setup '{name}' failed: {e.stderr.strip()}"
            ) from None
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Setup '{name}' timed out after 300s"
            ) from None


def _write_state() -> None:
    SETUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "tools": {t["name"]: shutil.which(t["name"]) for t in REQUIRED_TOOLS},
    }
    SETUP_PATH.write_text(json.dumps(state, indent=2) + "\n")
