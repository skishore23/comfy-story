#!/usr/bin/env python3
"""Install a source checkout with its pinned memory asset into an existing ComfyUI host."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from scripts.build_release import build_release_bundle, obtain_memory_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--memory-checkpoint", type=Path)
    parser.add_argument("--extra-model-paths-config", type=Path, action="append", default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root
    ):
        parser.error("Install from a clean committed checkout; local changes are not packaged")
    checkpoint = obtain_memory_checkpoint(root, args.memory_checkpoint)
    with tempfile.TemporaryDirectory(prefix="comfy-story-install-") as temporary:
        stage = Path(temporary)
        bundle = build_release_bundle(root, stage, memory_checkpoint=checkpoint)
        extracted = stage / "bundle"
        with zipfile.ZipFile(bundle) as archive:
            archive.extractall(extracted)
        command = [
            sys.executable,
            str(extracted / "install.py"),
            "--comfy-root",
            str(args.comfy_root),
        ]
        if args.check_only:
            command.append("--check-only")
        for config in args.extra_model_paths_config:
            command.extend(["--extra-model-paths-config", str(config)])
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
