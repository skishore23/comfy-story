"""Build the deterministic, dependency-safe Comfy Story distribution."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import subprocess
import tempfile
import tomllib
import zipfile
from pathlib import Path

_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_INTEGRATION_SUFFIXES = {".py", ".js", ".mjs", ".css", ".html", ".json", ".md"}
_INSTALLER = """#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re, shutil, stat, subprocess, sys
from importlib.metadata import PackageNotFoundError, version
from email.parser import BytesParser
from packaging.requirements import Requirement
import zipfile
from pathlib import Path, PurePosixPath

def verify_bundle(bundle):
    manifest_path = bundle / "manifest.json"
    if (manifest_path.is_symlink() or not manifest_path.is_file()
        or manifest_path.stat().st_size > 2 * 1024 * 1024):
        raise SystemExit("invalid bundle manifest file")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise SystemExit("invalid bundle manifest")
    files = manifest.get("files")
    if (manifest.get("format") != "comfy-story-bundle-v1"
        or not isinstance(files, dict) or not files):
        raise SystemExit("invalid bundle manifest")
    for name, expected in files.items():
        if (not isinstance(name, str) or not name or "\\\\" in name
            or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
            or PurePosixPath(name).as_posix() != name or name == "manifest.json"
            or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)):
            raise SystemExit("invalid bundle manifest path or digest")
    actual = set()
    for path in bundle.rglob("*"):
        if path.is_symlink():
            raise SystemExit("bundle must not contain symbolic links")
        if path.is_dir():
            continue
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise SystemExit("bundle must contain regular, unlinked files")
        name = path.relative_to(bundle).as_posix()
        if name != "manifest.json":
            actual.add(name)
    if actual != set(files) or not {"install.py", "INSTALL.md", "LICENSE"} <= actual:
        raise SystemExit("bundle file inventory does not match its manifest")
    for name, expected in files.items():
        digest = hashlib.sha256()
        with (bundle / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise SystemExit(f"bundle verification failed: {name}")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    payload = hashlib.sha256(encoded).hexdigest()
    if manifest.get("payload_sha256") != payload:
        raise SystemExit("bundle payload digest does not match its manifest")

def main() -> int:
    parser = argparse.ArgumentParser(description="Install Comfy Story into an existing ComfyUI")
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="Verify without installing")
    args = parser.parse_args()
    if sys.platform not in ("linux", "darwin"):
        raise SystemExit("Comfy Story currently requires Linux or macOS")
    if sys.version_info < (3, 11):
        raise SystemExit("Comfy Story requires Python 3.11 or newer")
    root = args.comfy_root.resolve()
    if not (root / "main.py").is_file() or not (root / "custom_nodes").is_dir():
        raise SystemExit("--comfy-root must be an existing ComfyUI root")
    bundle = Path(__file__).resolve().parent
    verify_bundle(bundle)
    wheels = sorted((bundle / "wheels").glob("comfy_story-*.whl"))
    if len(wheels) != 1:
        raise SystemExit("bundle must contain exactly one Comfy Story wheel")
    with zipfile.ZipFile(wheels[0]) as wheel:
        metadata_name = next(
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(wheel.read(metadata_name))
    failures = []
    for value in metadata.get_all("Requires-Dist", []):
        requirement = Requirement(value)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            installed = version(requirement.name)
        except PackageNotFoundError:
            failures.append(f"{requirement.name} is missing")
            continue
        if installed not in requirement.specifier:
            failures.append(
                f"{requirement.name} {installed} does not satisfy {requirement.specifier}"
            )
    if failures:
        raise SystemExit(
            "Comfy Story dependency preflight failed before installation: " + "; ".join(failures)
        )
    source = bundle / "custom_nodes" / "comfy_story"
    target = root / "custom_nodes" / "comfy_story"
    if target.exists() or target.is_symlink():
        raise SystemExit(
            "Comfy Story already exists; stage the new release separately "
            "and switch it after validation"
        )
    if args.check_only:
        print("Bundle, dependency, and destination checks passed; no files installed.")
        return 0
    command = [
        sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheels[0])
    ]
    subprocess.run(command, check=True)
    shutil.copytree(source, target)
    print(
        "Comfy Story installed. Restart ComfyUI, then add Comfy Story from Comfy / Story."
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""
_INSTALL = """# Install Comfy Story

Use an existing authorized MiniMax H3 installation. Read `docs/getting-started.md` for the
supported environment, setup, and first sequence.

1. Extract this bundle on the ComfyUI host.
2. With ComfyUI's Python, run
   `python install.py --comfy-root /path/to/ComfyUI --check-only`.
3. Resolve reported dependencies in that environment, preserving its Torch/CUDA build.
4. Run the same command without `--check-only`, then restart ComfyUI.
5. Add **Comfy Story** from **Comfy / Story**.

The installer verifies the exact file inventory and hashes before installation and uses `--no-deps`.
Models and access credentials are not included. Back up your complete story directory and workflow.
For upgrades, validate a staged installation before switching while the queue is empty.
"""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _tracked_files(root: Path, prefix: str, suffixes: set[str]) -> list[Path]:
    """Package reviewed source only; ignore local experiments and reject linked inputs."""
    names = (
        subprocess.check_output(["git", "ls-files", "-z", "--", prefix], cwd=root)
        .decode()
        .split("\0")
    )
    result = []
    for name in sorted(filter(None, names)):
        path = root / name
        if path.suffix not in suffixes:
            continue
        if any(part.is_symlink() for part in (path, *path.parents) if part != root.parent):
            raise ValueError("bundle source must not contain symbolic links")
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError("bundle source must be a regular file inside the repository")
        result.append(path)
    return result


def _copy_integration(root: Path, stage: Path) -> None:
    source = root / "integrations/comfy_story"
    target = stage / "custom_nodes/comfy_story"
    for path in _tracked_files(root, "integrations/comfy_story", _INTEGRATION_SUFFIXES):
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)


def _manifest(stage: Path, source_commit: str) -> bytes:
    files = {
        path.relative_to(stage).as_posix(): _sha256(path.read_bytes())
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path != stage / "manifest.json"
    }
    return json.dumps(
        {
            "files": files,
            "format": "comfy-story-bundle-v1",
            "source_commit": source_commit,
            "payload_sha256": _sha256(
                json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
            ),
            "runtime_profile": "comfy-minimax-h3",
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _record_digest(value: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=").decode()
    return f"sha256={encoded}"


def _build_wheel(root: Path, wheels: Path) -> Path:
    """Build the repository's pure-Python wheel without network or build isolation."""
    configuration = tomllib.loads((root / "pyproject.toml").read_text())
    files = {
        path.relative_to(root / "src").as_posix(): path.read_bytes()
        for path in _tracked_files(root, "src/comfy_story", {".py"})
    }
    project = configuration["project"]
    version = project["version"]
    dist_info = f"comfy_story-{version}.dist-info"
    runtime_profile = configuration["tool"]["comfy-story"]["story-runtime"]
    dependencies = [
        next(
            (
                override
                for name, override in runtime_profile.items()
                if value.startswith(f"{name}>=")
            ),
            value,
        )
        for value in project["dependencies"]
    ]

    metadata = [
        "Metadata-Version: 2.3",
        "Name: comfy-story",
        f"Version: {version}",
        "Summary: Comfy Story runtime for ComfyUI and MiniMax H3",
        f"Requires-Python: {project['requires-python']}",
        f"License: {project['license']}",
        *(f"Requires-Dist: {value}" for value in dependencies),
    ]
    files[f"{dist_info}/METADATA"] = ("\n".join(metadata) + "\n\n").encode()
    files[f"{dist_info}/licenses/LICENSE"] = (root / "LICENSE").read_bytes()
    files[f"{dist_info}/licenses/NOTICE"] = (root / "NOTICE").read_bytes()
    files[f"{dist_info}/licenses/LICENSING.md"] = (root / "LICENSING.md").read_bytes()
    files[f"{dist_info}/licenses/Apache-2.0.txt"] = (root / "licenses/Apache-2.0.txt").read_bytes()
    files[f"{dist_info}/WHEEL"] = (
        b"Wheel-Version: 1.0\nGenerator: comfy-story\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n"
    )
    # Ship only the customer film commands in the Comfy runtime distribution.
    files[f"{dist_info}/entry_points.txt"] = (
        "[console_scripts]\n"
        + "".join(
            f"{name} = {project['scripts'][name]}\n"
            for name in (
                "comfy-story-film",
                "comfy-story-audit",
            )
        )
    ).encode()
    record_name = f"{dist_info}/RECORD"
    records = [
        f"{name},{_record_digest(value)},{len(value)}" for name, value in sorted(files.items())
    ]
    records.append(f"{record_name},,")
    files[record_name] = ("\n".join(records) + "\n").encode()
    target = wheels / f"comfy_story-{version}-py3-none-any.whl"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in sorted(files.items()):
            info = zipfile.ZipInfo(name, _ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, value, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return target


def _zip_tree(stage: Path, target: Path) -> None:
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(value for value in stage.rglob("*") if value.is_file()):
            name = path.relative_to(stage).as_posix()
            info = zipfile.ZipInfo(name, _ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(
                info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
            )


def build_release_bundle(repository_root: Path, output_root: Path) -> Path:
    """Build and publish one deterministic release ZIP plus digest sidecar."""
    root = repository_root.resolve()
    output = output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    commit = _source_commit(root)
    with tempfile.TemporaryDirectory(prefix="comfy-story-build-") as temporary:
        stage = Path(temporary) / "bundle"
        wheels = stage / "wheels"
        wheels.mkdir(parents=True)
        _build_wheel(root, wheels)
        if len(tuple(wheels.glob("comfy_story-*.whl"))) != 1:
            raise RuntimeError("Comfy Story wheel build did not produce exactly one wheel")
        _copy_integration(root, stage)
        (stage / "install.py").write_text(_INSTALLER)
        (stage / "INSTALL.md").write_text(_INSTALL)
        (stage / "LICENSE").write_bytes((root / "LICENSE").read_bytes())
        (stage / "NOTICE").write_bytes((root / "NOTICE").read_bytes())
        (stage / "LICENSING.md").write_bytes((root / "LICENSING.md").read_bytes())
        (stage / "licenses").mkdir()
        (stage / "licenses/Apache-2.0.txt").write_bytes(
            (root / "licenses/Apache-2.0.txt").read_bytes()
        )
        (stage / "SECURITY.md").write_bytes((root / "SECURITY.md").read_bytes())
        (stage / "docs").mkdir()
        for name in (
            "getting-started.md",
            "prompting.md",
            "films.md",
        ):
            (stage / "docs" / name).write_bytes((root / "docs" / name).read_bytes())
        manifest = _manifest(stage, commit)
        (stage / "manifest.json").write_bytes(manifest)
        target = output / f"comfy-story-{commit[:12]}-{_sha256(manifest)[:12]}.zip"
        _zip_tree(stage, target)
    digest = _sha256(target.read_bytes())
    target.with_suffix(target.suffix + ".sha256").write_text(f"{digest}  {target.name}\n")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).parents[1]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root
    ):
        parser.error(
            "release builds require a clean committed checkout; local files are not published"
        )
    path = build_release_bundle(root, args.output_root)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
