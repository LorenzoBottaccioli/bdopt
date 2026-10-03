#!/usr/bin/env python3
"""Download and unpack an EnergyPlus release (8.7 ... latest) without admin rights.

    python tools/install_eplus.py 9.6                 # -> ~/.bdopt/eplus/EnergyPlus-9-6-0
    python tools/install_eplus.py 8.7 25.2 --dir /opt/eplus
    python tools/install_eplus.py --list

bdopt finds installations under ``~/.bdopt/eplus`` automatically (or set ``BDOPT_EPLUS_PATH``).
Standard library only. The release assets are the official ones from github.com/NREL/EnergyPlus.
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

#: version -> (git tag, short commit hash used in the asset names)
RELEASES: dict[str, tuple[str, str]] = {
    "8.7.0": ("v8.7.0", "78a111df4a"),
    "8.8.0": ("v8.8.0", "7c3bbe4830"),
    "8.9.0": ("v8.9.0", "40101eaafd"),
    "9.0.1": ("v9.0.1", "bb7ca4f0da"),
    "9.1.0": ("v9.1.0", "08d2e308bb"),
    "9.2.0": ("v9.2.0", "921312fa1d"),
    "9.3.0": ("v9.3.0", "baff08990c"),
    "9.4.0": ("v9.4.0", "998c4b761e"),
    "9.5.0": ("v9.5.0", "de239b2e5f"),
    "9.6.0": ("v9.6.0", "f420c06a69"),
    "22.1.0": ("v22.1.0", "ed759b17ee"),
    "22.2.0": ("v22.2.0", "c249759bad"),
    "23.1.0": ("v23.1.0", "87ed9199d4"),
    "23.2.0": ("v23.2.0", "7636e6b3e9"),
    "24.1.0": ("v24.1.0", "9d7789a3ac"),
    "24.2.0": ("v24.2.0a", "94a887817b"),
    "25.1.0": ("v25.1.0", "68a4a7c774"),
    "25.2.0": ("v25.2.0", "cf7368216c"),
    "26.1.0": ("v26.1.0", "6f2e40d102"),
    "26.2.0": ("v26.2.0", "4bd7a1f26f"),
}
BASE = "https://github.com/NREL/EnergyPlus/releases/download"


def resolve(spec: str) -> str:
    """``'9.6'`` -> ``'9.6.0'`` (newest patch level known)."""
    spec = spec.strip().lstrip("vV")
    if spec in RELEASES:
        return spec
    cands = [v for v in RELEASES if v == spec or v.startswith(spec + ".")]
    if not cands:
        raise SystemExit(f"unknown EnergyPlus version {spec!r}; known: {', '.join(RELEASES)}")
    return sorted(cands, key=lambda v: tuple(map(int, v.split("."))))[-1]


def _ubuntu_version() -> float | None:
    try:
        text = Path("/etc/os-release").read_text()
    except OSError:
        return None
    if "ubuntu" not in text.lower():
        return None
    m = re.search(r'VERSION_ID="?(\d+\.\d+)', text)
    return float(m.group(1)) if m else None


def candidates(version: str) -> list[str]:
    """Asset URLs to try, best first, for this operating system."""
    tag, sha = RELEASES[version]
    stem = f"{BASE}/{tag}/EnergyPlus-{version}-{sha}"
    sysname, machine = platform.system(), platform.machine().lower()
    if sysname == "Linux":
        host = _ubuntu_version() or 22.04
        ubuntu = sorted([18.04, 20.04, 22.04, 24.04], key=lambda u: (abs(u - host), -u))
        sufs = ["Linux-x86_64"] + [f"Linux-Ubuntu{u:.2f}-x86_64" for u in ubuntu]
        return [f"{stem}-{s}.{e}" for s in sufs for e in ("tar.gz", "sh")]
    if sysname == "Darwin":
        arm = machine in ("arm64", "aarch64")
        sufs = (["Darwin-macOS13-arm64"] if arm else []) + [
            "Darwin-macOS12.1-x86_64", "Darwin-macOS11.6-x86_64", "Darwin-macOS10.15-x86_64", "Darwin-x86_64"]
        return [f"{stem}-{s}.tar.gz" for s in sufs]
    if sysname == "Windows":
        return [f"{stem}-Windows-x86_64.zip", f"{stem}-Windows-x86_64.exe"]
    raise SystemExit(f"unsupported platform {sysname}")


def _exists(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30):
            return True
    except (urllib.error.URLError, OSError):
        return False


def _download(url: str, dest: Path) -> None:
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total and sys.stdout.isatty():
                print(f"\r  {done / 1e6:6.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    if sys.stdout.isatty():
        print()


def _extract_sh(path: Path, out: Path) -> None:
    """The CPack ``.sh`` installers are a shell header followed by a gzipped tar: unpack it directly."""
    with open(path, "rb") as fh:
        head = fh.read(200_000)
    m = re.search(rb'tail \$use_new_tail_syntax \+(\d+) "\$0" \| gunzip', head)
    if not m:
        raise SystemExit(f"cannot find the archive offset in {path.name}")
    skip = int(m.group(1)) - 1  # lines of header
    with open(path, "rb") as fh:
        for _ in range(skip):
            fh.readline()
        data = fh.read()
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(data)), mode="r:") as tf:
        tf.extractall(out)


def _find_root(tmp: Path) -> Path:
    for idd in sorted(tmp.rglob("Energy+.idd")):
        if (idd.parent / "ExpandObjects").exists() or any(idd.parent.glob("energyplus*")):
            return idd.parent
    raise SystemExit("downloaded archive does not contain an EnergyPlus installation")


def install(version: str, dest_dir: Path, force: bool = False) -> Path:
    target = dest_dir / f"EnergyPlus-{version.replace('.', '-')}"
    if target.exists() and not force:
        print(f"{target} already installed")
        return target
    urls = [u for u in candidates(version) if _exists(u)]
    if not urls:
        raise SystemExit(
            f"no downloadable asset found for EnergyPlus {version} on {platform.system()}; "
            "install it manually from https://energyplus.net/downloads"
        )
    url = urls[0]
    with tempfile.TemporaryDirectory(prefix="bdopt_eplus_") as t:
        tmp = Path(t)
        archive = tmp / url.rsplit("/", 1)[1]
        _download(url, archive)
        out = tmp / "out"
        out.mkdir()
        if archive.name.endswith(".tar.gz"):
            with tarfile.open(archive) as tf:
                tf.extractall(out)
        elif archive.suffix == ".sh":
            _extract_sh(archive, out)
        elif archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(out)
        elif archive.suffix == ".exe":  # NSIS installer: silent install into the target directory
            target.mkdir(parents=True, exist_ok=True)
            subprocess.run([str(archive), "/S", f"/D={target}"], check=True)
            print(f"installed {target}")
            return target
        root = _find_root(out)
        dest_dir.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(root), target)
    for f in target.glob("energyplus*"):
        if f.is_file() and not f.suffix:
            f.chmod(f.stat().st_mode | 0o111)
    print(f"installed {target}")
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("versions", nargs="*", help="e.g. 8.7 9.6 25.2")
    ap.add_argument("--dir", default=os.environ.get("BDOPT_EPLUS_DIR", str(Path.home() / ".bdopt" / "eplus")))
    ap.add_argument("--list", action="store_true", help="list the known versions")
    ap.add_argument("--force", action="store_true", help="reinstall")
    args = ap.parse_args()
    if args.list:
        print("\n".join(RELEASES))
        return 0
    if not args.versions:
        ap.error("give at least one version (or --list)")
    for v in args.versions:
        install(resolve(v), Path(args.dir), args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
