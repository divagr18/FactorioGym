"""Package the Lua mod into each worker's mod directory.

The user's global Factorio mod collection is never touched: workers get their
own mod directory containing exactly ``factoriorl_<version>.zip`` plus a
mod-list enabling it (PLAN.md 0.2, isolation).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

MOD_NAME = "factoriorl"


def mod_version() -> str:
    info = json.loads(
        (Path(__file__).resolve().parents[2] / "mod" / MOD_NAME / "info.json").read_text(
            encoding="utf-8"
        )
    )
    return info["version"]


def package_mod(target_mod_dir: Path) -> Path:
    """Zip the mod source into ``target_mod_dir`` and enable it. Returns zip path."""
    from factoriorl.paths import mod_source_dir

    source = mod_source_dir() / MOD_NAME
    if not (source / "info.json").is_file():
        raise FileNotFoundError(f"mod source missing: {source}")
    version = mod_version()
    target_mod_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_mod_dir / f"{MOD_NAME}_{version}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(source.rglob("*")):
            if file.is_file():
                archive.write(file, f"{MOD_NAME}_{version}/{file.relative_to(source)}")
    mod_list = target_mod_dir / "mod-list.json"
    # Base game only: DLCs come in via player-data.json feature flags; explicit
    # disable keeps the worker on base regardless of the user's profile.
    mod_list.write_text(
        json.dumps(
            {
                "mods": [
                    {"name": MOD_NAME, "enabled": True},
                    {"name": "space-age", "enabled": False},
                    {"name": "quality", "enabled": False},
                    {"name": "elevated-rails", "enabled": False},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return zip_path
