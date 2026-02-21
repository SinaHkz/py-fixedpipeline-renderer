from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    width: int = 1280
    height: int = 720
    fps_cap: int = 60
    title: str = "Renderer"

    # Project root = folder that contains /src
    project_root: Path = Path(__file__).resolve().parents[1]
    assets_dir: Path = project_root / "assets"
    models_dir: Path = assets_dir / "Models"

    mouse_sensitivity: float = 0.18
    move_speed: float = 4.5
    jump_speed: float = 6.0
    gravity: float = 16.0
