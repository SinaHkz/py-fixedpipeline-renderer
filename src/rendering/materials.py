from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple


Color = Tuple[float, float, float]


@dataclass
class Material:
    name: str
    kd: Color = (1.0, 1.0, 1.0)  # diffuse
    ka: Color = (0.1, 0.1, 0.1)  # ambient
    ks: Color = (0.0, 0.0, 0.0)  # specular
    ns: float = 1.0             # shininess
    map_kd: Optional[str] = None
