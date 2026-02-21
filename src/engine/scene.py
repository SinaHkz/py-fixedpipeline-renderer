from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from ..math_utils.linalg import (
    translation,
    rotation_euler_xyz,
    scale,
    look_at,
    orthographic,
    perspective,
)
from ..assets_io.obj_loader import Model


@dataclass
class Transform:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    rotation_euler: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))  # radians
    scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))

    def matrix(self) -> np.ndarray:
        t = translation(self.position)
        r = rotation_euler_xyz(
            float(self.rotation_euler[0]),
            float(self.rotation_euler[1]),
            float(self.rotation_euler[2]),
        )
        s = scale(self.scale)
        return t @ r @ s


@dataclass
class Actor:
    name: str
    model: Model
    transform: Transform = field(default_factory=Transform)
    is_static: bool = True


@dataclass
class Camera:
    position: np.ndarray = field(default_factory=lambda: np.array([0, 1.5, 5], dtype=np.float32))
    target: np.ndarray = field(default_factory=lambda: np.array([0, 1.0, 0], dtype=np.float32))
    up: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0], dtype=np.float32))
    fov_y_rad: float = np.deg2rad(60.0)
    aspect: float = 16 / 9
    near: float = 0.1
    far: float = 100.0
    ortho_size: float = 6.0
    use_ortho: bool = False

    def view_matrix(self) -> np.ndarray:
        return look_at(self.position, self.target, self.up)

    def projection_matrix(self) -> np.ndarray:
        if self.use_ortho:
            h = self.ortho_size
            w = h * self.aspect
            return orthographic(-w, w, -h, h, self.near, self.far)
        return perspective(self.fov_y_rad, self.aspect, self.near, self.far)


@dataclass
class Scene:
    actors: list[Actor] = field(default_factory=list)
    camera: Camera = field(default_factory=Camera)

    def add(self, actor: Actor) -> None:
        self.actors.append(actor)
