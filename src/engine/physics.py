from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class PlayerController:
    speed: float = 4.0
    jump_speed: float = 6.0
    gravity: float = 18.0

    room_min_x: float = -3.75
    room_max_x: float = 3.75
    room_min_z: float = -3.75
    room_max_z: float = 3.75

    vel: np.ndarray = None
    on_ground: bool = True

    def __post_init__(self) -> None:
        if self.vel is None:
            self.vel = np.zeros(3, dtype=np.float32)

    def step(self, actor_pos: np.ndarray, move_local: np.ndarray, yaw: float, jump: bool, dt: float) -> np.ndarray:
        siny, cosy = np.sin(yaw), np.cos(yaw)
        forward = np.array([siny, 0.0, -cosy], dtype=np.float32)
        right = np.array([cosy, 0.0, siny], dtype=np.float32)

        wish = right * float(move_local[0]) + forward * float(move_local[2])
        n = float(np.linalg.norm(wish))
        if n > 1e-6:
            wish /= n

        actor_pos = actor_pos + wish * (self.speed * dt)

        if jump and self.on_ground:
            self.vel[1] = self.jump_speed
            self.on_ground = False

        self.vel[1] -= self.gravity * dt
        actor_pos[1] += self.vel[1] * dt

        if actor_pos[1] < 0.0:
            actor_pos[1] = 0.0
            self.vel[1] = 0.0
            self.on_ground = True

        if actor_pos[0] < self.room_min_x:
            actor_pos[0] = self.room_min_x
            self.vel[0] = 0.0
        elif actor_pos[0] > self.room_max_x:
            actor_pos[0] = self.room_max_x
            self.vel[0] = 0.0

        if actor_pos[2] < self.room_min_z:
            actor_pos[2] = self.room_min_z
            self.vel[2] = 0.0
        elif actor_pos[2] > self.room_max_z:
            actor_pos[2] = self.room_max_z
            self.vel[2] = 0.0

        return actor_pos
