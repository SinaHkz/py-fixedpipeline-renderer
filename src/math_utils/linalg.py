from __future__ import annotations
import math
import numpy as np


def identity() -> np.ndarray:
    return np.identity(4, dtype=np.float32)


def translation(t: np.ndarray) -> np.ndarray:
    m = identity()
    m[0:3, 3] = t[0:3]
    return m


def scale(s: np.ndarray) -> np.ndarray:
    m = identity()
    m[0, 0] = float(s[0])
    m[1, 1] = float(s[1])
    m[2, 2] = float(s[2])
    return m


def rotation_euler_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = np.array(
        [[1, 0, 0, 0],
         [0, cx, -sx, 0],
         [0, sx, cx, 0],
         [0, 0, 0, 1]],
        dtype=np.float32,
    )
    Ry = np.array(
        [[cy, 0, sy, 0],
         [0, 1, 0, 0],
         [-sy, 0, cy, 0],
         [0, 0, 0, 1]],
        dtype=np.float32,
    )
    Rz = np.array(
        [[cz, -sz, 0, 0],
         [sz, cz, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]],
        dtype=np.float32,
    )

    return (Rz @ Ry @ Rx).astype(np.float32)


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = normalize(target - eye)
    r = normalize(np.cross(f, up))
    u = np.cross(r, f)

    m = identity()
    m[0, 0:3] = r
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[0:3, 3] = -m[0:3, 0:3] @ eye
    return m


def perspective(fov_y_rad: float, aspect: float, z_near: float, z_far: float) -> np.ndarray:
    f = 1.0 / math.tan(fov_y_rad / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (z_far + z_near) / (z_near - z_far)
    m[2, 3] = (2.0 * z_far * z_near) / (z_near - z_far)
    m[3, 2] = -1.0
    return m


def orthographic(left: float, right: float, bottom: float, top: float, z_near: float, z_far: float) -> np.ndarray:
    m = identity()
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (z_far - z_near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(z_far + z_near) / (z_far - z_near)
    return m


def to_gl(mat4: np.ndarray) -> np.ndarray:
    """
    Convert a 4x4 numpy matrix (row-major) to a flat array (column-major)
    for OpenGL.
    """
    return mat4.T.astype(np.float32).ravel()
