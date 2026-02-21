from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class Material:
    name: str
    kd: np.ndarray
    map_kd: Optional[str] = None


@dataclass
class Mesh:
    positions: np.ndarray          # (N,3)
    normals: np.ndarray            # (N,3)
    texcoords: np.ndarray          # (N,2)
    material_for_face: list[str]   # len = num_tris
    aabb_min: np.ndarray           # (3,)
    aabb_max: np.ndarray           # (3,)


@dataclass
class Model:
    meshes: list[Mesh]
    materials: dict[str, Material]


# ----------------------------
# MTL parsing
# ----------------------------

def _parse_mtl(mtl_path: Path) -> dict[str, Material]:
    mats: dict[str, Material] = {}
    current: Optional[Material] = None

    if not mtl_path.exists():
        return mats

    for raw in mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0]

        if key == "newmtl":
            name = " ".join(parts[1:]).strip()
            current = Material(name=name, kd=np.array([1.0, 1.0, 1.0], dtype=np.float32))
            mats[name] = current

        elif current and key == "Kd" and len(parts) >= 4:
            current.kd = np.array(list(map(float, parts[1:4])), dtype=np.float32)

        elif current and key == "map_Kd" and len(parts) >= 2:
            current.map_kd = " ".join(parts[1:]).strip()

    return mats


# ----------------------------
# OBJ helpers
# ----------------------------

def _triangulate(face: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    """Fan triangulation."""
    if len(face) < 3:
        return []
    if len(face) == 3:
        return [face]
    tris: list[list[tuple[int, int, int]]] = []
    for i in range(1, len(face) - 1):
        tris.append([face[0], face[i], face[i + 1]])
    return tris


def _resolve_index(i: int, n: int) -> int:
    """
    OBJ indices are 1-based. Negative indices are relative to the end.
    """
    if i > 0:
        return i - 1
    # negative
    return n + i


def _parse_face_vertex(tok: str, num_v: int, num_vt: int, num_vn: int) -> tuple[int, int, int]:
    """
    Token forms:
      v
      v/vt
      v//vn
      v/vt/vn
    Returns (vi, vti, vni) with -1 meaning missing.
    """
    comps = tok.split("/")
    vi_raw = int(comps[0])
    vi = _resolve_index(vi_raw, num_v)

    vti = -1
    vni = -1

    if len(comps) >= 2 and comps[1]:
        vti_raw = int(comps[1])
        vti = _resolve_index(vti_raw, num_vt)

    if len(comps) >= 3 and comps[2]:
        vni_raw = int(comps[2])
        vni = _resolve_index(vni_raw, num_vn)

    return (vi, vti, vni)


# ----------------------------
# Main loader
# ----------------------------

def load_obj(obj_path: str | Path) -> Model:
    """
    Loads a single OBJ into a single Mesh (triangles).
    Raises exceptions if parsing fails; never returns None.
    """
    obj_path = Path(obj_path)
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    base_dir = obj_path.parent
    lines = obj_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    positions: list[np.ndarray] = []
    texcoords: list[np.ndarray] = []
    normals: list[np.ndarray] = []

    out_pos: list[np.ndarray] = []
    out_uv: list[np.ndarray] = []
    out_n: list[np.ndarray] = []
    tri_mats: list[str] = []

    mtllib: Optional[Path] = None
    active_mtl: str = "default"

    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        tag = parts[0]

        try:
            if tag == "mtllib":
                mtllib = base_dir / (" ".join(parts[1:]).strip())

            elif tag == "usemtl":
                active_mtl = " ".join(parts[1:]).strip() or "default"

            elif tag == "v" and len(parts) >= 4:
                positions.append(np.array(list(map(float, parts[1:4])), dtype=np.float32))

            elif tag == "vt" and len(parts) >= 3:
                texcoords.append(np.array(list(map(float, parts[1:3])), dtype=np.float32))

            elif tag == "vn" and len(parts) >= 4:
                normals.append(np.array(list(map(float, parts[1:4])), dtype=np.float32))

            elif tag == "f":
                if len(parts) < 4:
                    continue

                face: list[tuple[int, int, int]] = []
                for tok in parts[1:]:
                    face.append(_parse_face_vertex(tok, len(positions), len(texcoords), len(normals)))

                for tri in _triangulate(face):
                    tri_pos = [positions[a] for (a, _, _) in tri]

                    tri_uv = [
                        texcoords[b] if 0 <= b < len(texcoords) else np.array([0.0, 0.0], dtype=np.float32)
                        for (_, b, _) in tri
                    ]

                    tri_n = [
                        normals[c] if 0 <= c < len(normals) else None
                        for (*_, c) in tri
                    ]

                    # If any normal missing, compute a face normal
                    if any(n is None for n in tri_n):
                        p0, p1, p2 = tri_pos
                        n = np.cross(p1 - p0, p2 - p0)
                        n_norm = float(np.linalg.norm(n))
                        if n_norm < 1e-10:
                            n = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                        else:
                            n = (n / n_norm).astype(np.float32)
                        tri_n = [n, n, n]

                    for p, uv, nrm in zip(tri_pos, tri_uv, tri_n):
                        out_pos.append(p)
                        out_uv.append(uv)
                        out_n.append(nrm.astype(np.float32))
                    tri_mats.append(active_mtl)

        except Exception as e:
            # Make failures obvious with location
            raise RuntimeError(
                f"OBJ parse error in {obj_path} at line {line_no}:\n"
                f"  {raw}\n"
                f"  error: {e}"
            ) from e

    if not out_pos:
        raise ValueError(f"OBJ has no faces / produced no triangles: {obj_path}")

    pos_arr = np.stack(out_pos, axis=0).astype(np.float32)
    uv_arr = np.stack(out_uv, axis=0).astype(np.float32)
    n_arr = np.stack(out_n, axis=0).astype(np.float32)

    aabb_min = pos_arr.min(axis=0)
    aabb_max = pos_arr.max(axis=0)

    materials = _parse_mtl(mtllib) if mtllib else {}
    if "default" not in materials:
        materials["default"] = Material(
            name="default",
            kd=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            map_kd=None,
        )

    mesh = Mesh(
        positions=pos_arr,
        normals=n_arr,
        texcoords=uv_arr,
        material_for_face=tri_mats,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
    )
    return Model(meshes=[mesh], materials=materials)
