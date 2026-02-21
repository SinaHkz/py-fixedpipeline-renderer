from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from OpenGL.GL import *
import pygame

from ..engine.scene import Scene, Actor
from ..math_utils.linalg import to_gl
from ..assets_io.obj_loader import Material


@dataclass
class GLResources:
    textures: dict[str, int]


class RendererGL:
    def __init__(self, asset_root: str | Path):
        self.asset_root = Path(asset_root)
        self.res = GLResources(textures={})

    def init_gl(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glDisable(GL_CULL_FACE)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (0.0, 5.0, 2.0, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        glEnable(GL_TEXTURE_2D)

    def _get_texture(self, rel_path: str) -> int:
        rel_path = rel_path.replace("\\", "/")
        if rel_path in self.res.textures:
            return self.res.textures[rel_path]

        img_path = (self.asset_root / rel_path).resolve()
        if not img_path.exists():
            img_path = (self.asset_root / "Models" / rel_path).resolve()

        if not img_path.exists():
            # Fail loudly: missing texture should be obvious
            raise FileNotFoundError(f"Texture not found: {rel_path}\nLooked at: {img_path}")

        surface = pygame.image.load(str(img_path)).convert_alpha()
        surface = pygame.transform.flip(surface, False, True)
        w, h = surface.get_size()
        data = pygame.image.tostring(surface, "RGBA", True)

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.res.textures[rel_path] = tex
        return tex

    def _apply_material(self, material: Material) -> None:
        glColor3f(float(material.kd[0]), float(material.kd[1]), float(material.kd[2]))
        if material.map_kd:
            tex = self._get_texture(material.map_kd)
            glBindTexture(GL_TEXTURE_2D, tex)
            glEnable(GL_TEXTURE_2D)
        else:
            glBindTexture(GL_TEXTURE_2D, 0)
            glDisable(GL_TEXTURE_2D)

    def render(self, scene: Scene, viewport_size: tuple[int, int]) -> None:
        w, h = viewport_size
        scene.camera.aspect = w / max(h, 1)

        glViewport(0, 0, w, h)
        glClearColor(0.08, 0.08, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(to_gl(scene.camera.projection_matrix()))

        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(to_gl(scene.camera.view_matrix()))

        for actor in scene.actors:
            self._draw_actor(actor)

    def _draw_actor(self, actor: Actor) -> None:
        # ✅ Safety: never crash on a bad actor
        if actor.model is None:
            return
        if not hasattr(actor.model, "meshes") or actor.model.meshes is None:
            return

        model_mat = actor.transform.matrix()

        for mesh in actor.model.meshes:
            verts = mesh.positions
            norms = mesh.normals
            uvs = mesh.texcoords
            mats = mesh.material_for_face

            glPushMatrix()
            glMultMatrixf(to_gl(model_mat))

            current = None
            glBegin(GL_TRIANGLES)
            for tri_idx in range(len(mats)):
                mname = mats[tri_idx]
                if mname != current:
                    glEnd()
                    material = actor.model.materials.get(mname) or actor.model.materials.get("default")
                    self._apply_material(material)
                    glBegin(GL_TRIANGLES)
                    current = mname

                base = tri_idx * 3
                for k in range(3):
                    i = base + k
                    glNormal3f(float(norms[i, 0]), float(norms[i, 1]), float(norms[i, 2]))
                    glTexCoord2f(float(uvs[i, 0]), float(uvs[i, 1]))
                    glVertex3f(float(verts[i, 0]), float(verts[i, 1]), float(verts[i, 2]))
            glEnd()

            glPopMatrix()
