from __future__ import annotations

from pathlib import Path
import traceback

import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL

from .assets_io.obj_loader import load_obj
from .engine.scene import Scene, Actor
from .rendering.gl_renderer import RendererGL
from .engine.physics import PlayerController
from .engine.framebuffer import save_framebuffer
from .engine.camera import CameraController


def _project_root() -> Path:
    # project_root/
    #   src/app.py   -> parents[1] is project_root
    return Path(__file__).resolve().parents[1]


ASSET_ROOT = _project_root() / "assets"


def must_load_obj(path: Path):
    """Load an OBJ and throw an explicit error if it fails."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model file:\n  {path}\n\n"
            f"Checked ASSET_ROOT={ASSET_ROOT}\n"
            f"Expected models in: {ASSET_ROOT / 'Models'}"
        )
    model = load_obj(path)
    if model is None:
        raise RuntimeError(f"OBJ loader returned None for:\n  {path}")
    # sanity check
    if not hasattr(model, "meshes") or model.meshes is None:
        raise RuntimeError(f"Loaded model is invalid (no meshes) for:\n  {path}")
    return model


def main() -> int:
    try:
        pygame.init()
        pygame.display.set_caption("URP - Universal Render Pipeline (Python3)")
        screen = pygame.display.set_mode((1280, 720), DOUBLEBUF | OPENGL)

        # Helpful diagnostics (prints once)
        print("[INFO] ASSET_ROOT =", ASSET_ROOT)
        print("[INFO] Models dir exists? ", (ASSET_ROOT / "Models").exists())

        renderer = RendererGL(ASSET_ROOT)
        renderer.init_gl()

        scene = Scene()

        # Load models using the exact names you had originally
        room_path = ASSET_ROOT / "Models" / "room.obj"
        egg_path = ASSET_ROOT / "Models" / "Egg.obj"

        room = must_load_obj(room_path)
        egg = must_load_obj(egg_path)

        room_actor = Actor("room", room, is_static=True)
        room_actor.transform.position[1] = -0.25
        scene.add(room_actor)

        player = Actor("player", egg, is_static=False)
        player.transform.scale *= 10
        player.transform.position = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        scene.add(player)

        controller = PlayerController()
        camera_controller = CameraController()

        mouse_look = False
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)
        screenshot_requested = False

        clock = pygame.time.Clock()
        running = True

        while running:
            dt = clock.tick(60) / 1000.0
            move = np.zeros(3, dtype=np.float32)
            jump = False

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False

                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_SPACE:
                        jump = True
                    elif ev.key == pygame.K_o:
                        scene.camera.use_ortho = True
                    elif ev.key == pygame.K_p:
                        scene.camera.use_ortho = False
                    elif ev.key == pygame.K_r:
                        camera_controller.reset()
                    elif ev.key == pygame.K_t:
                        if camera_controller.smooth_pos > 0:
                            camera_controller.smooth_pos = 0.0
                            camera_controller.smooth_target = 0.0
                            print("[INFO] Camera smoothing: OFF")
                        else:
                            camera_controller.smooth_pos = 14.0
                            camera_controller.smooth_target = 18.0
                            print("[INFO] Camera smoothing: ON")
                    elif ev.key == pygame.K_F12:
                        screenshot_requested = True

                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
                    mouse_look = True
                    pygame.event.set_grab(True)
                    pygame.mouse.set_visible(False)

                elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 3:
                    mouse_look = False
                    pygame.event.set_grab(False)
                    pygame.mouse.set_visible(True)

                elif ev.type == pygame.MOUSEMOTION and mouse_look:
                    camera_controller.handle_mouse(*ev.rel)
                elif ev.type == pygame.MOUSEWHEEL:
                    # wheel up/down zoom
                    camera_controller.zoom(ev.y)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                move[2] += 1
            if keys[pygame.K_s]:
                move[2] -= 1
            if keys[pygame.K_d]:
                move[0] += 1
            if keys[pygame.K_a]:
                move[0] -= 1

            # Player movement
            player.transform.position = controller.step(
                player.transform.position,
                move_local=move,
                yaw=camera_controller.yaw,
                jump=jump,
                dt=dt,
            )
            player.transform.rotation_euler[1] = camera_controller.yaw

            # Camera movement
            camera_controller.update(scene.camera, player.transform.position, dt=dt)

            w, h = screen.get_size()
            renderer.render(scene, (w, h))

            if screenshot_requested:
                save_framebuffer(w, h, "output_render.png")
                screenshot_requested = False
                print("[INFO] Saved screenshot to output_render.png")

            pygame.display.flip()

        pygame.quit()
        return 0

    except Exception:
        print("\n[ERROR] The renderer exited due to an exception:")
        traceback.print_exc()
        pygame.quit()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
