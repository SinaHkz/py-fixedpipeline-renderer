import numpy as np
from .scene import Camera


class CameraController:
    """
    3rd-person orbit camera:
    - RMB mouse look (yaw/pitch)
    - Mouse wheel zooms distance
    - Smooth follow (critically damped-ish exponential smoothing)
    """

    def __init__(
        self,
        distance: float = 4.5,
        height: float = 2.2,
        sensitivity: float = 0.003,
        zoom_speed: float = 0.55,
        min_distance: float = 1.5,
        max_distance: float = 14.0,
        smooth_pos: float = 14.0,
        smooth_target: float = 18.0,
        invert_y: bool = False,
    ):
        self.yaw = 0.0
        self.pitch = 0.15

        self.distance = float(distance)
        self.height = float(height)

        self.sensitivity = float(sensitivity)
        self.zoom_speed = float(zoom_speed)
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)

        # bigger = snappier
        self.smooth_pos = float(smooth_pos)
        self.smooth_target = float(smooth_target)

        self.invert_y = bool(invert_y)

        # internal smoothed state
        self._cam_pos = None
        self._cam_target = None

        # pitch clamp
        self._pitch_min = -0.6
        self._pitch_max = 0.6

    # ---------- input handlers ----------

    def handle_mouse(self, dx: float, dy: float) -> None:
        self.yaw += dx * self.sensitivity
        if self.invert_y:
            self.pitch += dy * self.sensitivity
        else:
            self.pitch -= dy * self.sensitivity
        self.pitch = float(np.clip(self.pitch, self._pitch_min, self._pitch_max))

    def zoom(self, wheel_y: float) -> None:
        """
        wheel_y: typically +1 / -1 from pygame MOUSEWHEEL event.
        Positive wheel_y zooms IN (smaller distance) by default.
        """
        self.distance *= (1.0 - float(wheel_y) * self.zoom_speed * 0.1)
        self.distance = float(np.clip(self.distance, self.min_distance, self.max_distance))

    def reset(self) -> None:
        self.yaw = 0.0
        self.pitch = 0.15
        self.distance = float(np.clip(4.5, self.min_distance, self.max_distance))
        self._cam_pos = None
        self._cam_target = None

    # ---------- update ----------

    @staticmethod
    def _exp_smooth(current: np.ndarray, target: np.ndarray, sharpness: float, dt: float) -> np.ndarray:
        """
        Exponential smoothing that is stable across framerates.
        sharpness: higher -> faster convergence.
        """
        if sharpness <= 0.0:
            return target
        # alpha = 1 - exp(-k*dt)
        a = 1.0 - float(np.exp(-sharpness * dt))
        return (current + (target - current) * a).astype(np.float32)

    def update(self, camera: Camera, target_pos: np.ndarray, dt: float = 1.0 / 60.0) -> None:
        """
        dt used for smoothing. Pass real dt each frame.
        """
        target_pos = target_pos.astype(np.float32)

        # Horizontal forward based on yaw
        forward_flat = np.array([np.sin(self.yaw), 0.0, -np.cos(self.yaw)], dtype=np.float32)

        # Desired camera position (orbit behind player)
        desired_pos = (
            target_pos
            - forward_flat * self.distance
            + np.array([0.0, self.height, 0.0], dtype=np.float32)
        )

        # Desired look direction (uses yaw+pitch)
        look_dir = np.array(
            [
                np.sin(self.yaw) * np.cos(self.pitch),
                np.sin(self.pitch),
                -np.cos(self.yaw) * np.cos(self.pitch),
            ],
            dtype=np.float32,
        )

        desired_target = (
            target_pos
            + np.array([0.0, 1.0, 0.0], dtype=np.float32)
            + look_dir * 2.0
        )

        # initialize smoothing buffers
        if self._cam_pos is None:
            self._cam_pos = desired_pos.copy()
        if self._cam_target is None:
            self._cam_target = desired_target.copy()

        # smooth
        self._cam_pos = self._exp_smooth(self._cam_pos, desired_pos, self.smooth_pos, dt)
        self._cam_target = self._exp_smooth(self._cam_target, desired_target, self.smooth_target, dt)

        camera.position = self._cam_pos
        camera.target = self._cam_target
