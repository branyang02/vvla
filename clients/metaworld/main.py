"""
clients/metaworld/main.py

Writes rollout videos to clients/metaworld/output/.
"""

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Literal

import gymnasium as gym
import metaworld  # noqa: F401
import numpy as np
import imageio.v3 as iio
from loguru import logger
from tqdm import tqdm
import tyro

from vlla.serving.websocket_policy_client import WebsocketPolicyClient

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


# https://metaworld.farama.org/rendering/rendering/#render-from-a-specific-camera
CAMERA_IDS = {
    "topview": 0,
    "corner": 1,
    "corner2": 2,
    "corner3": 3,
    "behindGripper": 4,
    "gripperPOV": 5,
}


@dataclass
class Args:
    host: str = "localhost"
    port: int = 8765
    env_name: str = "reach-v3"
    width: int = 224
    height: int = 224
    # Cameras to use for policy input
    policy_cameras: list[str] = field(
        default_factory=lambda: ["gripperPOV", "corner", "corner2"]
    )
    # The camera used for rendering the video output
    render_camera: Literal["corner", "corner2"] = "corner"

    num_envs: int = 6
    num_episodes: int = 2
    max_steps: int = 200
    seed: int = 42
    fps: int = 24


class MultiCameraWrapper(gym.Wrapper):
    """Wrapper that renders multiple cameras and includes images in info dict."""

    def __init__(self, env: gym.Env, camera_names: list[str]):
        super().__init__(env)
        self.camera_names = camera_names

    def _render_cameras(self) -> dict[str, np.ndarray]:
        renderer = self.unwrapped.mujoco_renderer
        images = {}
        for cam_name in self.camera_names:
            renderer.camera_id = CAMERA_IDS[cam_name]
            img = renderer.render(render_mode="rgb_array")
            images[cam_name] = img[::-1].copy()  # flip vertically
        return images

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info["cameras"] = self._render_cameras()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["cameras"] = self._render_cameras()
        return obs, reward, terminated, truncated, info


def make_env(
    env_name: str, width: int, height: int, seed_offset: int, camera_names: list[str]
) -> Callable[[], gym.Env]:
    def _init():
        env = gym.make(
            "Meta-World/MT1",
            env_name=env_name,
            seed=42 + seed_offset,
            render_mode="rgb_array",
            width=width,
            height=height,
        )
        env = MultiCameraWrapper(env, camera_names)
        return env

    return _init


def tile_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Arrange N frames into a grid image.

    Grid layout: cols = ceil(sqrt(N)), rows = ceil(N / cols).
    Empty slots are filled with black.
    """
    n = len(frames)
    h, w, c = frames[0].shape
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    grid = np.zeros((rows * h, cols * w, c), dtype=frames[0].dtype)
    for idx, frame in enumerate(frames):
        r, col = divmod(idx, cols)
        grid[r * h : (r + 1) * h, col * w : (col + 1) * w] = frame

    return grid


def main(args: Args) -> None:
    policy = WebsocketPolicyClient(host=args.host, port=args.port)
    logger.info(f"Server metadata: {policy.server_metadata}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env_fns = [
        make_env(args.env_name, args.width, args.height, i, args.policy_cameras)
        for i in range(args.num_envs)
    ]
    env = gym.vector.AsyncVectorEnv(env_fns)

    for episode in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed + episode)
        camera_views = info["cameras"]
        total_reward = np.zeros(args.num_envs)
        success = np.zeros(args.num_envs, dtype=bool)

        video_path = os.path.join(OUTPUT_DIR, f"episode_{episode:03d}.mp4")
        with iio.imopen(video_path, "w", plugin="pyav") as video:
            video.init_video_stream("h264", fps=args.fps)

            pbar = tqdm(
                range(args.max_steps), desc=f"Episode {episode + 1}/{args.num_episodes}"
            )
            for step in pbar:
                grid_frame = tile_frames(list(camera_views[args.render_camera]))
                video.write_frame(grid_frame)

                result = policy.infer(
                    {
                        "state": obs.astype(np.float32),
                        **{
                            f"image/{name}": camera_views[name]
                            for name in args.policy_cameras
                        },
                    }
                )
                action = np.clip(result["actions"], -1.0, 1.0).astype(np.float32)

                obs, reward, terminated, truncated, info = env.step(action)
                camera_views = info["cameras"]
                total_reward += reward
                success |= np.asarray(
                    info.get("success", np.zeros(args.num_envs)), dtype=bool
                )
                pbar.set_postfix(
                    reward=f"{total_reward.mean():.1f}", success=f"{success.mean():.0%}"
                )

        logger.info(
            f"Episode {episode + 1}/{args.num_episodes}: "
            f"mean_reward={total_reward.mean():.2f}, success_rate={success.mean():.2f}, "
            f"video={video_path}"
        )

    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
