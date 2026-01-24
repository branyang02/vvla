# Metaworld Client

## Requirements
```bash
sudo apt-get install libegl1-mesa
```

## Usage

Terminal window 1:
```bash
uv run python scripts/eval.py policy:meta-world
```

Terminal window 2:
```bash
MUJOCO_GL=egl uv run --group metaworld python clients/metaworld/main.py
```

## Environments

### `reach-v3`

- Observation space (39-dimensional):
    - Elements 0-2: Current hand position (x, y, z)
    - Element 3: Gripper distance (normalized between 0 and 1)
    - Elements 4-17: Object position and quaternion (padded to 14 elements)
    - Elements 18-35: Previous observation frame (frame stacking for temporal information)
    - Elements 36-38: Goal position (x, y, z)