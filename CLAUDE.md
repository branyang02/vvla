# vVLA

ML training and serving framework for visuomotor policies with modular datasets, models, policies, transforms, and WebSocket serving.

## Ground Truth

This file documents the current repository state. If code and docs disagree, trust code in:

- `scripts/`
- `src/vvla/`
- `tests/`

`clients/libero_client/LIBERO` is a git submodule (vendored upstream project), not first-party `vvla` core code.

## Quick Reference

```bash
git submodule update --init --recursive
uv sync
uv run pre-commit install

uv run pytest
uv run ruff check --fix .
uv run ruff format .
```

## CLI Workflows

## Training

`scripts/train.py` builds dataset, policy, and dataloader, then saves a checkpoint to `output/checkpoint`.

```bash
uv run python scripts/train.py \
  policy:libero \
  --dataset.repo-id lerobot/droid_100 \
  --policy.repo-id my-org/my-libero-policy
```

```bash
uv run python scripts/train.py \
  policy:metaworld \
  --dataset.repo-id lerobot/droid_100 \
  --policy.repo-id my-org/my-metaworld-policy
```

Training rules enforced by `make_policy(..., mode="train")`:

- `--policy.repo-id` is required for non-dummy policies.
- `norm_stats` is required (provided by dataset in `scripts/train.py`).
- `--policy.pretrained_repo_id_or_path` is optional and enables finetuning from checkpoint.

## Serving (Evaluation)

`scripts/eval.py` starts `WebsocketPolicyServer` and requires a policy config.

Dummy policy (no checkpoint required):

```bash
uv run python scripts/eval.py policy:dummy
```

Checkpoint-backed policy:

```bash
uv run python scripts/eval.py \
  policy:libero \
  --policy.pretrained_repo_id_or_path output/checkpoint
```

Evaluation rules enforced by `make_policy(..., mode="eval")`:

- `--policy.pretrained_repo_id_or_path` is required for non-dummy policies.
- `--policy.repo-id` must be unset.
- `norm_stats` must not be passed (loaded from checkpoint transforms).

## Project Structure

```text
src/vvla/
├── datasets/
│   ├── __init__.py
│   └── data_loader.py
├── models/
│   ├── __init__.py
│   ├── base_model.py
│   ├── pi0.py
│   └── pi05.py
├── policies/
│   ├── __init__.py
│   ├── base_policy.py
│   ├── dummy_policy.py
│   ├── libero_policy.py
│   ├── metaworld_policy.py
│   └── hub_utils.py
├── transforms/
│   ├── __init__.py
│   ├── transforms.py
│   ├── pi0_transforms.py
│   └── pi05_transforms.py
└── serving/
    ├── websocket_policy_server.py
    ├── websocket_policy_client.py
    └── msgpack_numpy.py

scripts/
├── train.py
└── eval.py

tests/
├── test_datasets.py
├── test_models.py
├── test_policies.py
├── test_transforms.py
└── test_msgpack.py
```

## Architecture Notes

## Factory + Config Pattern

Models, policies, and transforms use:

- dataclass configs
- union configs with Tyro subcommands
- `make_*` factory dispatch by config type

Key unions:

- `ModelConfig`: `pi0 | pi05`
- `PolicyConfig`: `libero | metaworld | dummy`
- `TransformsConfig`: `pi0 | pi05`

## Data Flow

Training:

1. `make_dataset(LeRobotDatasetConfig)` creates a dataset wrapper and extracts `norm_stats`.
2. `make_policy(config, mode="train", norm_stats=...)` builds or loads policy.
3. `make_dataloader(..., transforms=policy.input_transforms)` applies input transforms.

Serving:

1. `make_policy(config, mode="eval")` loads policy from checkpoint for non-dummy configs.
2. `WebsocketPolicyServer` serves inference over msgpack WebSocket.

## Current Policy Behavior

Current `infer()` behavior is placeholder/random in all policy implementations:

- `DummyPolicy`: random 1D action vector.
- `LiberoPolicy`: random action chunk shaped `(chunk_size, action_dim)`.
- `MetaworldPolicy`: random batched actions shaped `(batch_size, action_dim)`.

`train_forward()` in `LiberoPolicy` and `MetaworldPolicy` delegates to the model.

## Checkpoint and Hub Serialization

`BasePolicy.save_pretrained(save_dir)` currently writes:

- `model.safetensors`
- `transforms.json`

Transforms are serialized via `vvla.policies.hub_utils`.

Loading is done with `ClassName.from_config(config)` where `config.pretrained_repo_id_or_path` points to:

- local checkpoint directory, or
- Hugging Face repo id

`push_to_hub()` uploads a temporary saved folder and uses `config.repo_id` as target repo.

## Testing

```bash
uv run pytest
uv run pytest tests/test_policies.py
uv run pytest -k "make_policy"
uv run pytest --cov=vvla --cov-report=html
```

Coverage report viewer:

```bash
cd htmlcov
uv run python -m http.server 8000
```

## Conventions

- Python `>=3.11`
- Type hints throughout
- Tyro for CLI
- Ruff with import sorting (`[tool.ruff.lint] extend-select = ["I"]`)
- `msgpack` + explicit NumPy packing (`src/vvla/serving/msgpack_numpy.py`)
- Avoid pickle-based wire serialization
