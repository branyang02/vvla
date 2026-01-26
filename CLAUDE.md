# VLLA

ML training and serving framework for visuomotor policies.

## Commands

```bash
uv sync                              # Install dependencies
uv run pytest                        # Run tests
uv run python scripts/train.py       # Train a policy
uv run python scripts/eval.py        # Serve a policy over WebSocket
uv run ruff check --fix .            # Lint
uv run ruff format .                 # Format
```

## Project Structure

```
src/vlla/
    datasets/
        __init__.py            # LeRobotDatasetConfig, make_dataset(), make_dataloader()
        data_loader.py         # Dataset, TransformedDataset

    models/
        __init__.py            # ModelConfig (union), make_model()
        base_model.py          # BaseModel ABC
        pi0.py                 # Pi0ModelConfig, Pi0
        pi05.py                # Pi05ModelConfig, Pi05

    policies/
        __init__.py            # PolicyConfig (union), make_policy()
        base_policy.py         # BasePolicy ABC, DummyPolicy
        libero_policy.py       # LiberoPolicyConfig, LiberoPolicy
        metaworld_policy.py    # MetaworldPolicyConfig, MetaworldPolicy

    transforms/
        __init__.py            # TransformsConfig (union), make_transforms()
        transforms.py          # DataTransformFn protocol, NormStats, Normalize, Unnormalize
        pi0_transforms.py      # Pi0TransformsConfig
        pi05_transforms.py     # Pi05TransformsConfig

    serving/
        websocket_policy_server.py
        websocket_policy_client.py
        msgpack_numpy.py

scripts/
    train.py                   # Training entry point (tyro CLI)
    eval.py                    # WebSocket serving entry point (tyro CLI)
```

## Architecture Patterns

### Module Organization
Each module (models, policies, transforms) follows:
- **`__init__.py`**: Exports union config type + factory function
- **Base file**: ABC or Protocol definition
- **Implementation files**: Config dataclass + implementation class

### Config Pattern
- Configs are dataclasses with tyro-compatible defaults
- Union types use `Annotated[..., tyro.conf.subcommand()]` for CLI subcommands
- Factory functions dispatch on config type with isinstance checks

### Data Flow
- **Training**: `make_dataset()` -> `make_policy(norm_stats)` -> `make_dataloader(transforms)`
- **Inference**: `policy.infer(obs)` handles numpy<->tensor + transforms internally

### Key Interfaces
- **Models**: `dict[str, Tensor] -> dict[str, Tensor]` (pure nn.Module)
- **Transforms**: `dict[str, np.ndarray] -> dict[str, np.ndarray]` (DataTransformFn protocol)
- **Policies**: `infer(obs: dict) -> dict` (BasePolicy ABC)

## Conventions

- Python 3.11+, type hints throughout
- Linting/formatting: ruff with isort
- Config/CLI: tyro with dataclass configs
- Logging: loguru
