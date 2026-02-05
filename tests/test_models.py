"""
uv run pytest tests/test_models.py
"""

from typing import cast

import pytest
import torch
import tyro

from vvla.models import ModelConfig, make_model
from vvla.models.pi0 import Pi0, Pi0ModelConfig
from vvla.models.pi05 import Pi05, Pi05ModelConfig

#### Fixtures ####


@pytest.fixture
def pi0_config() -> Pi0ModelConfig:
    return Pi0ModelConfig(input_dim=10, output_dim=2, hidden_dim=32)


@pytest.fixture
def pi05_config() -> Pi05ModelConfig:
    return Pi05ModelConfig(input_dim=20, output_dim=5, hidden_dim=64)


# ==============================================================================
# Factory Tests
# ==============================================================================


def test_make_model_pi0(pi0_config):
    model = make_model(pi0_config)
    assert isinstance(model, Pi0)
    assert model.config == pi0_config


def test_make_model_pi05(pi05_config):
    model = make_model(pi05_config)
    assert isinstance(model, Pi05)
    assert model.config == pi05_config


def test_make_model_invalid():
    with pytest.raises(ValueError, match="Unknown model config"):
        make_model(cast(ModelConfig, "invalid_config_object"))


# ==============================================================================
# Tyro CLI Parsing Tests
# ==============================================================================


def test_tyro_selection_pi0():
    """Test that tyro correctly parses command line args for pi0."""
    # Simulate command line arguments: "pi0 --input-dim 50"
    args = ["pi0", "--input-dim", "50"]

    # We use tyro.cli with the ModelConfig union
    config = tyro.cli(ModelConfig, args=args)

    assert isinstance(config, Pi0ModelConfig)
    assert config.input_dim == 50
    # Check default was preserved
    assert config.hidden_dim == 128


def test_tyro_selection_pi05():
    """Test that tyro correctly parses command line args for pi05."""
    args = ["pi05", "--hidden-dim", "256"]
    config = tyro.cli(ModelConfig, args=args)

    assert isinstance(config, Pi05ModelConfig)
    assert config.hidden_dim == 256


# ==============================================================================
# Model Logic Tests (Forward Pass)
# ==============================================================================


@pytest.mark.parametrize(
    "model_class, config_cls",
    [(Pi0, Pi0ModelConfig), (Pi05, Pi05ModelConfig)],
)
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_model_forward_shape_and_stability(model_class, config_cls, batch_size, dtype):
    """
    Robust forward pass test verifying:
    1. Input/Output dictionary handling.
    2. Correct shape preservation.
    3. Numerical stability (no NaNs/Infs).
    4. Dtype consistency.
    """
    input_dim = 16
    output_dim = 3
    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = config_cls(input_dim=input_dim, output_dim=output_dim, hidden_dim=32)
    model = model_class(cfg).to(device=device, dtype=dtype)
    model.eval()

    # Create dummy input
    dummy_input = {
        "input": torch.randn(batch_size, input_dim, device=device, dtype=dtype)
    }

    # Forward pass
    output = model(dummy_input)

    # 1. Check Output Structure
    assert isinstance(output, dict)
    assert cfg.output_key in output
    out_tensor = output[cfg.output_key]

    # 2. Check Shape
    assert out_tensor.shape == (batch_size, output_dim)

    # 3. Check Dtype
    assert out_tensor.dtype == dtype

    # 4. Check Numerical Stability (No NaNs or Infs)
    assert torch.isfinite(out_tensor).all(), "Output contains NaNs or Infs"

    # 5. Check Device
    assert out_tensor.device.type == torch.device(device).type


@pytest.mark.parametrize(
    "model_class, config_cls",
    [(Pi0, Pi0ModelConfig), (Pi05, Pi05ModelConfig)],
)
def test_gradient_flow_and_updates(model_class, config_cls):
    """
    Robust gradient test verifying:
    1. Gradients are computed (not None).
    2. Gradients are finite (no NaNs).
    3. Gradients are non-zero (no dead neurons/disconnected graphs).
    4. Parameters actually update after an optimizer step.
    """
    cfg = config_cls(input_dim=5, output_dim=2, hidden_dim=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model_class(cfg).to(device)
    model.train()  # Ensure train mode

    inputs = torch.randn(4, 5, device=device)
    targets = torch.randn(4, 2, device=device)

    # Snapshot of parameters before update
    params_before = [p.clone() for p in model.parameters()]

    # Forward
    out = model({"input": inputs})[cfg.output_key]
    loss = torch.nn.functional.mse_loss(out, targets)
    loss.backward()

    # Optimizer Step
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()

    # Checks
    for i, (name, param) in enumerate(model.named_parameters()):
        # 1. Gradient Existence
        assert param.grad is not None, f"Parameter {name} has no gradient (None)."

        # 2. Gradient Stability
        assert torch.isfinite(param.grad).all(), (
            f"Parameter {name} gradient has NaNs/Infs."
        )

        # 3. Non-Zero Gradients (Skipping bias sometimes is safe, but weights should move)
        # Note: In rare random inits with ReLU, a neuron might be dead, but generally this should pass.
        assert param.grad.abs().sum() > 0, (
            f"Parameter {name} has zero gradient (disconnected?)."
        )

        # 4. Parameter Update Check
        assert not torch.equal(param, params_before[i]), (
            f"Parameter {name} did not update."
        )
