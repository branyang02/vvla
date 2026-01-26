"""
uv run pytest tests/test_policies.py
"""

from typing import cast

import numpy as np
import pytest
import torch
import tyro

from vlla.models.pi05 import Pi05, Pi05ModelConfig
from vlla.policies import PolicyConfig, load_policy, make_policy
from vlla.policies.base_policy import BasePolicy
from vlla.policies.dummy_policy import DummyPolicy, DummyPolicyConfig
from vlla.policies.libero_policy import LiberoPolicy, LiberoPolicyConfig
from vlla.policies.metaworld_policy import MetaworldPolicy, MetaworldPolicyConfig
from vlla.transforms.transforms import NormStats

#### Fixtures ####


@pytest.fixture
def norm_stats() -> dict[str, NormStats]:
    return {
        "observation": NormStats(
            mean=np.array([0.0] * 10),
            std=np.array([1.0] * 10),
            q01=None,
            q99=None,
        ),
        "action": NormStats(
            mean=np.array([0.0] * 4),
            std=np.array([1.0] * 4),
            q01=None,
            q99=None,
        ),
    }


@pytest.fixture
def pi05_model() -> Pi05:
    config = Pi05ModelConfig(input_dim=10, output_dim=4, hidden_dim=32)
    return Pi05(config)


@pytest.fixture
def libero_config() -> LiberoPolicyConfig:
    return LiberoPolicyConfig(
        model=Pi05ModelConfig(input_dim=10, output_dim=7, hidden_dim=32),
        action_dim=7,
        chunk_size=5,
    )


@pytest.fixture
def metaworld_config() -> MetaworldPolicyConfig:
    return MetaworldPolicyConfig(
        model=Pi05ModelConfig(input_dim=10, output_dim=4, hidden_dim=32),
        action_dim=4,
    )


# ==============================================================================
# BasePolicy Tests
# ==============================================================================


def test_dummy_policy_is_base_policy():
    policy = DummyPolicy(DummyPolicyConfig())
    assert isinstance(policy, BasePolicy)


def test_dummy_policy_infer_returns_dict():
    policy = DummyPolicy(DummyPolicyConfig(action_dim=4))
    obs = {"state": np.random.randn(10)}
    result = policy.infer(obs)
    assert isinstance(result, dict)
    assert "actions" in result


def test_dummy_policy_infer_shape():
    action_dim = 8
    policy = DummyPolicy(DummyPolicyConfig(action_dim=action_dim))
    obs = {"state": np.random.randn(10)}
    result = policy.infer(obs)
    assert result["actions"].shape == (action_dim,)
    assert result["actions"].dtype == np.float32


def test_dummy_policy_reset():
    policy = DummyPolicy(DummyPolicyConfig())
    # Should not raise
    policy.reset()


# ==============================================================================
# LiberoPolicy Tests
# ==============================================================================


def test_libero_policy_is_base_policy(libero_config, pi05_model):
    policy = LiberoPolicy(config=libero_config, model=pi05_model)
    assert isinstance(policy, BasePolicy)


def test_libero_policy_infer_returns_dict(libero_config):
    model = Pi05(Pi05ModelConfig(input_dim=10, output_dim=7, hidden_dim=32))
    policy = LiberoPolicy(config=libero_config, model=model)
    obs = {"state": np.random.randn(10)}
    result = policy.infer(obs)
    assert isinstance(result, dict)
    assert "actions" in result


def test_libero_policy_infer_shape(libero_config):
    model = Pi05(Pi05ModelConfig(input_dim=10, output_dim=7, hidden_dim=32))
    policy = LiberoPolicy(config=libero_config, model=model)
    obs = {"state": np.random.randn(10)}
    result = policy.infer(obs)
    # Libero returns action chunks
    assert result["actions"].shape == (
        libero_config.chunk_size,
        libero_config.action_dim,
    )
    assert result["actions"].dtype == np.float32


def test_libero_policy_train_forward(libero_config):
    model = Pi05(Pi05ModelConfig(input_dim=10, output_dim=7, hidden_dim=32))
    policy = LiberoPolicy(config=libero_config, model=model)

    batch = {"input": torch.randn(4, 10)}
    result = policy.train_forward(batch)
    assert isinstance(result, dict)


def test_libero_config_defaults():
    config = LiberoPolicyConfig()
    assert config.action_dim == 7
    assert config.chunk_size == 5


# ==============================================================================
# MetaworldPolicy Tests
# ==============================================================================


def test_metaworld_policy_is_base_policy(metaworld_config, pi05_model):
    policy = MetaworldPolicy(config=metaworld_config, model=pi05_model)
    assert isinstance(policy, BasePolicy)


def test_metaworld_policy_infer_returns_dict(metaworld_config, pi05_model):
    policy = MetaworldPolicy(config=metaworld_config, model=pi05_model)
    obs = {"state": np.random.randn(1, 10)}
    result = policy.infer(obs)
    assert isinstance(result, dict)
    assert "actions" in result


def test_metaworld_policy_infer_shape(metaworld_config, pi05_model):
    policy = MetaworldPolicy(config=metaworld_config, model=pi05_model)
    batch_size = 4
    obs = {"state": np.random.randn(batch_size, 10)}
    result = policy.infer(obs)
    assert result["actions"].shape == (batch_size, metaworld_config.action_dim)
    assert result["actions"].dtype == np.float32


def test_metaworld_policy_train_forward(metaworld_config, pi05_model):
    policy = MetaworldPolicy(config=metaworld_config, model=pi05_model)

    batch = {"input": torch.randn(4, 10)}
    result = policy.train_forward(batch)
    assert isinstance(result, dict)


def test_metaworld_config_defaults():
    config = MetaworldPolicyConfig()
    assert config.action_dim == 4


# ==============================================================================
# Factory Tests
# ==============================================================================


def test_make_policy_libero(norm_stats):
    config = LiberoPolicyConfig(
        model=Pi05ModelConfig(input_dim=10, output_dim=7, hidden_dim=32),
    )
    policy = make_policy(config, norm_stats)
    assert isinstance(policy, LiberoPolicy)
    assert policy.config == config


def test_make_policy_metaworld(norm_stats):
    config = MetaworldPolicyConfig(
        model=Pi05ModelConfig(input_dim=10, output_dim=4, hidden_dim=32),
    )
    policy = make_policy(config, norm_stats)
    assert isinstance(policy, MetaworldPolicy)
    assert policy.config == config


def test_make_policy_invalid(norm_stats):
    with pytest.raises(AttributeError):
        make_policy(cast(PolicyConfig, "invalid"), norm_stats)


# ==============================================================================
# Tyro CLI Parsing Tests
# ==============================================================================


def test_tyro_selection_libero():
    args = ["libero", "--action-dim", "8"]
    config = tyro.cli(PolicyConfig, args=args)
    assert isinstance(config, LiberoPolicyConfig)
    assert config.action_dim == 8


def test_tyro_selection_metaworld():
    args = ["metaworld", "--action-dim", "6"]
    config = tyro.cli(PolicyConfig, args=args)
    assert isinstance(config, MetaworldPolicyConfig)
    assert config.action_dim == 6


# ==============================================================================
# Save/Load Tests
# ==============================================================================


def test_libero_policy_save_load_roundtrip(tmp_path, libero_config, norm_stats):
    # Create policy
    model = Pi05(Pi05ModelConfig(input_dim=10, output_dim=7, hidden_dim=32))
    original_policy = LiberoPolicy(config=libero_config, model=model)

    # Save and load
    checkpoint_path = tmp_path / "libero_policy.pt"
    original_policy.save(checkpoint_path)
    loaded_policy = LiberoPolicy.load(checkpoint_path)

    # Verify config matches
    assert loaded_policy.config.action_dim == original_policy.config.action_dim
    assert loaded_policy.config.chunk_size == original_policy.config.chunk_size

    # Verify model weights match
    for key in original_policy.model.state_dict():
        assert torch.allclose(
            original_policy.model.state_dict()[key],
            loaded_policy.model.state_dict()[key],
        )


def test_metaworld_policy_save_load_roundtrip(tmp_path, metaworld_config, pi05_model):
    # Create policy
    original_policy = MetaworldPolicy(config=metaworld_config, model=pi05_model)

    # Save and load
    checkpoint_path = tmp_path / "metaworld_policy.pt"
    original_policy.save(checkpoint_path)
    loaded_policy = MetaworldPolicy.load(checkpoint_path)

    # Verify config matches
    assert loaded_policy.config.action_dim == original_policy.config.action_dim

    # Verify model weights match
    for key in original_policy.model.state_dict():
        assert torch.allclose(
            original_policy.model.state_dict()[key],
            loaded_policy.model.state_dict()[key],
        )


def test_load_policy_factory_libero(tmp_path, libero_config):
    # Create and save policy
    model = Pi05(Pi05ModelConfig(input_dim=10, output_dim=7, hidden_dim=32))
    original_policy = LiberoPolicy(config=libero_config, model=model)
    checkpoint_path = tmp_path / "policy.pt"
    original_policy.save(checkpoint_path)

    # Load using factory function
    loaded_policy = load_policy(checkpoint_path)
    assert isinstance(loaded_policy, LiberoPolicy)


def test_load_policy_factory_metaworld(tmp_path, metaworld_config, pi05_model):
    # Create and save policy
    original_policy = MetaworldPolicy(config=metaworld_config, model=pi05_model)
    checkpoint_path = tmp_path / "policy.pt"
    original_policy.save(checkpoint_path)

    # Load using factory function
    loaded_policy = load_policy(checkpoint_path)
    assert isinstance(loaded_policy, MetaworldPolicy)


def test_loaded_policy_infer_works(tmp_path, libero_config):
    # Create, save, and load policy
    model = Pi05(Pi05ModelConfig(input_dim=10, output_dim=7, hidden_dim=32))
    original_policy = LiberoPolicy(config=libero_config, model=model)
    checkpoint_path = tmp_path / "policy.pt"
    original_policy.save(checkpoint_path)
    loaded_policy = load_policy(checkpoint_path)

    # Verify infer works on loaded policy
    obs = {"state": np.random.randn(10)}
    result = loaded_policy.infer(obs)
    assert isinstance(result, dict)
    assert "actions" in result
    assert result["actions"].shape == (
        libero_config.chunk_size,
        libero_config.action_dim,
    )
