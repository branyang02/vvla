"""
uv run pytest tests/test_models.py
"""

from dataclasses import fields, is_dataclass
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from vvla.models import ModelConfig, make_model
from vvla.models.base_model import BaseModel
from vvla.models.pi0 import Pi0ModelConfig
from vvla.models.pi05 import Pi05ModelConfig

# ==============================================================================
# Config Structure Tests
# ==============================================================================


class TestPi0ModelConfig:
    """Tests for Pi0ModelConfig dataclass."""

    def test_is_dataclass(self):
        assert is_dataclass(Pi0ModelConfig)

    def test_default_values(self):
        config = Pi0ModelConfig()
        assert config.input_dim == 39
        assert config.output_dim == 4
        assert config.hidden_dim == 128
        assert config.output_key == "actions"

    def test_custom_values(self):
        config = Pi0ModelConfig(
            input_dim=100,
            output_dim=10,
            hidden_dim=256,
            output_key="custom_output",
        )
        assert config.input_dim == 100
        assert config.output_dim == 10
        assert config.hidden_dim == 256
        assert config.output_key == "custom_output"

    def test_has_expected_fields(self):
        field_names = {f.name for f in fields(Pi0ModelConfig)}
        assert "input_dim" in field_names
        assert "output_dim" in field_names
        assert "hidden_dim" in field_names
        assert "output_key" in field_names


class TestPi05ModelConfig:
    """Tests for Pi05ModelConfig dataclass."""

    def test_is_dataclass(self):
        assert is_dataclass(Pi05ModelConfig)

    def test_can_instantiate_with_defaults(self):
        # Should not raise
        config = Pi05ModelConfig()
        assert config is not None

    def test_has_nested_configs(self):
        config = Pi05ModelConfig()
        # Pi05 has nested VLM configs
        assert hasattr(config, "vlm_siglip_config")
        assert hasattr(config, "vlm_text_config")
        assert hasattr(config, "action_expert_config")


# ==============================================================================
# Factory Tests (with mocks)
# ==============================================================================


class TestMakeModel:
    """Tests for make_model factory function."""

    def test_dispatches_to_pi0(self):
        config = Pi0ModelConfig()

        with patch("vvla.models.pi0.Pi0") as MockPi0:
            mock_instance = MagicMock(spec=BaseModel)
            MockPi0.return_value = mock_instance

            result = make_model(config)

            MockPi0.assert_called_once_with(config)
            assert result == mock_instance

    def test_dispatches_to_pi05(self):
        config = Pi05ModelConfig()

        with patch("vvla.models.pi05.Pi05") as MockPi05:
            mock_instance = MagicMock(spec=BaseModel)
            MockPi05.return_value = mock_instance

            result = make_model(config)

            MockPi05.assert_called_once_with(config)
            assert result == mock_instance

    def test_raises_for_invalid_config(self):
        with pytest.raises(ValueError, match="Unknown model config"):
            make_model(cast(ModelConfig, "invalid_config_object"))

    def test_raises_for_none_config(self):
        with pytest.raises((ValueError, AttributeError)):
            make_model(cast(ModelConfig, None))


# ==============================================================================
# Tyro CLI Parsing Tests
# ==============================================================================


class TestTyroCLIParsing:
    """Tests for tyro CLI integration with ModelConfig union."""

    def test_parses_pi0_subcommand(self):
        import tyro

        args = ["pi0"]
        config = tyro.cli(ModelConfig, args=args)

        assert isinstance(config, Pi0ModelConfig)

    def test_parses_pi0_with_args(self):
        import tyro

        args = ["pi0", "--input-dim", "50", "--hidden-dim", "256"]
        config = tyro.cli(ModelConfig, args=args)

        assert isinstance(config, Pi0ModelConfig)
        assert config.input_dim == 50
        assert config.hidden_dim == 256

    def test_parses_pi05_subcommand(self):
        import tyro

        args = ["pi05"]
        config = tyro.cli(ModelConfig, args=args)

        assert isinstance(config, Pi05ModelConfig)

    def test_pi0_preserves_defaults(self):
        import tyro

        args = ["pi0", "--input-dim", "100"]
        config = tyro.cli(ModelConfig, args=args)

        assert isinstance(config, Pi0ModelConfig)
        assert config.input_dim == 100
        # Defaults preserved
        assert config.hidden_dim == 128
        assert config.output_dim == 4


# ==============================================================================
# BaseModel Interface Tests
# ==============================================================================


class TestBaseModelInterface:
    """Tests for BaseModel abstract base class."""

    def test_forward_is_abstract(self):
        # Check that forward is declared as an abstract method
        assert hasattr(BaseModel, "forward")
        assert getattr(BaseModel.forward, "__isabstractmethod__", False)

    def test_inherits_from_nn_module(self):
        import torch.nn as nn

        assert issubclass(BaseModel, nn.Module)

    def test_forward_signature(self):
        import inspect

        sig = inspect.signature(BaseModel.forward)
        params = list(sig.parameters.keys())
        # Should have self and x (dict input)
        assert "self" in params
        assert "x" in params


# ==============================================================================
# Model Registration Tests
# ==============================================================================


class TestModelRegistration:
    """Tests to verify models are properly registered in the module."""

    def test_model_config_is_union_type(self):
        # ModelConfig should accept both Pi0ModelConfig and Pi05ModelConfig
        from typing import get_origin

        # For Union types in Python 3.10+, we check if it's a UnionType
        origin = get_origin(ModelConfig)
        # Could be types.UnionType or typing.Union depending on Python version
        assert origin is not None or "|" in str(ModelConfig)

    def test_pi0_config_in_model_config(self):
        # Pi0ModelConfig should be a valid ModelConfig
        config = Pi0ModelConfig()
        # This should not raise in make_model's isinstance checks
        assert isinstance(config, Pi0ModelConfig)

    def test_pi05_config_in_model_config(self):
        # Pi05ModelConfig should be a valid ModelConfig
        config = Pi05ModelConfig()
        assert isinstance(config, Pi05ModelConfig)
