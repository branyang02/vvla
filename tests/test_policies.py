"""
uv run pytest tests/test_policies.py
"""

import json
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from vvla.models.base_model import BaseModel
from vvla.policies import PolicyConfig, make_policy
from vvla.policies.base_policy import BasePolicy
from vvla.policies.dummy_policy import DummyPolicy, DummyPolicyConfig
from vvla.policies.libero_policy import LiberoPolicy, LiberoPolicyConfig
from vvla.policies.metaworld_policy import MetaworldPolicy, MetaworldPolicyConfig
from vvla.transforms.transforms import Normalize, NormStats, Unnormalize

# ==============================================================================
# Test Model (simple PyTorch module for testing)
# ==============================================================================


class SimpleTestModel(BaseModel):
    """A minimal model for testing save/load without heavy dependencies."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.linear1 = nn.Linear(10, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 7)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        h = torch.relu(self.linear1(x["input"]))
        return {"output": self.linear2(h)}


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def norm_stats() -> dict[str, NormStats]:
    return {
        "observation": NormStats(
            mean=np.array([0.0, 1.0, 2.0], dtype=np.float32),
            std=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            q01=np.array([-1.0, -2.0, -3.0], dtype=np.float32),
            q99=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        ),
        "action": NormStats(
            mean=np.array([0.0, 0.5], dtype=np.float32),
            std=np.array([1.0, 1.5], dtype=np.float32),
            q01=None,
            q99=None,
        ),
    }


@pytest.fixture
def test_model() -> SimpleTestModel:
    return SimpleTestModel(hidden_dim=32)


@pytest.fixture
def libero_config() -> LiberoPolicyConfig:
    return LiberoPolicyConfig(
        repo_id="test-org/test-libero-policy",
        action_dim=7,
        chunk_size=5,
    )


@pytest.fixture
def metaworld_config() -> MetaworldPolicyConfig:
    return MetaworldPolicyConfig(
        repo_id="test-org/test-metaworld-policy",
        action_dim=4,
    )


@pytest.fixture
def libero_policy(libero_config, test_model, norm_stats) -> LiberoPolicy:
    """Create a LiberoPolicy with transforms for testing."""
    return LiberoPolicy(
        config=libero_config,
        model=test_model,
        input_transforms=[Normalize(norm_stats)],
        output_transforms=[Unnormalize(norm_stats)],
    )


@pytest.fixture
def metaworld_policy(metaworld_config, test_model, norm_stats) -> MetaworldPolicy:
    """Create a MetaworldPolicy with transforms for testing."""
    return MetaworldPolicy(
        config=metaworld_config,
        model=test_model,
        input_transforms=[Normalize(norm_stats)],
        output_transforms=[Unnormalize(norm_stats)],
    )


# ==============================================================================
# DummyPolicy Tests
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
    policy.reset()  # Should not raise


# ==============================================================================
# LiberoPolicy Tests
# ==============================================================================


def test_libero_policy_is_base_policy(libero_policy):
    assert isinstance(libero_policy, BasePolicy)


def test_libero_policy_infer_returns_dict(libero_policy):
    obs = {"state": np.random.randn(10)}
    result = libero_policy.infer(obs)
    assert isinstance(result, dict)
    assert "actions" in result


def test_libero_policy_infer_shape(libero_policy):
    obs = {"state": np.random.randn(10)}
    result = libero_policy.infer(obs)
    assert result["actions"].shape == (
        libero_policy.config.chunk_size,
        libero_policy.config.action_dim,
    )
    assert result["actions"].dtype == np.float32


def test_libero_config_defaults():
    config = LiberoPolicyConfig(repo_id="test/repo")
    assert config.action_dim == 7
    assert config.chunk_size == 5


# ==============================================================================
# MetaworldPolicy Tests
# ==============================================================================


def test_metaworld_policy_is_base_policy(metaworld_policy):
    assert isinstance(metaworld_policy, BasePolicy)


def test_metaworld_policy_infer_returns_dict(metaworld_policy):
    obs = {"state": np.random.randn(1, 10)}
    result = metaworld_policy.infer(obs)
    assert isinstance(result, dict)
    assert "actions" in result


def test_metaworld_policy_infer_shape(metaworld_policy):
    batch_size = 4
    obs = {"state": np.random.randn(batch_size, 10)}
    result = metaworld_policy.infer(obs)
    assert result["actions"].shape == (batch_size, metaworld_policy.config.action_dim)
    assert result["actions"].dtype == np.float32


def test_metaworld_config_defaults():
    config = MetaworldPolicyConfig(repo_id="test/repo")
    assert config.action_dim == 4


# ==============================================================================
# Factory Tests
# ==============================================================================


def test_make_policy_dummy():
    config = DummyPolicyConfig()
    policy = make_policy(config, mode="train")
    assert isinstance(policy, DummyPolicy)


def test_make_policy_requires_norm_stats_for_fresh_policy():
    config = LiberoPolicyConfig(repo_id="test/repo")
    with pytest.raises(ValueError, match="norm_stats"):
        make_policy(config, mode="train", norm_stats=None)


def test_make_policy_invalid():
    with pytest.raises(AttributeError):
        make_policy(cast(PolicyConfig, "invalid"), mode="train", norm_stats=None)


def test_make_policy_eval_requires_pretrained_repo_id_or_path():
    config = LiberoPolicyConfig(repo_id=None)
    with pytest.raises(ValueError, match="pretrained_repo_id_or_path"):
        make_policy(config, mode="eval", norm_stats=None)


def test_make_policy_eval_forbids_repo_id():
    config = LiberoPolicyConfig(
        repo_id="test/repo",
        pretrained_repo_id_or_path="test-org/test-policy",
    )
    with pytest.raises(ValueError, match="repo_id"):
        make_policy(config, mode="eval", norm_stats=None)


def test_make_policy_eval_forbids_norm_stats():
    config = LiberoPolicyConfig(
        repo_id=None,
        pretrained_repo_id_or_path="test-org/test-policy",
    )
    with pytest.raises(ValueError, match="norm_stats"):
        make_policy(config, mode="eval", norm_stats={})


def test_make_policy_eval_calls_from_config(libero_config):
    libero_config.repo_id = None
    libero_config.pretrained_repo_id_or_path = "test-org/test-policy"

    with patch("vvla.policies.LiberoPolicy.from_config") as mock_from_config:
        mock_from_config.return_value = LiberoPolicy(
            config=libero_config,
            model=SimpleTestModel(hidden_dim=32),
            input_transforms=[],
            output_transforms=[],
        )

        policy = make_policy(libero_config, mode="eval", norm_stats=None)

    mock_from_config.assert_called_once_with(libero_config)
    assert isinstance(policy, LiberoPolicy)


def test_make_policy_train_finetune_overwrites_transforms(libero_config, norm_stats):
    libero_config.pretrained_repo_id_or_path = "test-org/test-policy"
    original_policy = LiberoPolicy(
        config=libero_config,
        model=SimpleTestModel(hidden_dim=32),
        input_transforms=[Normalize(norm_stats)],
        output_transforms=[Unnormalize(norm_stats)],
    )

    new_input = [Normalize(norm_stats)]
    new_output = [Unnormalize(norm_stats)]

    with (
        patch("vvla.policies.LiberoPolicy.from_config") as mock_from_config,
        patch("vvla.policies.make_transforms") as mock_make_transforms,
    ):
        mock_from_config.return_value = original_policy
        mock_make_transforms.return_value = (new_input, new_output)

        policy = make_policy(libero_config, mode="train", norm_stats=norm_stats)

    assert policy.input_transforms is new_input
    assert policy.output_transforms is new_output


def test_make_policy_train_from_scratch_builds_fresh_policy(libero_config, norm_stats):
    libero_config.pretrained_repo_id_or_path = None

    fresh_model = SimpleTestModel(hidden_dim=32)
    fresh_input_transforms = [Normalize(norm_stats)]
    fresh_output_transforms = [Unnormalize(norm_stats)]

    with (
        patch("vvla.policies.make_model") as mock_make_model,
        patch("vvla.policies.make_transforms") as mock_make_transforms,
        patch("vvla.policies.LiberoPolicy.from_config") as mock_from_config,
    ):
        mock_make_model.return_value = fresh_model
        mock_make_transforms.return_value = (
            fresh_input_transforms,
            fresh_output_transforms,
        )

        policy = make_policy(libero_config, mode="train", norm_stats=norm_stats)

    mock_from_config.assert_not_called()
    mock_make_model.assert_called_once_with(libero_config.model)
    mock_make_transforms.assert_called_once_with(libero_config.transforms, norm_stats)

    assert isinstance(policy, LiberoPolicy)
    assert policy.model is fresh_model
    assert policy.input_transforms is fresh_input_transforms
    assert policy.output_transforms is fresh_output_transforms


def test_make_policy_train_missing_repo_id_raises():
    config = LiberoPolicyConfig(repo_id=None)
    with pytest.raises(ValueError, match="repo_id"):
        make_policy(config, mode="train", norm_stats={})


def test_make_policy_invalid_mode_raises():
    config = LiberoPolicyConfig(repo_id="test/repo")
    with pytest.raises(ValueError, match="mode"):
        make_policy(config, mode="invalid", norm_stats=None)


# ==============================================================================
# save_pretrained Tests
# ==============================================================================


class TestSavePretrained:
    """Tests for BasePolicy.save_pretrained()"""

    def test_creates_directory(self, tmp_path, libero_policy):
        save_dir = tmp_path / "new_dir" / "nested"
        libero_policy.save_pretrained(save_dir)
        assert save_dir.exists()

    def test_creates_model_safetensors(self, tmp_path, libero_policy):
        save_dir = tmp_path / "policy"
        libero_policy.save_pretrained(save_dir)

        model_path = save_dir / "model.safetensors"
        assert model_path.exists()
        assert model_path.stat().st_size > 0

    def test_creates_transforms_json(self, tmp_path, libero_policy):
        save_dir = tmp_path / "policy"
        libero_policy.save_pretrained(save_dir)

        transforms_path = save_dir / "transforms.json"
        assert transforms_path.exists()

        with open(transforms_path) as f:
            data = json.load(f)

        assert "input_transforms" in data
        assert "output_transforms" in data
        assert len(data["input_transforms"]) == 1
        assert data["input_transforms"][0]["_type"] == "Normalize"
        assert len(data["output_transforms"]) == 1
        assert data["output_transforms"][0]["_type"] == "Unnormalize"

    def test_transforms_contain_norm_stats(self, tmp_path, libero_policy):
        save_dir = tmp_path / "policy"
        libero_policy.save_pretrained(save_dir)

        with open(save_dir / "transforms.json") as f:
            data = json.load(f)

        input_transform = data["input_transforms"][0]
        assert "norm_stats" in input_transform
        assert "observation" in input_transform["norm_stats"]
        assert "action" in input_transform["norm_stats"]


# ==============================================================================
# from_config Tests
# ==============================================================================


class TestFromConfig:
    """Tests for BasePolicy.from_config()"""

    def test_loads_from_local_directory(self, tmp_path, libero_policy, test_model):
        save_dir = tmp_path / "policy"
        libero_policy.save_pretrained(save_dir)
        libero_policy.config.pretrained_repo_id_or_path = str(save_dir)

        with patch("vvla.policies.base_policy.make_model") as mock_make_model:
            mock_make_model.return_value = SimpleTestModel(hidden_dim=32)
            loaded = LiberoPolicy.from_config(config=libero_policy.config)

        assert isinstance(loaded, LiberoPolicy)
        assert loaded.policy_type == "libero"

    def test_loads_config_correctly(self, tmp_path, libero_policy):
        save_dir = tmp_path / "policy"
        libero_policy.save_pretrained(save_dir)
        libero_policy.config.pretrained_repo_id_or_path = str(save_dir)

        with patch("vvla.policies.base_policy.make_model") as mock_make_model:
            mock_make_model.return_value = SimpleTestModel(hidden_dim=32)
            loaded = LiberoPolicy.from_config(config=libero_policy.config)

        assert loaded.config.action_dim == libero_policy.config.action_dim
        assert loaded.config.chunk_size == libero_policy.config.chunk_size
        assert loaded.config.repo_id == libero_policy.config.repo_id

    def test_loads_model_weights_correctly(self, tmp_path, libero_policy):
        save_dir = tmp_path / "policy"
        libero_policy.save_pretrained(save_dir)
        libero_policy.config.pretrained_repo_id_or_path = str(save_dir)

        with patch("vvla.policies.base_policy.make_model") as mock_make_model:
            # Create a fresh model with same architecture
            fresh_model = SimpleTestModel(hidden_dim=32)
            mock_make_model.return_value = fresh_model
            loaded = LiberoPolicy.from_config(config=libero_policy.config)

        original_state = libero_policy.model.state_dict()
        loaded_state = loaded.model.state_dict()

        assert set(original_state.keys()) == set(loaded_state.keys())
        for key in original_state:
            assert torch.allclose(original_state[key], loaded_state[key])

    def test_loads_input_transforms(self, tmp_path, libero_policy):
        save_dir = tmp_path / "policy"
        libero_policy.save_pretrained(save_dir)
        libero_policy.config.pretrained_repo_id_or_path = str(save_dir)

        with patch("vvla.policies.base_policy.make_model") as mock_make_model:
            mock_make_model.return_value = SimpleTestModel(hidden_dim=32)
            loaded = LiberoPolicy.from_config(config=libero_policy.config)

        assert len(loaded.input_transforms) == 1
        assert isinstance(loaded.input_transforms[0], Normalize)

    def test_loads_output_transforms(self, tmp_path, libero_policy):
        save_dir = tmp_path / "policy"
        libero_policy.save_pretrained(save_dir)
        libero_policy.config.pretrained_repo_id_or_path = str(save_dir)

        with patch("vvla.policies.base_policy.make_model") as mock_make_model:
            mock_make_model.return_value = SimpleTestModel(hidden_dim=32)
            loaded = LiberoPolicy.from_config(config=libero_policy.config)

        assert len(loaded.output_transforms) == 1
        assert isinstance(loaded.output_transforms[0], Unnormalize)

    def test_loads_norm_stats_in_transforms(self, tmp_path, libero_policy, norm_stats):
        save_dir = tmp_path / "policy"
        libero_policy.save_pretrained(save_dir)
        libero_policy.config.pretrained_repo_id_or_path = str(save_dir)

        with patch("vvla.policies.base_policy.make_model") as mock_make_model:
            mock_make_model.return_value = SimpleTestModel(hidden_dim=32)
            loaded = LiberoPolicy.from_config(config=libero_policy.config)

        loaded_norm_stats = loaded.input_transforms[0].norm_stats
        assert "observation" in loaded_norm_stats
        assert "action" in loaded_norm_stats

        np.testing.assert_array_almost_equal(
            loaded_norm_stats["observation"].mean,
            norm_stats["observation"].mean,
        )
        np.testing.assert_array_almost_equal(
            loaded_norm_stats["observation"].std,
            norm_stats["observation"].std,
        )

    def test_metaworld_from_config(self, tmp_path, metaworld_policy):
        save_dir = tmp_path / "metaworld_policy"
        metaworld_policy.save_pretrained(save_dir)
        metaworld_policy.config.pretrained_repo_id_or_path = str(save_dir)

        with patch("vvla.policies.base_policy.make_model") as mock_make_model:
            mock_make_model.return_value = SimpleTestModel(hidden_dim=32)
            loaded = MetaworldPolicy.from_config(config=metaworld_policy.config)

        assert isinstance(loaded, MetaworldPolicy)
        assert loaded.config.action_dim == metaworld_policy.config.action_dim

    def test_loaded_policy_can_infer(self, tmp_path, libero_policy):
        save_dir = tmp_path / "policy"
        libero_policy.save_pretrained(save_dir)
        libero_policy.config.pretrained_repo_id_or_path = str(save_dir)

        with patch("vvla.policies.base_policy.make_model") as mock_make_model:
            mock_make_model.return_value = SimpleTestModel(hidden_dim=32)
            loaded = LiberoPolicy.from_config(config=libero_policy.config)

        obs = {"state": np.random.randn(10)}
        result = loaded.infer(obs)

        assert isinstance(result, dict)
        assert "actions" in result
        assert result["actions"].shape == (
            loaded.config.chunk_size,
            loaded.config.action_dim,
        )

    def test_from_config_downloads_from_hub(self, tmp_path, libero_policy):
        """Test that from_config calls snapshot_download for non-local paths."""
        save_dir = tmp_path / "policy"
        libero_policy.save_pretrained(save_dir)
        libero_policy.config.pretrained_repo_id_or_path = "my-org/my-policy"

        with (
            patch(
                "vvla.policies.base_policy.snapshot_download"
            ) as mock_snapshot_download,
            patch("vvla.policies.base_policy.make_model") as mock_make_model,
        ):
            mock_snapshot_download.return_value = str(save_dir)
            mock_make_model.return_value = SimpleTestModel(hidden_dim=32)

            loaded = LiberoPolicy.from_config(config=libero_policy.config)

            mock_snapshot_download.assert_called_once()
            call_kwargs = mock_snapshot_download.call_args
            assert call_kwargs[1]["repo_id"] == "my-org/my-policy"
            assert isinstance(loaded, LiberoPolicy)


# ==============================================================================
# push_to_hub Tests
# ==============================================================================


class TestPushToHub:
    """Tests for BasePolicy.push_to_hub()"""

    def test_creates_repo_with_correct_id(self, libero_policy):
        with patch("vvla.policies.base_policy.HfApi") as MockHfApi:
            mock_api = MagicMock()
            MockHfApi.return_value = mock_api
            mock_api.upload_large_folder.return_value = "https://huggingface.co/test"

            libero_policy.push_to_hub()

            mock_api.create_repo.assert_called_once_with(
                repo_id="test-org/test-libero-policy",
                private=False,
                exist_ok=True,
            )

    def test_uses_config_repo_id(self, test_model, norm_stats):
        custom_repo = "custom-org/custom-policy"
        config = LiberoPolicyConfig(repo_id=custom_repo)
        policy = LiberoPolicy(
            config=config,
            model=test_model,
            input_transforms=[Normalize(norm_stats)],
            output_transforms=[],
        )

        with patch("vvla.policies.base_policy.HfApi") as MockHfApi:
            mock_api = MagicMock()
            MockHfApi.return_value = mock_api
            mock_api.upload_large_folder.return_value = "https://huggingface.co/test"

            policy.push_to_hub()

            mock_api.create_repo.assert_called_once_with(
                repo_id=custom_repo,
                private=False,
                exist_ok=True,
            )

    def test_calls_upload_large_folder(self, libero_policy):
        with patch("vvla.policies.base_policy.HfApi") as MockHfApi:
            mock_api = MagicMock()
            MockHfApi.return_value = mock_api
            mock_api.upload_large_folder.return_value = "https://huggingface.co/test"

            libero_policy.push_to_hub()

            mock_api.upload_large_folder.assert_called_once()
            call_kwargs = mock_api.upload_large_folder.call_args[1]
            assert call_kwargs["repo_id"] == "test-org/test-libero-policy"
            assert call_kwargs["repo_type"] == "model"

    def test_saves_files_before_upload(self, libero_policy):
        with patch("vvla.policies.base_policy.HfApi") as MockHfApi:
            mock_api = MagicMock()
            MockHfApi.return_value = mock_api
            mock_api.upload_large_folder.return_value = "https://huggingface.co/test"

            libero_policy.push_to_hub()

            call_kwargs = mock_api.upload_large_folder.call_args[1]
            folder_path = call_kwargs["folder_path"]

            # The folder should have existed (temp dir is cleaned up after)
            # but we can verify the method was called with a folder path
            assert folder_path is not None

    def test_private_repo(self, libero_policy):
        with patch("vvla.policies.base_policy.HfApi") as MockHfApi:
            mock_api = MagicMock()
            MockHfApi.return_value = mock_api
            mock_api.upload_large_folder.return_value = "https://huggingface.co/test"

            libero_policy.push_to_hub(private=True)

            mock_api.create_repo.assert_called_once_with(
                repo_id="test-org/test-libero-policy",
                private=True,
                exist_ok=True,
            )

    def test_custom_commit_message(self, libero_policy):
        with patch("vvla.policies.base_policy.HfApi") as MockHfApi:
            mock_api = MagicMock()
            MockHfApi.return_value = mock_api
            mock_api.upload_large_folder.return_value = "https://huggingface.co/test"

            libero_policy.push_to_hub(commit_message="Custom commit")

            call_kwargs = mock_api.upload_large_folder.call_args[1]
            assert call_kwargs["commit_message"] == "Custom commit"

    def test_default_commit_message(self, libero_policy):
        with patch("vvla.policies.base_policy.HfApi") as MockHfApi:
            mock_api = MagicMock()
            MockHfApi.return_value = mock_api
            mock_api.upload_large_folder.return_value = "https://huggingface.co/test"

            libero_policy.push_to_hub()

            call_kwargs = mock_api.upload_large_folder.call_args[1]
            assert "libero" in call_kwargs["commit_message"]

    def test_create_pr_option(self, libero_policy):
        with patch("vvla.policies.base_policy.HfApi") as MockHfApi:
            mock_api = MagicMock()
            MockHfApi.return_value = mock_api
            mock_api.upload_large_folder.return_value = "https://huggingface.co/test"

            libero_policy.push_to_hub(create_pr=True)

            call_kwargs = mock_api.upload_large_folder.call_args[1]
            assert call_kwargs["create_pr"] is True

    def test_token_passed_to_api(self, libero_policy):
        with patch("vvla.policies.base_policy.HfApi") as MockHfApi:
            mock_api = MagicMock()
            MockHfApi.return_value = mock_api
            mock_api.upload_large_folder.return_value = "https://huggingface.co/test"

            libero_policy.push_to_hub(token="my-token")

            MockHfApi.assert_called_once_with(token="my-token")

    def test_returns_url(self, libero_policy):
        with patch("vvla.policies.base_policy.HfApi") as MockHfApi:
            mock_api = MagicMock()
            MockHfApi.return_value = mock_api
            expected_url = "https://huggingface.co/test-org/test-libero-policy"
            mock_api.upload_large_folder.return_value = expected_url

            result = libero_policy.push_to_hub()

            assert result == expected_url

    def test_metaworld_push_to_hub(self, metaworld_policy):
        with patch("vvla.policies.base_policy.HfApi") as MockHfApi:
            mock_api = MagicMock()
            MockHfApi.return_value = mock_api
            mock_api.upload_large_folder.return_value = "https://huggingface.co/test"

            metaworld_policy.push_to_hub()

            mock_api.create_repo.assert_called_once_with(
                repo_id="test-org/test-metaworld-policy",
                private=False,
                exist_ok=True,
            )
            call_kwargs = mock_api.upload_large_folder.call_args[1]
            assert "metaworld" in call_kwargs["commit_message"]


# ==============================================================================
# Roundtrip Tests
# ==============================================================================


class TestRoundtrip:
    """End-to-end save/load roundtrip tests."""

    def test_libero_full_roundtrip(self, tmp_path, libero_policy, norm_stats):
        save_dir = tmp_path / "policy"
        libero_policy.save_pretrained(save_dir)
        libero_policy.config.pretrained_repo_id_or_path = str(save_dir)

        with patch("vvla.policies.base_policy.make_model") as mock_make_model:
            mock_make_model.return_value = SimpleTestModel(hidden_dim=32)
            loaded = LiberoPolicy.from_config(config=libero_policy.config)

        # Config roundtrip
        assert loaded.config.action_dim == libero_policy.config.action_dim
        assert loaded.config.chunk_size == libero_policy.config.chunk_size
        assert loaded.config.repo_id == libero_policy.config.repo_id

        # Model weights roundtrip
        for key in libero_policy.model.state_dict():
            assert torch.allclose(
                libero_policy.model.state_dict()[key],
                loaded.model.state_dict()[key],
            )

        # Transforms roundtrip
        assert len(loaded.input_transforms) == len(libero_policy.input_transforms)
        assert len(loaded.output_transforms) == len(libero_policy.output_transforms)

        # NormStats roundtrip
        original_stats = libero_policy.input_transforms[0].norm_stats
        loaded_stats = loaded.input_transforms[0].norm_stats

        for key in original_stats:
            np.testing.assert_array_almost_equal(
                original_stats[key].mean,
                loaded_stats[key].mean,
            )
            np.testing.assert_array_almost_equal(
                original_stats[key].std,
                loaded_stats[key].std,
            )

    def test_metaworld_full_roundtrip(self, tmp_path, metaworld_policy):
        save_dir = tmp_path / "policy"
        metaworld_policy.save_pretrained(save_dir)
        metaworld_policy.config.pretrained_repo_id_or_path = str(save_dir)

        with patch("vvla.policies.base_policy.make_model") as mock_make_model:
            mock_make_model.return_value = SimpleTestModel(hidden_dim=32)
            loaded = MetaworldPolicy.from_config(config=metaworld_policy.config)

        assert loaded.config.action_dim == metaworld_policy.config.action_dim
        assert loaded.config.repo_id == metaworld_policy.config.repo_id

        for key in metaworld_policy.model.state_dict():
            assert torch.allclose(
                metaworld_policy.model.state_dict()[key],
                loaded.model.state_dict()[key],
            )

    def test_policy_without_transforms_roundtrip(
        self, tmp_path, libero_config, test_model
    ):
        policy = LiberoPolicy(
            config=libero_config,
            model=test_model,
            input_transforms=[],
            output_transforms=[],
        )

        save_dir = tmp_path / "policy"
        policy.save_pretrained(save_dir)
        libero_config.pretrained_repo_id_or_path = str(save_dir)

        with patch("vvla.policies.base_policy.make_model") as mock_make_model:
            mock_make_model.return_value = SimpleTestModel(hidden_dim=32)
            loaded = LiberoPolicy.from_config(config=libero_config)

        assert len(loaded.input_transforms) == 0
        assert len(loaded.output_transforms) == 0
