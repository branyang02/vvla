"""
uv run pytest tests/test_transforms.py
"""

import numpy as np
import pytest

from vlla.transforms import make_transforms
from vlla.transforms.pi0_transforms import Pi0TransformsConfig
from vlla.transforms.pi05_transforms import Pi05TransformsConfig
from vlla.transforms.transforms import (
    CompositeTransform,
    DataTransformFn,
    Normalize,
    NormStats,
    ResizeTransform,
    Unnormalize,
    compose,
)

#### Fixtures ####


@pytest.fixture
def norm_stats() -> dict[str, NormStats]:
    return {
        "observation": NormStats(
            mean=np.array([0.0, 1.0, 2.0]),
            std=np.array([1.0, 2.0, 3.0]),
            q01=np.array([-2.0, -1.0, 0.0]),
            q99=np.array([2.0, 3.0, 4.0]),
        ),
        "action": NormStats(
            mean=np.array([0.5, 0.5]),
            std=np.array([0.1, 0.2]),
            q01=None,
            q99=None,
        ),
    }


@pytest.fixture
def sample_data() -> dict[str, np.ndarray]:
    return {
        "observation": np.array([1.0, 2.0, 3.0]),
        "action": np.array([0.6, 0.7]),
    }


# ==============================================================================
# Protocol Tests
# ==============================================================================


def test_normalize_is_data_transform_fn(norm_stats):
    transform = Normalize(norm_stats=norm_stats)
    assert isinstance(transform, DataTransformFn)


def test_unnormalize_is_data_transform_fn(norm_stats):
    transform = Unnormalize(norm_stats=norm_stats)
    assert isinstance(transform, DataTransformFn)


def test_resize_is_data_transform_fn():
    transform = ResizeTransform(height=224, width=224)
    assert isinstance(transform, DataTransformFn)


def test_composite_is_data_transform_fn(norm_stats):
    transforms = [Normalize(norm_stats=norm_stats), Unnormalize(norm_stats=norm_stats)]
    composite = CompositeTransform(transforms=transforms)
    assert isinstance(composite, DataTransformFn)


# ==============================================================================
# Compose Tests
# ==============================================================================


def test_compose_returns_composite_transform(norm_stats):
    transforms = [Normalize(norm_stats=norm_stats)]
    result = compose(transforms)
    assert isinstance(result, CompositeTransform)


def test_composite_transform_applies_in_order(sample_data):
    """Test that transforms are applied in sequence."""
    call_order = []

    class TrackingTransform:
        def __init__(self, name: str):
            self.name = name

        def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            call_order.append(self.name)
            return data

    transforms = [TrackingTransform("first"), TrackingTransform("second")]
    composite = CompositeTransform(transforms=transforms)
    composite(sample_data)

    assert call_order == ["first", "second"]


def test_composite_transform_passes_data_through(sample_data):
    """Test that data flows through the transform chain."""

    class AddOneTransform:
        def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            return {k: v + 1 for k, v in data.items()}

    transforms = [AddOneTransform(), AddOneTransform()]
    composite = CompositeTransform(transforms=transforms)
    result = composite(sample_data)

    np.testing.assert_array_equal(result["observation"], sample_data["observation"] + 2)
    np.testing.assert_array_equal(result["action"], sample_data["action"] + 2)


def test_compose_empty_list():
    """Test composing an empty list of transforms."""
    composite = compose([])
    data = {"x": np.array([1, 2, 3])}
    result = composite(data)
    np.testing.assert_array_equal(result["x"], data["x"])


# ==============================================================================
# Transform Class Tests
# ==============================================================================


def test_normalize_callable(norm_stats, sample_data):
    transform = Normalize(norm_stats=norm_stats)
    result = transform(sample_data)
    assert isinstance(result, dict)


def test_unnormalize_callable(norm_stats, sample_data):
    transform = Unnormalize(norm_stats=norm_stats)
    result = transform(sample_data)
    assert isinstance(result, dict)


def test_resize_transform_callable(sample_data):
    transform = ResizeTransform(height=128, width=128)
    result = transform(sample_data)
    assert isinstance(result, dict)


def test_normalize_is_frozen(norm_stats):
    transform = Normalize(norm_stats=norm_stats)
    with pytest.raises(AttributeError):
        transform.norm_stats = {}


def test_unnormalize_is_frozen(norm_stats):
    transform = Unnormalize(norm_stats=norm_stats)
    with pytest.raises(AttributeError):
        transform.norm_stats = {}


def test_resize_is_frozen():
    transform = ResizeTransform(height=224, width=224)
    with pytest.raises(AttributeError):
        transform.height = 128


# ==============================================================================
# NormStats Tests
# ==============================================================================


def test_norm_stats_fields():
    stats = NormStats(
        mean=np.array([0.0]),
        std=np.array([1.0]),
        q01=np.array([-1.0]),
        q99=np.array([1.0]),
    )
    np.testing.assert_array_equal(stats.mean, np.array([0.0]))
    np.testing.assert_array_equal(stats.std, np.array([1.0]))
    np.testing.assert_array_equal(stats.q01, np.array([-1.0]))
    np.testing.assert_array_equal(stats.q99, np.array([1.0]))


def test_norm_stats_optional_quantiles():
    stats = NormStats(
        mean=np.array([0.0]),
        std=np.array([1.0]),
        q01=None,
        q99=None,
    )
    assert stats.q01 is None
    assert stats.q99 is None


# ==============================================================================
# Config Build Tests
# ==============================================================================


def test_pi0_config_build(norm_stats):
    config = Pi0TransformsConfig()
    input_transforms, output_transforms = config.build(norm_stats)

    assert len(input_transforms) == 1
    assert len(output_transforms) == 1
    assert isinstance(input_transforms[0], Normalize)
    assert isinstance(output_transforms[0], Unnormalize)


def test_pi05_config_build(norm_stats):
    config = Pi05TransformsConfig()
    input_transforms, output_transforms = config.build(norm_stats)

    assert len(input_transforms) == 1
    assert len(output_transforms) == 1
    assert isinstance(input_transforms[0], Normalize)
    assert isinstance(output_transforms[0], Unnormalize)


def test_pi05_config_custom_image_settings():
    config = Pi05TransformsConfig(
        image_keys=["image1", "image2"],
        image_height=128,
        image_width=256,
    )
    assert config.image_keys == ["image1", "image2"]
    assert config.image_height == 128
    assert config.image_width == 256


# ==============================================================================
# Factory Tests
# ==============================================================================


def test_make_transforms_pi0(norm_stats):
    config = Pi0TransformsConfig()
    input_transforms, output_transforms = make_transforms(config, norm_stats)

    assert len(input_transforms) >= 1
    assert len(output_transforms) >= 1


def test_make_transforms_pi05(norm_stats):
    config = Pi05TransformsConfig()
    input_transforms, output_transforms = make_transforms(config, norm_stats)

    assert len(input_transforms) >= 1
    assert len(output_transforms) >= 1


# ==============================================================================
# Normalization Correctness Tests
# ==============================================================================


def test_normalize_computes_correct_values():
    """Verify (x - mean) / std formula."""
    stats = {
        "x": NormStats(
            mean=np.array([10.0, 20.0]),
            std=np.array([2.0, 5.0]),
            q01=None,
            q99=None,
        )
    }
    data = {"x": np.array([12.0, 30.0])}
    transform = Normalize(norm_stats=stats)

    result = transform(data)

    # (12 - 10) / 2 = 1.0, (30 - 20) / 5 = 2.0
    expected = np.array([1.0, 2.0])
    np.testing.assert_allclose(result["x"], expected)


def test_unnormalize_computes_correct_values():
    """Verify x * std + mean formula."""
    stats = {
        "x": NormStats(
            mean=np.array([10.0, 20.0]),
            std=np.array([2.0, 5.0]),
            q01=None,
            q99=None,
        )
    }
    data = {"x": np.array([1.0, 2.0])}
    transform = Unnormalize(norm_stats=stats)

    result = transform(data)

    # 1.0 * 2 + 10 = 12.0, 2.0 * 5 + 20 = 30.0
    expected = np.array([12.0, 30.0])
    np.testing.assert_allclose(result["x"], expected)


def test_normalize_unnormalize_roundtrip():
    """Normalize then unnormalize returns original."""
    stats = {
        "x": NormStats(
            mean=np.array([5.0, 10.0, 15.0]),
            std=np.array([1.0, 2.0, 3.0]),
            q01=None,
            q99=None,
        )
    }
    original = np.array([7.5, 14.0, 21.0])
    data = {"x": original.copy()}

    normalize = Normalize(norm_stats=stats)
    unnormalize = Unnormalize(norm_stats=stats)

    normalized = normalize(data)
    result = unnormalize(normalized)

    np.testing.assert_allclose(result["x"], original)


def test_normalize_skips_missing_keys():
    """Keys not in norm_stats are unchanged."""
    stats = {
        "x": NormStats(
            mean=np.array([0.0]),
            std=np.array([1.0]),
            q01=None,
            q99=None,
        )
    }
    original_y = np.array([100.0, 200.0])
    data = {"x": np.array([5.0]), "y": original_y.copy()}

    transform = Normalize(norm_stats=stats)
    result = transform(data)

    # y should be unchanged since it's not in norm_stats
    np.testing.assert_array_equal(result["y"], original_y)


def test_normalize_handles_zero_std():
    """Epsilon prevents division by zero."""
    stats = {
        "x": NormStats(
            mean=np.array([5.0]),
            std=np.array([0.0]),  # zero std
            q01=None,
            q99=None,
        )
    }
    data = {"x": np.array([10.0])}

    transform = Normalize(norm_stats=stats)
    result = transform(data)

    # (10 - 5) / (0 + 1e-8) = 5e8
    expected = np.array([5.0 / 1e-8])
    np.testing.assert_allclose(result["x"], expected)
