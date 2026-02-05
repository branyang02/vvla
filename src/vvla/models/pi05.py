"""
uv run python src/vvla/models/pi05.py
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

from vvla.models.base_model import BaseModel


@dataclass
class SiglipVisionConfig:
    attention_dropout: float = 0.0
    # hidden_act: str = "gelu_pytorch_tanh"
    hidden_size: int = 1152
    image_size: int = 224
    intermediate_size: int = 4304
    layer_norm_eps: float = 1e-6
    # model_type: str = "siglip_vision_model"
    num_attention_heads: int = 16
    num_channels: int = 3
    num_hidden_layers: int = 27
    patch_size: int = 14
    projection_dim: int = 2048
    # projector_hidden_act: str = "gelu_fast"
    # torch_dtype: torch.dtype = torch.float32
    # transformers_version: str = "4.53.2"
    # vision_use_head: bool = False
    vocab_size: int = 257152


@dataclass
class GemmaTextConfig:
    adarms_cond_dim: int | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    # bos_token_id: int = 2
    # eos_token_id: int = 1
    head_dim: int = 256
    # hidden_act: str = "gelu_pytorch_tanh"
    # hidden_activation: str = "gelu_pytorch_tanh"
    hidden_size: int = 2048
    # initializer_range: float = 0.02
    intermediate_size: int = 16384
    max_position_embeddings: int = 8192
    # model_type: str = "gemma"
    num_attention_heads: int = 8
    num_hidden_layers: int = 18
    # num_image_tokens: int = 256
    num_key_value_heads: int = 1
    pad_token_id: int = 0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    # torch_dtype: str = "float32"
    use_adarms: bool = False
    use_cache: bool = True
    vocab_size: int = 257152


@dataclass
class GemmaExpertConfig:
    adarms_cond_dim: int | None = 1024
    attention_bias: bool = False
    attention_dropout: float = 0.0
    # bos_token_id: int = 2
    # eos_token_id: int = 1
    head_dim: int = 256
    # hidden_act: str = "gelu_pytorch_tanh"
    # hidden_activation: str = "gelu_pytorch_tanh"
    hidden_size: int = 1024
    # initializer_range: float = 0.02
    intermediate_size: int = 4096
    max_position_embeddings: int = 8192
    # model_type: str = "gemma"
    num_attention_heads: int = 8
    num_hidden_layers: int = 18
    num_key_value_heads: int = 1
    pad_token_id: int = 0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    # torch_dtype: str = "float32"
    # transformers_version: str = "4.53.2"
    use_adarms: bool = True
    use_cache: bool = True
    vocab_size: int = 257152


@dataclass
class Pi05ModelConfig:
    vlm_siglip_config: SiglipVisionConfig = field(default_factory=SiglipVisionConfig)
    vlm_text_config: GemmaTextConfig = field(default_factory=GemmaTextConfig)
    action_expert_config: GemmaExpertConfig = field(default_factory=GemmaExpertConfig)


##########################################################################################


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        print(
            "`sdpa` attention does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    # query = query.contiguous()
    # key = key.contiguous()
    # value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        # The last condition is for encoder (decoder) models which specify this by passing their own `is_causal` flag
        # This is mainly due to those models having mixed implementations for encoder, decoder, and encoder-decoder attns
        is_causal = (
            query.shape[2] > 1
            and attention_mask is None
            and getattr(module, "is_causal", True)
        )

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


def torch_int(x):
    """
    Casts an input to a torch int64 tensor if we are in a tracing context, otherwise to a Python int.
    """
    return (
        x.to(torch.int64)
        if torch.jit.is_tracing() and isinstance(x, torch.Tensor)
        else int(x)
    )


@dataclass
class ModelOutput:
    last_hidden_state: None = None
    hidden_states: None = None
    past_key_values: None = None


class PytorchGELUTanh(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.gelu(input, approximate="tanh")


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = PytorchGELUTanh()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        """Input shape: Batch x Time x Channel"""

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        keys = keys.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        values = values.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_output, attn_weights = sdpa_attention_forward(
            self,
            queries,
            keys,
            values,
            attention_mask=None,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )

        attn_output = attn_output.reshape(
            batch_size, seq_length, embed_dim
        ).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, None


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
    ):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
            )

            hidden_states = layer_outputs[0]

        return ModelOutput(
            last_hidden_state=hidden_states,
        )


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:  # noqa: FBT002
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(
            pixel_values.to(dtype=target_dtype)
        )  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values,
    ):
        hidden_states = self.embeddings(pixel_values)
        # Convert to bfloat16 if the encoder uses bfloat16
        if (
            len(self.encoder.layers) > 0
            and self.encoder.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
        ):
            hidden_states = hidden_states.to(torch.bfloat16)

        encoder_outputs = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return ModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
        )


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values):
        return self.vision_model(pixel_values=pixel_values)


##########################################################################################


def gated_residual(x, y, gate):
    """
    Applies gated residual connection with optional gate parameter.

    Args:
        x: Input tensor (residual)
        y: Output tensor to be added
        gate: Optional gate tensor to modulate the addition

    Returns:
        x + y if gate is None, otherwise x + y * gate
    """
    if x is None and y is None:
        return None
    if x is None or y is None:
        return x if x is not None else y
    if gate is None:
        return x + y
    return x + y * gate


def compute_default_rope_parameters(
    config=None,
    device=None,
    seq_len=None,
    **rope_kwargs,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = (
            config.partial_rotary_factor
            if hasattr(config, "partial_rotary_factor")
            else 1.0
        )
        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    return inv_freq, attention_factor


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is None:
        raise ValueError(
            "Attention mask is required for causal attention to ensure proper masking of future tokens."
        )

    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class GemmaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: GemmaTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value=None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool = False,  # noqa: FBT001, FBT002
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # Use cache if provided
        if past_key_value is not None:
            if use_cache:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            else:
                key_states = torch.cat(
                    [past_key_value[self.layer_idx][0], key_states], dim=2
                )
                value_states = torch.cat(
                    [past_key_value[self.layer_idx][1], value_states], dim=2
                )

        attention_interface = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = PytorchGELUTanh()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        cond_dim = (
            getattr(config, "adarms_cond_dim", None)
            if getattr(config, "use_adarms", False)
            else None
        )
        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            cond_dim=cond_dim,
            config=config,
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            cond_dim=cond_dim,
            config=config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: torch.Tensor | None = None,
        use_cache: bool | None = False,  # noqa: FBT002
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor]
        | None = None,  # necessary, but kept here for BC
        adarms_cond: torch.Tensor | None = None,
    ):
        residual = hidden_states
        hidden_states, gate = self.input_layernorm(hidden_states, adarms_cond)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,  # originally passed in as kwargs to self.self_attn
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = gated_residual(residual, hidden_states, gate)

        # Fully Connected
        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(hidden_states, adarms_cond)
        hidden_states = self.mlp(hidden_states)
        hidden_states = gated_residual(residual, hidden_states, gate)

        outputs = (hidden_states,)
        return outputs


class GemmaRMSNorm(nn.Module):
    def __init__(
        self, dim: int, eps: float = 1e-6, cond_dim: int | None = None, config=None
    ):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim
        self.config = config

        # Dense layer for adaptive normalization (if cond_dim is provided)
        if cond_dim is not None:
            # self.dense = nn.Linear(cond_dim, dim * 3, bias=True, dtype=torch.bfloat16)
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            # Initialize with zeros (matches source implementation)
            nn.init.zeros_(self.dense.weight)
        else:
            self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))
            self.dense = None

    def _norm(self, x):
        # Compute variance in float32 (like the source implementation)
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        # Compute normalization in float32
        return x * torch.rsqrt(var + self.eps)

    def forward(self, x, cond=None):
        dtype = x.dtype  # original dtype, could be half-precision
        normed_inputs = self._norm(x)

        if cond is None or self.dense is None:
            # regular RMSNorm
            # scale by learned parameter in float32 (matches source implementation)
            normed_inputs = normed_inputs * (1.0 + self.weight.float())
            return normed_inputs.to(
                dtype
            ), None  # return in original dtype with None gate

        # adaptive RMSNorm (if cond is provided and dense layer exists)
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(
                f"Expected cond dimension {self.cond_dim}, got {cond.shape[-1]}"
            )

        # self.dense.to(dtype=torch.bfloat16).to(dtype=torch.float32)
        modulation = self.dense(cond)
        # Reshape modulation to broadcast properly: [batch, 1, features] for [batch, seq, features]
        if len(x.shape) == 3:  # [batch, seq, features]
            modulation = modulation.unsqueeze(1)

        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)

        # Apply adaptive normalization: use model weight dtype to ensure compatibility
        # model_dtype = self.dense.weight.dtype  # Use the model's dtype (bfloat16)
        # scale = scale.to(model_dtype)
        # shift = shift.to(model_dtype)
        # gate = gate.to(model_dtype)
        # normed_inputs = normed_inputs.to(model_dtype)  # Convert normed_inputs to model dtype

        normed_inputs = normed_inputs * (1 + scale.to(torch.float32)) + shift.to(
            torch.float32
        )

        return normed_inputs.to(dtype), gate.to(dtype)


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, config: GemmaTextConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = compute_default_rope_parameters

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaTextConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                GemmaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        cond_dim = (
            getattr(config, "adarms_cond_dim", None)
            if getattr(config, "use_adarms", False)
            else None
        )
        self.norm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            cond_dim=cond_dim,
            config=config,
        )
        self.rotary_emb = GemmaRotaryEmbedding(config=config)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        adarms_cond: torch.Tensor | None = None,
    ) -> ModelOutput:
        """
        adarms_cond (`torch.Tensor` of shape `(batch_size, cond_dim)`, *optional*):
            Condition for ADARMS.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )  # 0 in prefix pass
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # embed positions
        hidden_states = inputs_embeds
        # Convert to bfloat16 if the first layer uses bfloat16
        if (
            len(self.layers) > 0
            and self.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
        ):
            hidden_states = hidden_states.to(torch.bfloat16)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # normalized
        # Gemma downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(  # noqa: F841
            self.config.hidden_size**0.5, dtype=hidden_states.dtype
        )
        # hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                adarms_cond=adarms_cond,
            )

            hidden_states = layer_outputs[
                0
            ]  # TODO: Since we are doing a weird tuple thing in decoder layer

        hidden_states, _ = self.norm(hidden_states, adarms_cond)

        return ModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


##########################################################################################


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: Pi05ModelConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.vlm_siglip_config.hidden_size,
            config.vlm_siglip_config.projection_dim,
            bias=True,
        )

    def forward(self, image_features):
        hidden_states = self.linear(image_features)

        return hidden_states


class PaliGemmaVLM(nn.Module):
    def __init__(self, config: Pi05ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vlm_siglip_config)
        self.language_model = GemmaModel(config=config.vlm_text_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)

    def get_image_features(self, pixel_values: torch.FloatTensor):
        image_outputs = self.vision_tower(pixel_values)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features

    def forward(self, x):
        pass


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(self, config: Pi05ModelConfig):
        super().__init__()
        self.config = config

        self.vlm = PaliGemmaVLM(config)
        self.action_expert = GemmaModel(config.action_expert_config)

    def forward(x):
        pass


class Pi05(BaseModel):
    """A simple MLP model for Pi0 environments."""

    def __init__(self, config: Pi05ModelConfig):
        super().__init__()
        self.config = config
        self.paligemma_with_expert = PaliGemmaWithExpertModel(config)

        self.action_in_proj = nn.Linear(32, config.action_expert_config.hidden_size)
        self.action_out_proj = nn.Linear(config.action_expert_config.hidden_size, 32)
        self.time_mlp_in = nn.Linear(
            config.action_expert_config.hidden_size,
            config.action_expert_config.hidden_size,
        )
        self.time_mlp_out = nn.Linear(
            config.action_expert_config.hidden_size,
            config.action_expert_config.hidden_size,
        )

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        pass


if __name__ == "__main__":
    config = Pi05ModelConfig()
    model = Pi05(config)
    print(model)
