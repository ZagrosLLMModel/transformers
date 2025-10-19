# coding=utf-8
# Copyright 2025 The HumanV Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HumanV model configuration"""

from typing import Optional

from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...modeling_rope_utils import RopeParameters, rope_config_validation, standardize_rope_params
from ...utils import logging


logger = logging.get_logger(__name__)


class HumanVConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HumanVModel`]. It is used to instantiate a
    HumanV model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the HumanV model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`HumanVModel`]
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 2816):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `8`.
        head_dim (`int`, *optional*, defaults to 64):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionaty should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        use_sliding_window (`bool`, *optional*, defaults to `True`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 20):
            The number of layers using full attention. The first `max_window_layers` layers will use full attention, while any
            additional layer afterwards will use SWA (Sliding Window Attention).
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import HumanVModel, HumanVConfig

    >>> # Initializing a HumanV style configuration
    >>> configuration = HumanVConfig()

    >>> # Initializing a model from the HumanV-0.6B style configuration
    >>> model = HumanVModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "humanv"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `HumanV`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 1024,
        intermediate_size: int = 2816,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: Optional[int] = 8,
        head_dim: int = 64,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        rope_parameters: Optional[RopeParameters | dict[RopeParameters]] = None,
        attention_bias: bool = False,
        use_sliding_window: bool = True,
        sliding_window: Optional[int] = 4096,
        max_window_layers: Optional[int] = 20,
        layer_types: Optional[list[str]] = None,
        attention_dropout: float = 0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility, but adjusted for GQA
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads // 2

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Try to set `rope_scaling` if available, otherwise use `rope_parameters`
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or rope_parameters

        self.layer_types = layer_types
        if self.layer_types is None:
            if self.use_sliding_window and self.max_window_layers is not None:
                self.layer_types = [
                    "sliding_attention"
                    if i >= self.max_window_layers
                    else "full_attention"
                    for i in range(self.num_hidden_layers)
                ]
            else:
                self.layer_types = ["full_attention"] * self.num_hidden_layers
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        # Validate the correctness of rotary position embeddings parameters
        rope_theta = kwargs.get("rope_theta", 10000000.0)  # Optimized to 1e7 for better long-context performance
        standardize_rope_params(self, rope_theta=rope_theta)
        rope_config_validation(self)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["HumanVConfig"]