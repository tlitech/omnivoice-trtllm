"""TRT-LLM model definition for OmniVoice's Qwen3 backbone.

The TRT engine handles only the 28 Qwen3 transformer layers + final RMSNorm.
Audio embeddings, audio heads, and the iterative unmasking loop stay in PyTorch.

Engine I/O:
    Input:  hidden_states [B, S, 1024], attention_mask (via input_lengths [B]),
            rope_cos [B, S, 128], rope_sin [B, S, 128]
    Output: output [B, S, 1024]
"""
from __future__ import annotations

from collections import OrderedDict

import tensorrt as trt

from ..._utils import str_dtype_to_trt
from ...functional import Tensor, concat
from ...module import ModuleList
from ..modeling_utils import PretrainedConfig, PretrainedModel
from .modules import Qwen3Block, RMSNorm


class OmniVoice(PretrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.dtype = str_dtype_to_trt(config.dtype)

        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size

        self.layers = ModuleList(
            [
                Qwen3Block(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    num_kv_heads=config.num_kv_heads,
                    head_dim=config.head_dim,
                    intermediate_size=config.intermediate_size,
                    rms_norm_eps=config.rms_norm_eps,
                    dtype=self.dtype,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.final_norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, self.dtype
        )

    def forward(
        self,
        hidden_states,
        rope_cos,
        rope_sin,
        input_lengths,
    ):
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, rope_cos, rope_sin, input_lengths
            )

        output = self.final_norm(hidden_states)
        output.mark_output("output", self.dtype)
        return output

    def prepare_inputs(self, **kwargs):
        max_batch_size = kwargs.get("max_batch_size", 16)
        hidden_size = self.hidden_size
        head_dim = 128
        max_seq_len = 4096

        batch_size_range = [1, max_batch_size // 2, max_batch_size]
        seq_len_range = [1, max_seq_len // 2, max_seq_len]

        hidden_states = Tensor(
            name="hidden_states",
            dtype=self.dtype,
            shape=[-1, -1, hidden_size],
            dim_range=OrderedDict(
                [
                    ("batch_size", [batch_size_range]),
                    ("seq_len", [seq_len_range]),
                    ("hidden_size", [hidden_size]),
                ]
            ),
        )

        rope_cos = Tensor(
            name="rope_cos",
            dtype=self.dtype,
            shape=[-1, -1, head_dim],
            dim_range=OrderedDict(
                [
                    ("batch_size", [batch_size_range]),
                    ("seq_len", [seq_len_range]),
                    ("head_dim", [head_dim]),
                ]
            ),
        )

        rope_sin = Tensor(
            name="rope_sin",
            dtype=self.dtype,
            shape=[-1, -1, head_dim],
            dim_range=OrderedDict(
                [
                    ("batch_size", [batch_size_range]),
                    ("seq_len", [seq_len_range]),
                    ("head_dim", [head_dim]),
                ]
            ),
        )

        input_lengths = Tensor(
            name="input_lengths",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict(
                [("batch_size", [batch_size_range])]
            ),
        )

        return {
            "hidden_states": hidden_states,
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
            "input_lengths": input_lengths,
        }
