"""OmniVoice TRT-LLM runtime wrapper.

Loads a TRT engine for the Qwen3 transformer backbone and runs it via
tensorrt_llm Session (same pattern as F5-TTS). Used by model.py to
monkey-patch OmniVoice.llm.forward.
"""

import json
import os
from functools import wraps
from typing import Optional

import tensorrt as trt
import tensorrt_llm
import torch
from tensorrt_llm._utils import str_dtype_to_torch, trt_dtype_to_torch
from tensorrt_llm.runtime.session import Session


def cuda_stream_guard(func):
    """Sync external stream and set current stream to the session's stream."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        external_stream = torch.cuda.current_stream()
        if external_stream != self.stream:
            external_stream.synchronize()
            torch.cuda.set_stream(self.stream)
        ret = func(self, *args, **kwargs)
        if external_stream != self.stream:
            self.stream.synchronize()
            torch.cuda.set_stream(external_stream)
        return ret

    return wrapper


class OmniVoiceTRTLLM:
    """TRT-LLM accelerated OmniVoice transformer backbone."""

    def __init__(
        self,
        trtllm_config: dict,
        tllm_model_dir: str,
        model_dir: str,
        device: torch.device = torch.device("cuda"),
        stream: Optional[torch.cuda.Stream] = None,
        debug: bool = False,
    ):
        self.device = device
        self.dtype = trtllm_config["pretrained_config"]["dtype"]
        self.debug = debug

        # CUDA stream
        self.stream = stream or torch.cuda.Stream(self.device)
        torch.cuda.set_stream(self.stream)

        # Load TRT engine via Session
        engine_file = os.path.join(tllm_model_dir, "rank0.engine")
        with open(engine_file, "rb") as f:
            engine_buffer = f.read()
        self.session = Session.from_serialized_engine(engine_buffer)

        # Validate tensor names
        expected = {"hidden_states", "rope_cos", "rope_sin", "input_lengths", "output"}
        found = {
            self.session.engine.get_tensor_name(i)
            for i in range(self.session.engine.num_io_tensors)
        }
        if expected != found:
            raise RuntimeError(
                f"Engine tensor mismatch — expected {expected}, found {found}"
            )

        # Model config for RoPE
        with open(os.path.join(model_dir, "config.json")) as f:
            hf_config = json.load(f)
        llm_config = hf_config.get("llm_config", {})
        self.head_dim = llm_config["head_dim"]
        rope_theta = llm_config.get("rope_parameters", {}).get(
            "rope_theta", 1000000
        )

        # Precompute RoPE
        self.rope_cos, self.rope_sin = self._precompute_rope(4096, rope_theta)

        # I/O buffers
        self.inputs = {}
        self.outputs = {}

    def _precompute_rope(self, max_seq_len, rope_theta):
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                / self.head_dim
            )
        )
        freqs = torch.outer(
            torch.arange(max_seq_len, dtype=torch.float32), inv_freq
        )
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().half().to(self.device)
        sin = emb.sin().half().to(self.device)
        return cos, sin

    def _tensor_dtype(self, name):
        return trt_dtype_to_torch(self.session.engine.get_tensor_dtype(name))

    def _setup(self, batch_size, seq_len):
        for i in range(self.session.engine.num_io_tensors):
            name = self.session.engine.get_tensor_name(i)
            if self.session.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = list(self.session.engine.get_tensor_shape(name))
                shape[0] = batch_size
                shape[1] = seq_len
                self.outputs[name] = torch.empty(
                    shape, dtype=self._tensor_dtype(name), device=self.device
                )

    @cuda_stream_guard
    def forward_trt(self, hidden_states, input_lengths):
        """Run a single forward pass through the TRT engine.

        Args:
            hidden_states: [B, S, H] — input embeddings
            input_lengths: [B] int32 — valid lengths for attention masking
        Returns:
            output: [B, S, H] float16
        """
        B, S = hidden_states.shape[0], hidden_states.shape[1]
        input_type = str_dtype_to_torch(self.dtype)

        self._setup(B, S)

        rope_cos = self.rope_cos[:S].unsqueeze(0).expand(B, -1, -1).contiguous()
        rope_sin = self.rope_sin[:S].unsqueeze(0).expand(B, -1, -1).contiguous()

        self.inputs = {
            "hidden_states": hidden_states.to(input_type).contiguous(),
            "rope_cos": rope_cos.to(input_type),
            "rope_sin": rope_sin.to(input_type),
            "input_lengths": input_lengths.to(torch.int32).contiguous(),
        }

        self.session.set_shapes(self.inputs)
        ok = self.session.run(self.inputs, self.outputs, self.stream.cuda_stream)
        assert ok, "TRT engine execution failed"

        out = self.outputs["output"]

        if self.debug and not hasattr(self, "_debug_done"):
            self._debug_done = True
            print(
                f"[TRT DEBUG] input  — shape={hidden_states.shape} "
                f"mean={hidden_states.float().mean().item():.6f} "
                f"std={hidden_states.float().std().item():.6f}",
                flush=True,
            )
            print(
                f"[TRT DEBUG] output — shape={out.shape} "
                f"mean={out.float().mean().item():.6f} "
                f"std={out.float().std().item():.6f}",
                flush=True,
            )

        return out
