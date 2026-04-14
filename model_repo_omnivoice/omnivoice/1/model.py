import json
import os
import sys
import time

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack, to_dlpack

_model_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_model_dir, "_lib"))


class TritonPythonModel:
    def initialize(self, args):
        self.device = torch.device("cuda")

        parameters = json.loads(args["model_config"])["parameters"]
        for key, value in parameters.items():
            parameters[key] = value["string_value"]

        model_path = parameters["model_path"]
        load_asr = parameters.get("load_asr", "false").lower() == "true"
        self.reference_sample_rate = int(
            parameters.get("reference_audio_sample_rate", "24000")
        )
        debug = parameters.get("debug", "false").lower() == "true"

        tllm_model_dir = parameters.get("tllm_model_dir", "")
        self.use_trtllm = bool(tllm_model_dir)

        if self.use_trtllm:
            self._init_trtllm(model_path, tllm_model_dir, load_asr, debug)
        else:
            self._init_pytorch(model_path, load_asr)

    def _init_pytorch(self, model_path, load_asr):
        """Pure PyTorch mode."""
        from omnivoice import OmniVoice

        self.model = OmniVoice.from_pretrained(
            model_path,
            device_map=self.device,
            dtype=torch.float16,
            load_asr=load_asr,
        )
        self.sampling_rate = self.model.sampling_rate

    def _init_trtllm(self, model_path, tllm_model_dir, load_asr, debug):
        """TRT-LLM mode — swap LLM backbone with TRT engine."""
        from omnivoice import OmniVoice
        from omnivoice_trtllm import OmniVoiceTRTLLM

        config_file = os.path.join(tllm_model_dir, "config.json")
        with open(config_file) as f:
            trtllm_config = json.load(f)

        self.trtllm = OmniVoiceTRTLLM(
            trtllm_config=trtllm_config,
            tllm_model_dir=tllm_model_dir,
            model_dir=model_path,
            device=self.device,
            debug=debug,
        )

        self.model = OmniVoice.from_pretrained(
            model_path,
            device_map=self.device,
            dtype=torch.float16,
            load_asr=load_asr,
        )
        self.sampling_rate = self.model.sampling_rate

        if debug:
            self._validate_trt_vs_pytorch()

        # Monkey-patch only the LLM forward — generation logic stays untouched
        trtllm = self.trtllm

        def _trt_llm_forward(inputs_embeds=None, attention_mask=None, **kwargs):
            B, S, _ = inputs_embeds.shape
            if attention_mask is not None and attention_mask.dim() == 4:
                input_lengths = attention_mask[:, 0, 0, :].sum(dim=-1).to(torch.int32)
            else:
                input_lengths = torch.full(
                    (B,), S, dtype=torch.int32, device=inputs_embeds.device
                )
            hidden_states = trtllm.forward_trt(inputs_embeds, input_lengths)
            from transformers.modeling_outputs import BaseModelOutputWithPast
            return BaseModelOutputWithPast(last_hidden_state=hidden_states)

        self.model.llm.forward = _trt_llm_forward

        # Note: can't free LLM weights — OmniVoice still accesses
        # llm.model.embed_tokens via get_input_embeddings()

    def _validate_trt_vs_pytorch(self):
        """Compare TRT engine vs PyTorch LLM output for the same input."""
        S, H = 32, self.model.llm.config.hidden_size
        test_embeds = torch.randn(1, S, H, dtype=torch.float16, device=self.device)
        test_mask = torch.ones(1, 1, S, S, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            pt_out = self.model.llm(
                inputs_embeds=test_embeds,
                attention_mask=test_mask,
                return_dict=True,
            )[0].float()
            trt_out = self.trtllm.forward_trt(
                test_embeds,
                torch.tensor([S], dtype=torch.int32, device=self.device),
            ).float()

        cos_sim = torch.nn.functional.cosine_similarity(
            pt_out.flatten().unsqueeze(0), trt_out.flatten().unsqueeze(0)
        ).item()
        max_diff = (pt_out - trt_out).abs().max().item()

        print(
            f"[VALIDATE] TRT vs PyTorch: cos_sim={cos_sim:.6f} "
            f"max_diff={max_diff:.6f}",
            flush=True,
        )
        if cos_sim < 0.9:
            print("[VALIDATE] WARNING: TRT output differs significantly!", flush=True)

    def _parse_string(self, request, name):
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return None
        s = tensor.as_numpy()[0][0].decode("utf-8")
        return s if s.strip() else None

    def execute(self, requests):
        logger = pb_utils.Logger
        t_total_start = time.perf_counter()

        batch = len(requests)
        responses = []
        audio_duration = 0

        for i in range(batch):
            request = requests[i]
            t_start = time.perf_counter()

            target_text = self._parse_string(request, "target_text")
            if not target_text:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError("target_text is required")
                    )
                )
                continue

            ref_text = self._parse_string(request, "reference_text")
            language = self._parse_string(request, "language")
            instruct = self._parse_string(request, "instruct")

            ref_audio = None
            wav_tensor = pb_utils.get_input_tensor_by_name(
                request, "reference_wav"
            )
            if wav_tensor is not None:
                wav = from_dlpack(wav_tensor.to_dlpack())
                wav_lens = pb_utils.get_input_tensor_by_name(
                    request, "reference_wav_len"
                )
                if wav_lens is not None:
                    wav_len = from_dlpack(wav_lens.to_dlpack()).squeeze().item()
                    wav = wav[:, :wav_len]
                ref_audio = (wav.cpu().numpy(), self.reference_sample_rate)

            kwargs = {"text": target_text}
            if language is not None:
                kwargs["language"] = language
            if instruct is not None:
                kwargs["instruct"] = instruct
            if ref_audio is not None:
                kwargs["ref_audio"] = ref_audio
                if ref_text is not None:
                    kwargs["ref_text"] = ref_text

            try:
                audios = self.model.generate(**kwargs)
            except Exception as e:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f"{type(e).__name__}: {e}")
                    )
                )
                continue

            audio = audios[0]
            audio_duration = len(audio) / self.sampling_rate
            t_infer = (time.perf_counter() - t_start) * 1000

            audio_out = torch.from_numpy(audio).unsqueeze(0).float()
            audio_pb = pb_utils.Tensor.from_dlpack(
                "waveform", to_dlpack(audio_out)
            )
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[audio_pb])
            )

            mode = "trtllm" if self.use_trtllm else "pytorch"
            logger.log_info(
                f"[Item {i}] mode={mode} | text={len(target_text)} chars | "
                f"audio={audio_duration:.2f}s | infer={t_infer:.1f}ms"
            )

        t_total = (time.perf_counter() - t_total_start) * 1000
        rtf = (t_total / 1000) / audio_duration if audio_duration > 0 else 0

        logger.log_info(
            f"[Timing] batch={batch} | total={t_total:.1f}ms | "
            f"audio={audio_duration:.2f}s | RTF={rtf:.3f}x"
        )

        return responses
