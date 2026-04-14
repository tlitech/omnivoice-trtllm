from pathlib import Path

import modal


LOCAL_DIR = Path(__file__).parent
scripts_dir = LOCAL_DIR / "scripts"
patch_dir = LOCAL_DIR / "patch"
model_repo_dir = LOCAL_DIR / "model_repo_omnivoice"
client_http_path = LOCAL_DIR / "client_http.py"
voices_dir = LOCAL_DIR / "voices"

scripts_remote_dir = Path("/root/scripts")
patch_remote_dir = Path("/root/patch")
model_repo_remote_dir = Path("/root/model_repo_omnivoice")
client_http_remote_path = Path("/root/client_http.py")
voices_remote_dir = Path("/root/voices")

trtllm_image = modal.Image.from_registry(
    "nvcr.io/nvidia/tritonserver:25.04-trtllm-python-py3",
    add_python="3.12",
).entrypoint([])

# tensorrt-llm is pre-installed in tritonserver:25.04 — only install extra deps
trtllm_image = trtllm_image.run_commands(
    "/usr/bin/python3 -m pip install "
    "tritonclient[grpc] "
    "torchaudio "
    "librosa soundfile pydub accelerate "
    "hf-transfer "
    "huggingface_hub ",
)

trtllm_image = trtllm_image.pip_install(
    "tritonclient[grpc]",
    "torchaudio",
    "librosa",
    "soundfile",
    "pydub",
    "accelerate",
    "hf-transfer",
    "huggingface_hub",
)

# Volume for model weights and engines
volume = modal.Volume.from_name(
    "omnivoice-triton-server", create_if_missing=True
)

VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"
OMNIVOICE_MODEL_PATH = MODELS_PATH / "OmniVoice"
TRTLLM_CKPT_PATH = MODELS_PATH / "trtllm_ckpt"
TRTLLM_ENGINE_PATH = MODELS_PATH / "trtllm_engine"
OUTPUT_AUDIO_PATH = VOLUME_PATH / "output_audio"

MODEL_ID = "k2-fsa/OmniVoice"

trtllm_image = trtllm_image.env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": str(MODELS_PATH),
    }
)

# Serving image: extends trtllm_image with correct deps for Triton Python backend
serving_image = trtllm_image.run_commands(
    "/usr/bin/python3 -m pip install 'transformers>=5.3.0'",
    "/usr/bin/python3 -m pip install --force-reinstall --no-deps "
    "torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128",
    # Patch tensorrt_llm: stub AutoModelForVision2Seq (dropped in transformers 5.x)
    '/usr/bin/python3 -c "' "import pathlib,re;"
    "p=pathlib.Path('/usr/local/lib/python3.12/dist-packages/tensorrt_llm/models/gpt/convert.py');"
    "t=p.read_text();"
    r"t=re.sub(r'AutoModelForVision2Seq,\s*','',t,count=1);"  # remove from import
    "t=t.replace('AutoModelForVision2Seq','None');"  # stub remaining usages
    "p.write_text(t);"
    "compile(t,'convert.py','exec');"
    "print('PATCH OK: convert.py compiles')" '"',
)

# Add local files to both images
def _add_local_files(image):
    return (
        image.add_local_dir(scripts_dir, remote_path=str(scripts_remote_dir))
        .add_local_dir(patch_dir, remote_path=str(patch_remote_dir))
        .add_local_dir(model_repo_dir, remote_path=str(model_repo_remote_dir))
        .add_local_file(client_http_path, remote_path=str(client_http_remote_path))
        .add_local_dir(voices_dir, remote_path=str(voices_remote_dir))
    )


trtllm_image = _add_local_files(trtllm_image)
serving_image = _add_local_files(serving_image)

N_GPUS = 1
GPU_CONFIG = f"L4:{N_GPUS}"
MINUTES = 60

app = modal.App("omnivoice-triton-server")


# ---------------------------------------------------------------------------
# Stage 0: Download model
# ---------------------------------------------------------------------------
@app.function(
    image=trtllm_image,
    volumes={VOLUME_PATH: volume},
    timeout=60 * MINUTES,
)
def download_model():
    from huggingface_hub import snapshot_download

    print(f"Downloading OmniVoice model: {MODEL_ID}")
    snapshot_download(MODEL_ID, local_dir=OMNIVOICE_MODEL_PATH)
    volume.commit()
    print(f"Model downloaded to {OMNIVOICE_MODEL_PATH}")


# ---------------------------------------------------------------------------
# Stage 1: Build TRT-LLM engine
# ---------------------------------------------------------------------------
@app.function(
    image=trtllm_image,
    gpu=GPU_CONFIG,
    volumes={VOLUME_PATH: volume},
    timeout=60 * MINUTES,
)
def build_trtllm_engine():
    import os
    import shutil
    import subprocess
    import sys

    # Add /usr/bin/python3's site-packages to sys.path so imports find tensorrt_llm
    result = subprocess.run(
        ["/usr/bin/python3", "-c",
         "import site; print('\\n'.join(site.getsitepackages()))"],
        capture_output=True, text=True, check=True,
    )
    for p in result.stdout.strip().splitlines():
        if p not in sys.path:
            sys.path.insert(0, p)

    os.environ["PATH"] = "/usr/bin:" + os.environ.get("PATH", "")
    os.environ["PYTHON_EXECUTABLE"] = "/usr/bin/python3"

    # Step 1: Convert checkpoint
    print("Converting checkpoint to TRT-LLM format")
    subprocess.run(
        [
            "/usr/bin/python3",
            str(scripts_remote_dir / "convert_checkpoint.py"),
            "--model_dir",
            str(OMNIVOICE_MODEL_PATH),
            "--output_dir",
            str(TRTLLM_CKPT_PATH),
        ],
        check=True,
    )

    # Step 2: Find tensorrt_llm path (use pip show to avoid importing it)
    result = subprocess.run(
        ["/usr/bin/python3", "-m", "pip", "show", "tensorrt_llm"],
        capture_output=True, text=True, check=True,
    )
    location = [l.split(": ", 1)[1] for l in result.stdout.splitlines() if l.startswith("Location:")][0]
    trtllm_path = Path(location) / "tensorrt_llm"
    target_dir = trtllm_path / "models" / "omnivoice"

    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(patch_remote_dir / "omnivoice", target_dir)
    print(f"  Copied patch/omnivoice/ to {target_dir}")

    # Step 2b: Register OmniVoice in tensorrt_llm MODEL_MAP
    init_file = trtllm_path / "models" / "__init__.py"
    init_content = init_file.read_text()
    register_line = "\nfrom .omnivoice.model import OmniVoice\nMODEL_MAP['OmniVoice'] = OmniVoice\n"
    if "OmniVoice" not in init_content:
        with open(init_file, "a") as f:
            f.write(register_line)
        print("  Registered OmniVoice in MODEL_MAP")

    # Step 3: Build TRT-LLM engine
    print("Building TRT-LLM engine")
    subprocess.run(
        [
            "trtllm-build",
            "--checkpoint_dir",
            str(TRTLLM_CKPT_PATH),
            "--max_batch_size",
            "16",
            "--output_dir",
            str(TRTLLM_ENGINE_PATH),
            "--remove_input_padding",
            "disable",
        ],
        check=True,
    )

    volume.commit()
    print(f"Engine built at {TRTLLM_ENGINE_PATH}")


# ---------------------------------------------------------------------------
# Stage 2: Start Triton + test
# ---------------------------------------------------------------------------
@app.function(
    image=serving_image,
    gpu=GPU_CONFIG,
    volumes={VOLUME_PATH: volume},
    scaledown_window=20 * MINUTES,
)
def start_and_test_triton_server(use_trtllm: bool = True, debug: bool = False):
    import os
    import shutil
    import subprocess
    import time

    import requests

    os.environ["PATH"] = "/usr/bin:" + os.environ.get("PATH", "")
    os.environ["PYTHON_EXECUTABLE"] = "/usr/bin/python3"

    model_repo_dest = MODELS_PATH / "model_repo"

    print("\nBuilding Triton model repository")

    if model_repo_dest.exists():
        shutil.rmtree(model_repo_dest)
    shutil.copytree(model_repo_remote_dir, model_repo_dest)

    # Fill config template
    config_file = model_repo_dest / "omnivoice" / "config.pbtxt"
    trtllm_val = str(TRTLLM_ENGINE_PATH) if use_trtllm else ""
    debug_val = "true" if debug else "false"
    subprocess.run(
        [
            "python3",
            str(scripts_remote_dir / "fill_template.py"),
            "-i",
            str(config_file),
            f"model:{OMNIVOICE_MODEL_PATH},load_asr:true,trtllm:{trtllm_val},debug:{debug_val}",
        ],
        check=True,
    )

    print(f"Model repository prepared at {model_repo_dest}")
    volume.commit()

    triton_process = subprocess.Popen(
        ["tritonserver", f"--model-repository={model_repo_dest}"]
    )

    print("Waiting for Triton server to start...")
    max_retries = 120
    server_ready = False

    for retry_count in range(1, max_retries + 1):
        try:
            response = requests.get("http://localhost:8000/v2/health/ready")
            if response.status_code == 200:
                server_ready = True
                print("Triton server is ready")
                break
        except Exception:
            pass
        time.sleep(5)
        print(f"  Checking server health... ({retry_count}/{max_retries})")

    if not server_ready:
        triton_process.kill()
        raise RuntimeError("Triton server failed to start within timeout")

    # Test with voice samples
    print("\nTesting Triton server with voice samples")
    OUTPUT_AUDIO_PATH.mkdir(parents=True, exist_ok=True)

    voices_sys = voices_remote_dir / "system"
    test_samples = sorted(
        f.stem for f in Path(voices_sys).glob("*.txt")
    )
    print(f"Found {len(test_samples)} voice samples")

    def _build_cmd(name):
        wav_path = voices_sys / f"{name}.wav"
        txt_path = voices_sys / f"{name}.txt"
        out_path = OUTPUT_AUDIO_PATH / f"{name}_gen.wav"
        lines = txt_path.read_text().strip().splitlines()

        if name.startswith("design_"):
            # Voice design: line1=instruct, line2=target
            instruct = lines[0].strip()
            target_text = lines[1].strip() if len(lines) > 1 else instruct
            cmd = [
                "python3", str(client_http_remote_path),
                "--instruct", instruct,
                "--target-text", target_text,
                "--output-audio", str(out_path),
            ]
        elif name.startswith("fine_"):
            # Fine-grained: line1=target text with special tokens, no ref audio
            target_text = lines[0].strip()
            cmd = [
                "python3", str(client_http_remote_path),
                "--target-text", target_text,
                "--output-audio", str(out_path),
            ]
        else:
            # Voice clone: line1=ref text, line2=target, needs wav
            ref_text = lines[0].strip()
            target_text = lines[1].strip() if len(lines) > 1 else ref_text
            cmd = [
                "python3", str(client_http_remote_path),
                "--reference-audio", str(wav_path),
                "--reference-text", ref_text,
                "--target-text", target_text,
                "--output-audio", str(out_path),
            ]
        return cmd, target_text

    # Warm-up: run first sample to initialize CUDA kernels / caches
    if test_samples:
        cmd, _ = _build_cmd(test_samples[0])
        if cmd:
            print("  [warm-up] running first sample...")
            subprocess.run(cmd, check=True)
            print("  [warm-up] done")

    # Benchmark all samples
    latencies = []
    for name in test_samples:
        cmd, target_text = _build_cmd(name)
        if cmd is None:
            print(f"  Skipping {name} — no txt file")
            continue

        t0 = time.perf_counter()
        subprocess.run(cmd, check=True)
        elapsed = time.perf_counter() - t0
        latencies.append((name, elapsed))
        print(f"  [{name}] {elapsed:.2f}s | {target_text[:60]}...")

    # Summary
    if latencies:
        times = [t for _, t in latencies]
        avg = sum(times) / len(times)
        print(f"\n{'='*50}")
        print(f"Samples: {len(latencies)}")
        print(f"Latency — mean={avg:.2f}s | min={min(times):.2f}s | max={max(times):.2f}s")

    volume.commit()
    print(f"All tests completed. Outputs saved to {OUTPUT_AUDIO_PATH}")


# ---------------------------------------------------------------------------
# Production serve
# ---------------------------------------------------------------------------
@app.function(
    image=serving_image,
    gpu=GPU_CONFIG,
    volumes={VOLUME_PATH: volume},
    scaledown_window=20 * MINUTES,
)
@modal.concurrent(max_inputs=10)
@modal.web_server(port=8000, startup_timeout=10 * MINUTES)
def serve():
    import os
    import shutil
    import subprocess

    os.environ["PATH"] = "/usr/bin:" + os.environ.get("PATH", "")
    os.environ["PYTHON_EXECUTABLE"] = "/usr/bin/python3"

    model_repo_dest = MODELS_PATH / "model_repo"

    if model_repo_dest.exists():
        shutil.rmtree(model_repo_dest)
    shutil.copytree(model_repo_remote_dir, model_repo_dest)

    config_file = model_repo_dest / "omnivoice" / "config.pbtxt"
    # Use TRT-LLM engine if available, otherwise fall back to PyTorch
    trtllm_val = str(TRTLLM_ENGINE_PATH) if TRTLLM_ENGINE_PATH.exists() else ""
    subprocess.run(
        [
            "python3",
            str(scripts_remote_dir / "fill_template.py"),
            "-i",
            str(config_file),
            f"model:{OMNIVOICE_MODEL_PATH},load_asr:true,trtllm:{trtllm_val},debug:false",
        ],
        check=True,
    )

    print(f"Model repository prepared at {model_repo_dest}")
    volume.commit()

    subprocess.Popen(
        ["tritonserver", f"--model-repository={model_repo_dest}"]
    )


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    # # Stage 0: Download model
    # print("Stage 0: Downloading OmniVoice model")
    # download_model.remote()
    # print("Model downloaded")

    # # Stage 1: Build TRT-LLM engine
    # print("\nStage 1: Building TRT-LLM engine")
    # build_trtllm_engine.remote()
    # print("Engine built")

    # Stage 2: Start Triton and test
    print("\nStage 2: Starting Triton Inference Server and testing")
    start_and_test_triton_server.remote(use_trtllm=False, debug=False)

    print("\nOmniVoice TRT-LLM serving ready!")
    print("To deploy: modal deploy omnivoice_modal.py")
    print("To test:   python test_client_http.py")
