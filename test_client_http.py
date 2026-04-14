import argparse
import time

import numpy as np
import requests
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--server-url",
        type=str,
        required=True,
        help="Address of the server (e.g. https://your-app--omnivoice-triton-server-serve.modal.run)",
    )

    parser.add_argument(
        "--reference-audio",
        type=str,
        default=None,
        help="Path to reference audio file (24kHz) for voice cloning",
    )

    parser.add_argument(
        "--reference-text",
        type=str,
        default=None,
        help="Reference text matching the audio",
    )

    parser.add_argument(
        "--target-text",
        type=str,
        default="I don't really care what you call me. "
        "I've been a silent spectator, watching species evolve, "
        "empires rise and fall. But always remember, "
        "I am mighty and enduring.",
        help="Target text to synthesize",
    )

    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language name or code",
    )

    parser.add_argument(
        "--instruct",
        type=str,
        default=None,
        help="Voice design instruction",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="omnivoice",
        help="Triton model name to request",
    )

    parser.add_argument(
        "--output-audio",
        type=str,
        default="output.wav",
        help="Path to save the output audio",
    )

    parser.add_argument(
        "--loop",
        type=int,
        default=5,
        help="Number of iterations to run",
    )

    parser.add_argument(
        "--voices-dir",
        type=str,
        default=None,
        help="Run all voice samples from this directory (e.g. voices/system). "
             "Overrides --reference-audio/--target-text.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="voices/output",
        help="Directory to save generated audio when using --voices-dir",
    )

    return parser.parse_args()


def prepare_request(
    samples=None,
    reference_text=None,
    target_text="",
    language=None,
    instruct=None,
):
    inputs = [
        {
            "name": "target_text",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [target_text],
        },
    ]

    if samples is not None:
        lengths = np.array([[len(samples)]], dtype=np.int32)
        samples_2d = samples.reshape(1, -1).astype(np.float32)
        inputs.append(
            {
                "name": "reference_wav",
                "shape": list(samples_2d.shape),
                "datatype": "FP32",
                "data": samples_2d.tolist(),
            }
        )
        inputs.append(
            {
                "name": "reference_wav_len",
                "shape": list(lengths.shape),
                "datatype": "INT32",
                "data": lengths.tolist(),
            }
        )

    if reference_text is not None:
        inputs.append(
            {
                "name": "reference_text",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [reference_text],
            }
        )

    if language is not None:
        inputs.append(
            {
                "name": "language",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [language],
            }
        )

    if instruct is not None:
        inputs.append(
            {
                "name": "instruct",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [instruct],
            }
        )

    return {"inputs": inputs}


def load_audio(wav_path, target_sample_rate=24000):
    samples, sample_rate = sf.read(wav_path)
    if sample_rate != target_sample_rate:
        from scipy.signal import resample

        num_samples = int(len(samples) * (target_sample_rate / sample_rate))
        samples = resample(samples, num_samples)
    return samples.astype(np.float32), target_sample_rate


def check_health(server_url, max_retries=30, retry_interval=2):
    health_url = f"{server_url}/v2/health/ready"
    print(f"Checking server health at {health_url}...")

    for attempt in range(1, max_retries + 1):
        try:
            rsp = requests.get(health_url, verify=False, timeout=10)
            if rsp.status_code == 200:
                print("Server is healthy and ready")
                return True
            else:
                print(
                    f"Attempt {attempt}/{max_retries}: "
                    f"status code {rsp.status_code}"
                )
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt}/{max_retries}: {e}")

        if attempt < max_retries:
            time.sleep(retry_interval)

    print(f"Server health check failed after {max_retries} attempts")
    return False


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def run_inference(url, data, iteration):
    start_time = time.time()
    rsp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=data,
        verify=False,
        params={"request_id": str(iteration)},
    )
    latency = time.time() - start_time

    if rsp.status_code != 200:
        return None, latency, 0, 0, f"Error: {rsp.status_code}"

    result = rsp.json()
    if "error" in result:
        return None, latency, 0, 0, f"Error: {result['error']}"

    audio = result["outputs"][0]["data"]
    audio = np.array(audio, dtype=np.float32)

    num_samples = len(audio)
    duration = num_samples / 24000
    size_bytes = num_samples * 4

    return audio, latency, duration, size_bytes, None


def run_single_text(args, url):
    """Run benchmark on a single target text."""
    samples = None
    if args.reference_audio is not None:
        samples, sr = load_audio(args.reference_audio, target_sample_rate=24000)

    data = prepare_request(
        samples=samples,
        reference_text=args.reference_text,
        target_text=args.target_text,
        language=args.language,
        instruct=args.instruct,
    )

    print(f"\n{'='*60}")
    print(f"Running {args.loop} iterations")
    print(f"Target text: {args.target_text[:80]}...")
    mode = "voice clone" if samples is not None else (
        "voice design" if args.instruct else "auto"
    )
    print(f"Mode: {mode}")
    print(f"{'='*60}\n")

    # Warm-up
    print("[warm-up] ...", end=" ", flush=True)
    _, warmup_lat, _, _, err = run_inference(url, data, 0)
    if err:
        print(f"FAILED - {err}")
    else:
        print(f"done ({warmup_lat:.2f}s)\n")

    latencies = []
    durations = []

    for i in range(1, args.loop + 1):
        print(f"[{i}/{args.loop}] ", end="", flush=True)
        audio, latency, duration, size_bytes, error = run_inference(url, data, i)
        if error:
            print(f"FAILED - {error}")
            continue
        latencies.append(latency)
        durations.append(duration)
        rtf = latency / duration
        print(f"latency={latency:.2f}s | audio={duration:.2f}s | RTF={rtf:.3f}x")
        if i == args.loop:
            sf.write(args.output_audio, audio, 24000, "PCM_16")

    if latencies:
        print(f"\n{'='*60}")
        print(f"Runs: {len(latencies)}/{args.loop}")
        print(f"Latency — mean={np.mean(latencies):.2f}s | min={np.min(latencies):.2f}s | max={np.max(latencies):.2f}s")
        print(f"Audio   — mean={np.mean(durations):.2f}s")
        print(f"RTF     — mean={np.mean(latencies)/np.mean(durations):.3f}x")
        print(f"Saved to {args.output_audio}")


def run_voices(args, url):
    """Run all voice samples from a directory."""
    import os
    from pathlib import Path

    voices_dir = Path(args.voices_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(voices_dir.glob("*.wav"))
    txt_files = sorted(voices_dir.glob("*.txt"))
    # Include design samples (txt only, no wav)
    all_names = sorted({f.stem for f in wav_files} | {f.stem for f in txt_files})

    print(f"\n{'='*60}")
    print(f"Voice samples: {len(all_names)} from {voices_dir}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}\n")

    # Warm-up with first wav sample
    first_wav = next((n for n in all_names if (voices_dir / f"{n}.wav").exists()), None)
    if first_wav:
        txt_path = voices_dir / f"{first_wav}.txt"
        if txt_path.exists():
            lines = txt_path.read_text().strip().splitlines()
            ref_text = lines[0].strip()
            target_text = lines[1].strip() if len(lines) > 1 else ref_text
            samples, _ = load_audio(str(voices_dir / f"{first_wav}.wav"))
            data = prepare_request(samples=samples, reference_text=ref_text, target_text=target_text)
            print("[warm-up] ...", end=" ", flush=True)
            _, warmup_lat, _, _, err = run_inference(url, data, 0)
            if err:
                print(f"FAILED - {err}")
            else:
                print(f"done ({warmup_lat:.2f}s)\n")

    latencies = []
    for name in all_names:
        wav_path = voices_dir / f"{name}.wav"
        txt_path = voices_dir / f"{name}.txt"
        out_path = output_dir / f"{name}_gen.wav"

        if not txt_path.exists():
            print(f"  [{name}] SKIP — no txt file")
            continue

        lines = txt_path.read_text().strip().splitlines()
        ref_text = lines[0].strip()
        target_text = lines[1].strip() if len(lines) > 1 else ref_text

        # Build request
        if name.startswith("design_"):
            # Voice design: no ref audio, use instruct
            data = prepare_request(instruct=ref_text, target_text=target_text)
            mode = "design"
        elif wav_path.exists():
            samples, _ = load_audio(str(wav_path))
            data = prepare_request(samples=samples, reference_text=ref_text, target_text=target_text)
            mode = "clone"
        else:
            print(f"  [{name}] SKIP — no wav file")
            continue

        audio, latency, duration, _, error = run_inference(url, data, name)
        if error:
            print(f"  [{name}] FAILED — {error}")
            continue

        sf.write(str(out_path), audio, 24000, "PCM_16")
        rtf = latency / duration if duration > 0 else 0
        latencies.append((name, latency, duration))

        print(f"  [{name}] {mode} | {latency:.2f}s | audio={duration:.2f}s | RTF={rtf:.3f}x")
        print(f"    ref:    {ref_text[:80]}")
        print(f"    target: {target_text[:80]}")
        print(f"    output: {out_path}")

    if latencies:
        times = [t for _, t, _ in latencies]
        durs = [d for _, _, d in latencies]
        print(f"\n{'='*60}")
        print(f"Samples: {len(latencies)}/{len(all_names)}")
        print(f"Latency — mean={np.mean(times):.2f}s | min={np.min(times):.2f}s | max={np.max(times):.2f}s")
        print(f"Audio   — mean={np.mean(durs):.2f}s")
        print(f"RTF     — mean={np.mean(times)/np.mean(durs):.3f}x")
        print(f"Output  — {output_dir}")


if __name__ == "__main__":
    args = get_args()
    server_url = args.server_url
    if not server_url.startswith(("http://", "https://")):
        server_url = f"http://{server_url}"

    if not check_health(server_url):
        print("Exiting due to health check failure")
        exit(1)

    url = f"{server_url}/v2/models/{args.model_name}/infer"

    if args.voices_dir:
        run_voices(args, url)
    else:
        run_single_text(args, url)
