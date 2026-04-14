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
        default="localhost:8000",
        help="Address of the server",
    )
    parser.add_argument(
        "--reference-audio",
        type=str,
        default=None,
        help="Path to reference audio file for voice cloning",
    )
    parser.add_argument(
        "--reference-text",
        type=str,
        default=None,
        help="Transcript of the reference audio (auto-transcribed if omitted)",
    )
    parser.add_argument(
        "--target-text",
        type=str,
        required=True,
        help="Text to synthesize",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language name or code (e.g. 'English', 'en')",
    )
    parser.add_argument(
        "--instruct",
        type=str,
        default=None,
        help="Voice design instruction (e.g. 'Male, British Accent')",
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
        "--num-runs",
        type=int,
        default=1,
        help="Number of inference runs (1 warm-up + N benchmark)",
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
        samples = samples.reshape(1, -1).astype(np.float32)
        inputs.append(
            {
                "name": "reference_wav",
                "shape": list(samples.shape),
                "datatype": "FP32",
                "data": samples.tolist(),
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


def send_request(url, data):
    rsp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=data,
        verify=False,
    )
    result = rsp.json()
    if "error" in result:
        return None, result["error"]
    audio = np.array(result["outputs"][0]["data"], dtype=np.float32)
    return audio, None


if __name__ == "__main__":
    args = get_args()
    server_url = args.server_url
    if not server_url.startswith(("http://", "https://")):
        server_url = f"http://{server_url}"

    url = f"{server_url}/v2/models/{args.model_name}/infer"

    samples = None
    if args.reference_audio is not None:
        samples, sr = load_audio(args.reference_audio)

    data = prepare_request(
        samples=samples,
        reference_text=args.reference_text,
        target_text=args.target_text,
        language=args.language,
        instruct=args.instruct,
    )

    num_runs = args.num_runs

    if num_runs <= 1:
        # Single run
        t0 = time.perf_counter()
        audio, err = send_request(url, data)
        latency = time.perf_counter() - t0
        if err:
            print(f"Error: {err}")
            exit(1)
        audio_dur = len(audio) / 24000
        rtf = latency / audio_dur if audio_dur > 0 else 0
        sf.write(args.output_audio, audio, 24000, "PCM_16")
        print(
            f"latency={latency:.3f}s | audio={audio_dur:.2f}s | "
            f"RTF={rtf:.3f}x | saved to {args.output_audio}"
        )
    else:
        # Warm-up
        print("[warm-up] ...", end=" ", flush=True)
        audio, err = send_request(url, data)
        if err:
            print(f"Error: {err}")
            exit(1)
        print("done")

        # Benchmark
        latencies = []
        audio_durations = []

        for i in range(num_runs):
            t0 = time.perf_counter()
            audio, err = send_request(url, data)
            latency = time.perf_counter() - t0

            if err:
                print(f"[{i+1}/{num_runs}] Error: {err}")
                continue

            audio_dur = len(audio) / 24000
            rtf = latency / audio_dur if audio_dur > 0 else 0
            latencies.append(latency)
            audio_durations.append(audio_dur)

            print(
                f"[{i+1}/{num_runs}] latency={latency:.3f}s | "
                f"audio={audio_dur:.2f}s | RTF={rtf:.3f}x"
            )

        # Save last audio
        if audio is not None:
            sf.write(args.output_audio, audio, 24000, "PCM_16")

        # Summary
        if latencies:
            lat = np.array(latencies)
            dur = np.array(audio_durations)
            rtfs = lat / dur

            print(f"\n{'='*50}")
            print(f"Runs: {len(latencies)}/{num_runs}")
            print(
                f"Latency  — mean={lat.mean():.3f}s | "
                f"median={np.median(lat):.3f}s | "
                f"min={lat.min():.3f}s | max={lat.max():.3f}s"
            )
            print(f"Audio    — mean={dur.mean():.2f}s")
            print(
                f"RTF      — mean={rtfs.mean():.3f}x | "
                f"median={np.median(rtfs):.3f}x | "
                f"min={rtfs.min():.3f}x | max={rtfs.max():.3f}x"
            )
            print(f"Saved last audio to {args.output_audio}")
