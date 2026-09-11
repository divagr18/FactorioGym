"""Separate T0 vLLM serving compatibility probe for the selected Gemma checkpoint."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path


DEFAULT_MODEL = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
PROMPT = (
    "<bos><start_of_turn>user\n"
    "Reply with one JSON action that inspects nearby iron ore.\n"
    "<end_of_turn>\n<start_of_turn>model\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test direct vLLM Gemma 4 rollout serving")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": "T0-vllm",
        "model": args.model,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "passed": False,
        "started_at_unix": time.time(),
    }
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams

        report["vllm"] = vllm.__version__
        report["cuda_before_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
        engine = LLM(
            model=args.model,
            quantization="bitsandbytes",
            dtype="bfloat16",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=True,
        )
        result = engine.generate(
            [PROMPT], SamplingParams(temperature=0.0, max_tokens=16, stop_token_ids=[])
        )
        text = result[0].outputs[0].text
        if not text.strip():
            raise RuntimeError("vLLM generated an empty completion")
        report.update(
            {
                "passed": True,
                "completion": text,
                "completion_tokens": len(result[0].outputs[0].token_ids),
                "cuda_after_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            }
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        report["finished_at_unix"] = time.time()
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
