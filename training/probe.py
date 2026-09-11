"""T0 Gemma model-fit probe.

This is intentionally smaller than a training job. It proves the whole update
cycle needed by multi-turn GRPO before we build an environment bridge or collect
teacher data. Gemma 4's supported Unsloth training route currently does not use
in-process vLLM. We do not launch a second vLLM process because the RTX 4060
must also hold the trainable model and its activations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import traceback
from pathlib import Path


DEFAULT_MODEL = "unsloth/gemma-4-E2B-it"
LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
PROBE_MESSAGES = [
    {
        "role": "user",
        "content": (
            "You control a factory through structured actions. Reply with one concise JSON action "
            "that inspects nearby iron ore."
        ),
    }
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FactorioRL T0 adapter update-cycle probe")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="load only a multimodal model's language decoder; required for text-only agents",
    )
    parser.add_argument(
        "--force-single-gpu",
        action="store_true",
        help="request an all-GPU device map instead of automatic CPU/disk offload",
    )
    parser.add_argument(
        "--fast-inference",
        action="store_true",
        help="enable Unsloth's in-process vLLM integration for a compatible model only",
    )
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.35)
    return parser.parse_args()


def _fingerprint_trainable_parameters(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def _cuda_report(torch) -> dict:
    report = {"available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        report.update(
            {
                "device": properties.name,
                "total_gb": round(properties.total_memory / 1e9, 2),
                "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
            }
        )
    return report


def _restore_text_decoder_architecture(model) -> bool:
    """Repair the stripped text-only config expected by Unsloth's generate wrapper.

    `text_only=True` correctly removes Gemma 4's unused multimodal towers but
    the current loader leaves `architectures=None`. Its generation wrapper
    subsequently iterates that field, although the loaded decoder is the normal
    `Gemma4ForCausalLM` architecture. This only restores configuration metadata;
    it does not alter a parameter or the device map.
    """

    repaired = False
    candidates = [
        getattr(model, "config", None),
        getattr(getattr(model, "base_model", None), "config", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "config", None),
    ]
    for config in candidates:
        if config is not None and getattr(config, "architectures", None) is None:
            config.architectures = ["Gemma4ForCausalLM"]
            repaired = True
    return repaired


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": "T0",
        "model": args.model,
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "lora_rank": args.lora_rank,
        "load_in_4bit": args.load_in_4bit,
        "text_only": args.text_only,
        "force_single_gpu": args.force_single_gpu,
        "vllm": {
            "enabled": args.fast_inference,
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "topology": "single-process shared with trainable model",
        },
        "platform": platform.platform(),
        "started_at_unix": time.time(),
        "passed": False,
    }
    report_path = args.output / "report.json"

    try:
        import torch
        from unsloth import FastVisionModel

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; T0 requires the configured local GPU")

        report["torch"] = torch.__version__
        report["packages"] = {
            package: importlib.metadata.version(package)
            for package in ("transformers", "trl", "vllm", "unsloth", "unsloth-zoo")
        }
        report["cuda_before"] = _cuda_report(torch)
        load_kwargs = {}
        if args.force_single_gpu:
            load_kwargs["device_map"] = {"": 0}
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
            text_only=args.text_only,
            fast_inference=args.fast_inference,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            **load_kwargs,
        )
        model = FastVisionModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=list(LORA_TARGETS),
            lora_alpha=args.lora_rank * 2,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        if args.text_only:
            report["text_only_architecture_metadata_repaired"] = _restore_text_decoder_architecture(
                model
            )
        prompt = tokenizer.apply_chat_template(
            PROBE_MESSAGES, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        model.eval()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                use_cache=True,
            )
        generated_tokens = generated[:, inputs["input_ids"].shape[1] :]
        if generated_tokens.numel() == 0:
            raise RuntimeError("generation produced no tokens")
        report["generated_tokens"] = int(generated_tokens.numel())

        labels = generated.clone()
        labels[:, : inputs["input_ids"].shape[1]] = -100
        model.train()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
        )
        before = _fingerprint_trainable_parameters(model)
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=generated, attention_mask=torch.ones_like(generated), labels=labels)
        loss = output.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite recomputed token loss: {loss.item()}")
        loss.backward()
        grad_norm_sq = sum(
            float(parameter.grad.detach().float().norm().item()) ** 2
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        )
        if grad_norm_sq == 0:
            raise RuntimeError("all trainable gradients are zero")
        optimizer.step()
        after = _fingerprint_trainable_parameters(model)
        if before == after:
            raise RuntimeError("optimizer step did not change any trainable adapter weights")

        adapter_dir = args.output / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        if not (adapter_dir / "adapter_config.json").is_file():
            raise RuntimeError("adapter save did not produce adapter_config.json")
        model.load_adapter(str(adapter_dir), adapter_name="t0-resume", is_trainable=False)
        model.set_adapter("t0-resume")
        model.eval()
        with torch.no_grad():
            resumed = model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=True)
        if resumed.shape[1] <= inputs["input_ids"].shape[1]:
            raise RuntimeError("adapter reload produced no continuation")

        report.update(
            {
                "passed": True,
                "loss": float(loss.detach().cpu()),
                "gradient_l2_norm": grad_norm_sq**0.5,
                "adapter": str(adapter_dir),
                "cuda_after": _cuda_report(torch),
            }
        )
    except Exception as error:  # Keep a machine-readable failure artifact for the gate.
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        try:
            import torch

            report["cuda_after"] = _cuda_report(torch)
        except ImportError:
            pass
    finally:
        report["finished_at_unix"] = time.time()
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
