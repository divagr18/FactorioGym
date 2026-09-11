"""Run one frozen Gemma same-scene group through the Windows bridge."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from bridge_client import BridgeClient
from rollout_artifacts import RolloutWriter
from rollout_collector import Sample, SequentialGroupCollector
from unsloth import FastVisionModel

SYSTEM = """You control Factorio using exactly one catalog action. Reply only JSON:
{"index": integer, "arguments": object}. Select a currently legal action and use only
the exact arguments declared by that action. Do not explain."""


def _repair(model) -> None:
    for candidate in (
        getattr(model, "config", None),
        getattr(getattr(model, "base_model", None), "config", None),
    ):
        if candidate is not None and getattr(candidate, "architectures", None) is None:
            candidate.architectures = ["Gemma4ForCausalLM"]


class GemmaSampler:
    def __init__(self, model_name: str, revision: str):
        self.policy_revision = revision
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,
            text_only=True,
            device_map={"": 0},
        )
        _repair(self.model)
        self.model.eval()

    def sample(self, state: dict) -> Sample:
        prompt = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": SYSTEM + "\nSTATE:\n" + json.dumps(state, separators=(",", ":")),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            sequence = self.model.generate(
                **inputs, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9
            )
            prompt_len = inputs["input_ids"].shape[1]
            completion = sequence[0, prompt_len:]
            logits = self.model(input_ids=sequence).logits[0]
            logprobs = [
                float(torch.log_softmax(logits[prompt_len - 1 + i], -1)[token])
                for i, token in enumerate(completion)
            ]
        text = self.tokenizer.decode(completion, skip_special_tokens=True).strip()
        try:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            action = json.loads(match.group(0) if match else text)
            index, arguments = action["index"], action["arguments"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(f"model emitted invalid action: {text[:240]!r}") from exc
        return Sample(
            inputs["input_ids"][0].tolist(), completion.tolist(), logprobs, index, arguments
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("FACTORIORL_BRIDGE_URL"))
    parser.add_argument("--token", default=os.environ.get("FACTORIORL_BRIDGE_TOKEN"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--model", default="unsloth/gemma-4-E2B-it-unsloth-bnb-4bit")
    args = parser.parse_args()
    if not args.base_url or not args.token:
        raise SystemExit("bridge URL and token are required through arguments or environment")
    sampler = GemmaSampler(args.model, revision=f"{args.model}:base")
    writer = RolloutWriter(
        args.output,
        run_id="gemma-base-group",
        task_version="construct-smelting-line-v1",
        policy_revision=sampler.policy_revision,
    )
    digests = SequentialGroupCollector(
        BridgeClient(args.base_url, args.token, timeout_seconds=120), writer, sampler
    ).collect_group(seed=args.seed)
    print(json.dumps({"episodes": len(digests), "output": str(args.output)}))


if __name__ == "__main__":
    main()
