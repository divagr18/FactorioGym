"""Collect one frozen Gemma group through the canonical Factorio policy interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from factoriorl.agent.parsing import ParseFailure, parse_action
from factoriorl.agent.summary import LegalAction
from factoriorl.agent.transcript import Transcript

from bridge_client import BridgeClient
from rollout_artifacts import LOGPROB_RECOMPUTE_TOLERANCE, RolloutWriter, audit_run
from rollout_collector import Sample, SequentialGroupCollector
from unsloth import FastVisionModel
from transformers.generation.logits_process import TemperatureLogitsWarper, TopPLogitsWarper


def _repair(model: Any) -> None:
    """Work around an Unsloth metadata omission without changing model weights."""
    for candidate in (getattr(model, "config", None), getattr(getattr(model, "base_model", None), "config", None)):
        if candidate is not None and getattr(candidate, "architectures", None) is None:
            candidate.architectures = ["Gemma4ForCausalLM"]


def _digest(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(
        json.dumps(messages[:1], separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


class GemmaSampler:
    """Frozen 4-bit Gemma sampler with append-only canonical prompt history."""

    def __init__(self, model_name: str, revision: str, *, max_new_tokens: int, temperature: float, top_p: float):
        self.policy_revision = revision
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=model_name,
            max_seq_length=4096,
            load_in_4bit=True,
            text_only=True,
            device_map={"": 0},
        )
        _repair(self.model)
        self.model.eval()
        self._transcript: Transcript | None = None
        self._episode_nonce: int | None = None
        self._prompt_digest = ""

    def _sampling_logprobs(self, sequence: torch.Tensor, prompt_len: int, completion: torch.Tensor) -> list[float]:
        """Recompute the same temperature/top-p distribution used for sampling."""
        # A causal LM logit at position i predicts token i + 1.  The first
        # generated token is therefore scored by the final prompt logit, not
        # by the first logit returned for the completion.  Requesting only the
        # trailing logits shifted this alignment by one and could assign
        # -inf after top-p filtering to a token that was valid when sampled.
        all_logits = self.model(input_ids=sequence).logits[0]
        logits = all_logits[prompt_len - 1 : prompt_len - 1 + completion.numel()]
        temperature = TemperatureLogitsWarper(self.temperature)
        top_p = TopPLogitsWarper(self.top_p)
        values: list[float] = []
        for index, token in enumerate(completion):
            scores = logits[index].unsqueeze(0)
            prefix = sequence[:, : prompt_len + index]
            scores = temperature(prefix, scores)
            scores = top_p(prefix, scores)
            values.append(float(torch.log_softmax(scores[0], -1)[token]))
        return values

    def _ensure_episode(self, state: dict[str, Any]) -> None:
        policy = state["policy"]
        nonce = int(state["episode_nonce"])
        if self._episode_nonce == nonce:
            return
        self._episode_nonce = nonce
        self._transcript = Transcript(
            system=policy["system_prompt"], static_prefix=policy["static_prefix"]
        )
        self._prompt_digest = str(policy["prompt_digest"])

    def sample(self, state: dict[str, Any]) -> Sample:
        self._ensure_episode(state)
        assert self._transcript is not None
        policy = state["policy"]
        if policy["prompt_digest"] != self._prompt_digest:
            raise RuntimeError("bridge changed the canonical prompt inside an episode")
        self._transcript.append_user(str(policy["summary"]))
        messages = self._transcript.record_sent()
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                output_scores=True,
                return_dict_in_generate=True,
            )
            sequence = generated.sequences
            prompt_len = inputs["input_ids"].shape[1]
            completion = sequence[0, prompt_len:]
            logprobs = [
                float(torch.log_softmax(scores[0], -1)[token])
                for scores, token in zip(generated.scores, completion, strict=True)
            ]
            recomputed = self._sampling_logprobs(sequence, prompt_len, completion)
            parity_error = max(abs(left - right) for left, right in zip(logprobs, recomputed, strict=True))
            if parity_error > LOGPROB_RECOMPUTE_TOLERANCE:
                raise RuntimeError(
                    f"sampled-token logprob recomputation diverged by {parity_error:.6g}"
                )
        text = self.tokenizer.decode(completion, skip_special_tokens=True).strip()
        self._transcript.append_assistant(text)
        vocabulary = tuple((str(key), str(description)) for key, description in policy["vocabulary"])
        legal = tuple(LegalAction(**entry) for entry in policy["legal_actions"])
        requires = {key: tuple(values) for key, values in policy["requires"].items()}
        outcome = parse_action(
            text,
            legal,
            vocabulary,
            targetable=frozenset(policy["targetable_actions"]),
            handles=frozenset(policy["visible_handles"]),
            requires=requires,
            domains=policy["raw_domains"],
        )
        common = {
            "messages": messages,
            "rendered_prompt": prompt,
            "sampled_token_mask": [True] * completion.numel(),
            "completion_text": text,
            "prompt_digest": self._prompt_digest,
            "logprob_recompute_max_abs_error": parity_error,
        }
        if isinstance(outcome, ParseFailure):
            return Sample(
                inputs["input_ids"][0].tolist(),
                completion.tolist(),
                logprobs,
                None,
                {},
                parse_failure=outcome.to_dict(),
                **common,
            )
        return Sample(
            inputs["input_ids"][0].tolist(),
            completion.tolist(),
            logprobs,
            outcome.index,
            outcome.arguments,
            target=outcome.target,
            parsed_action=outcome.to_dict(),
            **common,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("FACTORIORL_BRIDGE_URL"))
    parser.add_argument("--token", default=os.environ.get("FACTORIORL_BRIDGE_TOKEN"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--max-turns", type=int, default=32)
    parser.add_argument("--model", default="unsloth/gemma-4-E2B-it-unsloth-bnb-4bit")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()
    if not args.base_url or not args.token:
        raise SystemExit("bridge URL and token are required through arguments or environment")
    sampler = GemmaSampler(
        args.model,
        revision=f"{args.model}:base",
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    writer = RolloutWriter(
        args.output,
        run_id="gemma-base-group",
        task_version="construct-smelting-line-v1",
        policy_revision=sampler.policy_revision,
        metadata={
            "model": args.model,
            "sampling": {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens},
            "collector": "sequential-same-scene-v2",
        },
    )
    digests = SequentialGroupCollector(
        BridgeClient(args.base_url, args.token, timeout_seconds=120), writer, sampler
    ).collect_group(seed=args.seed, max_turns=args.max_turns)
    print(json.dumps({"episodes": len(digests), "output": str(args.output), "audit": audit_run(args.output)}))


if __name__ == "__main__":
    main()
