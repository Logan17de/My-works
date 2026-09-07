from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import interleave_datasets, load_dataset
from transformers import AutoTokenizer


DATASET_ALIASES = {
    "k2-horizon": "IFM/K2-Horizon-Pretrain-Data",
    "k2-txt360-v2": "IFM/TxT360-v2",
    "txt360": "LLM360/TxT360",
    "k2-code": "IFM/Code-Reasoning",
    "k2-math": "IFM/Math-Reasoning",
    "k2-behaviors": "IFM/Pretrain-Behaviors",
    "k2-sft": "IFM/SFT-Reasoning",
}


def resolve_source(source: str) -> str:
    if source in DATASET_ALIASES:
        return DATASET_ALIASES[source]
    if source.startswith("hf:"):
        return source[3:]
    return source


def _messages_to_text(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
    parts: list[str] = []
    for item in messages:
        if isinstance(item, dict):
            role = str(item.get("role", ""))
            content = item.get("content", "")
            if isinstance(content, list):
                content = " ".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in content)
            parts.append(f"{role}: {content}" if role else str(content))
        else:
            parts.append(str(item))
    return "\n".join(parts) if parts else None


def extract_text(example: dict[str, Any], preferred: str | None = None) -> str | None:
    if preferred:
        value: Any = example
        try:
            for part in preferred.split("."):
                value = value[part]
        except Exception:
            value = None
        if isinstance(value, str) and value.strip():
            return value
    for key in ("text", "content", "document", "completion", "response", "answer"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            if key in {"response", "answer"} and isinstance(example.get("prompt"), str):
                return f"{example['prompt']}\n{value}"
            if key == "answer" and isinstance(example.get("question"), str):
                return f"{example['question']}\n{value}"
            return value
    msg = _messages_to_text(example.get("messages"))
    if msg:
        return msg
    prompt = example.get("prompt")
    if isinstance(prompt, str):
        response = example.get("response") or example.get("completion") or example.get("answer")
        return f"{prompt}\n{response}" if isinstance(response, str) else prompt
    return None


def _load_one(spec: dict[str, Any], default_cfg: dict[str, Any]):
    source = resolve_source(spec.get("source", default_cfg["source"]))
    config_name = spec.get("config_name", default_cfg.get("config_name"))
    split = spec.get("split", default_cfg.get("split", "train"))
    token_env = default_cfg.get("hf_token_env", "HF_TOKEN")
    token = os.environ.get(token_env) or None
    if source.startswith("local:"):
        path = source[6:]
        return load_dataset("json", data_files=path, split=split, streaming=True)
    if Path(source).exists():
        return load_dataset("json", data_files=source, split=split, streaming=True)
    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if token:
        kwargs["token"] = token
    if config_name:
        return load_dataset(source, config_name, **kwargs)
    return load_dataset(source, **kwargs)


def build_streaming_dataset(cfg: dict[str, Any], *, resume_raw_examples: int = 0):
    mixture_file = cfg.get("mixture_file")
    if mixture_file:
        payload = json.loads(Path(mixture_file).read_text(encoding="utf-8"))
        components = payload.get("components", payload)
        streams = []
        probs = []
        for item in components:
            streams.append(_load_one(item, cfg))
            probs.append(float(item.get("weight", 1.0)))
        total = sum(probs)
        probs = [p / total for p in probs]
        ds = interleave_datasets(streams, probabilities=probs, seed=int(cfg.get("seed", 17)), stopping_strategy="all_exhausted")
    else:
        ds = _load_one(cfg, cfg)
    buffer = int(cfg.get("shuffle_buffer", 0) or 0)
    if buffer > 1:
        ds = ds.shuffle(seed=int(cfg.get("seed", 17)), buffer_size=buffer)
    if resume_raw_examples > 0:
        ds = ds.skip(resume_raw_examples)
    return ds


class PackedTokenStream:
    def __init__(self, cfg: dict[str, Any], *, resume_state: dict | None = None) -> None:
        self.cfg = cfg
        self.sequence_length = int(cfg["sequence_length"])
        self.text_field = cfg.get("text_field")
        self.eos_between = bool(cfg.get("eos_between_documents", True))
        self.raw_examples_seen = int((resume_state or {}).get("raw_examples_seen", 0))
        self.buffer: list[int] = list((resume_state or {}).get("buffer", []))
        self.dataset = build_streaming_dataset(cfg, resume_raw_examples=self.raw_examples_seen)
        self.iterator = iter(self.dataset)
        self.tokenizer = AutoTokenizer.from_pretrained(resolve_source(cfg.get("tokenizer", "Qwen/Qwen3.8-Flash-Next")))
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id for document packing")

    def _fill(self) -> None:
        while len(self.buffer) < self.sequence_length + 1:
            example = next(self.iterator)
            self.raw_examples_seen += 1
            text = extract_text(example, self.text_field)
            if not text:
                continue
            ids = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            if self.eos_between:
                ids.append(self.tokenizer.eos_token_id)
            self.buffer.extend(ids)

    def next_sequence(self) -> torch.Tensor:
        self._fill()
        ids = self.buffer[: self.sequence_length + 1]
        del self.buffer[: self.sequence_length + 1]
        return torch.tensor(ids, dtype=torch.long)

    def next_microbatch(self, microbatch_size: int) -> torch.Tensor:
        rows = [self.next_sequence() for _ in range(microbatch_size)]
        return torch.stack(rows, dim=0)

    def state_dict(self) -> dict[str, Any]:
        return {"raw_examples_seen": self.raw_examples_seen, "buffer": self.buffer}


def probe_dataset(cfg: dict[str, Any], n: int = 3) -> list[dict[str, Any]]:
    ds = build_streaming_dataset(cfg)
    out = []
    for idx, example in enumerate(ds):
        text = extract_text(example, cfg.get("text_field"))
        out.append({"index": idx, "keys": sorted(example.keys()), "text_preview": (text or "")[:400]})
        if len(out) >= n:
            break
    return out
