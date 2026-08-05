from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


class DynamicByteTokenizer:
    """UTF-8 byte fallback plus append-only concept tokens."""

    SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
    BYTE_PREFIX = "<byte:"
    BYTE_SUFFIX = ">"

    def __init__(self, extra_tokens: Iterable[str] | None = None) -> None:
        base = list(self.SPECIAL_TOKENS)
        base.extend(f"{self.BYTE_PREFIX}{value:02x}{self.BYTE_SUFFIX}" for value in range(256))
        self.id_to_token = base
        self.token_to_id = {token: index for index, token in enumerate(base)}
        self._added_tokens: list[str] = []
        self._prefix_index: dict[str, list[str]] = {}
        if extra_tokens:
            self.add_tokens(extra_tokens)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def _rebuild_index(self) -> None:
        index: dict[str, list[str]] = {}
        for token in self._added_tokens:
            if token:
                index.setdefault(token[0], []).append(token)
        for values in index.values():
            values.sort(key=len, reverse=True)
        self._prefix_index = index

    def add_tokens(self, tokens: Iterable[str]) -> list[int]:
        new_ids = []
        for raw in tokens:
            token = str(raw)
            if not token or token in self.token_to_id:
                continue
            token_id = len(self.id_to_token)
            self.id_to_token.append(token)
            self.token_to_id[token] = token_id
            self._added_tokens.append(token)
            new_ids.append(token_id)
        if new_ids:
            self._rebuild_index()
        return new_ids

    def discover_tokens(self, text: str, min_frequency: int = 2, max_new_tokens: int = 2000) -> list[str]:
        pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        counts = Counter(piece for piece in pieces if len(piece) >= 2)
        return [token for token, count in counts.most_common() if count >= min_frequency and token not in self.token_to_id][:max_new_tokens]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.bos_token_id] if add_bos else []
        position = 0
        while position < len(text):
            matched = next((candidate for candidate in self._prefix_index.get(text[position], ()) if text.startswith(candidate, position)), None)
            if matched is not None:
                ids.append(self.token_to_id[matched])
                position += len(matched)
            else:
                ids.extend(4 + byte for byte in text[position].encode("utf-8"))
                position += 1
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int], stop_at_eos: bool = True) -> str:
        chunks: list[str] = []
        buffer = bytearray()

        def flush() -> None:
            if buffer:
                chunks.append(buffer.decode("utf-8", errors="replace"))
                buffer.clear()

        for raw in ids:
            token_id = int(raw)
            if token_id < 0 or token_id >= self.vocab_size:
                flush(); chunks.append("�"); continue
            token = self.id_to_token[token_id]
            if token == "<eos>" and stop_at_eos:
                break
            if token in {"<pad>", "<bos>", "<eos>"}:
                continue
            if 4 <= token_id < 260:
                buffer.append(token_id - 4)
            else:
                flush(); chunks.append(token)
        flush()
        return "".join(chunks)

    def to_dict(self) -> dict:
        return {"version": 1, "tokens": self.id_to_token}

    @classmethod
    def from_dict(cls, data: dict) -> "DynamicByteTokenizer":
        tokenizer = cls()
        tokens = list(data["tokens"])
        if tokens[:260] != tokenizer.id_to_token:
            raise ValueError("Tokenizer base vocabulary is incompatible")
        tokenizer.add_tokens(tokens[260:])
        return tokenizer

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
