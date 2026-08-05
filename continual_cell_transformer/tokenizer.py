from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


class DynamicByteTokenizer:
    """
    Hybrid tokenizer with two useful properties:

    1. Any Unicode input is always representable through 256 UTF-8 byte tokens.
    2. New words or multi-word concepts can be appended later without changing
       any existing token ID.

    Added tokens are matched greedily. Old IDs never move.
    """

    SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
    BYTE_PREFIX = "<byte:"
    BYTE_SUFFIX = ">"

    def __init__(self, extra_tokens: Iterable[str] | None = None) -> None:
        base = list(self.SPECIAL_TOKENS)
        base.extend(f"{self.BYTE_PREFIX}{value:02x}{self.BYTE_SUFFIX}" for value in range(256))
        self.id_to_token: list[str] = base
        self.token_to_id: dict[str, int] = {token: idx for idx, token in enumerate(base)}
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
    def unk_token_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def _rebuild_prefix_index(self) -> None:
        index: dict[str, list[str]] = {}
        for token in self._added_tokens:
            if not token:
                continue
            index.setdefault(token[0], []).append(token)
        for values in index.values():
            values.sort(key=len, reverse=True)
        self._prefix_index = index

    def add_tokens(self, tokens: Iterable[str]) -> list[int]:
        """Append tokens while preserving every existing ID."""
        new_ids: list[int] = []
        for raw_token in tokens:
            token = str(raw_token)
            if not token or token in self.token_to_id:
                continue
            token_id = len(self.id_to_token)
            self.id_to_token.append(token)
            self.token_to_id[token] = token_id
            self._added_tokens.append(token)
            new_ids.append(token_id)
        if new_ids:
            self._rebuild_prefix_index()
        return new_ids

    def discover_tokens(
        self,
        text: str,
        min_frequency: int = 2,
        max_new_tokens: int = 2_000,
        min_length: int = 2,
    ) -> list[str]:
        """
        Find reusable word-like units. Byte fallback still represents everything,
        so discovery is optional rather than required for correctness.
        """
        pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        counts = Counter(piece for piece in pieces if len(piece) >= min_length)
        candidates = [
            token
            for token, count in counts.most_common()
            if count >= min_frequency and token not in self.token_to_id
        ]
        return candidates[:max_new_tokens]

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_token_id)

        position = 0
        while position < len(text):
            matched: str | None = None
            candidates = self._prefix_index.get(text[position], ())
            for candidate in candidates:
                if text.startswith(candidate, position):
                    matched = candidate
                    break

            if matched is not None:
                ids.append(self.token_to_id[matched])
                position += len(matched)
                continue

            # Fallback is one Unicode character encoded as one or more UTF-8 bytes.
            char = text[position]
            for byte in char.encode("utf-8"):
                ids.append(4 + byte)
            position += 1

        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int], stop_at_eos: bool = True) -> str:
        chunks: list[str] = []
        byte_buffer = bytearray()

        def flush_bytes() -> None:
            if byte_buffer:
                chunks.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for raw_id in ids:
            token_id = int(raw_id)
            if token_id < 0 or token_id >= self.vocab_size:
                flush_bytes()
                chunks.append("�")
                continue

            token = self.id_to_token[token_id]
            if token == "<eos>" and stop_at_eos:
                break
            if token in {"<pad>", "<bos>", "<eos>"}:
                continue
            if token == "<unk>":
                flush_bytes()
                chunks.append("�")
                continue

            if 4 <= token_id < 260:
                byte_buffer.append(token_id - 4)
            else:
                flush_bytes()
                chunks.append(token)

        flush_bytes()
        return "".join(chunks)

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "tokens": self.id_to_token,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DynamicByteTokenizer":
        tokens = list(data["tokens"])
        tokenizer = cls()
        expected_base = tokenizer.id_to_token
        if tokens[: len(expected_base)] != expected_base:
            raise ValueError("Tokenizer base vocabulary is incompatible.")
        tokenizer.add_tokens(tokens[len(expected_base) :])
        return tokenizer

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "DynamicByteTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
