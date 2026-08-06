from __future__ import annotations

"""Continual arithmetic training with safe objective migration.

This entry point can resume a mastered addition checkpoint trained with the
legacy ``math_answer_only_v1`` objective, while training the new task with the
cleaner ``math_answer_eos_v2`` objective. Model weights are preserved; only the
training objective used for new-task and retention batches changes.
"""

import train_math
from config import ModelConfig
from math_objective_v2 import OBJECTIVE_VERSION, encode_math_records
from model import ContinualCellTransformer
from tokenizer import DynamicByteTokenizer
from zero_impact_growth import apply_zero_impact_growth_patch


LEGACY_OBJECTIVES = {"math_answer_only_v1", "math_answer_eos_v2"}
_original_build_model_and_tokenizer = train_math.build_model_and_tokenizer


# Ensure micro-neuron insertion preserves logits even though the pool
# normalizes by sqrt(active_micro_count).
apply_zero_impact_growth_patch()


def build_model_and_tokenizer_compatible(args, train_text: str):
    if not args.resume:
        return _original_build_model_and_tokenizer(args, train_text)

    checkpoint = train_math.load_checkpoint(args.resume)
    source_objective = checkpoint.get("training_objective")
    if source_objective not in LEGACY_OBJECTIVES:
        raise ValueError(
            f"Checkpoint objective {source_objective!r} cannot be migrated. "
            f"Supported objectives: {sorted(LEGACY_OBJECTIVES)!r}."
        )

    tokenizer = DynamicByteTokenizer.from_dict(checkpoint["tokenizer"])
    old_vocab = tokenizer.vocab_size
    tokens = list(args.add_token)
    if args.auto_add_words:
        tokens += tokenizer.discover_tokens(train_text)
    tokenizer.add_tokens(tokens)

    model = ContinualCellTransformer(
        ModelConfig.from_dict(checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if tokenizer.vocab_size > old_vocab:
        model.resize_vocabulary(tokenizer.vocab_size)

    previous_step = int(checkpoint.get("step", 0))
    print(
        f"resume objective migration: {source_objective!r} -> "
        f"{OBJECTIVE_VERSION!r}; preserved model weights from step={previous_step}"
    )
    return model, tokenizer, previous_step


# Patch every path used by train_math.main(): encoding, resume validation, and
# checkpoint metadata all use the V2 objective after this point.
train_math.OBJECTIVE_VERSION = OBJECTIVE_VERSION
train_math.encode_math_records = encode_math_records
train_math.build_model_and_tokenizer = build_model_and_tokenizer_compatible


if __name__ == "__main__":
    train_math.main()
