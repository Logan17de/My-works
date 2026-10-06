from __future__ import annotations

import torch

from config import ModelConfig
from learner_variant_transformer import LearnerVariantTransformer, VARIANTS


def main() -> None:
    config = ModelConfig(
        vocab_size=260,
        d_model=128,
        n_heads=4,
        d_ff=512,
        max_seq_len=32,
        dropout=0.0,
        min_depth=1,
        max_depth=8,
    )
    x = torch.randint(0, config.vocab_size, (2, 16))

    for variant in VARIANTS:
        model = LearnerVariantTransformer(
            config,
            num_layers=8,
            variant=variant,
            learner_dim=32,
        )
        result = model(x)
        expected = (2, 16, config.vocab_size)
        actual = tuple(result["logits"].shape)
        if actual != expected:
            raise AssertionError(
                f"{variant}: expected logits {expected}, got {actual}"
            )
        if int(result["used_depth"]) != 8:
            raise AssertionError(f"{variant}: unexpected depth {result['used_depth']}")
        print(
            f"{variant:20} logits={actual} "
            f"params={model.parameter_count():,} "
            f"learner_params={model.learner_parameter_count():,}"
        )

    print("Learner variant smoke test passed.")


if __name__ == "__main__":
    main()
