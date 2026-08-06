from __future__ import annotations

import train
from mastery_eval import evaluate_math_mastery


# Replace the original evaluator before train.main() starts.
train.evaluate_math_mastery = evaluate_math_mastery


if __name__ == "__main__":
    train.main()
