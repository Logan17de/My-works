from __future__ import annotations

import train_math
from column_addition_eval import evaluate_mastery
from column_addition_objective import OBJECTIVE_VERSION, encode_records, parse_records
from growth_safety import zero_impact
from zero_impact_growth import apply_zero_impact_growth_patch


# Reuse the proven batching, optimizer, growth, checkpoint, and early-stop loop,
# while replacing the arithmetic record format and objective.
train_math.OBJECTIVE_VERSION = OBJECTIVE_VERSION
train_math.parse_math_records = parse_records
train_math.encode_math_records = encode_records
train_math.evaluate_math_mastery = evaluate_mastery
train_math.zero_impact = zero_impact
apply_zero_impact_growth_patch()


if __name__ == "__main__":
    train_math.main()
