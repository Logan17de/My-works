from __future__ import annotations

import train_math
from math_objective_v2 import OBJECTIVE_VERSION, encode_math_records


# Patch the trainer before main() starts so every code path—including resume
# validation and checkpoint metadata—uses the corrected objective.
train_math.OBJECTIVE_VERSION = OBJECTIVE_VERSION
train_math.encode_math_records = encode_math_records


if __name__ == "__main__":
    train_math.main()
