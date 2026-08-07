from __future__ import annotations

import train_fixed  # applies math_answer_eos_v2 and growth patches
import train_math
from plain_math_compat import parse_math_records


train_math.parse_math_records = parse_math_records


if __name__ == "__main__":
    train_math.main()
