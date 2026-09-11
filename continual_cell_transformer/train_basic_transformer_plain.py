from __future__ import annotations

import train_basic_transformer as trainer
from plain_math_compat import parse_math_records


trainer.parse_math_records = parse_math_records


if __name__ == "__main__":
    trainer.main()
