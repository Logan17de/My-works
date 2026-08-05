from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from config import ModelConfig
from model import ContinualCellTransformer
from tokenizer import DynamicByteTokenizer


def load_checkpoint(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def prepare_prompt(prompt: str, raw_prompt: bool) -> str:
    prompt = prompt.strip()
    if raw_prompt:
        return prompt
    if prompt.startswith("<Q>"):
        return prompt if "<A>" in prompt else f"{prompt}\n<A>"
    return f"<Q> {prompt}\n<A>"


@torch.no_grad()
def generate_until_stop(
    model: ContinualCellTransformer,
    tokenizer: DynamicByteTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    stop_texts: tuple[str, ...],
) -> str:
    generated = input_ids
    answer_ids: list[int] = []

    for _ in range(max_new_tokens):
        context = generated[:, -model.config.max_seq_len :]
        logits = model(context)["logits"][:, -1, :]

        if temperature <= 0.0 or top_k == 1:
            next_token = torch.argmax(
                logits,
                dim=-1,
                keepdim=True,
            )
        else:
            logits = logits / max(temperature, 1e-5)
            if top_k > 0:
                selected = min(top_k, logits.size(-1))
                threshold = torch.topk(
                    logits,
                    k=selected,
                    dim=-1,
                ).values[:, -1:]
                logits = logits.masked_fill(
                    logits < threshold,
                    float("-inf"),
                )
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

        token_id = int(next_token.item())
        generated = torch.cat((generated, next_token), dim=1)
        answer_ids.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break

        decoded = tokenizer.decode(answer_ids)
        if any(marker in decoded for marker in stop_texts):
            break

    decoded = tokenizer.decode(answer_ids)
    earliest_stop = len(decoded)
    for marker in stop_texts:
        position = decoded.find(marker)
        if position >= 0:
            earliest_stop = min(earliest_stop, position)

    answer = decoded[:earliest_stop].strip()
    if answer.startswith("<A>"):
        answer = answer[len("<A>") :].strip()
    return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 uses deterministic greedy decoding.",
    )
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--stop-text",
        action="append",
        default=None,
    )
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument(
        "--show-routing",
        action="store_true",
        help="Show active shared-population cells and activity statistics.",
    )
    args = parser.parse_args()

    stop_texts = tuple(
        args.stop_text or ("<END>", "<Q>")
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = load_checkpoint(args.checkpoint)
    tokenizer = DynamicByteTokenizer.from_dict(
        checkpoint["tokenizer"]
    )
    model = ContinualCellTransformer(
        ModelConfig.from_dict(
            checkpoint["model_config"]
        )
    )
    missing, unexpected = model.load_compatible_state_dict(
        checkpoint["model_state"]
    )
    if missing:
        print("Initialized missing architecture fields:", missing)
    if unexpected:
        print("Ignored unexpected fields:", unexpected)

    model.to(device).eval()
    print("Continual Cell Transformer V4. Type /quit to exit.")
    print(
        "Shared population:",
        [pool.active_count for pool in model.pools()],
    )

    while True:
        try:
            user_prompt = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_prompt.strip().lower() in {"/quit", "/exit"}:
            break
        if not user_prompt.strip():
            continue

        model_prompt = prepare_prompt(
            user_prompt,
            args.raw_prompt,
        )
        prompt_ids = tokenizer.encode(
            model_prompt,
            add_bos=True,
        )
        input_ids = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=device,
        )

        if args.show_routing:
            routing = model(input_ids)
            print("Active cells:", routing["active_cell_ids"])
            print(
                "Active fraction:",
                float(routing["active_fraction"]),
            )
            print("Mean gate:", float(routing["mean_gate"]))
            print(
                "Plastic gate:",
                float(routing["plastic_gate_mean"]),
            )
            print(
                "Plastic output RMS:",
                float(routing["plastic_output_rms"]),
            )

        answer = generate_until_stop(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stop_texts=stop_texts,
        )
        print("Model:", answer or "[no answer generated]")


if __name__ == "__main__":
    main()
