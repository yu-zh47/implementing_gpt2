#!/usr/bin/env python3
"""Simple generation script for the toy GPT model.

Example
-------
python generate.py \
    --ckpt_dir ckpts_400steps \
    --prompt "Hello, I'm a language model" \
    --num_return_sequences 5 \
    --max_length 50

The script loads the most recent checkpoint in ``ckpt_dir`` (or a specific
``--ckpt_file``) onto the first available CUDA device and produces ``N``
continuations.
"""
import argparse
import glob
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import tiktoken

# Re-use the model definition from training.
from model import GPT, GPTConfig

# EXPORT DISABLE_ADDMM_CUDA_LT=1   # tells PyTorch to skip the buggy fused path
# pip install --upgrade --pre torch torchaudio torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

def find_checkpoint(ckpt_dir: Path, ckpt_file: str | None) -> Path:
    """Return the checkpoint path to load.

    If *ckpt_file* is given, return it; otherwise, pick the latest *.pt* in the
    directory based on lexicographic order (names contain the step number).
    """
    if ckpt_file is not None:
        p = Path(ckpt_file)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    candidates = sorted(ckpt_dir.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoint found in {ckpt_dir}")
    return candidates[-1]  # latest by name


def generate(model: GPT, enc, prompt: str, num_sequences: int, max_length: int, device: torch.device):
    model.eval()

    # Encode prompt and repeat for the batch.
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    tokens = tokens.repeat(num_sequences, 1)  # (B, prompt_len)

    while tokens.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(tokens)  # (B, T, vocab)
            logits = logits[:, -1, :]  # next-token logits (B, vocab)
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
            next_tokens = torch.multinomial(topk_probs, num_samples=1)  # (B,1)
            next_idx = torch.gather(topk_indices, dim=-1, index=next_tokens)
            tokens = torch.cat((tokens, next_idx), dim=1)

    # Decode and print.
    for i in range(num_sequences):
        output = enc.decode(tokens[i, :max_length].tolist())
        print(f"[Sample {i+1}] {output}")


def main():
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory that holds checkpoints")
    parser.add_argument("--ckpt_file", type=str, default=None, help="Load this specific checkpoint file instead of latest in dir")
    parser.add_argument("--prompt", type=str, default="Hello, I'm a language model", help="Prompt text to start generation")
    parser.add_argument("--num_return_sequences", type=int, default=5, help="How many sequences to sample")
    parser.add_argument("--max_length", type=int, default=30, help="Maximum generated length (including prompt)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = find_checkpoint(Path(args.ckpt_dir), args.ckpt_file)
    print(f"Loading checkpoint {ckpt_path} on {device}…")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Some checkpoints may be saved from a `torch.compile`d or `torch.nn.parallel.DistributedDataParallel`
    # model. In these cases the parameter names are prefixed (e.g. "_orig_mod." or "module.").
    # Strip such prefixes so that they match the vanilla GPT instance created below.
    def clean_state_dict(state_dict):
        """Remove common wrapper prefixes from state-dict keys."""
        prefixes = ("_orig_mod.", "module.")
        example_key = next(iter(state_dict))
        if example_key.startswith(prefixes):
            cleaned = {k.split('.', 1)[1]: v for k, v in state_dict.items()}
            return cleaned
        return state_dict

    checkpoint_state = clean_state_dict(checkpoint["model"]) if "model" in checkpoint else clean_state_dict(checkpoint)

    # Recreate the model – the config must match training.
    config = GPTConfig(block_size=1024, vocab_size=50304)
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint_state, strict=True)
    print("Model weights loaded.")

    # Tokeniser
    enc = tiktoken.get_encoding("gpt2")

    # Generate
    generate(model, enc, args.prompt, args.num_return_sequences, args.max_length, device)


if __name__ == "__main__":
    main()