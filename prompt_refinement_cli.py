#!/usr/bin/env python3
"""
Prompt Refinement CLI using MPT-R
Author: Atiksh Chawla
Created: May 2, 2025

This script takes a rough user prompt (from a file or stdin) and refines it using the MPT-R language model.
Usage:
  $ prompt_refinement_cli.py [input.txt]
  $ echo "my rough prompt" | prompt_refinement_cli.py
"""

import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def load_model(model_name: str = "mosaicml/mpt-7b-instruct"):
    """
    Load the MPT-R model and tokenizer, and return a text-generation pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        return_full_text=False  # only return the generated continuation
    )
    return gen


def refine_prompt(prompt: str, generator, max_length: int = 256, temperature: float = 0.7) -> str:
    """
    Wrap the user prompt with an instruction for refinement and generate the improved version.
    """
    instruction = (
        "Refine the following user prompt to be clear, specific, and creative:\n"
        "---\n"
        f"{prompt}\n"
        "---\n"
        "Provide only the improved prompt."
    )
    outputs = generator(
        instruction,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        num_return_sequences=1,
        truncation=True
    )
    # outputs is a list of {'generated_text': ...}
    refined = outputs[0]["generated_text"].strip()
    return refined


def main():
    parser = argparse.ArgumentParser(description="Prompt Refinement CLI using MPT-R")
    parser.add_argument(
        "input",
        nargs="?",
        help="Text file with the prompt. If omitted, reads from stdin."
    )
    parser.add_argument(
        "--model",
        default="mosaicml/mpt-7b-instruct",
        help="Hugging Face model ID (default: mosaicml/mpt-7b-instruct)"
    )
    args = parser.parse_args()

    if args.input:
        try:
            with open(args.input, 'r') as f:
                prompt = f.read().strip()
        except IOError:
            print(f"Error: Could not read file {args.input}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Enter your prompt (Ctrl-Z then Enter to finish):")
        prompt = sys.stdin.read().strip()

    if not prompt:
        print("No prompt provided. Exiting.", file=sys.stderr)
        sys.exit(1)

    generator = load_model(args.model)
    refined = refine_prompt(prompt, generator)

    print("\n=== Refined Prompt ===\n")
    print(refined)


if __name__ == "__main__":
    main()
