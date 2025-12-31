#!/usr/bin/env python3
"""
Training script for BPE tokenizer on TinyStories dataset.

This script trains a BPE tokenizer with vocab_size=10,000 on the TinyStories dataset
and measures training time and memory usage.
"""

import sys
import os
import time
import json
import psutil
from pathlib import Path

# Add parent directory to path to import cs336_basics
sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.tokenizers.bpe import BPETokenizerTrainer


def get_memory_usage_gb():
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)


def save_vocab(vocab, output_path):
    """Save vocabulary to JSON file in GPT-2 style (UTF-8 strings as keys)."""
    # Convert to GPT-2 style: token_string -> token_id
    vocab_serializable = {}
    for token_id, token_bytes in vocab.items():
        try:
            # Decode to UTF-8 string
            token_str = token_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # For non-UTF-8 bytes, use escaped representation
            token_str = token_bytes.decode('utf-8', errors='replace')
        vocab_serializable[token_str] = token_id

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_serializable, f, indent=2, ensure_ascii=False)
    print(f"Vocabulary saved to {output_path}")


def save_merges(merges, output_path):
    """Save merges to text file in GPT-2 style (UTF-8 strings)."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for token1, token2 in merges:
            try:
                # Decode to UTF-8 strings
                token1_str = token1.decode('utf-8')
                token2_str = token2.decode('utf-8')
            except UnicodeDecodeError:
                # For non-UTF-8 bytes, use escaped representation
                token1_str = token1.decode('utf-8', errors='replace')
                token2_str = token2.decode('utf-8', errors='replace')
            f.write(f"{token1_str} {token2_str}\n")
    print(f"Merges saved to {output_path}")


def find_longest_token(vocab):
    """Find the longest token in vocabulary."""
    longest_token = max(vocab.items(), key=lambda x: len(x[1]))
    return longest_token


def main():
    # Configuration
    input_path = "/ocean/projects/cis250265p/xli45/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    output_dir = "/ocean/projects/cis250265p/xli45/assignment1-basics/trained_tokenizers/tinystories"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Training BPE Tokenizer on TinyStories Dataset")
    print("=" * 80)
    print(f"Input file: {input_path}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Measure initial memory
    initial_memory = get_memory_usage_gb()
    print(f"\nInitial memory usage: {initial_memory:.2f} GB")

    # Train tokenizer
    print("\nStarting training...")
    start_time = time.time()

    trainer = BPETokenizerTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_workers=8
    )
    vocab, merges = trainer.train(input_path)

    end_time = time.time()
    training_time = end_time - start_time

    # Measure peak memory
    peak_memory = get_memory_usage_gb()
    memory_used = peak_memory - initial_memory

    print("\nTraining completed!")
    print("=" * 80)
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Peak memory usage: {peak_memory:.2f} GB")
    print(f"Memory used: {memory_used:.2f} GB")
    print(f"Final vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print("=" * 80)

    # Find longest token
    longest_id, longest_token = find_longest_token(vocab)
    print(f"\nLongest token:")
    print(f"  Token ID: {longest_id}")
    print(f"  Length: {len(longest_token)} bytes")
    print(f"  Token (repr): {longest_token}")
    try:
        decoded = longest_token.decode('utf-8', errors='replace')
        print(f"  Token (UTF-8): {repr(decoded)}")
    except Exception as e:
        print(f"  Could not decode as UTF-8: {e}")

    # Save vocabulary and merges
    print("\nSaving tokenizer...")
    vocab_path = os.path.join(output_dir, "vocab.json")
    merges_path = os.path.join(output_dir, "merges.txt")

    save_vocab(vocab, vocab_path)
    save_merges(merges, merges_path)

    # Save training metrics
    metrics = {
        "dataset": "TinyStories",
        "input_path": input_path,
        "vocab_size": vocab_size,
        "special_tokens": special_tokens,
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60,
        "peak_memory_gb": peak_memory,
        "memory_used_gb": memory_used,
        "final_vocab_size": len(vocab),
        "num_merges": len(merges),
        "longest_token_id": longest_id,
        "longest_token_length": len(longest_token)
    }

    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Training metrics saved to {metrics_path}")

    print("\n" + "=" * 80)
    print("All done!")
    print("=" * 80)


if __name__ == "__main__":
    main()