"""
This script provides a CLI interface for interacting with a language model using DRAM attention.
It supports two modes:

1. Prefill: Takes a long prompt file as input, runs the model's prefill stage, and saves KV caches 
   for each layer to disk for later reuse.

2. Generate: Loads the saved KV caches and generates responses in an interactive chat session.

Example usage:
    # First prefill with long context
    python chat.py prefill --prompt-file prompt.txt
    
    # Then generate responses
    python chat.py generate --top-k $((4*1024)) --lru-hbm-cache-size $((32*1024)) --temperature 0.7 --top-p 0.9
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from dram_attention import DRAMAttention, LRUCache
from termcolor import colored

from model import ModelArgs, Transformer
from tokenizer import ChatFormat, Tokenizer

# Set maximum GPU memory usage to 24GB
total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
mem_fraction = min(1.0, 24.0 / total_mem_gb)
torch.cuda.set_per_process_memory_fraction(mem_fraction)

# Set random seed and model defaults
torch.manual_seed(42)
torch.set_default_dtype(torch.bfloat16)
torch.set_grad_enabled(False)
torch.set_default_device("cuda")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model(args):
    """
    Load model weights and configuration from checkpoint directory.

    Args:
        args: Parsed command line arguments containing model configuration

    Returns:
        tuple: (model, tokenizer) - Loaded model and tokenizer instances
    """
    ckpt_dir = args.model_dir
    device = "cpu" if args.command == "prefill" else "cuda"

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=args.seq_len,
        cache_top_k=getattr(args, "top_k", None),
        max_batch_size=1,
        cache_dir=args.cache_dir,
        lru_hbm_cache_size=getattr(args, "lru_hbm_cache_size", None),
        **params,
    )

    tokenizer = Tokenizer(model_path=str(Path(ckpt_dir) / "tokenizer.model"))
    assert model_args.vocab_size == tokenizer.n_words

    with torch.device("meta"):
        model = Transformer(model_args).to_empty(device=device)

    checkpoint = torch.load(
        Path(ckpt_dir) / "consolidated.00.pth",
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    model.load_state_dict(checkpoint, strict=False)
    logger.info(f"Model loaded from {ckpt_dir}")
    return model, tokenizer


def prefill(model, tokenizer, prompt_file, cache_dir):
    """
    Run model prefill stage on prompt and save KV caches.

    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        prompt_file: Path to text file containing prompt
        cache_dir: Directory to save KV caches
    """
    with open(prompt_file) as f:
        prompt = f.read()

    chat_format = ChatFormat(tokenizer)
    dialog = [{"role": "system", "content": prompt}]
    tokens = torch.tensor(
        chat_format.encode_dialog_prompt(
            dialog, add_bos=True, add_assistant_header=False
        ),
        dtype=torch.long,
        device="cuda",
    ).unsqueeze(0)

    # Log sample of input text
    decoded_text = tokenizer.decode(tokens[0][:100].tolist())
    logger.info("Decoded first 100 tokens of input text:")
    logger.info(decoded_text)
    logger.info(f"Input tokens length: {tokens.shape[1]}")

    # Run prefill and save KV caches
    os.makedirs(cache_dir, exist_ok=True)
    logits = model.forward(tokens, start_pos=0, stage="prefill")
    del logits


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs: Probability distribution tensor
        p: Probability threshold for top-p sampling

    Returns:
        Sampled token indices and their probabilities
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    next_token_prob = torch.gather(probs, -1, next_token)
    return next_token, next_token_prob


def sample(logits, temperature=0.7, top_p=0.9, return_prob=False):
    """
    Sample next token from logits using temperature and top-p sampling.

    Args:
        logits: Raw logits output from model
        temperature: Temperature for softmax scaling
        top_p: Probability threshold for top-p sampling
        return_prob: Whether to return probability of sampled token

    Returns:
        Sampled token index and optionally its probability
    """
    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
    next_token, next_token_prob = sample_top_p(probs, top_p)
    if return_prob:
        return next_token, next_token_prob
    return next_token


def print_token(text, prob=None, colorize=False):
    """Print token text with optional color based on probability."""
    if not colorize:
        print(text, end="", flush=True)
        return

    if prob is None:
        print(colored(text, "green"), end="", flush=True)
        return

    if prob > 0.95:
        color = "white"
    elif prob > 0.70:
        color = "green"
    elif prob > 0.30:
        color = "yellow"
    else:
        color = "red"

    print(colored(text, color), end="", flush=True)


@torch.inference_mode()
def generate(
    model, tokenizer, cache_dir, max_output_length, temperature, top_p, colorize=False
):
    """
    Run interactive chat generation using loaded KV caches.

    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        cache_dir: Directory containing saved KV caches
        max_output_length: Maximum length of generated responses
        temperature: Sampling temperature
        top_p: Top-p sampling threshold
        colorize: Whether to colorize assistant responses
    """
    args = model.params
    eot_token_id = tokenizer.special_tokens["<|eot_id|>"]
    start_header_id = tokenizer.special_tokens["<|start_header_id|>"]
    logger.info(f"EOT token id: {eot_token_id}")

    # Load KV caches into model layers
    for i, layer in enumerate(model.layers):
        cache_path = os.path.join(cache_dir, f"{i:02d}_cache.pth")
        cache = torch.load(
            cache_path, map_location="cuda", mmap=False, weights_only=True
        )
        layer.attention.dram_attention = DRAMAttention(
            max_output_length=max_output_length,
            lru_hbm_cache_size=args.lru_hbm_cache_size,
            local_cache_size=args.local_cache_size,
            page_size=args.cache_page_size,
            top_k=args.cache_top_k,
            n_kv_heads=args.n_kv_heads,
            head_dim=layer.attention.head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        layer.attention.dram_attention.init_cache(cache["cache_k"], cache["cache_v"])
        logger.info(f"Loaded KV cache for layer {i:02d} from {cache_path}")
        cache_len = cache["cache_k"].shape[1]
        del cache

    logger.info(f"Cache length: {cache_len}")
    chat_format = ChatFormat(tokenizer)

    # Create CUDA graph for single token inference
    single_token = torch.zeros((1, 1), dtype=torch.long, device="cuda")
    cur_pos = torch.zeros(1, dtype=torch.int32, device="cuda")
    cur_pos.fill_(cache_len)
    single_token[0, 0].fill_(start_header_id)

    # Warmup and capture CUDA graph
    logits = model.forward(single_token, start_pos=cur_pos, stage="generate")

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        logits.copy_(model.forward(single_token, start_pos=cur_pos, stage="generate"))

    # Interactive chat loop
    while True:
        user_text = input(colored("User: ", "light_red"))
        if user_text.lower() in ["quit", "exit"]:
            logger.info("Quitting...")
            break

        # Tokenize and process user input
        messages = [{"role": "user", "content": user_text}]
        tokens = torch.tensor(
            chat_format.encode_dialog_prompt(
                messages,
                add_bos=False,
                add_assistant_header=True,
            ),
            dtype=torch.long,
            device="cuda",
        ).unsqueeze(0)

        for pos in range(tokens.shape[1]):
            single_token.copy_(tokens[:, pos : pos + 1])
            g.replay()
            cur_pos.add_(1)

        # Generate response
        start_time = time.time()
        next_token, prob = sample(
            logits, temperature=temperature, top_p=top_p, return_prob=True
        )
        single_token.copy_(next_token)
        generated_text = [single_token.item()]
        print_token("Assistant: ", colorize=colorize)

        while True:
            if single_token.item() != eot_token_id:
                decoded = tokenizer.decode([single_token.item()])
                print_token(decoded, prob.item(), colorize)

            g.replay()
            next_token, prob = sample(
                logits, temperature=temperature, top_p=top_p, return_prob=True
            )
            cur_pos.add_(1)

            if single_token.item() == eot_token_id:
                break

            if cur_pos.item() >= cache_len + max_output_length:
                raise RuntimeError("Reached maximum sequence length")
            
            generated_text.append(next_token.item())
            single_token.copy_(next_token)

        end_time = time.time()
        elapsed_time = end_time - start_time
        tokens_generated = len(generated_text)
        tokens_per_second = tokens_generated / elapsed_time
        print(flush=True)
        print("---")
        msg = f"Generated {tokens_generated} tokens in {elapsed_time:.2f}s ({tokens_per_second:.2f} tokens/s)"
        print(colored(msg, "yellow"))


def main():
    """Parse command line arguments and run appropriate mode."""
    parser = argparse.ArgumentParser(description="Chat with model with long context")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/tmp/model/original",
        help="Directory containing model files",
    )
    parser.add_argument(
        "--seq-len", type=int, default=128 * 1024, help="Maximum sequence length"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/tmp/cache",
        help="Directory to store/load KV caches",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Prefill command
    prefill_parser = subparsers.add_parser(
        "prefill", help="Prefill the model with a prompt"
    )
    prefill_parser.add_argument(
        "--prompt-file",
        type=str,
        default="prompt.txt",
        help="Path to file containing the prompt",
    )

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate response")
    generate_parser.add_argument(
        "--top-k",
        type=int,
        default=4 * 1024,
        help="Number of top pages to retrieve from DRAM cache",
    )
    generate_parser.add_argument(
        "--lru-hbm-cache-size",
        type=int,
        default=32 * 1024,
        help="Size of HBM LRU cache in number of tokens",
    )
    generate_parser.add_argument(
        "--max-output-length",
        type=int,
        default=1024 * 4,
        help="Maximum length of generated response in tokens",
    )
    generate_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling (higher = more random)",
    )
    generate_parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling threshold (higher = more diverse)",
    )
    generate_parser.add_argument(
        "--colorize",
        action="store_true",
        help="Colorize assistant responses based on token probabilities",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args)

    if args.command == "prefill":
        prefill(model, tokenizer, args.prompt_file, args.cache_dir)
    elif args.command == "generate":
        generate(
            model,
            tokenizer,
            args.cache_dir,
            max_output_length=args.max_output_length,
            temperature=args.temperature,
            top_p=args.top_p,
            colorize=args.colorize,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
