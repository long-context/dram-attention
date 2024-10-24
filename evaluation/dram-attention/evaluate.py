import json
import logging
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from model import ModelArgs, Transformer
from tokenizer import ChatFormat, Tokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

torch.manual_seed(42)
torch.set_default_dtype(torch.bfloat16)
torch.set_grad_enabled(False)
torch.set_default_device("cuda")

parser = ArgumentParser()
parser.add_argument("--model-dir", default="/tmp/model/original")
parser.add_argument("--seq-len", type=int, default=8192)
parser.add_argument("--top-k", type=int, default=4096)
parser.add_argument("--page-size", type=int, default=16)
parser.add_argument("--run-name", type=str, default=None)
parser.add_argument("--local-attn-window", type=int, default=4096)
parser.add_argument("--data-dir", default="/tmp/data/ruler/niah_multikey_2")
FLAGS = parser.parse_args()

assert FLAGS.local_attn_window % FLAGS.page_size == 0

logger.info("Running evaluation with the following parameters:")
logger.info(f"  Model directory: {FLAGS.model_dir}")
logger.info(f"  Sequence length: {FLAGS.seq_len}")
logger.info(f"  Top-k value: {FLAGS.top_k}")
logger.info(f"  Page size: {FLAGS.page_size}")
logger.info(f"  Run name: {FLAGS.run_name}")
logger.info(f"  Local attention window: {FLAGS.local_attn_window}")
logger.info(f"  Data directory: {FLAGS.data_dir}")

data_file = f"{FLAGS.data_dir}/validation_{FLAGS.seq_len}.jsonl"


def load_model(ckpt_dir, max_seq_len: int):
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=1,
        **params,
    )
    tokenizer = Tokenizer(model_path=str(Path(ckpt_dir) / "tokenizer.model"))
    assert model_args.vocab_size == tokenizer.n_words

    with torch.device("meta"):
        model = Transformer(model_args).to_empty(device="cuda")

    checkpoint = torch.load(
        Path(ckpt_dir) / "consolidated.00.pth",
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    model.load_state_dict(checkpoint, strict=False)
    logger.info(f"Model loaded from {ckpt_dir}")
    return model, tokenizer


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prefix_prompt_tokens: List[int],
    suffix_prompt_tokens: List[int],
    max_gen_len: int,
) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
    params = model.params
    bsz = 1
    prompt_tokens = prefix_prompt_tokens + suffix_prompt_tokens
    prompt_len = len(prompt_tokens)
    assert prompt_len <= params.max_seq_len
    total_len = min(params.max_seq_len, max_gen_len + prompt_len)

    pad_id = tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
    tokens[0, :prompt_len] = torch.tensor(
        prompt_tokens, dtype=torch.long, device="cuda"
    )
    input_text_mask = tokens != pad_id

    stop_tokens = list(tokenizer.stop_tokens)
    aligned_start_pos = len(prefix_prompt_tokens) // FLAGS.page_size * FLAGS.page_size

    logits = model.forward(tokens[:, 0:aligned_start_pos], 0)
    next_token = torch.argmax(logits[:, -1], dim=-1)
    next_token = torch.where(
        input_text_mask[:, aligned_start_pos], tokens[:, aligned_start_pos], next_token
    )
    tokens[:, aligned_start_pos] = next_token

    assert FLAGS.top_k % FLAGS.page_size == 0
    torch.cuda.synchronize()

    for cur_pos in range(aligned_start_pos, total_len - 1):
        logits = model.forward(tokens[:, cur_pos : cur_pos + 1], cur_pos)
        next_token = torch.argmax(logits[:, -1], dim=-1)
        torch.cuda.synchronize()
        next_token = next_token.reshape(-1)
        next_token = torch.where(
            input_text_mask[:, cur_pos + 1], tokens[:, cur_pos + 1], next_token
        )
        tokens[:, cur_pos + 1] = next_token
        if next_token.item() in stop_tokens:
            break
        if next_token.item() >= 128000:
            break
    return tokens[0].tolist()[len(prompt_tokens) : cur_pos + 1]


def read_jsonl(file_path):
    with open(file_path, "r") as file:
        for line in file.readlines():
            yield json.loads(line.strip())


model, tokenizer = load_model(FLAGS.model_dir, max_seq_len=FLAGS.seq_len)
formatter = ChatFormat(tokenizer)
logger.info(f"Processing data from {data_file}")
total_recall = 0
total_items = 0
results_file_name = f"eval_results_{FLAGS.seq_len}"
if FLAGS.run_name is not None:
    results_file_name += f"_{FLAGS.run_name}"
results_file_name += ".jsonl"

with open(results_file_name, "w") as results_file:
    for item in read_jsonl(data_file):
        logger.info(f"==== #{item['index']} ====")
        lines = item["input"].split("\n")
        prefix_prompt = "\n".join(lines[:-1]) + "\n"
        suffix_prompt = lines[-1]
        prefix_prompt_tokens = tokenizer.encode(prefix_prompt, bos=True, eos=False)
        suffix_prompt_tokens = tokenizer.encode(suffix_prompt, bos=False, eos=False)
        prompt_tokens = prefix_prompt_tokens + suffix_prompt_tokens
        logger.info(f"Prompt length: {len(prompt_tokens)}")
        logger.info("Last line of input: {}".format(suffix_prompt))
        output_tokens = generate(
            model,
            tokenizer,
            prefix_prompt_tokens=prefix_prompt_tokens,
            suffix_prompt_tokens=suffix_prompt_tokens,
            max_gen_len=32,
        )
        output = tokenizer.decode(output_tokens)
        logger.info("Generated output (first line): {}".format(output.split("\n")[0]))
        match = re.search(r"\d+", output)
        model_output = match.group() if match else None
        answers = item["answer"]
        recall = 1 if model_output in answers and model_output is not None else 0
        total_recall += recall
        total_items += 1
        result = {
            "index": item["index"],
            "query": item["query"],
            "answer": answers,
            "model_output": model_output,
            "recall": recall,
        }
        logger.info(f"Result: {json.dumps(result)}")
        json.dump(result, results_file)
        results_file.write("\n")
        results_file.flush()

final_recall = total_recall / total_items if total_items > 0 else 0
logger.info(f"Average recall: {final_recall}")
