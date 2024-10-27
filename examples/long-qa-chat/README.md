# Long Question Answering Chat

This example demonstrates how to use dram-attention in a simple application that enables users to interact with and ask questions about a very long document.

```
pip install termcolor
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --include "original/*" --local-dir /tmp/model

python chat.py prefill --prompt-file prompt.txt
python chat.py generate --top-k $((2*1024)) --lru-hbm-cache-size $((16*1024))
# Type 'quit' to exit the chat
```