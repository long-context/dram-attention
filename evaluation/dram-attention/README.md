# Introduction

Our evaluation uses the `Llama-3.1-8B-Instruct` model on the `niah_multikey_2` synthetic task from the RULER benchmark (https://arxiv.org/abs/2404.06654). This is a simple key-value retrieval task, where each key is a two-word phrase mapped to a numeric value. The model receives a list of key-value pairs and must retrieve the correct value when given a specific key.


```
pip install huggingface-hub
huggingface-cli login --token [INSERT_YOUR_HF_TOKEN]
bash download.sh
bash run_eval.sh
```
