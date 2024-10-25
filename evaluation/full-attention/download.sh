#!/bin/bash

set -e # Exit on error

(
    cd /tmp
    wget -c https://huggingface.co/datasets/princeton-nlp/HELMET/resolve/main/data.tar.gz
    tar -xvzf data.tar.gz --no-same-owner data/ruler/niah_multikey_2
)

huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --include "original/*" --local-dir /tmp/model
