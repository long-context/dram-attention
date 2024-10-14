#!/bin/bash

set -e # Exit on error

for seq_len in 8192 16384 32768 65536 131072; do
    echo "Running evaluation with sequence length: $seq_len"
    python evaluate.py --seq-len $seq_len
done

echo "Generating summary report..."
python read_eval_result.py | tee report.txt
