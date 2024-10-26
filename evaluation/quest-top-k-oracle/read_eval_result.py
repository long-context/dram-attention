import glob
import json


def calculate_average_recall(file_path):
    total_recall = 0
    total_items = 0

    with open(file_path, "r") as file:
        for line in file:
            try:
                data = json.loads(line)
                recall = data.get("recall")
                if recall is not None:
                    total_recall += recall
                    total_items += 1
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")

    if total_items > 0:
        average_recall = total_recall / total_items
        return average_recall
    else:
        return 0


# Find all evaluation result files
eval_files = glob.glob("eval_results_*.jsonl")

# Calculate average recall for each file
results = []
for file in eval_files:
    seq_len = file.split("_")[2].split(".")[0]  # Extract seq_len from filename
    average_recall = calculate_average_recall(file)
    results.append([seq_len, f"{average_recall:.4f}"])

# Sort results by seq_len (as integer)
results.sort(key=lambda x: int(x[0]))

# Print results as a table
print("Sequence Length | Average Recall")
print("-" * 16 + "|" + "-" * 15)
for seq_len, recall in results:
    print(f"{seq_len:>15} | {recall:>14}")
