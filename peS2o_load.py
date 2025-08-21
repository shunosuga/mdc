import json
import os

from datasets import load_dataset

# ストリーミングで読み込み
dataset = load_dataset("allenai/peS2o", streaming=True, trust_remote_code=True)

# ディレクトリ作成
os.makedirs("./data/peS2o", exist_ok=True)

# JSONL形式で保存（1行に1レコード）
with open("./data/peS2o/papers.jsonl", "w", encoding="utf-8") as f:
    for i, example in enumerate(dataset["train"]):
        # 1行に1つのJSONオブジェクトを書き込み
        json.dump(example, f, ensure_ascii=False)
        f.write("\n")

        if (i + 1) % 1000 == 0:
            print(f"Saved {i + 1} papers...")

        if i >= 9999:  # 10000件で停止
            break

print("Saved as JSONL format - easy to browse with text editors")
