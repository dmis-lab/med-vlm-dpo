import os, json
from glob import glob
import openai
from utils import load_config, build_paths, apply_api_key

config = load_config()
paths = build_paths(config)
apply_api_key(config)

input_dir = paths["batch_input_dir"]
jsonl_files = glob(os.path.join(input_dir, "*.jsonl"))

log_path = os.path.join(input_dir, "batch_log.json")
batch_log = {}

client = openai.OpenAI()

for file_path in jsonl_files:
    file_name = os.path.basename(file_path)

    try:
        with open(file_path, "rb") as f:
            upload = client.files.create(file=f, purpose="batch")

        batch = client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_log[file_name] = batch.id
        print(f"[02] Uploaded {file_name} → {batch.id}")

    except Exception as e:
        print(f"❌ Error uploading {file_name}: {e}")

with open(log_path, "w", encoding="utf-8") as f:
    json.dump(batch_log, f, indent=4)

print(f"[02] Saved batch log → {log_path}")
