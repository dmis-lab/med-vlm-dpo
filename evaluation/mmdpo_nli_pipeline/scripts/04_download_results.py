import os, json, openai
from utils import load_config, apply_api_key, build_paths, ensure_dir

config = load_config()
paths = build_paths(config)
apply_api_key(config)

input_dir = paths["batch_input_dir"]
output_dir = paths["batch_output_dir"]
ensure_dir(output_dir)

log_path = os.path.join(input_dir, "batch_log.json")
batch_log = json.load(open(log_path, "r", encoding="utf-8"))

client = openai.OpenAI()

for fname, batch_id in batch_log.items():
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        print(f"[04] SKIP {fname} (status={batch.status})")
        continue

    out_file_id = batch.output_file_id
    content = client.files.content(out_file_id)

    output_path = os.path.join(output_dir, fname.replace("_batch_input", "_batch_output"))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content.text)

    print(f"[04] Saved → {output_path}")
