import os
import json
import random
import string
from utils import load_config, build_paths, ensure_dir

config = load_config()
paths = build_paths(config)

input_dir = paths["inference_dir"]
output_dir = paths["batch_input_dir"]
atomic_file = paths["atomic_facts"]
prompt_template_file = paths["prompt_template"]

ensure_dir(output_dir)

# Load prompt template
with open(prompt_template_file, "r", encoding="utf-8") as f:
    prompt_template = f.read()

# Load atomic facts
with open(atomic_file, "r", encoding="utf-8") as f:
    reference = json.load(f)
id_to_facts = {x["id"]: x["atomic_facts"] for x in reference}

used_ids = set()

for filename in os.listdir(input_dir):
    if not filename.endswith(".jsonl"):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename.replace(".jsonl", "_batch_input.jsonl"))

    count = 0
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            entry = json.loads(line)
            sample_id = entry.get("id")
            premise = entry.get("caption_generated", "").strip()
            hypothesis = id_to_facts.get(sample_id)

            if not sample_id or not premise or not hypothesis:
                continue

            custom_id = str(sample_id)
            while custom_id in used_ids:
                custom_id = f"{sample_id}_{''.join(random.choices(string.ascii_lowercase+string.digits, k=4))}"
            used_ids.add(custom_id)

            prompt = prompt_template.format(
                premise=premise,
                hypothesis=json.dumps(hypothesis, ensure_ascii=False)
            )

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": config["model"],
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1024
                }
            }

            outfile.write(json.dumps(request, ensure_ascii=False) + "\n")
            count += 1

    print(f"[01] {filename}: {count} requests → {output_path}")
