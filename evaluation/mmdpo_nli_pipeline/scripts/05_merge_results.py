import os, json
from utils import load_config, build_paths, ensure_dir

config = load_config()
paths = build_paths(config)

inf_dir = paths["inference_dir"]
batch_out_dir = paths["batch_output_dir"]
atomic_file = paths["atomic_facts"]
merged_dir = paths["merged_output_dir"]

ensure_dir(merged_dir)

# Load atomic facts
# Atomic Map을 만들 때도 안전하게 키를 문자열로 변환합니다.
with open(atomic_file, "r", encoding="utf-8") as f:
    atomic_data = json.load(f)
atomic_map = {str(x["id"]): x for x in atomic_data}

print(f"[05] Merging results...")

for fname in os.listdir(inf_dir):
    if not fname.endswith(".jsonl"):
        continue

    prefix = fname.replace(".jsonl", "")
    inf_path = os.path.join(inf_dir, fname)
    batch_path = os.path.join(batch_out_dir, prefix + "_batch_output.jsonl")
    out_path = os.path.join(merged_dir, prefix + ".json")

    # Read batch outputs: custom_id → gpt4_nli
    batch_map = {}
    
    # 배치 파일이 없는 경우 건너뛰기 (예외 처리)
    if not os.path.exists(batch_path):
        print(f"⚠️ Warning: Batch output not found for {fname}. Skipping.")
        continue

    with open(batch_path, "r", encoding="utf-8") as bf:
        for line in bf:
            obj = json.loads(line)
            cid = obj["custom_id"]
            try:
                content = obj["response"]["body"]["choices"][0]["message"]["content"]
            except:
                content = None
            batch_map[cid] = content

    merged = []
    with open(inf_path, "r", encoding="utf-8") as inf:
        for line in inf:
            entry = json.loads(line)
            
            # 🔥 [수정] 여기서 문자열로 확실하게 변환!
            sid = str(entry["id"])

            # atomic_map 키도 위에서 문자열로 바꿨으므로 여기서도 문자열로 조회
            fact = atomic_map.get(sid)
            if not fact:
                continue

            # Find custom_id (prefix match)
            # 이제 sid가 문자열이므로 startswith가 정상 작동합니다.
            target_cid = next((cid for cid in batch_map if cid.startswith(sid)), None)
            nli = batch_map.get(target_cid)

            merged.append({
                "id": sid,
                "ground_truth": fact.get("gold_report") or fact.get("gold_caption"),
                "atomic_facts": fact["atomic_facts"],
                "model_response": entry["caption_generated"],
                "gpt4_nli": nli
            })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=4, ensure_ascii=False)

    print(f" Saved → {out_path}")

print("[05] Merge completed.")