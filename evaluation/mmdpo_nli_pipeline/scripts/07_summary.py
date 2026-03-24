import os, json
from utils import load_config, build_paths

config = load_config()
paths = build_paths(config)

src_dir = paths["nli_eval_dir"]
summary_path = os.path.join(src_dir, "summary_scores.txt")

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("file\tavg_comp\tavg_hall\n")

    for fname in sorted(os.listdir(src_dir)):
        if not fname.endswith(".json"):
            continue

        data = json.load(open(os.path.join(src_dir, fname), "r", encoding="utf-8"))
        comps = [e["nli_scores"]["comp"] for e in data]
        halls = [e["nli_scores"]["hall"] for e in data]

        avg_c = round(sum(comps)/len(comps), 4)
        avg_h = round(sum(halls)/len(halls), 4)

        f.write(f"{fname}\t{avg_c}\t{avg_h}\n")

print(f"[07] Summary saved → {summary_path}")
