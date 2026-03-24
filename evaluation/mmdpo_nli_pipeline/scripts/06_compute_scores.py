import os, json, re
from utils import load_config, build_paths, ensure_dir

def compute_scores(nli_input, n):
    # [수정 포인트 1] 입력이 문자열이면 리스트로 변환 시도
    if isinstance(nli_input, str):
        try:
            nli_list = json.loads(nli_input)
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 (예: 형식이 깨진 경우) 빈 리스트 처리
            nli_list = []
    elif isinstance(nli_input, list):
        nli_list = nli_input
    else:
        nli_list = []

    comp = hall = 0
    
    # [수정 포인트 2] Neutral 등 처리되지 않은 라벨 로깅을 위해 확인
    for item in nli_list:
        if not isinstance(item, str):
            continue
            
        # "1. Contradiction" -> "Contradiction"으로 정리
        label = re.sub(r"^\d+\.\s*", "", item).strip()
        
        # 대소문자 무시를 위해 lower() 사용 권장 (선택사항)
        if label == "Entailment":
            comp += 1
        elif label == "Partial Entailment":
            comp += 0.5
        elif label == "Contradiction":
            hall += 1
        # Neutral은 점수에 영향 없음 (0점)

    # 0으로 나누기 방지
    if n == 0:
        return 0.0, 0.0
        
    return comp/n, hall/n

config = load_config()
paths = build_paths(config)

src_dir = paths["merged_output_dir"]
dst_dir = paths["nli_eval_dir"]
ensure_dir(dst_dir)

print("[06] Calculating scores...")

for fname in os.listdir(src_dir):
    if not fname.endswith(".json"):
        continue

    # 파일 읽기
    file_path = os.path.join(src_dir, fname)
    data = json.load(open(file_path, "r", encoding="utf-8"))
    
    # 점수 계산
    for entry in data:
        facts = entry.get("atomic_facts", [])
        gpt_response = entry.get("gpt4_nli")
        
        comp, hall = compute_scores(gpt_response, len(facts))
        
        entry["nli_scores"] = {
            "comp": round(comp, 4),
            "hall": round(hall, 4)
        }

    # 결과 저장
    out_path = os.path.join(dst_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f" -> Processed {fname}")

print("[06] Done.")