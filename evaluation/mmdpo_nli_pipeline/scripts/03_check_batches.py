import os, json, time, openai
from utils import load_config, apply_api_key, build_paths

config = load_config()
paths = build_paths(config)
apply_api_key(config)

log_path = os.path.join(paths["batch_input_dir"], "batch_log.json")
batch_log = json.load(open(log_path, "r", encoding="utf-8"))

client = openai.OpenAI()

print("[03] Checking batch statuses and progress...")

while True:
    all_done = True
    current_time = time.strftime("%H:%M:%S")
    print(f"\n[{current_time}] --------------------------------------------------")

    for fname, batch_id in batch_log.items():
        try:
            # 배치 정보 전체 가져오기
            batch = client.batches.retrieve(batch_id)
            status = batch.status
            
            # 진행률 계산 로직 추가
            counts = batch.request_counts
            progress_msg = ""
            
            if counts and counts.total is not None:
                completed = counts.completed
                total = counts.total
                failed = counts.failed
                percent = (completed / total * 100) if total > 0 else 0
                progress_msg = f"| Progress: {completed}/{total} ({percent:.1f}%) | Failed: {failed}"
            else:
                progress_msg = "| Preparing..."

            # 한 줄로 상태 출력
            print(f" - {fname}\n   └─ Status: {status} {progress_msg}")

            # 실패 시 에러 로그 출력 (기존 기능 유지)
            if status == "failed":
                print(f"      🚨 ERRORS: {batch.errors}")
                if batch.error_file_id:
                    try:
                        error_content = client.files.content(batch.error_file_id).text
                        print(f"      📄 Detail:\n{error_content}")
                    except:
                        pass

            # 종료 조건 체크 (완료/실패/만료/취소)
            if status not in ["completed", "failed", "expired", "cancelled"]:
                all_done = False
        
        except Exception as e:
            print(f" - {fname}: ⚠️ Error checking status ({e})")
            all_done = False

    if all_done:
        print("\n[03] All batches finished.")
        break

    time.sleep(60)