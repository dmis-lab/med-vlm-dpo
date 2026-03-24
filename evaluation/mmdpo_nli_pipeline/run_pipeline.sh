#!/bin/bash
set -e

echo "========================================"
echo "  MMDPO NLI Evaluation Pipeline Running  "
echo "========================================"

echo ""
echo "🔧 Loading configuration..."
python3 - << 'EOF'
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'scripts'))

try:
    from utils import load_config, build_paths
except ImportError:
    sys.path.append(os.getcwd())
    from utils import load_config, build_paths
    
config = load_config()
paths = build_paths(config)
print("✔ Dataset:", config["dataset"])
print("✔ Backbone:", config["backbone"])
print("✔ Model:", config["model"])
print("✔ Inference Dir:", paths["inference_dir"])
print("✔ Batch Input Dir:", paths["batch_input_dir"])
EOF
echo ""

echo "========================================"
echo " STEP 01: Generate batch input JSONL     "
echo "========================================"
python3 scripts/01_gen_batch_input.py
echo ""

echo "========================================"
echo " STEP 02: Upload batch requests to GPT   "
echo "========================================"
export OPENAI_API_KEY="Your API key"

python3 scripts/02_upload_batches.py
echo ""

echo "========================================"
echo " STEP 03: Check batch status continuously"
echo "   (This may take minutes to hours)      "
echo "========================================"
python3 scripts/03_check_batches.py
echo ""

echo "========================================"
echo " STEP 04: Download batch results         "
echo "========================================"
python3 scripts/04_download_results.py
echo ""

echo "========================================"
echo " STEP 05: Merge inference + GPT results  "
echo "========================================"
python3 scripts/05_merge_results.py
echo ""

echo "========================================"
echo " STEP 06: Compute NLI scores             "
echo "========================================"
python3 scripts/06_compute_scores.py
echo ""

echo "========================================"
echo " STEP 07: Generate summary               "
echo "========================================"
python3 scripts/07_summary.py
echo ""

echo "========================================"
echo "   🎉 Pipeline completed successfully!   "
echo "========================================"
