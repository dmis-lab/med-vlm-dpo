set -e

export CUDA_VISIBLE_DEVICES=2

PYTHONPATH=. python llava/eval/inference_mimic.py \
    --conv-mode mistral_instruct \
    --model-path microsoft/llava-med-v1.5-mistral-7b \
    --question-file data/benchmark_question_file/mimic-cxr-findings_test_single_image.jsonl \
    --image-folder data/mimic/images \
    --answers-file inference_output/mimic_inference.jsonl \
    --temperature 0.8 \
    --batch-size 1

echo "All top-n batch processing complete."