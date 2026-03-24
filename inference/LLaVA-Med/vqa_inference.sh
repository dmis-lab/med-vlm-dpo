set -e

export CUDA_VISIBLE_DEVICES=2

PYTHONPATH=. python llava/eval/model_vqa.py \
    --conv-mode mistral_instruct \
    --model-path microsoft/llava-med-v1.5-mistral-7b \
    --question-file data/benchmark_question_file/pvqa_test_llavamed_format.jsonl \
    --image-folder data/VQA/vqarad/images \
    --answers-file inference_output/vqarad_inference.jsonl \
    --temperature 0.8 \
    --batch-size 1

echo "All top-n batch processing complete."