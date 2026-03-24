#!/bin/bash

# Deepspeed DPO Training Script for HuatuoGPT-Vision-7B model

export PYTHONPATH=$PYTHONPATH:<YOUR_WORKSPACE_PATH>/MMDPO

wandb login <YOUR_WANDB_API_KEY>

# global batch size 16
GLOBAL_BATCH_SIZE="16"
CUDA="0,1,2,3"
PER_DEVICE_TRAIN_BATCH_SIZE="4"
PER_DEVICE_EVAL_BATCH_SIZE="4"
GRADIENT_ACCUMULATION_STEPS="1"


DATA_PATH="<YOUR_WORKSPACE_PATH>/mmdpo/data/sample/huatuo/huatuo-IT_dpo___wth_gpt4o_gen_rejected.json"  

IMAGE_FOLDER="<YOUR_WORKSPACE_PATH>/data/pubmedvision"

OUTPUT_DIR="<YOUR_WORKSPACE_PATH>/checkpoints/mmdpo/huatuo/instruction_tuning/dpo+nll/gpt4o_gen_rejected"
RUN_NAME="huatuo-IT___gpt4o_gen_rejected_dpo+nll"

NUM_TRAIN_EPOCHS="3"
SAVE_TOTAL_LIMIT="3"

LEARNING_RATE="1e-7"
MM_PROJECTOR_LR="0"

CHECKPOINT="<YOUR_WORKSPACE_PATH>/models/HuatuoGPT-Vision-7B"

deepspeed --include localhost:$CUDA --master_port $((RANDOM + 30000)) mmdpo/models/HuatuoGPT-Vision-7B/train_dpo.py \
    --mm_projector_lr $MM_PROJECTOR_LR \
    --deepspeed mmdpo/models/HuatuoGPT-Vision-7B/scripts/zero3.json \
    --model_name_or_path $CHECKPOINT \
    --version qwen_2 \
    --image_folder $IMAGE_FOLDER \
    --data_path $DATA_PATH \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}/${RUN_NAME}___lr${LEARNING_RATE}_gb${GLOBAL_BATCH_SIZE}" \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 50000 \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "${RUN_NAME}___lr${LEARNING_RATE}_gb${GLOBAL_BATCH_SIZE}" \
    --beta 0.1 \
    --sft_weight 1.0