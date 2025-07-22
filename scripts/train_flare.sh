#!/bin/bash

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: bash train_flare.sh <CONFIG_NAME>"
    echo "例如: bash train_flare.sh my_task_flare"
    exit 1
fi

CONFIG_NAME="$1"
CONFIG_FILE="model_config/$CONFIG_NAME.yml"

echo "🚀 开始FLARE训练..."
echo "📋 配置文件: $CONFIG_FILE"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件 $CONFIG_FILE 不存在!"
    echo "请先运行: python scripts/generate_flare_config.py <model_name>"
    exit 1
fi

# 环境变量设置
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0
export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="../weights/RDT/siglip-so400m-patch14-384"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export WANDB_PROJECT="RDT_FLARE"
export WANDB_DEFAULT_RUN_NAME=$CONFIG_NAME
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 从YAML配置文件读取参数
echo "📖 读取配置参数..."
PRETRAINED_MODEL_NAME=$(python scripts/read_yaml.py "$CONFIG_FILE" pretrained_model_name_or_path)
TRAIN_BATCH_SIZE=$(python scripts/read_yaml.py "$CONFIG_FILE" train_batch_size)
SAMPLE_BATCH_SIZE=$(python scripts/read_yaml.py "$CONFIG_FILE" sample_batch_size)
MAX_TRAIN_STEPS=$(python scripts/read_yaml.py "$CONFIG_FILE" max_train_steps)
CHECKPOINTING_PERIOD=$(python scripts/read_yaml.py "$CONFIG_FILE" checkpointing_period)
SAMPLE_PERIOD=$(python scripts/read_yaml.py "$CONFIG_FILE" sample_period)
CHECKPOINTS_TOTAL_LIMIT=$(python scripts/read_yaml.py "$CONFIG_FILE" checkpoints_total_limit)
LEARNING_RATE=$(python scripts/read_yaml.py "$CONFIG_FILE" learning_rate)
DATALOADER_NUM_WORKERS=$(python scripts/read_yaml.py "$CONFIG_FILE" dataloader_num_workers)
DATASET_TYPE=$(python scripts/read_yaml.py "$CONFIG_FILE" dataset_type)
STATE_NOISE_SNR=$(python scripts/read_yaml.py "$CONFIG_FILE" state_noise_snr)
GRAD_ACCUM_STEPS=$(python scripts/read_yaml.py "$CONFIG_FILE" gradient_accumulation_steps)
OUTPUT_DIR=$(python scripts/read_yaml.py "$CONFIG_FILE" checkpoint_path)
CUDA_USE=$(python scripts/read_yaml.py "$CONFIG_FILE" cuda_visible_device)

# FLARE特定参数
NUM_FUTURE_TOKENS=$(python scripts/read_yaml.py "$CONFIG_FILE" num_future_tokens 2>/dev/null || echo "32")
ACTIVATION_LAYER=$(python scripts/read_yaml.py "$CONFIG_FILE" activation_layer 2>/dev/null || echo "6")
ALIGNMENT_LOSS_WEIGHT=$(python scripts/read_yaml.py "$CONFIG_FILE" alignment_loss_weight 2>/dev/null || echo "0.1")
NUM_VL_FUSION_LAYERS=$(python scripts/read_yaml.py "$CONFIG_FILE" num_vl_fusion_layers 2>/dev/null || echo "4")
NUM_QFORMER_LAYERS=$(python scripts/read_yaml.py "$CONFIG_FILE" num_qformer_layers 2>/dev/null || echo "6")
ALIGNMENT_TEMPERATURE=$(python scripts/read_yaml.py "$CONFIG_FILE" alignment_temperature 2>/dev/null || echo "0.07")

# 清理引号
PRETRAINED_MODEL_NAME=$(echo "$PRETRAINED_MODEL_NAME" | tr -d '"')
CUDA_USE=$(echo "$CUDA_USE" | tr -d '"')
OUTPUT_DIR=$(echo "$OUTPUT_DIR" | tr -d '"')

# 创建输出目录
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "📁 创建输出目录: $OUTPUT_DIR"
else
    echo "📁 输出目录已存在: $OUTPUT_DIR"
fi

# 设置GPU
export CUDA_VISIBLE_DEVICES=$CUDA_USE
echo "🔧 使用GPU: $CUDA_USE"

# 计算数据集统计信息
echo "📊 计算数据集统计信息..."
python -m data.compute_dataset_stat_hdf5 --task_name $CONFIG_NAME

# 启动FLARE训练
echo "🎯 启动FLARE训练..."
accelerate launch --main_process_port=28499 main.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_model_name_or_path=$PRETRAINED_MODEL_NAME \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --future_vision_encoder_path /home/deng_xiang/qian_daichao/RoboTwin/policy/RDT_flare/siglip2-large-patch16-256 \
    --current_vision_image_size 384 \
    --future_vision_image_size 256 \
    --max_text_length 32 \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --sample_batch_size=$SAMPLE_BATCH_SIZE \
    --max_train_steps=$MAX_TRAIN_STEPS \
    --checkpointing_period=$CHECKPOINTING_PERIOD \
    --sample_period=$SAMPLE_PERIOD \
    --checkpoints_total_limit=$CHECKPOINTS_TOTAL_LIMIT \
    --lr_scheduler="constant" \
    --learning_rate=$LEARNING_RATE \
    --mixed_precision="bf16" \
    --dataloader_num_workers=$DATALOADER_NUM_WORKERS \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=$STATE_NOISE_SNR \
    --load_from_hdf5 \
    --report_to=wandb \
    --precomp_lang_embed \
    --gradient_accumulation_steps=$GRAD_ACCUM_STEPS \
    --model_config_path=$CONFIG_FILE \
    --CONFIG_NAME=$CONFIG_NAME \
    --enable_flare \
    --num_future_tokens=$NUM_FUTURE_TOKENS \
    --activation_layer=$ACTIVATION_LAYER \
    --alignment_loss_weight=$ALIGNMENT_LOSS_WEIGHT \
    --num_vl_fusion_layers=$NUM_VL_FUSION_LAYERS \
    --num_qformer_layers=$NUM_QFORMER_LAYERS \
    --alignment_temperature=$ALIGNMENT_TEMPERATURE

echo "🎉 FLARE训练完成!"