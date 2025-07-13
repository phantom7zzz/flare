#!/usr/bin/env python
"""
FLARE配置生成器
"""
import os
import yaml
import argparse
from datetime import datetime

def generate_flare_config(model_name, base_config_path=None):
    """生成FLARE任务配置文件"""
    finetune_data_path = os.path.join("training_data/", f"{model_name}")
    checkpoint_path = os.path.join("checkpoints/", f"{model_name}_flare")
    
    # FLARE配置数据
    data = {
        "model": f"{model_name}_flare",
        "data_path": finetune_data_path,
        "checkpoint_path": checkpoint_path,
        "pretrained_model_name_or_path": "../weights/RDT/rdt-1b",
        "cuda_visible_device": "0,1,2,3",
        "train_batch_size": 16,
        "sample_batch_size": 32,
        "max_train_steps": 25000,
        "checkpointing_period": 2500,
        "sample_period": 100,
        "checkpoints_total_limit": 40,
        "learning_rate": 8e-5,
        "dataloader_num_workers": 8,
        "state_noise_snr": 40,
        "gradient_accumulation_steps": 2,
        "dataset_type": "finetune",
        
        # FLARE特定参数
        "num_future_tokens": 32,
        "activation_layer": 6,
        "alignment_loss_weight": 0.1,
        "num_vl_fusion_layers": 4,
        "num_qformer_layers": 6,
        "alignment_temperature": 0.07,
        "future_obs_prob": 0.8,
        "enable_flare": True,
    }
    
    # 创建配置文件路径
    if not os.path.exists("model_config"):
        os.makedirs("model_config")
    
    task_config_path = os.path.join("model_config/", f"{model_name}_flare.yml")
    
    # 添加时间戳注释
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_comment = f"# FLARE配置生成于 {current_time}\n"
    
    # 写入配置文件
    with open(task_config_path, "w") as f:
        f.write(time_comment)
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    # 创建数据目录
    if not os.path.exists(finetune_data_path):
        os.makedirs(finetune_data_path)
    
    print(f"✅ FLARE配置已生成: {task_config_path}")
    print(f"✅ 数据目录已创建: {finetune_data_path}")
    return task_config_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FLARE configuration.")
    parser.add_argument("model_name", type=str, help="Name of the model/task")
    parser.add_argument("--base_config", type=str, default=None, help="Base config to inherit from")
    
    args = parser.parse_args()
    generate_flare_config(args.model_name, args.base_config)