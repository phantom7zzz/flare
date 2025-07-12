# model_config/_generate_flare_config.py
import os
import yaml
import argparse
from datetime import datetime

def generate_flare_config(model_name):
    """
    生成FLARE任务配置文件
    """
    fintune_data_path = os.path.join("training_data/", f"{model_name}")
    checkpoint_path = os.path.join("checkpoints/", f"{model_name}_flare")
    
    # FLARE特定的配置
    data = {
        "model": f"{model_name}_flare",
        "data_path": fintune_data_path,
        "checkpoint_path": checkpoint_path,
        "pretrained_model_name_or_path": "../weights/RDT/rdt-1b",
        "cuda_visible_device": "0,1,2,3",
        "train_batch_size": 16,  # 可能需要减小batch size以适应FLARE的额外内存需求
        "sample_batch_size": 32,
        "max_train_steps": 25000,  # 可能需要更多训练步数
        "checkpointing_period": 2500,
        "sample_period": 100,
        "checkpoints_total_limit": 40,
        "learning_rate": 8e-5,  # 稍微降低学习率以确保稳定训练
        "dataloader_num_workers": 8,
        "state_noise_snr": 40,
        "gradient_accumulation_steps": 2,  # 增加梯度累积以应对更小的batch size
        "dataset_type": "finetune",
        
        # FLARE特定参数
        "num_future_tokens": 32,
        "activation_layer": 6,
        "alignment_loss_weight": 0.1,
    }
    
    task_config_path = os.path.join("model_config/", f"{model_name}_flare.yml")
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_comment = f"# FLARE任务配置模板\n# 生成时间: {current_time}\n\n"
    
    with open(task_config_path, "w") as f:
        f.write(time_comment)
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    if not os.path.exists(fintune_data_path):
        os.makedirs(fintune_data_path)
    
    print(f"FLARE配置已生成: {task_config_path}")
    return task_config_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FLARE finetune config.")
    parser.add_argument("model_name", type=str, help="The name of the task (e.g., beat_block_hammer)")
    args = parser.parse_args()
    model_name = args.model_name
    
    generate_flare_config(model_name)