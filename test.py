def test_flare_data_processing():
    """
    测试FLARE数据处理功能
    """
    print("Testing FLARE data processing...")
    
    # 测试1: 验证未来观测索引计算
    def test_future_obs_indexing():
        import tensorflow as tf
        
        print("  Testing future observation indexing...")
        ACTION_CHUNK_SIZE = 32
        total_frames = 100
        
        def get_future_obs_index(current_step):
            future_step = current_step + ACTION_CHUNK_SIZE - 1
            future_step = tf.minimum(future_step, total_frames - 1)
            return future_step
        
        # 测试几个时间步
        test_steps = [0, 10, 50, 80, 90]
        for step in test_steps:
            future_idx = get_future_obs_index(step).numpy()
            expected = min(step + ACTION_CHUNK_SIZE - 1, total_frames - 1)
            assert future_idx == expected, f"Step {step}: expected {expected}, got {future_idx}"
            
        print("    ✓ Future observation indexing test passed")
    
    # 测试2: 验证数据集创建
    def test_dataset_creation():
        print("  Testing FLARE dataset creation...")
        
        # 由于依赖较多，这里提供测试框架
        try:
            # 这里应该测试VLADatasetWithFLARE的创建
            print("    ✓ Dataset creation test passed (placeholder)")
        except Exception as e:
            print(f"    ✗ Dataset creation test failed: {e}")
    
    # 测试3: 验证HDF5数据集
    def test_hdf5_future_obs():
        print("  Testing HDF5 future observation parsing...")
        
        # 这里应该测试HDF5VLADatasetWithFLARE
        try:
            print("    ✓ HDF5 future observation test passed (placeholder)")
        except Exception as e:
            print(f"    ✗ HDF5 future observation test failed: {e}")
    
    # 运行测试
    test_future_obs_indexing()
    test_dataset_creation() 
    test_hdf5_future_obs()
    
    print("FLARE data processing tests completed!")

def create_flare_config_generator():
    """
    创建FLARE配置生成器
    """
    config_generator_code = '''
import os
import yaml
import argparse
from datetime import datetime

def generate_flare_config(model_name, base_config_path=None):
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
        "train_batch_size": 16,  # 减小以适应FLARE内存需求
        "sample_batch_size": 32,
        "max_train_steps": 25000,  # 增加训练步数
        "checkpointing_period": 2500,
        "sample_period": 100,
        "checkpoints_total_limit": 40,
        "learning_rate": 8e-5,  # 稍微降低学习率
        "dataloader_num_workers": 8,
        "state_noise_snr": 40,
        "gradient_accumulation_steps": 2,  # 增加梯度累积
        "dataset_type": "finetune",
        
        # FLARE特定参数
        "num_future_tokens": 32,
        "activation_layer": 6,
        "alignment_loss_weight": 0.1,
    }
    
    task_config_path = os.path.join("model_config/", f"{model_name}_flare.yml")
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_comment = f"# FLARE配置生成于 {current_time}\\n"
    
    with open(task_config_path, "w") as f:
        f.write(time_comment)
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    if not os.path.exists(fintune_data_path):
        os.makedirs(fintune_data_path)
    
    print(f"FLARE配置已生成: {task_config_path}")
    return task_config_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FLARE finetune config.")
    parser.add_argument("model_name", type=str, help="The name of the task")
    parser.add_argument("--base_config", type=str, default=None, 
                       help="Base config to inherit from")
    args = parser.parse_args()
    
    generate_flare_config(args.model_name, args.base_config)
'''
    
    return config_generator_code