#!/usr/bin/env python3
"""
FLARE训练诊断脚本
用于检查alignment_loss为0的原因
"""

import torch
import yaml
import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def debug_flare_data_loading(model_config_path="model_config/your_task_flare.yml"):
    """检查数据加载阶段的FLARE配置"""
    print("🔍 检查数据加载配置...")
    
    try:
        # 加载配置
        with open("configs/base.yaml", "r") as f:
            base_config = yaml.safe_load(f)
        
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)
        
        print(f"✅ 配置文件加载成功")
        print(f"   - Base config loaded")
        print(f"   - Model config: {model_config_path}")
        
        # 检查FLARE相关配置
        flare_enabled = model_config.get("enable_flare", False)
        print(f"📋 FLARE配置:")
        print(f"   - enable_flare: {flare_enabled}")
        print(f"   - num_future_tokens: {model_config.get('num_future_tokens', 'Not set')}")
        print(f"   - activation_layer: {model_config.get('activation_layer', 'Not set')}")
        print(f"   - alignment_loss_weight: {model_config.get('alignment_loss_weight', 'Not set')}")
        
        return base_config, model_config
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return None, None

def debug_dataset_creation(base_config, model_config_path):
    """检查数据集创建和未来观测生成"""
    print("\n🔍 检查数据集创建...")
    
    try:
        from train.dataset import VLAConsumerDatasetWithFLARE
        from transformers import AutoTokenizer
        from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
        
        # 创建编码器
        vision_encoder = SiglipVisionTower(
            vision_tower="google/siglip-so400m-patch14-384", 
            args=None
        )
        image_processor = vision_encoder.image_processor
        tokenizer = None  # 使用预计算的语言嵌入
        
        # 创建数据集（使用FLARE参数）
        dataset = VLAConsumerDatasetWithFLARE(
            model_config_path=model_config_path,
            config=base_config["dataset"],
            tokenizer=tokenizer,
            image_processor=image_processor,
            num_cameras=base_config["common"]["num_cameras"],
            img_history_size=base_config["common"]["img_history_size"],
            dataset_type="finetune",
            enable_future_obs=True,          # 强制启用
            future_obs_prob=1.0,            # 100%概率用于调试
            action_chunk_size=base_config["common"]["action_chunk_size"],
            use_hdf5=True,
            use_precomp_lang_embed=True,
        )
        
        print(f"✅ 数据集创建成功")
        print(f"   - enable_future_obs: {dataset.enable_future_obs}")
        print(f"   - future_obs_prob: {dataset.future_obs_prob}")
        print(f"   - action_chunk_size: {dataset.action_chunk_size}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_single_sample(dataset):
    """测试单个样本的未来观测生成"""
    print("\n🔍 测试单个样本...")
    
    try:
        # 获取一个样本
        sample = dataset[0]
        
        print(f"📊 样本数据检查:")
        print(f"   - 样本keys: {list(sample.keys())}")
        
        # 检查关键字段
        future_obs_image = sample.get('future_obs_image')
        has_future_obs = sample.get('has_future_obs', False)
        text_instruction = sample.get('text_instruction', '')
        
        print(f"   - has_future_obs: {has_future_obs}")
        print(f"   - future_obs_image: {future_obs_image is not None}")
        if future_obs_image is not None:
            print(f"   - future_obs_image shape: {future_obs_image.shape}")
            print(f"   - future_obs_image dtype: {future_obs_image.dtype}")
            print(f"   - future_obs_image range: [{future_obs_image.min():.3f}, {future_obs_image.max():.3f}]")
        
        print(f"   - text_instruction: '{text_instruction[:50]}...' (长度: {len(text_instruction)})")
        
        return sample
        
    except Exception as e:
        print(f"❌ 样本测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_dataloader(dataset):
    """测试数据加载器批次处理"""
    print("\n🔍 测试数据加载器...")
    
    try:
        from train.dataset import DataCollatorForVLAConsumerDatasetWithFLARE
        from torch.utils.data import DataLoader
        
        data_collator = DataCollatorForVLAConsumerDatasetWithFLARE(tokenizer=None)
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=0,
        )
        
        # 获取一个批次
        batch = next(iter(dataloader))
        
        print(f"📊 批次数据检查:")
        print(f"   - 批次keys: {list(batch.keys())}")
        
        # 检查未来观测相关字段
        if 'has_future_obs' in batch:
            has_future_obs = batch['has_future_obs']
            print(f"   - has_future_obs: {has_future_obs}")
            print(f"   - 有效未来观测数量: {has_future_obs.sum()}/{len(has_future_obs)}")
        else:
            print(f"   - ❌ has_future_obs字段缺失!")
        
        if 'future_obs_images' in batch:
            future_obs_images = batch['future_obs_images']
            print(f"   - future_obs_images shape: {future_obs_images.shape}")
            print(f"   - future_obs_images dtype: {future_obs_images.dtype}")
        else:
            print(f"   - ❌ future_obs_images字段缺失!")
        
        if 'text_instructions' in batch:
            text_instructions = batch['text_instructions']
            print(f"   - text_instructions数量: {len(text_instructions)}")
            print(f"   - 示例指令: '{text_instructions[0][:50]}...'")
        else:
            print(f"   - ❌ text_instructions字段缺失!")
        
        return batch
        
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_model_creation(base_config, model_config):
    """检查FLARE模型创建"""
    print("\n🔍 检查FLARE模型创建...")
    
    try:
        from models.rdt_runner import RDTRunnerWithFLARE
        from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
        
        vision_encoder = SiglipVisionTower(
            vision_tower="google/siglip-so400m-patch14-384", 
            args=None
        )
        
        img_cond_len = (base_config["common"]["img_history_size"] * 
                       base_config["common"]["num_cameras"] * 
                       vision_encoder.num_patches)
        
        # 创建FLARE模型
        rdt = RDTRunnerWithFLARE(
            action_dim=base_config["common"]["state_dim"],
            pred_horizon=base_config["common"]["action_chunk_size"],
            config=base_config["model"],
            lang_token_dim=base_config["model"]["lang_token_dim"],
            img_token_dim=base_config["model"]["img_token_dim"],
            state_token_dim=base_config["model"]["state_token_dim"],
            max_lang_cond_len=base_config["dataset"]["tokenizer_max_length"],
            img_cond_len=img_cond_len,
            # FLARE参数
            num_future_tokens=model_config.get('num_future_tokens', 32),
            activation_layer=model_config.get('activation_layer', 6),
            alignment_loss_weight=model_config.get('alignment_loss_weight', 0.1),
            enable_flare=model_config.get('enable_flare', True),
            dtype=torch.bfloat16,
        )
        
        print(f"✅ FLARE模型创建成功")
        print(f"   - enable_flare: {rdt.enable_flare}")
        print(f"   - num_future_tokens: {rdt.num_future_tokens}")
        print(f"   - activation_layer: {rdt.activation_layer}")
        print(f"   - alignment_loss_weight: {rdt.alignment_loss_weight}")
        
        # 检查FLARE组件
        print(f"   - VL token generator: {hasattr(rdt.model, 'vl_token_generator')}")
        print(f"   - Target generator: {hasattr(rdt.model, 'target_generator')}")
        print(f"   - Future obs tokens: {hasattr(rdt.model, 'future_obs_tokens')}")
        
        return rdt
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_loss_computation(rdt, batch):
    """测试损失计算过程"""
    print("\n🔍 测试损失计算过程...")
    
    try:
        # 模拟训练参数
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weight_dtype = torch.bfloat16
        
        # 准备数据
        lang_tokens = batch["lang_embeds"].to(dtype=weight_dtype)
        lang_attn_mask = batch["lang_attn_mask"]
        img_tokens = torch.randn(2, 1000, 1152).to(dtype=weight_dtype)  # 模拟图像token
        state_tokens = batch["states"].to(dtype=weight_dtype)[:, -1:, :]
        action_gt = batch["actions"].to(dtype=weight_dtype)
        state_elem_mask = batch["state_elem_mask"].to(dtype=weight_dtype).unsqueeze(1)
        ctrl_freqs = batch["ctrl_freqs"]
        
        # FLARE数据
        future_obs_images = batch.get("future_obs_images")
        text_instructions = batch.get("text_instructions", [""] * img_tokens.shape[0])
        has_future_obs = batch.get("has_future_obs")
        
        print(f"📊 损失计算输入检查:")
        print(f"   - future_obs_images: {future_obs_images is not None}")
        print(f"   - text_instructions: {len(text_instructions)} items")
        print(f"   - has_future_obs: {has_future_obs}")
        
        if future_obs_images is not None:
            print(f"   - future_obs_images shape: {future_obs_images.shape}")
            
            # 模拟视觉编码
            future_vision_embeds = torch.randn(2, 729, 1152).to(dtype=weight_dtype)
            print(f"   - future_vision_embeds shape: {future_vision_embeds.shape}")
        else:
            future_vision_embeds = None
        
        # 尝试计算损失
        print(f"\n🎯 计算FLARE损失...")
        total_loss, loss_dict = rdt.compute_loss_with_flare(
            lang_tokens=lang_tokens,
            lang_attn_mask=lang_attn_mask,
            img_tokens=img_tokens,
            state_tokens=state_tokens,
            action_gt=action_gt,
            action_mask=state_elem_mask,
            ctrl_freqs=ctrl_freqs,
            future_vision_tokens=future_vision_embeds,
            text_instructions=text_instructions,
            has_future_obs=has_future_obs,
        )
        
        print(f"✅ 损失计算成功!")
        print(f"   - total_loss: {total_loss:.6f}")
        print(f"   - diffusion_loss: {loss_dict.get('diffusion_loss', 'N/A'):.6f}")
        print(f"   - alignment_loss: {loss_dict.get('alignment_loss', 'N/A'):.6f}")
        print(f"   - used_flare: {loss_dict.get('used_flare', 'N/A')}")
        print(f"   - alignment_loss_weight: {loss_dict.get('alignment_loss_weight', 'N/A')}")
        
        # 诊断alignment_loss为0的原因
        if loss_dict.get('alignment_loss', 0) == 0:
            print(f"\n⚠️  ALIGNMENT LOSS为0的可能原因:")
            if not loss_dict.get('used_flare', False):
                print(f"   - FLARE未被使用 (used_flare=False)")
                print(f"   - 检查: enable_flare, future_vision_tokens, text_instructions")
            else:
                print(f"   - FLARE被使用但alignment_loss=0")
                print(f"   - 可能原因: 激活提取失败、目标生成失败、对齐计算错误")
        
        return total_loss, loss_dict
        
    except Exception as e:
        print(f"❌ 损失计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """主诊断流程"""
    print("🚀 FLARE训练诊断开始...")
    print("=" * 60)
    
    # 检查配置文件（修改为你的配置文件路径）
    model_config_path = input("请输入你的模型配置文件路径 (例: model_config/your_task_flare.yml): ").strip()
    if not model_config_path:
        model_config_path = "model_config/example_flare.yml"
    
    if not os.path.exists(model_config_path):
        print(f"❌ 配置文件不存在: {model_config_path}")
        return
    
    # Step 1: 检查配置
    base_config, model_config = debug_flare_data_loading(model_config_path)
    if base_config is None or model_config is None:
        return
    
    # Step 2: 检查数据集
    dataset = debug_dataset_creation(base_config, model_config_path)
    if dataset is None:
        return
    
    # Step 3: 测试单个样本
    sample = debug_single_sample(dataset)
    if sample is None:
        return
    
    # Step 4: 测试批次处理
    batch = debug_dataloader(dataset)
    if batch is None:
        return
    
    # Step 5: 检查模型
    rdt = debug_model_creation(base_config, model_config)
    if rdt is None:
        return
    
    # Step 6: 测试损失计算
    total_loss, loss_dict = debug_loss_computation(rdt, batch)
    
    print("\n" + "=" * 60)
    print("🎊 诊断完成!")
    
    if loss_dict and loss_dict.get('alignment_loss', 0) > 0:
        print("✅ FLARE训练配置正常，alignment_loss > 0")
    else:
        print("❌ 发现问题，alignment_loss = 0")
        print("\n🔧 建议检查:")
        print("   1. 确保enable_flare=True")
        print("   2. 检查数据集中future_obs_prob设置")
        print("   3. 验证HDF5数据集是否包含足够的时间步")
        print("   4. 确认模型配置中FLARE参数设置正确")

if __name__ == "__main__":
    main()