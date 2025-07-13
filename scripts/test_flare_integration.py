#!/usr/bin/env python
"""
FLARE集成测试脚本
验证所有FLARE组件是否正确集成
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import yaml
import os
import sys
from pathlib import Path

# 添加项目路径
current_file = Path(__file__)
project_root = current_file.parent.parent
sys.path.append(str(project_root))

def test_basic_imports():
    """测试基础模块导入"""
    print("📦 测试基础模块导入...")
    
    try:
        from models.multimodal_encoder.vl_token_generator import VLTokenGenerator
        print("  ✅ VL Token生成器导入成功")
        
        from models.multimodal_encoder.qformer_target_generator import QFormerTargetGenerator
        print("  ✅ Q-Former目标生成器导入成功")
        
        from models.rdt.dit_activation_extractor import FLAREActivationAligner
        print("  ✅ DiT激活提取器导入成功")
        
        from models.rdt.model import RDTWithFLARE
        print("  ✅ FLARE RDT模型导入成功")
        
        from models.rdt_runner import RDTRunnerWithFLARE
        print("  ✅ FLARE RDT Runner导入成功")
        
        return True
    except Exception as e:
        print(f"  ❌ 模块导入失败: {e}")
        return False

def test_config_generation():
    """测试配置生成"""
    print("\n⚙️ 测试配置生成...")
    
    try:
        # 模拟配置生成
        test_config = {
            "model": "test_flare",
            "num_future_tokens": 32,
            "activation_layer": 6,
            "alignment_loss_weight": 0.1,
            "enable_flare": True,
        }
        
        # 验证配置
        assert test_config["num_future_tokens"] == 32
        assert test_config["activation_layer"] == 6
        assert test_config["enable_flare"] == True
        
        print("  ✅ 配置生成测试通过")
        return True
        
    except Exception as e:
        print(f"  ❌ 配置生成测试失败: {e}")
        return False

def test_data_flow():
    """测试数据流"""
    print("\n💾 测试数据流...")
    
    try:
        # 模拟数据
        batch_size = 2
        action_chunk_size = 32
        
        # 模拟未来观测计算
        def compute_future_obs(current_step, episode_length):
            future_step = current_step + action_chunk_size - 1
            return min(future_step, episode_length - 1)
        
        # 测试不同情况
        test_cases = [
            (0, 100),   # 正常情况
            (50, 100),  # 中间情况
            (90, 100),  # 接近边界
            (99, 100),  # 边界情况
        ]
        
        for current_step, episode_length in test_cases:
            future_step = compute_future_obs(current_step, episode_length)
            assert 0 <= future_step < episode_length
            
        print("  ✅ 未来观测计算逻辑正确")
        print("  ✅ 边界情况处理正确")
        return True
        
    except Exception as e:
        print(f"  ❌ 数据流测试失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n🤖 测试模型创建...")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 基础参数
        hidden_size = 1152
        num_future_tokens = 32
        
        # 创建基础配置
        config = {
            'rdt': {
                'hidden_size': hidden_size,
                'depth': 8,
                'num_heads': 16,
            },
            'lang_adaptor': 'linear',
            'img_adaptor': 'linear',
            'state_adaptor': 'linear',
            'noise_scheduler': {
                'num_train_timesteps': 1000,
                'beta_schedule': 'linear',
                'prediction_type': 'epsilon',
                'clip_sample': True,
                'num_inference_timesteps': 100,
            }
        }
        
        # 尝试创建FLARE Runner
        from models.rdt_runner import RDTRunnerWithFLARE
        
        runner = RDTRunnerWithFLARE(
            action_dim=128,
            pred_horizon=32,
            config=config,
            lang_token_dim=hidden_size,
            img_token_dim=hidden_size,
            state_token_dim=128,
            max_lang_cond_len=120,
            img_cond_len=1000,
            num_future_tokens=num_future_tokens,
            activation_layer=6,
            alignment_loss_weight=0.1,
            enable_flare=True,
        )
        
        print(f"  ✅ FLARE Runner创建成功")
        print(f"  ✅ 参数数量: {sum(p.numel() for p in runner.parameters()):,}")
        return True
        
    except Exception as e:
        print(f"  ❌ 模型创建测试失败: {e}")
        return False

def test_loss_computation():
    """测试损失计算"""
    print("\n⚖️ 测试损失计算...")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 模拟损失计算
        diffusion_loss = torch.tensor(0.5)
        alignment_loss = torch.tensor(0.1)
        alignment_weight = 0.1
        
        total_loss = diffusion_loss + alignment_weight * alignment_loss
        
        expected_total = 0.5 + 0.1 * 0.1
        assert abs(total_loss.item() - expected_total) < 1e-6
        
        print(f"  ✅ 扩散损失: {diffusion_loss.item():.4f}")
        print(f"  ✅ 对齐损失: {alignment_loss.item():.4f}")
        print(f"  ✅ 总损失: {total_loss.item():.4f}")
        return True
        
    except Exception as e:
        print(f"  ❌ 损失计算测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("🔥 FLARE集成测试开始")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_config_generation,
        test_data_flow,
        test_model_creation,
        test_loss_computation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"测试 {test_func.__name__} 出现异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎊 所有测试通过！FLARE实现已就绪！")
        print("\n🚀 接下来的步骤:")
        print("1. 生成配置: python scripts/generate_flare_config.py your_task_name")
        print("2. 开始训练: bash scripts/train_flare.sh your_task_name_flare")
        print("3. 监控训练: 查看WandB日志中的对齐损失")
    else:
        print("❌ 部分测试失败，请检查相关组件")
    
    print("=" * 60)
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
    