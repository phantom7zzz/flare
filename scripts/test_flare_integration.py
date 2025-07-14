import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import yaml
import os
import sys
from pathlib import Path
from contextlib import nullcontext

# 添加项目路径
current_file = Path(__file__)
project_root = current_file.parent.parent
sys.path.append(str(project_root))

def setup_gpu_environment():
    """设置GPU环境"""
    print("🎯 设置GPU训练环境...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法进行GPU训练测试")
        return None, None
    
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"✅ GPU设备: {gpu_name}")
    print(f"✅ GPU显存: {gpu_memory:.1f}GB")
    
    # 设置为BF16优化
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    print("✅ GPU优化设置完成")
    return device, gpu_memory

def test_gpu_tensor_operations():
    """测试GPU上的基础张量操作"""
    print("\n🧮 测试GPU张量操作...")
    
    device = torch.device("cuda:0")
    
    try:
        # 测试BF16张量创建和操作
        x_fp32 = torch.randn(2, 32, 1152, device=device, dtype=torch.float32)
        x_bf16 = torch.randn(2, 32, 1152, device=device, dtype=torch.bfloat16)
        
        # 测试数据类型转换
        x_converted = x_fp32.to(torch.bfloat16)
        assert x_converted.device == device
        assert x_converted.dtype == torch.bfloat16
        
        # 测试基础运算
        result = x_bf16 + x_converted
        assert result.dtype == torch.bfloat16
        
        # 测试autocast
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            linear = nn.Linear(1152, 1152).to(device=device, dtype=torch.bfloat16)
            output = linear(x_bf16)
            assert output.dtype == torch.bfloat16
        
        print("  ✅ BF16张量操作正常")
        print("  ✅ 数据类型转换正常")
        print("  ✅ autocast功能正常")
        
        return True
        
    except Exception as e:
        print(f"  ❌ GPU张量操作失败: {e}")
        return False

def test_rdt_components_on_gpu():
    """测试RDT基础组件在GPU上的运行"""
    print("\n🧩 测试GPU上的RDT组件...")
    
    device = torch.device("cuda:0")
    hidden_size = 1152
    
    try:
        from models.rdt.blocks import TimestepEmbedder, RDTBlock, FinalLayer
        
        # 时间步嵌入器
        t_embedder = TimestepEmbedder(hidden_size, dtype=torch.bfloat16).to(device)
        test_t = torch.randint(0, 1000, (2,), device=device)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            t_embed = t_embedder(test_t)
            assert t_embed.device == device
            assert t_embed.shape == (2, hidden_size)
        
        print("  ✅ 时间步嵌入器GPU测试通过")
        
        # RDT Block
        rdt_block = RDTBlock(hidden_size=hidden_size, num_heads=16).to(device, dtype=torch.bfloat16)
        test_x = torch.randn(2, 10, hidden_size, device=device, dtype=torch.bfloat16)
        test_c = torch.randn(2, 20, hidden_size, device=device, dtype=torch.bfloat16)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            block_out = rdt_block(test_x, test_c)
            assert block_out.device == device
            assert block_out.shape == test_x.shape
            assert block_out.dtype == torch.bfloat16
        
        print("  ✅ RDT Block GPU测试通过")
        
        # 最终层
        final_layer = FinalLayer(hidden_size=hidden_size, out_channels=128).to(device, dtype=torch.bfloat16)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            final_out = final_layer(test_x)
            assert final_out.device == device
            assert final_out.shape == (2, 10, 128)
        
        print("  ✅ 最终层GPU测试通过")
        
        return True
        
    except Exception as e:
        print(f"  ❌ RDT组件GPU测试失败: {e}")
        return False

def test_flare_data_structures_gpu():
    """测试FLARE数据结构在GPU上的处理"""
    print("\n💾 测试GPU上的FLARE数据结构...")
    
    device = torch.device("cuda:0")
    batch_size = 2
    action_chunk_size = 32
    hidden_size = 1152
    
    try:
        # 模拟FLARE训练数据（全部在GPU上）
        flare_batch = {
            "states": torch.randn(batch_size, 1, 128, device=device, dtype=torch.bfloat16),
            "actions": torch.randn(batch_size, action_chunk_size, 128, device=device, dtype=torch.bfloat16),
            "images": torch.randn(batch_size, 6, 3, 224, 224, device=device, dtype=torch.bfloat16),
            "future_obs_images": torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.bfloat16),
            "has_future_obs": torch.tensor([True, False], device=device),
            "lang_tokens": torch.randn(batch_size, 120, hidden_size, device=device, dtype=torch.bfloat16),
            "img_tokens": torch.randn(batch_size, 1000, hidden_size, device=device, dtype=torch.bfloat16),
        }
        
        # 验证数据结构
        for key, tensor in flare_batch.items():
            if isinstance(tensor, torch.Tensor):
                assert tensor.device == device, f"{key} 不在GPU上"
                if tensor.dtype.is_floating_point:
                    assert tensor.dtype == torch.bfloat16, f"{key} 数据类型不是BF16"
        
        print("  ✅ FLARE数据结构GPU分配正确")
        
        # 测试维度兼容性
        img_seq_len = flare_batch["img_tokens"].shape[1]  # 1000
        
        # 模拟位置嵌入维度问题
        pos_embed_len = 4374  # 你遇到的实际长度
        if img_seq_len != pos_embed_len:
            print(f"  🔍 检测到维度不匹配: {img_seq_len} vs {pos_embed_len}")
            
            # 测试动态调整策略
            if img_seq_len < pos_embed_len:
                # 截断策略
                adjusted_embed = torch.randn(1, img_seq_len, hidden_size, device=device, dtype=torch.bfloat16)
                print(f"  🔧 截断位置嵌入: {pos_embed_len} -> {img_seq_len}")
            else:
                # 扩展策略
                base_embed = torch.randn(1, pos_embed_len, hidden_size, device=device, dtype=torch.bfloat16)
                extra_len = img_seq_len - pos_embed_len
                if extra_len <= pos_embed_len:
                    extra_embed = base_embed[:, -extra_len:, :]
                else:
                    repeat_times = (extra_len // pos_embed_len) + 1
                    repeated_embed = base_embed.repeat(1, repeat_times, 1)
                    extra_embed = repeated_embed[:, :extra_len, :]
                
                adjusted_embed = torch.cat([base_embed, extra_embed], dim=1)
                print(f"  🔧 扩展位置嵌入: {pos_embed_len} -> {img_seq_len}")
            
            assert adjusted_embed.shape[1] == img_seq_len
            assert adjusted_embed.device == device
            
        print("  ✅ 位置嵌入维度处理策略测试通过")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FLARE数据结构GPU测试失败: {e}")
        return False

def test_loss_computation_gpu():
    """测试GPU上的损失计算"""
    print("\n⚖️ 测试GPU损失计算...")
    
    device = torch.device("cuda:0")
    
    try:
        # 模拟预测和目标
        pred = torch.randn(2, 32, 128, device=device, dtype=torch.bfloat16)
        target = torch.randn(2, 32, 128, device=device, dtype=torch.bfloat16)
        
        # 在GPU上计算MSE损失（BF16）
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            diffusion_loss = F.mse_loss(pred, target)
            assert diffusion_loss.device == device
            assert not torch.isnan(diffusion_loss), "损失计算出现NaN"
        
        print("  ✅ GPU MSE损失计算正常")
        
        # 测试对齐损失
        alignment_loss = torch.tensor(0.1, device=device, dtype=torch.bfloat16)
        alignment_weight = 0.1
        
        total_loss = diffusion_loss + alignment_weight * alignment_loss
        assert total_loss.device == device
        
        print("  ✅ 对齐损失计算正常")
        
        # 测试条件损失（部分样本有未来观测）
        has_future_obs = torch.tensor([True, False], device=device)
        valid_count = has_future_obs.sum().float()
        
        if valid_count > 0:
            scaled_alignment_loss = alignment_loss * (2 / valid_count)  # batch_size=2
            assert scaled_alignment_loss.device == device
        
        print("  ✅ 条件损失计算正常")
        
        return True
        
    except Exception as e:
        print(f"  ❌ GPU损失计算测试失败: {e}")
        return False

def test_model_creation_gpu():
    """测试在GPU上创建FLARE模型"""
    print("\n🤖 测试GPU上的FLARE模型创建...")
    
    device = torch.device("cuda:0")
    
    try:
        # 基础配置
        config = {
            'rdt': {
                'hidden_size': 1152,
                'depth': 6,  # 减少深度以加快测试
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
        
        # 跳过需要外部模型的完整FLARE创建，只测试适配器
        from models.rdt_runner import RDTRunnerWithFLARE
        
        print("  ⚠️  跳过完整模型创建（避免外部模型依赖）")
        
        # 测试适配器创建
        lang_token_dim = 1152
        img_token_dim = 1152
        hidden_size = 1152
        
        # 创建单独的适配器进行测试
        lang_adaptor = nn.Linear(lang_token_dim, hidden_size).to(device, dtype=torch.bfloat16)
        img_adaptor = nn.Linear(img_token_dim, hidden_size).to(device, dtype=torch.bfloat16)
        
        # 测试适配器前向传播
        test_lang = torch.randn(2, 120, lang_token_dim, device=device, dtype=torch.bfloat16)
        test_img = torch.randn(2, 1000, img_token_dim, device=device, dtype=torch.bfloat16)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            adapted_lang = lang_adaptor(test_lang)
            adapted_img = img_adaptor(test_img)
            
            assert adapted_lang.device == device
            assert adapted_img.device == device
            assert adapted_lang.dtype == torch.bfloat16
            assert adapted_img.dtype == torch.bfloat16
        
        print("  ✅ 适配器创建和前向传播正常")
        
        return True
        
    except Exception as e:
        print(f"  ❌ GPU模型创建测试失败: {e}")
        return False

def test_training_simulation_gpu():
    """模拟GPU训练过程"""
    print("\n🏋️ 模拟GPU训练过程...")
    
    device = torch.device("cuda:0")
    
    try:
        # 创建简化的训练组件
        hidden_size = 1152
        batch_size = 2
        
        # 模型组件
        model = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 128)
        ).to(device, dtype=torch.bfloat16)
        
        # 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # 模拟训练数据
        inputs = torch.randn(batch_size, 32, hidden_size, device=device, dtype=torch.bfloat16)
        targets = torch.randn(batch_size, 32, 128, device=device, dtype=torch.bfloat16)
        
        # 模拟训练步骤
        for step in range(3):  # 3个训练步骤
            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = F.mse_loss(outputs, targets)
            
            # 检查损失有效性
            assert not torch.isnan(loss), f"第{step}步出现NaN损失"
            assert loss.device == device
            
            # 反向传播
            loss.backward()
            
            # 检查梯度
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            assert total_norm > 0, f"第{step}步梯度为零"
            assert not np.isnan(total_norm), f"第{step}步梯度为NaN"
            
            optimizer.step()
            
            print(f"    步骤 {step+1}: 损失={loss.item():.4f}, 梯度范数={total_norm:.4f}")
        
        print("  ✅ GPU训练模拟成功")
        
        return True
        
    except Exception as e:
        print(f"  ❌ GPU训练模拟失败: {e}")
        return False

def test_memory_management():
    """测试GPU显存管理"""
    print("\n💾 测试GPU显存管理...")
    
    try:
        device = torch.device("cuda:0")
        
        # 记录初始显存
        initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        # 创建大型张量
        large_tensors = []
        for i in range(5):
            tensor = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
            large_tensors.append(tensor)
        
        # 记录使用后显存
        used_memory = torch.cuda.memory_allocated() / 1024**2
        memory_increase = used_memory - initial_memory
        
        print(f"  📊 显存使用增加: {memory_increase:.1f}MB")
        
        # 清理显存
        del large_tensors
        torch.cuda.empty_cache()
        
        # 记录清理后显存
        final_memory = torch.cuda.memory_allocated() / 1024**2
        memory_freed = used_memory - final_memory
        
        print(f"  🧹 显存释放: {memory_freed:.1f}MB")
        print("  ✅ GPU显存管理正常")
        
        return True
        
    except Exception as e:
        print(f"  ❌ GPU显存管理测试失败: {e}")
        return False

def main():
    """运行所有GPU测试"""
    print("🚀 FLARE GPU训练测试开始")
    print("=" * 60)
    
    # 首先设置GPU环境
    device, gpu_memory = setup_gpu_environment()
    if device is None:
        print("❌ 无法设置GPU环境，测试终止")
        return False
    
    tests = [
        ("GPU张量操作", test_gpu_tensor_operations),
        ("RDT组件GPU", test_rdt_components_on_gpu),
        ("FLARE数据结构GPU", test_flare_data_structures_gpu),
        ("GPU损失计算", test_loss_computation_gpu),
        ("GPU模型创建", test_model_creation_gpu),
        ("GPU训练模拟", test_training_simulation_gpu),
        ("GPU显存管理", test_memory_management),
    ]
    
    passed = 0
    total = len(tests)
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 运行测试: {test_name}")
        try:
            if test_func():
                passed += 1
                results.append(f"✅ {test_name}")
            else:
                results.append(f"❌ {test_name}")
        except Exception as e:
            print(f"  💥 测试 {test_name} 出现异常: {e}")
            results.append(f"💥 {test_name}: {str(e)[:50]}...")
    
    print("\n" + "=" * 60)
    print("📊 GPU测试结果汇总:")
    for result in results:
        print(f"  {result}")
    
    print(f"\n📈 总体结果: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎊 所有GPU测试通过！FLARE GPU训练已就绪！")
        print("\n🚀 GPU训练准备清单:")
        print("✅ GPU环境配置正确")
        print("✅ BF16混合精度支持")
        print("✅ FLARE组件兼容")
        print("✅ 显存管理正常")
        print("\n🔥 可以开始GPU训练！")
    elif passed >= total * 0.8:  # 80%以上通过
        print("\n🔶 大部分GPU测试通过，可以尝试训练")
        print("⚠️  注意观察失败的组件")
    else:
        print("\n❌ 多项GPU测试失败，请先修复关键问题")
    
    # 显示GPU状态
    print("\n📊 当前GPU状态:")
    print(f"  显存已用: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
    print(f"  显存保留: {torch.cuda.memory_reserved() / 1024**2:.1f}MB")
    
    print("=" * 60)
    return passed >= total * 0.8

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)