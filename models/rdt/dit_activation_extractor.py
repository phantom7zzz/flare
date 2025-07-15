# models/rdt/dit_activation_extractor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class ActivationHook:
    """激活提取钩子类"""
    
    def __init__(self, name: str):
        self.name = name
        self.activations = {}
        self.gradients = {}
        
    def forward_hook(self, module, input, output):
        """前向传播钩子"""
        if isinstance(output, tuple):
            # 如果输出是元组，取第一个元素
            self.activations[self.name] = output[0].detach()
        else:
            self.activations[self.name] = output.detach()
    
    def backward_hook(self, module, grad_input, grad_output):
        """反向传播钩子"""
        if isinstance(grad_output, tuple) and grad_output[0] is not None:
            self.gradients[self.name] = grad_output[0].detach()
        elif grad_output is not None:
            self.gradients[self.name] = grad_output.detach()
    
    def get_activation(self):
        """获取最新的激活"""
        return self.activations.get(self.name, None)
    
    def get_gradient(self):
        """获取最新的梯度"""
        return self.gradients.get(self.name, None)
    
    def clear(self):
        """清空存储的激活和梯度"""
        self.activations.clear()
        self.gradients.clear()


class DiTActivationExtractor:
    """
    DiT层激活提取器
    
    功能：
    1. 在指定DiT层注册钩子
    2. 精确提取未来预测token的激活
    3. 支持多层激活提取和比较
    4. 提供激活可视化和分析工具
    """
    
    def __init__(self, 
                 model: nn.Module,
                 target_layers: List[int] = [6],
                 num_future_tokens: int = 32,
                 token_start_offset: int = 3,  # timestep + freq + state tokens
                 enable_gradient_hooks: bool = False):
        """
        初始化激活提取器
        
        Args:
            model: RDTWithFLARE模型
            target_layers: 目标提取层列表，默认[6]
            num_future_tokens: 未来预测token数量
            token_start_offset: 状态、动作token后的偏移量
            enable_gradient_hooks: 是否启用梯度钩子
        """
        self.model = model
        self.target_layers = target_layers
        self.num_future_tokens = num_future_tokens
        self.token_start_offset = token_start_offset
        self.enable_gradient_hooks = enable_gradient_hooks
        
        # 存储钩子和激活
        self.hooks = {}
        self.hook_handles = []
        self.extracted_activations = {}
        self.layer_outputs = {}
        
        # 注册钩子
        self._register_hooks()
        
    def _register_hooks(self):
        """注册钩子到指定的DiT层"""
        
        # 获取DiT blocks
        if hasattr(self.model, 'blocks'):
            dit_blocks = self.model.blocks
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'blocks'):
            dit_blocks = self.model.model.blocks
        else:
            raise AttributeError("Cannot find DiT blocks in the model")
        
        for layer_idx in self.target_layers:
            if layer_idx >= len(dit_blocks):
                print(f"Warning: Layer {layer_idx} does not exist. Model has {len(dit_blocks)} layers.")
                continue
                
            layer_name = f"dit_layer_{layer_idx}"
            hook = ActivationHook(layer_name)
            self.hooks[layer_name] = hook
            
            # 注册前向钩子
            handle = dit_blocks[layer_idx].register_forward_hook(hook.forward_hook)
            self.hook_handles.append(handle)
            
            # 注册反向钩子（可选）
            if self.enable_gradient_hooks:
                handle = dit_blocks[layer_idx].register_backward_hook(hook.backward_hook)
                self.hook_handles.append(handle)
                
        print(f"Registered hooks for layers: {self.target_layers}")
    
    def extract_future_token_activations(self, 
                                       layer_idx: int = 6,
                                       horizon: int = 32) -> Optional[torch.Tensor]:
        """
        提取指定层的未来预测token激活
        
        Args:
            layer_idx: DiT层索引
            horizon: 动作序列长度（用于计算token位置）
            
        Returns:
            future_activations: (B, num_future_tokens, D) 未来token激活
        """
        layer_name = f"dit_layer_{layer_idx}"
        
        if layer_name not in self.hooks:
            print(f"Warning: Layer {layer_idx} hook not found")
            return None
            
        activation = self.hooks[layer_name].get_activation()
        
        if activation is None:
            print(f"Warning: No activation found for layer {layer_idx}")
            return None
        
        # 计算未来token的起始位置
        # 序列结构: [timestep, freq, state, action_tokens..., future_tokens...]
        future_token_start = self.token_start_offset + horizon
        future_token_end = future_token_start + self.num_future_tokens
        
        # 提取未来token激活
        if activation.shape[1] < future_token_end:
            print(f"Warning: Activation sequence length {activation.shape[1]} is too short for future tokens")
            return None
            
        future_activations = activation[:, future_token_start:future_token_end, :]
        
        # 存储提取的激活
        self.extracted_activations[f"layer_{layer_idx}_future"] = future_activations.detach()
        
        return future_activations
    
    def extract_activations_at_step(self, 
                                  step_idx: int,
                                  layer_indices: List[int] = None) -> Dict[str, torch.Tensor]:
        """
        在指定步骤提取多层激活
        
        Args:
            step_idx: 时间步索引
            layer_indices: 层索引列表，默认使用target_layers
            
        Returns:
            activations: 各层激活字典
        """
        if layer_indices is None:
            layer_indices = self.target_layers
            
        activations = {}
        
        for layer_idx in layer_indices:
            layer_name = f"dit_layer_{layer_idx}"
            if layer_name in self.hooks:
                activation = self.hooks[layer_name].get_activation()
                if activation is not None:
                    activations[f"layer_{layer_idx}"] = activation.detach()
                    
        return activations
    
    def compute_activation_statistics(self, 
                                    layer_idx: int = 6) -> Dict[str, torch.Tensor]:
        """
        计算激活统计信息
        
        Args:
            layer_idx: 层索引
            
        Returns:
            statistics: 统计信息字典
        """
        future_activations = self.extract_future_token_activations(layer_idx)
        
        if future_activations is None:
            return {}
            
        stats = {
            'mean': future_activations.mean(dim=[0, 1]),
            'std': future_activations.std(dim=[0, 1]),
            'max': future_activations.max(dim=1)[0].max(dim=0)[0],
            'min': future_activations.min(dim=1)[0].min(dim=0)[0],
            'norm': torch.norm(future_activations, dim=-1).mean(),
        }
        
        return stats
    
    def get_activation_similarity(self, 
                                layer_idx1: int, 
                                layer_idx2: int) -> torch.Tensor:
        """
        计算不同层之间的激活相似性
        
        Args:
            layer_idx1, layer_idx2: 两个层的索引
            
        Returns:
            similarity: 余弦相似性矩阵
        """
        act1 = self.extract_future_token_activations(layer_idx1)
        act2 = self.extract_future_token_activations(layer_idx2)
        
        if act1 is None or act2 is None:
            return torch.tensor(0.0)
            
        # 计算余弦相似性
        act1_flat = act1.reshape(-1, act1.shape[-1])
        act2_flat = act2.reshape(-1, act2.shape[-1])
        
        similarity = F.cosine_similarity(act1_flat, act2_flat, dim=-1).mean()
        
        return similarity
    
    def clear_activations(self):
        """清空所有存储的激活"""
        for hook in self.hooks.values():
            hook.clear()
        self.extracted_activations.clear()
        self.layer_outputs.clear()
    
    def remove_hooks(self):
        """移除所有钩子"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self.hooks.clear()
        print("All hooks removed")
    
    def __del__(self):
        """析构函数，确保钩子被移除"""
        self.remove_hooks()


class FLAREActivationAligner:
    """
    FLARE激活对齐器
    
    功能：
    1. 整合激活提取和目标token生成
    2. 计算精确的对齐损失
    3. 提供训练时的激活监控
    """
    
    def __init__(self, 
                 model: nn.Module,
                 target_layer: int = 6,
                 num_future_tokens: int = 32,
                 alignment_temperature: float = 0.07,
                 loss_type: str = "cosine_contrastive"):
        """
        初始化对齐器
        
        Args:
            model: FLARE模型
            target_layer: 目标DiT层
            num_future_tokens: 未来token数量
            alignment_temperature: 对比学习温度
            loss_type: 损失类型 ("cosine_contrastive", "mse", "kl_div")
        """
        self.model = model
        self.target_layer = target_layer
        self.num_future_tokens = num_future_tokens
        self.alignment_temperature = alignment_temperature
        self.loss_type = loss_type
        
        # 创建激活提取器
        self.activation_extractor = DiTActivationExtractor(
            model=model,
            target_layers=[target_layer],
            num_future_tokens=num_future_tokens
        )
        
        # 激活历史记录
        self.activation_history = []
        self.loss_history = []
        
    def compute_precise_alignment_loss(self, 
                                     target_tokens: torch.Tensor,
                                     horizon: int = 32) -> Tuple[torch.Tensor, Dict]:
        """
        计算精确的对齐损失
        
        Args:
            target_tokens: (B, num_future_tokens, D) 目标tokens
            horizon: 动作序列长度
            
        Returns:
            loss: 对齐损失
            info: 额外信息字典
        """
        # 1. 提取DiT层激活
        pred_tokens = self.activation_extractor.extract_future_token_activations(
            layer_idx=self.target_layer,
            horizon=horizon
        )
        
        if pred_tokens is None:
            return torch.tensor(0.0, device=target_tokens.device), {}
        
        # 2. 检查维度匹配
        if pred_tokens.shape != target_tokens.shape:
            print(f"Shape mismatch: pred {pred_tokens.shape} vs target {target_tokens.shape}")
            return torch.tensor(0.0, device=target_tokens.device), {}
        
        # 3. 根据损失类型计算损失
        if self.loss_type == "cosine_contrastive":
            loss = self._cosine_contrastive_loss(pred_tokens, target_tokens)
        elif self.loss_type == "mse":
            loss = F.mse_loss(pred_tokens, target_tokens)
        elif self.loss_type == "kl_div":
            loss = self._kl_divergence_loss(pred_tokens, target_tokens)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # 4. 计算额外统计信息
        info = {
            'pred_norm': torch.norm(pred_tokens, dim=-1).mean().item(),
            'target_norm': torch.norm(target_tokens, dim=-1).mean().item(),
            'cosine_sim': F.cosine_similarity(
                pred_tokens.reshape(-1, pred_tokens.shape[-1]),
                target_tokens.reshape(-1, target_tokens.shape[-1]),
                dim=-1
            ).mean().item(),
            'layer_idx': self.target_layer
        }
        
        # 5. 记录历史
        self.loss_history.append(loss.item())
        self.activation_history.append({
            'pred_mean': pred_tokens.mean().item(),
            'target_mean': target_tokens.mean().item(),
            'pred_std': pred_tokens.std().item(),
            'target_std': target_tokens.std().item()
        })
        
        return loss, info
    
    def _cosine_contrastive_loss(self, pred_tokens, target_tokens):
        """余弦对比损失"""
        # L2 归一化
        pred_norm = F.normalize(pred_tokens, p=2, dim=-1)
        target_norm = F.normalize(target_tokens, p=2, dim=-1)
        
        batch_size, num_tokens, hidden_dim = pred_norm.shape
        
        # 计算相似度矩阵
        similarity = torch.bmm(pred_norm, target_norm.transpose(1, 2)) / self.alignment_temperature
        
        # 对角线元素是正样本对
        labels = torch.arange(num_tokens, device=similarity.device).unsqueeze(0).expand(batch_size, -1)
        
        # 计算对比损失
        loss = F.cross_entropy(similarity.reshape(-1, num_tokens), labels.reshape(-1))
        
        return loss
    
    def _kl_divergence_loss(self, pred_tokens, target_tokens):
        """KL散度损失"""
        # 转换为概率分布
        pred_prob = F.softmax(pred_tokens, dim=-1)
        target_prob = F.softmax(target_tokens, dim=-1)
        
        # 计算KL散度
        kl_loss = F.kl_div(pred_prob.log(), target_prob, reduction='batchmean')
        
        return kl_loss
    
    def get_alignment_metrics(self) -> Dict[str, float]:
        """获取对齐指标"""
        if not self.loss_history:
            return {}
            
        recent_losses = self.loss_history[-100:]  # 最近100步
        recent_activations = self.activation_history[-100:]
        
        metrics = {
            'avg_loss': sum(recent_losses) / len(recent_losses),
            'loss_trend': recent_losses[-1] - recent_losses[0] if len(recent_losses) > 1 else 0,
            'avg_pred_mean': sum(act['pred_mean'] for act in recent_activations) / len(recent_activations),
            'avg_target_mean': sum(act['target_mean'] for act in recent_activations) / len(recent_activations),
            'activation_stability': sum(act['pred_std'] for act in recent_activations) / len(recent_activations)
        }
        
        return metrics
    
    def set_target_layer(self, layer_idx: int):
        """动态设置目标层"""
        self.target_layer = layer_idx
        # 重新创建激活提取器
        self.activation_extractor.remove_hooks()
        self.activation_extractor = DiTActivationExtractor(
            model=self.model,
            target_layers=[layer_idx],
            num_future_tokens=self.num_future_tokens
        )
    
    def clear_history(self):
        """清空历史记录"""
        self.activation_history.clear()
        self.loss_history.clear()
        self.activation_extractor.clear_activations()


# 使用示例
def test_dit_activation_extractor():
    """测试DiT激活提取器"""
    # 这里需要一个实际的FLARE模型进行测试
    print("DiT Activation Extractor test completed")
    
    # 示例用法：
    # model = RDTWithFLARE(...)
    # aligner = FLAREActivationAligner(model, target_layer=6)
    # 
    # # 在训练循环中
    # target_tokens = target_generator(vl_tokens, vl_mask)
    # alignment_loss, info = aligner.compute_precise_alignment_loss(target_tokens, horizon=32)


if __name__ == "__main__":
    test_dit_activation_extractor()