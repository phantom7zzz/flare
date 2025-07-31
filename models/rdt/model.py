import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from pathlib import Path
import sys
import os
import traceback
# 导入所有必要的模块
from models.rdt.blocks import (FinalLayer, RDTBlock, TimestepEmbedder, get_1d_sincos_pos_embed_from_grid,
                              get_multimodal_cond_pos_embed)
from models.rdt.dit_activation_extractor import FLAREActivationAligner
from transformers import AutoModel, AutoImageProcessor

class RDTWithFLARE(nn.Module):
    """
    FLARE增强RDT模型
    
    功能：
    1. 标准的RDT动作预测
    2. SigLIP2视觉特征生成
    3. DiT层激活提取和对齐
    4. 联合损失优化
    """

    def __init__(self,
                 output_dim=128,
                 horizon=32,
                 hidden_size=1152,
                 depth=28,
                 num_heads=16,
                 max_lang_cond_len=32,
                 img_cond_len=4096,
                 lang_pos_embed_config=None,
                 img_pos_embed_config=None,
                 dtype=torch.bfloat16,
                 # FLARE相关参数
                 num_future_tokens=32,
                 activation_layer=21,
                 alignment_temperature=0.07,
                 future_vision_model_name=None,
                 future_text_model_name=None,
                 future_vision_image_size=256,
                 use_pooling=True,
                 target_tokens=64,
                 # 兼容性参数（不使用但需要接收）
                 num_vl_fusion_layers=4,
                 num_qformer_layers=2,
                 **kwargs):
        super().__init__()
        
        # 基础参数
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.max_lang_cond_len = max_lang_cond_len
        self.img_cond_len = img_cond_len
        self.dtype = dtype
        self.lang_pos_embed_config = lang_pos_embed_config
        self.img_pos_embed_config = img_pos_embed_config
        
        # FLARE相关参数
        self.num_future_tokens = num_future_tokens
        self.activation_layer = activation_layer
        self.alignment_temperature = alignment_temperature
        self.use_pooling = use_pooling
        self.target_tokens = target_tokens
        
        print(f"🔧 初始化FLARE模型: DiT={num_future_tokens}tokens, SigLIP2={target_tokens}tokens")
        
        # 基础RDT组件
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        self.freq_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        
        self.vision_feature_adapter = nn.Linear(1152, 2048, bias=False)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.vision_feature_adapter.weight)
        
        # 序列结构定义
        self.state_token_len = 1
        self.seq_structure = {
            'timestep': 1,
            'freq': 1, 
            'state': self.state_token_len,
            'action': self.horizon,
            'future_obs': self.num_future_tokens
        }
        
        # 计算总序列长度
        total_seq_len = sum(self.seq_structure.values())
        self.x_pos_embed = nn.Parameter(torch.zeros(1, total_seq_len, hidden_size))
        
        # 预计算索引位置
        self._compute_sequence_indices()
        
        # 条件位置编码
        self.lang_cond_pos_embed = nn.Parameter(torch.zeros(1, max_lang_cond_len, hidden_size))
        self.img_cond_pos_embed = nn.Parameter(torch.zeros(1, img_cond_len, hidden_size))

        # Transformer blocks
        self.blocks = nn.ModuleList([RDTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, output_dim)
        
        # FLARE组件：SigLIP2目标生成器
        self._initialize_siglip2_model(future_vision_model_name)
        
        # 未来观测token初始化
        self.future_obs_tokens = nn.Parameter(torch.randn(1, num_future_tokens, hidden_size))
        
        # 未来观测token的MLP处理器
        self.future_obs_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_size)
        )
        
        # 激活对齐器（延迟初始化）
        self.activation_aligner = None
        
        self.initialize_weights()
        self._ensure_bf16_consistency()
        
        # 模型参数统计
        total_params = sum(p.numel() for p in self.parameters())
        dit_params = sum(p.numel() for p in self.blocks.parameters())
        siglip2_params = sum(p.numel() for p in self.siglip2_model.parameters()) if self.siglip2_model else 0
        
        print(f"📊 模型参数: 总计{total_params:,}, DiT{dit_params:,}, SigLIP2{siglip2_params:,}(冻结)")

    def _initialize_siglip2_model(self, siglip2_path):
        """初始化SigLIP2视觉编码器"""
        if siglip2_path is None:
            raise ValueError("future_vision_model_name 不能为空！")
        
        if not os.path.exists(siglip2_path):
            raise FileNotFoundError(f"SigLIP2模型路径不存在: {siglip2_path}")
        
        print(f"🔧 加载SigLIP2模型: {siglip2_path}")
        
        # 加载完整模型并提取视觉编码器
        full_model = AutoModel.from_pretrained(siglip2_path, local_files_only=True)
        self.siglip2_model = full_model.vision_model  # 只要视觉部分
        self.siglip2_model.eval()
        self.siglip2_model.requires_grad_(False)
        
        # 获取视觉编码器的hidden_size
        vision_config = full_model.config.vision_config
        hidden_size = getattr(vision_config, 'hidden_size', 1024)
        
        self.siglip2_adapter = nn.Linear(hidden_size, self.hidden_size, bias=False)
        print(f"✅ SigLIP2视觉编码器已加载，维度: {hidden_size} → {self.hidden_size}")

    def _compute_sequence_indices(self):
        """预计算序列中各部分的索引位置"""
        self.indices = {}
        start_idx = 0
        for key, length in self.seq_structure.items():
            self.indices[key] = (start_idx, start_idx + length)
            start_idx += length
            
    def _generate_siglip2_targets(self, future_obs_images):
        """使用SigLIP2视觉编码器生成目标tokens"""
        if future_obs_images is None or self.siglip2_model is None:
            batch_size = future_obs_images.shape[0] if future_obs_images is not None else 1
            device = future_obs_images.device if future_obs_images is not None else next(self.parameters()).device
            return torch.zeros(batch_size, self.target_tokens, self.hidden_size, device=device, dtype=self.dtype)
        
        batch_size = future_obs_images.shape[0]
        device = future_obs_images.device
        
        # 确保图像尺寸是256x256
        if future_obs_images.shape[-1] != 256:
            future_obs_images = F.interpolate(future_obs_images, size=(256, 256), mode='bilinear', align_corners=False)
        
        # 确保设备和数据类型匹配
        model_device = next(self.siglip2_model.parameters()).device
        model_dtype = next(self.siglip2_model.parameters()).dtype
        future_obs_images = future_obs_images.to(device=model_device, dtype=model_dtype)
        
        with torch.no_grad():
            # 调用SigLIP2视觉编码器
            vision_outputs = self.siglip2_model(future_obs_images)
            
            # 获取视觉特征
            if hasattr(vision_outputs, 'last_hidden_state'):
                features = vision_outputs.last_hidden_state
            elif hasattr(vision_outputs, 'pooler_output'):
                features = vision_outputs.pooler_output.unsqueeze(1)
            else:
                features = vision_outputs
            
            # 调整到目标token数量
            seq_len = features.shape[1]
            if seq_len >= self.target_tokens:
                features = features[:, :self.target_tokens, :]
            else:
                repeat_times = (self.target_tokens + seq_len - 1) // seq_len
                features = features.repeat(1, repeat_times, 1)[:, :self.target_tokens, :]
            
            # 维度适配
            target_tokens = self.siglip2_adapter(features.to(self.dtype))
            
            return target_tokens
    
    def _ensure_bf16_consistency(self):
        """确保模型所有组件都使用BF16"""
        target_dtype = self.dtype
        
        # 转换所有参数和缓冲区
        for name, param in self.named_parameters():
            if param.dtype != target_dtype:
                param.data = param.data.to(target_dtype)
                
        for name, buffer in self.named_buffers():
            if buffer.dtype != target_dtype and buffer.dtype.is_floating_point:
                buffer.data = buffer.data.to(target_dtype)
        
        # 处理SigLIP2模型
        if hasattr(self, 'siglip2_model') and self.siglip2_model is not None:
            self.siglip2_model = self.siglip2_model.to(target_dtype)
            
    def initialize_weights(self):
        """初始化模型权重"""
        # 初始化transformer层
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # 初始化位置编码
        x_pos_embed = get_multimodal_cond_pos_embed(embed_dim=self.hidden_size,
                                                    mm_cond_lens=OrderedDict([
                                                        ('timestep', 1),
                                                        ('ctrl_freq', 1),
                                                        ('state', 1),
                                                        ('action', self.horizon),
                                                        ('future_obs', self.num_future_tokens),
                                                    ]))
        self.x_pos_embed.data.copy_(torch.from_numpy(x_pos_embed).float().unsqueeze(0))

        # 语言位置编码
        if self.lang_pos_embed_config is None:
            lang_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size,
                                                                    torch.arange(self.max_lang_cond_len))
        else:
            lang_cond_pos_embed = get_multimodal_cond_pos_embed(embed_dim=self.hidden_size,
                                                                mm_cond_lens=OrderedDict(self.lang_pos_embed_config),
                                                                embed_modality=False)
        self.lang_cond_pos_embed.data.copy_(torch.from_numpy(lang_cond_pos_embed).float().unsqueeze(0))

        # 图像位置编码
        if self.img_pos_embed_config is None:
            img_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, torch.arange(self.img_cond_len))
        else:
            img_cond_pos_embed = get_multimodal_cond_pos_embed(embed_dim=self.hidden_size,
                                                               mm_cond_lens=OrderedDict(self.img_pos_embed_config),
                                                               embed_modality=False)
        self.img_cond_pos_embed.data.copy_(torch.from_numpy(img_cond_pos_embed).float().unsqueeze(0))

        # 初始化timestep和freq embedding
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[2].weight, std=0.02)

        # 初始化最终层
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)
        
        # 初始化FLARE组件
        nn.init.normal_(self.future_obs_tokens, std=0.02)

        # 移动到指定数据类型
        self.to(self.dtype)
        
    def _initialize_activation_aligner(self):
        """延迟初始化激活对齐器"""
        if self.activation_aligner is None:
            self.activation_aligner = FLAREActivationAligner(
                model=self,
                target_layer=self.activation_layer,
                num_future_tokens=self.num_future_tokens,
                alignment_temperature=self.alignment_temperature,
                loss_type="cosine_contrastive"
            )

    def forward(self, x, freq, t, lang_c, img_c, lang_mask=None, img_mask=None, 
                future_obs_image=None, return_alignment_loss=False, **kwargs):
        """FLARE模型前向传播"""
        # 统一数据类型处理
        target_dtype = self.dtype
        device = x.device
        batch_size = x.shape[0]

        # 确保所有输入张量使用相同的数据类型
        x = x.to(dtype=target_dtype, device=device)
        if isinstance(freq, torch.Tensor):
            freq = freq.to(dtype=target_dtype, device=device)
        if isinstance(t, torch.Tensor):
            t = t.to(dtype=target_dtype, device=device)
        lang_c = lang_c.to(dtype=target_dtype, device=device)
        img_c = img_c.to(dtype=target_dtype, device=device)
        if future_obs_image is not None:
            future_obs_image = future_obs_image.to(dtype=target_dtype, device=device)
        
        # 处理时间步扩展
        if t.shape[0] == 1 and batch_size != 1:
            t = t.expand(batch_size)
            
        # 1. 编码时间步和频率
        t_embed = self.t_embedder(t).unsqueeze(1)  # (B, 1, D)
        freq_embed = self.freq_embedder(freq).unsqueeze(1)  # (B, 1, D)
        
        # 2. 分离状态和动作
        state_start, state_end = 0, self.state_token_len
        action_start, action_end = self.state_token_len, x.shape[1]
        
        state_tokens = x[:, state_start:state_end]  # (B, state_len, D)
        action_tokens = x[:, action_start:action_end]  # (B, action_len, D)
        
        # 3. 处理未来观测token
        future_obs_tokens = self._process_future_obs_tokens(batch_size, device, target_dtype)
        
        # 4. 序列拼接
        sequence_parts = [t_embed, freq_embed, state_tokens, action_tokens, future_obs_tokens]
        sequence = torch.cat(sequence_parts, dim=1)  # (B, total_seq_len, D)
        
        # 5. 添加位置编码
        sequence = sequence + self.x_pos_embed
        
        # 6. 准备条件
        conds = [lang_c, img_c]
        masks = [lang_mask, img_mask]
        
        # 7. FLARE: 生成SigLIP2目标tokens
        target_future_tokens = None
        if return_alignment_loss and future_obs_image is not None:
            target_future_tokens = self._generate_siglip2_targets(future_obs_image)
            print(f"🎯 SigLIP2目标tokens: {target_future_tokens.shape}")

        # 8. 通过transformer blocks
        for i, block in enumerate(self.blocks):
            condition_idx = i % len(conds)
            c, mask = conds[condition_idx], masks[condition_idx]
            sequence = block(sequence, c, mask)

        # 9. 最终层处理
        sequence = self.final_layer(sequence)

        # 10. 只返回动作token
        action_start_idx, action_end_idx = self.indices['action']
        action_tokens = sequence[:, action_start_idx:action_end_idx]
        
        # 11. 计算对齐损失
        alignment_loss = None
        if return_alignment_loss and target_future_tokens is not None:
            self._initialize_activation_aligner()
            
            # 使用正确的未来token位置
            future_start_idx, future_end_idx = self.indices['future_obs']
            alignment_loss, _ = self.activation_aligner.compute_precise_alignment_loss(
                target_future_tokens, 
                horizon=self.horizon,
                future_token_indices=(future_start_idx, future_end_idx)
            )
            print(f"✅ 对齐损失: {alignment_loss.item():.4f}")
        
        if return_alignment_loss:
            return action_tokens, alignment_loss
        else:
            return action_tokens
        
    def _process_future_obs_tokens(self, batch_size, device, target_dtype):
        """处理未来观测tokens - DiT序列中的占位符"""
        # 确保future_obs_tokens参数存在且维度正确
        if not hasattr(self, 'future_obs_tokens') or self.future_obs_tokens.shape[-1] != self.hidden_size:
            self.future_obs_tokens = nn.Parameter(
                torch.randn(1, self.num_future_tokens, self.hidden_size) * 0.02
            ).to(device=device, dtype=target_dtype)
        
        # 扩展到batch size
        future_obs_tokens = self.future_obs_tokens.expand(batch_size, -1, -1)
        
        # 通过MLP处理
        future_obs_tokens = self.future_obs_mlp(future_obs_tokens)
        
        return future_obs_tokens.to(device=device, dtype=target_dtype)