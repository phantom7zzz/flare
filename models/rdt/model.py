# models/rdt/model_flare.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from pathlib import Path
import sys, os

# get current workspace
current_file = Path(__file__)
sys.path.append(current_file.parent.parent)

from rdt.blocks import (FinalLayer, RDTBlock, TimestepEmbedder, get_1d_sincos_pos_embed_from_grid,
                        get_multimodal_cond_pos_embed)


class CrossModalFusion(nn.Module):
    """跨模态融合模块，用于融合视觉和语言特征"""
    
    def __init__(self, hidden_size, num_heads=8, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 跨模态注意力层
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers * 2)
        ])
        
        # FFN
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
        
    def forward(self, vision_tokens, lang_tokens, vision_mask=None, lang_mask=None):
        """
        Args:
            vision_tokens: (B, V, D) 视觉特征
            lang_tokens: (B, L, D) 语言特征
            vision_mask: (B, V) 视觉特征掩码
            lang_mask: (B, L) 语言特征掩码
        Returns:
            fused_tokens: (B, V+L, D) 融合后的特征
        """
        # 拼接视觉和语言特征
        fused_tokens = torch.cat([vision_tokens, lang_tokens], dim=1)  # (B, V+L, D)
        
        # 创建联合掩码
        if vision_mask is not None and lang_mask is not None:
            joint_mask = torch.cat([vision_mask, lang_mask], dim=1)  # (B, V+L)
        else:
            joint_mask = None
            
        # 通过跨模态注意力层
        for i, (cross_attn, ln1, ln2, ffn) in enumerate(
            zip(self.cross_attention_layers, self.layer_norms[::2], self.layer_norms[1::2], self.ffns)
        ):
            # Self-attention with cross-modal interaction
            residual = fused_tokens
            fused_tokens = ln1(fused_tokens)
            
            attn_out, _ = cross_attn(
                fused_tokens, fused_tokens, fused_tokens,
                key_padding_mask=joint_mask if joint_mask is not None else None
            )
            fused_tokens = residual + attn_out
            
            # FFN
            residual = fused_tokens
            fused_tokens = ln2(fused_tokens)
            fused_tokens = residual + ffn(fused_tokens)
            
        return fused_tokens


class FutureObsPredictor(nn.Module):
    """未来观测预测模块"""
    
    def __init__(self, hidden_size, num_future_tokens=32):
        super().__init__()
        self.num_future_tokens = num_future_tokens
        
        # 压缩网络：将融合特征压缩到未来观测token
        self.compressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # 学习到的查询向量
        self.query_tokens = nn.Parameter(torch.randn(1, num_future_tokens, hidden_size))
        
        # 注意力池化
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 最终投影
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, fused_features, feature_mask=None):
        """
        Args:
            fused_features: (B, N, D) 融合后的特征
            feature_mask: (B, N) 特征掩码
        Returns:
            future_obs_tokens: (B, num_future_tokens, D) 预测的未来观测token
        """
        batch_size = fused_features.shape[0]
        
        # 压缩特征
        compressed_features = self.compressor(fused_features)  # (B, N, D)
        
        # 扩展查询token
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)  # (B, num_future_tokens, D)
        
        # 注意力池化：用查询token从压缩特征中提取信息
        future_obs_tokens, _ = self.attention_pool(
            query_tokens, compressed_features, compressed_features,
            key_padding_mask=feature_mask if feature_mask is not None else None
        )
        
        # 最终投影
        future_obs_tokens = self.final_proj(future_obs_tokens)
        
        return future_obs_tokens


class RDTWithFLARE(nn.Module):
    """
    FLARE增强的RDT模型
    """

    def __init__(self,
                 output_dim=128,
                 horizon=32,
                 hidden_size=1152,
                 depth=28,
                 num_heads=16,
                 max_lang_cond_len=1024,
                 img_cond_len=4096,
                 lang_pos_embed_config=None,
                 img_pos_embed_config=None,
                 dtype=torch.bfloat16,
                 # FLARE相关参数
                 num_future_tokens=32,
                 activation_layer=6):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.max_lang_cond_len = max_lang_cond_len
        self.img_cond_len = img_cond_len
        self.dtype = dtype
        self.lang_pos_embed_config = lang_pos_embed_config
        self.img_pos_embed_config = img_pos_embed_config
        
        # FLARE相关属性
        self.num_future_tokens = num_future_tokens
        self.activation_layer = activation_layer

        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        self.freq_embedder = TimestepEmbedder(hidden_size, dtype=dtype)

        # 位置编码：[timestep; state; action; future_obs]
        self.x_pos_embed = nn.Parameter(torch.zeros(1, horizon + 3 + num_future_tokens, hidden_size))
        
        # 条件位置编码
        self.lang_cond_pos_embed = nn.Parameter(torch.zeros(1, max_lang_cond_len, hidden_size))
        self.img_cond_pos_embed = nn.Parameter(torch.zeros(1, img_cond_len, hidden_size))

        # Transformer blocks
        self.blocks = nn.ModuleList([RDTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, output_dim)
        
        # FLARE组件
        # 跨模态融合模块
        self.cross_modal_fusion = CrossModalFusion(hidden_size)
        
        # 未来观测预测器
        self.future_obs_predictor = FutureObsPredictor(hidden_size, num_future_tokens)
        
        # 未来观测token初始化
        self.future_obs_tokens = nn.Parameter(torch.randn(1, num_future_tokens, hidden_size))
        
        # 未来观测token的MLP处理器
        self.future_obs_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(0.1)
        )
        
        self.initialize_weights()

    def initialize_weights(self):
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

    def compute_alignment_loss(self, pred_future_tokens, target_future_tokens, temperature=0.07):
        """
        计算对齐损失，使用对比学习
        
        Args:
            pred_future_tokens: (B, M, D) 预测的未来观测token
            target_future_tokens: (B, M, D) 目标未来观测token
            temperature: 温度参数
        """
        # L2归一化
        pred_norm = F.normalize(pred_future_tokens, p=2, dim=-1)  # (B, M, D)
        target_norm = F.normalize(target_future_tokens, p=2, dim=-1)  # (B, M, D)
        
        batch_size, num_tokens, hidden_dim = pred_norm.shape
        
        # 计算相似度矩阵 (B, M, M)
        similarity = torch.bmm(pred_norm, target_norm.transpose(1, 2)) / temperature
        
        # 对角线元素是正样本对
        labels = torch.arange(num_tokens, device=similarity.device).unsqueeze(0).expand(batch_size, -1)
        
        # 计算对比损失
        loss = F.cross_entropy(similarity.view(-1, num_tokens), labels.view(-1))
        
        return loss

    def forward(self, x, freq, t, lang_c, img_c, lang_mask=None, img_mask=None, 
                future_vision_tokens=None, return_alignment_loss=False):
        """
        Forward pass
        
        Args:
            x: (B, T, D) 状态和动作序列
            freq: (B,) 控制频率
            t: (B,) 时间步
            lang_c: (B, L, D) 语言条件
            img_c: (B, I, D) 图像条件
            lang_mask: (B, L) 语言掩码
            img_mask: (B, I) 图像掩码
            future_vision_tokens: (B, V, D) 未来观测的视觉token
            return_alignment_loss: 是否返回对齐损失
        """
        batch_size = x.shape[0]
        
        # 编码时间步和频率
        t = self.t_embedder(t).unsqueeze(1)  # (B, 1, D)
        freq = self.freq_embedder(freq).unsqueeze(1)  # (B, 1, D)
        
        # 初始化未来观测token
        future_obs_tokens = self.future_obs_tokens.expand(batch_size, -1, -1)  # (B, M, D)
        future_obs_tokens = self.future_obs_mlp(future_obs_tokens)
        
        # 扩展时间步到batch
        if t.shape[0] == 1:
            t = t.expand(batch_size, -1, -1)
        
        # 拼接所有token: [timestep, freq, state+action, future_obs]
        x = torch.cat([t, freq, x, future_obs_tokens], dim=1)  # (B, T+3+M, D)

        # 添加位置编码
        x = x + self.x_pos_embed
        lang_c = lang_c + self.lang_cond_pos_embed[:, :lang_c.shape[1]]
        img_c = img_c + self.img_cond_pos_embed

        # 如果提供了未来观测，计算目标token用于对齐损失
        target_future_tokens = None
        if future_vision_tokens is not None and return_alignment_loss:
            # 跨模态融合
            fused_features = self.cross_modal_fusion(
                future_vision_tokens, lang_c, 
                vision_mask=None, lang_mask=lang_mask
            )  # (B, V+L, D)
            
            # 生成目标未来观测token
            target_future_tokens = self.future_obs_predictor(fused_features)  # (B, M, D)

        # 通过transformer blocks
        conds = [lang_c, img_c]
        masks = [lang_mask, img_mask]
        alignment_loss = None
        
        for i, block in enumerate(self.blocks):
            c, mask = conds[i % 2], masks[i % 2]
            x = block(x, c, mask)  # (B, T+3+M, D)
            
            # 在指定层计算对齐损失
            if (i == self.activation_layer and target_future_tokens is not None and return_alignment_loss):
                # 提取当前的未来观测token
                current_future_tokens = x[:, -(self.num_future_tokens):]  # (B, M, D)
                alignment_loss = self.compute_alignment_loss(current_future_tokens, target_future_tokens)

        # 最终层处理
        x = self.final_layer(x)  # (B, T+3+M, out_channels)

        # 只返回动作token，去除时间步、频率和未来观测token
        action_tokens = x[:, 2:2+self.horizon]  # (B, horizon, out_channels)
        
        if return_alignment_loss:
            return action_tokens, alignment_loss
        else:
            return action_tokens