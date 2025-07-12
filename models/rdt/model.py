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


class QFormerCompressor(nn.Module):
    """Q-Former压缩器，用于将视觉token和语言token压缩到32个VL token"""
    
    def __init__(self, hidden_size, num_query_tokens=32, num_layers=2):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))
        
        # Q-Former层
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=16,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, vision_tokens, lang_tokens, vision_mask=None, lang_mask=None):
        """
        Args:
            vision_tokens: (B, V, D) 视觉token
            lang_tokens: (B, L, D) 语言token  
            vision_mask: (B, V) 视觉token掩码
            lang_mask: (B, L) 语言token掩码
        Returns:
            vl_tokens: (B, 32, D) 压缩后的VL token
        """
        batch_size = vision_tokens.shape[0]
        
        # 扩展query tokens
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # 拼接视觉和语言token作为memory
        memory = torch.cat([vision_tokens, lang_tokens], dim=1)  # (B, V+L, D)
        
        # 拼接掩码
        if vision_mask is not None and lang_mask is not None:
            memory_mask = torch.cat([vision_mask, lang_mask], dim=1)  # (B, V+L)
        else:
            memory_mask = None
            
        # 通过Q-Former层
        output = query_tokens
        for layer in self.layers:
            output = layer(output, memory, memory_key_padding_mask=memory_mask)
            
        output = self.layer_norm(output)
        return output


class RDTWithFLARE(nn.Module):
    """
    FLARE增强的RDT模型，添加了未来观测token和对齐损失
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
                 num_future_tokens=32,  # 未来观测token数量
                 activation_layer=6):   # 激活层索引
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

        # 修改位置编码以包含未来观测token
        # [timestep; state; action; future_obs]
        self.x_pos_embed = nn.Parameter(torch.zeros(1, horizon + 3 + num_future_tokens, hidden_size))
        
        # Language conditions
        self.lang_cond_pos_embed = nn.Parameter(torch.zeros(1, max_lang_cond_len, hidden_size))
        # Image conditions
        self.img_cond_pos_embed = nn.Parameter(torch.zeros(1, img_cond_len, hidden_size))

        self.blocks = nn.ModuleList([RDTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, output_dim)
        
        # FLARE相关组件
        # Q-Former压缩器，用于压缩视觉和语言token
        self.qformer_compressor = QFormerCompressor(hidden_size, num_future_tokens)
        
        # 未来观测token的MLP处理器
        self.future_obs_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(0.1)
        )
        
        # 初始化未来观测token
        self.future_obs_tokens = nn.Parameter(torch.randn(1, num_future_tokens, hidden_size))
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize pos_embed by sin-cos embedding - 修改以包含未来观测token
        x_pos_embed = get_multimodal_cond_pos_embed(embed_dim=self.hidden_size,
                                                    mm_cond_lens=OrderedDict([
                                                        ('timestep', 1),
                                                        ('ctrl_freq', 1),
                                                        ('state', 1),
                                                        ('action', self.horizon),
                                                        ('future_obs', self.num_future_tokens),  # 新增
                                                    ]))
        self.x_pos_embed.data.copy_(torch.from_numpy(x_pos_embed).float().unsqueeze(0))

        if self.lang_pos_embed_config is None:
            lang_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size,
                                                                    torch.arange(self.max_lang_cond_len))
        else:
            lang_cond_pos_embed = get_multimodal_cond_pos_embed(embed_dim=self.hidden_size,
                                                                mm_cond_lens=OrderedDict(self.lang_pos_embed_config),
                                                                embed_modality=False)
        self.lang_cond_pos_embed.data.copy_(torch.from_numpy(lang_cond_pos_embed).float().unsqueeze(0))

        if self.img_pos_embed_config is None:
            img_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, torch.arange(self.img_cond_len))
        else:
            img_cond_pos_embed = get_multimodal_cond_pos_embed(embed_dim=self.hidden_size,
                                                               mm_cond_lens=OrderedDict(self.img_pos_embed_config),
                                                               embed_modality=False)
        self.img_cond_pos_embed.data.copy_(torch.from_numpy(img_cond_pos_embed).float().unsqueeze(0))

        # Initialize timestep and control freq embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[2].weight, std=0.02)

        # Initialize the final layer: zero-out the final linear layer
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)
        
        # 初始化FLARE相关组件
        nn.init.normal_(self.future_obs_tokens, std=0.02)

        # Move all the params to given data type:
        self.to(self.dtype)

    def compute_alignment_loss(self, future_obs_tokens, vl_tokens, margin=0.2):
        """
        计算未来观测token和VL token之间的对齐损失
        使用论文中提到的对比学习损失
        """
        # 归一化特征
        future_obs_norm = F.normalize(future_obs_tokens, p=2, dim=-1)  # (B, 32, D)
        vl_norm = F.normalize(vl_tokens, p=2, dim=-1)  # (B, 32, D)
        
        # 计算相似度矩阵
        similarity = torch.bmm(future_obs_norm, vl_norm.transpose(1, 2))  # (B, 32, 32)
        
        # 对角线元素是正样本相似度
        positive_sim = torch.diagonal(similarity, dim1=1, dim2=2)  # (B, 32)
        
        # 计算对比损失
        # 对于每个查询，其他所有位置都是负样本
        batch_size, seq_len = positive_sim.shape
        
        # 创建掩码，排除对角线元素
        mask = torch.eye(seq_len, device=similarity.device).unsqueeze(0).expand(batch_size, -1, -1)
        negative_sim = similarity.masked_fill(mask.bool(), float('-inf'))
        
        # 计算InfoNCE损失
        logits = torch.cat([positive_sim.unsqueeze(-1), negative_sim.flatten(2)], dim=-1)  # (B, 32, 1+31*32)
        labels = torch.zeros(batch_size, seq_len, dtype=torch.long, device=logits.device)
        
        alignment_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return alignment_loss

    def forward(self, x, freq, t, lang_c, img_c, lang_mask=None, img_mask=None, 
                future_vision_tokens=None, return_alignment_loss=False):
        """
        Forward pass of RDT with FLARE.
        
        新增参数:
            future_vision_tokens: (B, V_future, D) 未来观测的视觉token
            return_alignment_loss: 是否返回对齐损失
        """
        batch_size = x.shape[0]
        
        t = self.t_embedder(t).unsqueeze(1)  # (B, 1, D) or (1, 1, D)
        freq = self.freq_embedder(freq).unsqueeze(1)  # (B, 1, D)
        
        # 初始化未来观测token
        future_obs_tokens = self.future_obs_tokens.expand(batch_size, -1, -1)  # (B, 32, D)
        future_obs_tokens = self.future_obs_mlp(future_obs_tokens)
        
        # Append timestep and future obs tokens to the input tokens
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1, -1)
        
        # 拼接所有token: [timestep, freq, state+action, future_obs]
        x = torch.cat([t, freq, x, future_obs_tokens], dim=1)  # (B, T+3+32, D)

        # Add multimodal position embeddings
        x = x + self.x_pos_embed
        lang_c = lang_c + self.lang_cond_pos_embed[:, :lang_c.shape[1]]
        img_c = img_c + self.img_cond_pos_embed

        # 如果提供了未来观测的视觉token，计算VL token用于对齐损失
        alignment_loss = None
        if future_vision_tokens is not None and return_alignment_loss:
            # 使用Q-Former压缩视觉和语言token
            vl_tokens = self.qformer_compressor(
                future_vision_tokens, lang_c, 
                vision_mask=None, lang_mask=lang_mask
            )  # (B, 32, D)
            
            # 提取第activation_layer层的未来观测token用于计算对齐损失
            future_tokens_for_alignment = x[:, -(self.num_future_tokens):]  # (B, 32, D)

        # Forward pass through transformer blocks
        conds = [lang_c, img_c]
        masks = [lang_mask, img_mask]
        
        for i, block in enumerate(self.blocks):
            c, mask = conds[i % 2], masks[i % 2]
            x = block(x, c, mask)  # (B, T+3+32, D)
            
            # 在第activation_layer层激活未来观测token并计算对齐损失
            if i == self.activation_layer and future_vision_tokens is not None and return_alignment_loss:
                current_future_tokens = x[:, -(self.num_future_tokens):]  # (B, 32, D)
                alignment_loss = self.compute_alignment_loss(current_future_tokens, vl_tokens)

        # 最终层处理
        x = self.final_layer(x)  # (B, T+3+32, out_channels)

        # 只保留动作token，排除未来观测token
        action_tokens = x[:, 2:2+self.horizon]  # 跳过timestep和freq，取horizon个动作token
        
        if return_alignment_loss:
            return action_tokens, alignment_loss
        else:
            return action_tokens