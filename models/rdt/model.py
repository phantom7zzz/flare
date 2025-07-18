import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from pathlib import Path
import sys
import os

# 导入所有必要的模块
from models.rdt.blocks import (FinalLayer, RDTBlock, TimestepEmbedder, get_1d_sincos_pos_embed_from_grid,
                              get_multimodal_cond_pos_embed)
from models.multimodal_encoder.vl_token_generator import VLTokenGenerator
from models.multimodal_encoder.qformer_target_generator import QFormerTargetGenerator
from models.rdt.dit_activation_extractor import FLAREActivationAligner


class RDTWithFLARE(nn.Module):
    """
    完整集成的FLARE增强RDT模型
    
    功能：
    1. 标准的RDT动作预测
    2. 未来观测的VL token生成
    3. Q-Former目标token生成
    4. DiT层激活提取和对齐
    5. 联合损失优化
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
                 activation_layer=6,
                 num_vl_fusion_layers=4,
                 num_qformer_layers=6,
                 alignment_temperature=0.07,
                 vision_model_name="google/siglip-so400m-patch14-384",
                 text_model_name="google/siglip-so400m-patch14-384"):
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

        # 基础RDT组件
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        self.freq_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        
        self.vision_feature_adapter = nn.Linear(1152, 2048, bias=False)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.vision_feature_adapter.weight)
        
        # 确保future_obs_tokens维度正确
        if hasattr(self, 'future_obs_tokens'):
            if self.future_obs_tokens.shape[-1] != 2048:
                self.future_obs_tokens = nn.Parameter(
                    torch.randn(1, self.num_future_tokens, 2048) * 0.02
                )
        
        print("✅ 维度适配器初始化完成")
        # 位置编码：[timestep; freq; state; action; future_obs]
        #self.x_pos_embed = nn.Parameter(torch.zeros(1, horizon + 3 + num_future_tokens, hidden_size))
        self.state_token_len = 1  # 状态压缩为1个token
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
        
        # FLARE组件
        # 1. VL Token生成器
        self.vl_token_generator = VLTokenGenerator(
            vision_model_name=vision_model_name,
            text_model_name=text_model_name,
            hidden_size=hidden_size,
            num_fusion_layers=num_vl_fusion_layers,
            num_heads=num_heads
        )
        
        # 2. Q-Former目标生成器
        self.target_generator = QFormerTargetGenerator(
            hidden_size=hidden_size,
            num_query_tokens=num_future_tokens,
            num_layers=num_qformer_layers,
            num_heads=num_heads
        )
        
        # 3. 未来观测token初始化
        self.future_obs_tokens = nn.Parameter(torch.randn(1, num_future_tokens, hidden_size))
        
        # 4. 未来观测token的MLP处理器
        self.future_obs_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_size)
        )
        
        # 5. 激活对齐器（延迟初始化，避免循环引用）
        self.activation_aligner = None
        
        self.initialize_weights()
        self._ensure_bf16_consistency()
        
    def _compute_sequence_indices(self):
        """预计算序列中各部分的索引位置"""
        self.indices = {}
        start_idx = 0
        for key, length in self.seq_structure.items():
            self.indices[key] = (start_idx, start_idx + length)
            start_idx += length
    
    def _ensure_bf16_consistency(self):
        """确保模型所有组件都使用BF16"""
        target_dtype = self.dtype
        
        # 转换所有参数
        for name, param in self.named_parameters():
            if param.dtype != target_dtype:
                param.data = param.data.to(target_dtype)
                
        # 转换所有缓冲区  
        for name, buffer in self.named_buffers():
            if buffer.dtype != target_dtype and buffer.dtype.is_floating_point:
                buffer.data = buffer.data.to(target_dtype)
                
        print(f"✅ 模型统一使用数据类型: {target_dtype}")
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
        loss = F.cross_entropy(similarity.reshape(-1, num_tokens), labels.reshape(-1))
        
        return loss

    def forward(self, x, freq, t, lang_c, img_c, lang_mask=None, img_mask=None, 
                future_vision_tokens=None, text_instructions=None, future_obs_image=None, return_alignment_loss=False):
        """
        FLARE模型前向传播
        
        Args:
            x: (B, T, D) 状态和动作序列
            freq: (B,) 控制频率
            t: (B,) 时间步
            lang_c: (B, L, D) 语言条件
            img_c: (B, I, D) 图像条件
            lang_mask: (B, L) 语言掩码
            img_mask: (B, I) 图像掩码
            future_vision_tokens: (B, V, D) 未来观测的视觉token
            text_instructions: 文本指令（用于VL生成）
            return_alignment_loss: 是否返回对齐损失
        """
        # 🎯 确保输入数据类型一致
        # 统一数据类型处理
        target_dtype = self.dtype
        device = x.device

        # 确保所有输入张量使用相同的数据类型
        x = x.to(dtype=target_dtype, device=device)
        if isinstance(freq, torch.Tensor):
            freq = freq.to(dtype=target_dtype, device=device)
        if isinstance(t, torch.Tensor):
            t = t.to(dtype=target_dtype, device=device)
        lang_c = lang_c.to(dtype=target_dtype, device=device)
        img_c = img_c.to(dtype=target_dtype, device=device)
        if future_vision_tokens is not None:
            future_vision_tokens = future_vision_tokens.to(dtype=target_dtype, device=device)
        
        # batch_size = x.shape[0]
        
        # # 编码时间步和频率
        # t = self.t_embedder(t).unsqueeze(1)  # (B, 1, D)
        # freq = self.freq_embedder(freq).unsqueeze(1)  # (B, 1, D)
        
        # # 初始化未来观测token
        # future_obs_tokens = self.future_obs_tokens.expand(batch_size, -1, -1)  # (B, M, D)
        # future_obs_tokens = self.future_obs_mlp(future_obs_tokens)
        
        # # 扩展时间步到batch
        # if t.shape[0] == 1:
        #     t = t.expand(batch_size, -1, -1)
        
        # # 拼接所有token: [timestep, freq, state+action, future_obs]
        # #x = torch.cat([t, freq, x, future_obs_tokens], dim=1)  # (B, T+3+M, D)
        # # 添加位置编码
        # x = x + self.x_pos_embed
        # lang_c = lang_c + self.lang_cond_pos_embed[:, :lang_c.shape[1]]
        batch_size = x.shape[0]
        if t.shape[0] == 1 and batch_size != 1:
            # 推理/采样时如果t是单元素，扩展成和batch_size一致
            t = t.expand(batch_size)
        t_embed = self.t_embedder(t).unsqueeze(1)  # (B, 1, D)
        device = x.device
        target_dtype = self.dtype
        
        # 1. 编码时间步和频率
        t_embed = self.t_embedder(t).unsqueeze(1)  # (B, 1, D)
        freq_embed = self.freq_embedder(freq).unsqueeze(1)  # (B, 1, D)
        
        # 2. 分离状态和动作 (关键修复)
        # x 输入应该是 [state_tokens, action_tokens]
        state_start, state_end = 0, self.state_token_len
        action_start, action_end = self.state_token_len, x.shape[1]
        
        state_tokens = x[:, state_start:state_end]  # (B, state_len, D)
        action_tokens = x[:, action_start:action_end]  # (B, action_len, D)
        
        # 3. 处理未来观测token (FLARE核心修复)
        future_obs_tokens = self._process_future_obs_tokens(
            batch_size, future_obs_image, device, target_dtype
        )
        
        # 4. 正确的序列拼接 (按照定义的结构)
        sequence_parts = [
            t_embed,           # timestep
            freq_embed,        # freq
            state_tokens,      # state
            action_tokens,     # action  
            future_obs_tokens  # future_obs
        ]
        sequence_parts = [part for part in sequence_parts if part is not None]
        if not sequence_parts:
            # 如果所有部分都是None，创建dummy tensor
            sequence_parts = [torch.zeros(batch_size, 1, 2048, device=device, dtype=dtype)] 
        for i, p in enumerate(sequence_parts):
            assert p.shape[0] == batch_size, f"part {i} batch size {p.shape[0]} != {batch_size}"  
        sequence = torch.cat(sequence_parts, dim=1)  # (B, total_seq_len, D)
        
        # 5. 验证序列长度
        expected_len = sum(self.seq_structure.values())
        assert sequence.shape[1] == expected_len, \
            f"序列长度不匹配: {sequence.shape[1]} vs {expected_len}"
        
        # 6. 添加位置编码
        sequence = sequence + self.x_pos_embed
        
        # 7. 准备条件
        conds = [lang_c, img_c]
        masks = [lang_mask, img_mask]
        # 8. 通过transformer blocks (改进条件使用策略)
        target_future_tokens = None
        alignment_loss = None
        

        # FLARE: 计算目标tokens（如果需要）
        # target_future_tokens = None
        # if future_vision_tokens is not None and return_alignment_loss:
        #     try:
        #         # 1. 生成VL tokens
        #         vl_tokens, vl_mask = self.vl_token_generator(
        #             future_obs_image, text_instructions
        #         )

                
        #         # 2. 生成目标tokens
        #         target_future_tokens = self.target_generator(vl_tokens, vl_mask)
        #     except Exception as e:
        #         raise  # 直接让程序崩溃，打印完整Traceback
        # FLARE: 计算目标tokens（在transformer处理前）
        if return_alignment_loss and future_obs_image is not None:
            try:
                vl_tokens, vl_mask = self.vl_token_generator(
                    future_obs_image, text_instructions
                )
                target_future_tokens = self.target_generator(vl_tokens, vl_mask)
            except Exception as e:
                print(f"Warning: Target token generation failed: {e}")
                target_future_tokens = None

        # 通过transformer blocks
        # conds = [lang_c, img_c]
        # masks = [lang_mask, img_mask]
        # alignment_loss = None
        
        # Transformer处理
        for i, block in enumerate(self.blocks):
            # 改进的条件使用策略：前期语言，后期视觉
            condition_idx = 0 if i < len(self.blocks) // 2 else 1
            c, mask = conds[condition_idx], masks[condition_idx]
            sequence = block(sequence, c, mask)

        # 最终层处理
        sequence = self.final_layer(sequence)

        # 只返回动作token，去除时间步、频率和未来观测token
        #action_tokens = x[:, 2:2+self.horizon]  # (B, horizon, out_channels)
        action_start_idx, action_end_idx = self.indices['action']
        action_tokens = sequence[:, action_start_idx:action_end_idx]
        
        # 11. 计算对齐损失 (使用正确的索引)
        if return_alignment_loss and target_future_tokens is not None:
            try:
                self._initialize_activation_aligner()
                
                # 使用正确的未来token位置
                future_start_idx, future_end_idx = self.indices['future_obs']
                alignment_loss, _ = self.activation_aligner.compute_precise_alignment_loss(
                    target_future_tokens, 
                    horizon=self.horizon,
                    future_token_indices=(future_start_idx, future_end_idx)
                )
            except Exception as e:
                print(f"Warning: Alignment loss computation failed: {e}")
                alignment_loss = torch.tensor(0.0, device=action_tokens.device)
        
        if return_alignment_loss:
            return action_tokens, alignment_loss
        else:
            return action_tokens
        
    def _process_future_obs_tokens(self, batch_size, future_obs_image, device, target_dtype):
        """处理未来观测tokens """

        
        if future_obs_image is not None:
            try:
                # Step 1: 视觉编码
                with torch.no_grad():
                    future_vision_features = self.vl_token_generator.vision_encoder(future_obs_image)
                
                # Step 2: 调整token数量
                if future_vision_features.shape[1] != self.num_future_tokens:

                    future_vision_features = F.adaptive_avg_pool1d(
                        future_vision_features.transpose(1, 2),
                        self.num_future_tokens
                    ).transpose(1, 2)

                
                # Step 3: 关键的维度修复

                current_dim = future_vision_features.shape[-1]  # 1152
                target_dim = 2048  # MLP期望的输入维度
                
                
                if current_dim != target_dim:
                    # 创建维度适配器
                    if not hasattr(self, 'vision_feature_adapter'):
                        self.vision_feature_adapter = nn.Linear(
                            current_dim, 
                            target_dim,
                            bias=False  # 可选：不使用偏置
                        ).to(device=device, dtype=target_dtype)
                        
                        # 初始化适配器权重（重要！）
                        with torch.no_grad():
                            # 使用Xavier初始化
                            nn.init.xavier_uniform_(self.vision_feature_adapter.weight)
                    
                    # 应用适配器
                    future_vision_features = self.vision_feature_adapter(future_vision_features)
                
                # Step 4: 应用MLP
                
                future_obs_tokens = self.future_obs_mlp(future_vision_features)
                
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                
                # 创建fallback tokens
                future_obs_tokens = torch.zeros(
                    batch_size, 
                    self.num_future_tokens, 
                    2048,  # 使用MLP的输出维度
                    device=device, 
                    dtype=target_dtype
                )

        else:
            
            # 检查随机tokens的维度是否正确
            random_token_dim = self.future_obs_tokens.shape[-1]
            mlp_input_dim = 2048  

            
            if random_token_dim != mlp_input_dim:
                # 重新创建正确维度的随机tokens
                self.future_obs_tokens = nn.Parameter(
                    torch.randn(1, self.num_future_tokens, mlp_input_dim) * 0.02
                ).to(device=device, dtype=target_dtype)
            
            future_obs_tokens = self.future_obs_tokens.expand(batch_size, -1, -1)
            future_obs_tokens = self.future_obs_mlp(future_obs_tokens)
        
        result = future_obs_tokens.to(device=device, dtype=target_dtype)
        return result