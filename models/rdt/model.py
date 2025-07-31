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
from models.multimodal_encoder.vl_token_generator import VLTokenGenerator
from models.multimodal_encoder.qformer_target_generator import QFormerTargetGenerator
from models.rdt.dit_activation_extractor import FLAREActivationAligner


# class RDTWithFLARE(nn.Module):
#     """
#     完整集成的FLARE增强RDT模型
    
#     功能：
#     1. 标准的RDT动作预测
#     2. 未来观测的VL token生成
#     3. Q-Former目标token生成
#     4. DiT层激活提取和对齐
#     5. 联合损失优化
#     """

#     def __init__(self,
#                  output_dim=128,
#                  horizon=32,
#                  hidden_size=1152,
#                  depth=28,
#                  num_heads=16,
#                  max_lang_cond_len=1024,
#                  img_cond_len=4096,
#                  lang_pos_embed_config=None,
#                  img_pos_embed_config=None,
#                  dtype=torch.bfloat16,
#                  # FLARE相关参数
#                  num_future_tokens=32,
#                  activation_layer=6,
#                  num_vl_fusion_layers=4,
#                  num_qformer_layers=2,
#                  alignment_temperature=0.07,
#                  vision_model_name="google/siglip-so400m-patch14-384",
#                  text_model_name="google/siglip-so400m-patch14-384"):
#         super().__init__()
        
#         # 基础参数
#         self.horizon = horizon
#         self.hidden_size = hidden_size
#         self.max_lang_cond_len = max_lang_cond_len
#         self.img_cond_len = img_cond_len
#         self.dtype = dtype
#         self.lang_pos_embed_config = lang_pos_embed_config
#         self.img_pos_embed_config = img_pos_embed_config
        
#         # FLARE相关参数
#         self.num_future_tokens = num_future_tokens
#         self.activation_layer = activation_layer
#         self.alignment_temperature = alignment_temperature

#         # 基础RDT组件
#         self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
#         self.freq_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        
#         self.vision_feature_adapter = nn.Linear(1152, 2048, bias=False)
#         with torch.no_grad():
#             nn.init.xavier_uniform_(self.vision_feature_adapter.weight)
        
#         # 确保future_obs_tokens维度正确
#         if hasattr(self, 'future_obs_tokens'):
#             if self.future_obs_tokens.shape[-1] != 2048:
#                 self.future_obs_tokens = nn.Parameter(
#                     torch.randn(1, self.num_future_tokens, 2048) * 0.02
#                 )
        
#         print("✅ 维度适配器初始化完成")
#         # 位置编码：[timestep; freq; state; action; future_obs]
#         #self.x_pos_embed = nn.Parameter(torch.zeros(1, horizon + 3 + num_future_tokens, hidden_size))
#         self.state_token_len = 1  # 状态压缩为1个token
#         self.seq_structure = {
#             'timestep': 1,
#             'freq': 1, 
#             'state': self.state_token_len,
#             'action': self.horizon,
#             'future_obs': self.num_future_tokens
#         }
        
#         # 计算总序列长度
#         total_seq_len = sum(self.seq_structure.values())
#         self.x_pos_embed = nn.Parameter(torch.zeros(1, total_seq_len, hidden_size))
        
#         # 预计算索引位置
#         self._compute_sequence_indices()
        
        
        
        
        
#         # 条件位置编码
#         self.lang_cond_pos_embed = nn.Parameter(torch.zeros(1, max_lang_cond_len, hidden_size))
#         self.img_cond_pos_embed = nn.Parameter(torch.zeros(1, img_cond_len, hidden_size))

#         # Transformer blocks
#         self.blocks = nn.ModuleList([RDTBlock(hidden_size, num_heads) for _ in range(depth)])
#         self.final_layer = FinalLayer(hidden_size, output_dim)
        
#         # FLARE组件
#         # 1. VL Token生成器
#         self.vl_token_generator = VLTokenGenerator(
#             vision_model_name=vision_model_name,
#             text_model_name=text_model_name,
#             hidden_size=hidden_size,
#             num_fusion_layers=num_vl_fusion_layers,
#             num_heads=num_heads
#         )
        
#         # 2. Q-Former目标生成器
#         self.target_generator = QFormerTargetGenerator(
#             hidden_size=hidden_size,
#             num_query_tokens=num_future_tokens,
#             num_layers=num_qformer_layers,
#             num_heads=num_heads
#         )
        
#         # 3. 未来观测token初始化
#         self.future_obs_tokens = nn.Parameter(torch.randn(1, num_future_tokens, hidden_size))
        
#         # 4. 未来观测token的MLP处理器
#         self.future_obs_mlp = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size * 2),
#             nn.GELU(),
#             nn.Linear(hidden_size * 2, hidden_size),
#             nn.Dropout(0.1),
#             nn.LayerNorm(hidden_size)
#         )
        
#         # 5. 激活对齐器（延迟初始化，避免循环引用）
#         self.activation_aligner = None
        
#         self.initialize_weights()
#         self._ensure_bf16_consistency()

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
                 depth=28,                    # 🎯 默认28层DiT
                 num_heads=16,
                 max_lang_cond_len=32,
                 img_cond_len=4096,
                 lang_pos_embed_config=None,
                 img_pos_embed_config=None,
                 dtype=torch.bfloat16,
                 # FLARE相关参数
                 num_future_tokens=32,
                 activation_layer=21,
                 num_vl_fusion_layers=4,
                 num_qformer_layers=2,        # 🎯 默认2层Q-Former
                 alignment_temperature=0.07,
                 # 🔧 只接收未来观测编码器参数
                 future_vision_model_name=None,
                 future_text_model_name=None,
                 future_vision_image_size=256,
                 # SigLIP2相关参数
                 siglip2_model_name="google/siglip-large-patch16-256",
                 use_pooling=True,
                 target_tokens=64,  # 2x2池化后的token数量):
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
        # 🔧 未来观测编码器路径
        self.future_vision_path = future_vision_model_name or "./models/siglip2-large-patch16-256"
        self.future_text_path = future_text_model_name or self.future_vision_path
        self.future_vision_image_size = future_vision_image_size
        
        self.use_pooling = use_pooling
        self.target_tokens = target_tokens
        
        print(f"🔧 初始化FLARE-SigLIP2模型:")
        print(f"   DiT未来tokens: {num_future_tokens}")
        print(f"   SigLIP2目标tokens: {target_tokens}")
        print(f"   使用池化: {use_pooling}")
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
        
        # 序列结构定义
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

        # 🎯 28层Transformer blocks
        print(f"🏗️  构建{depth}层DiT blocks...")
        self.blocks = nn.ModuleList([RDTBlock(hidden_size, num_heads) for _ in range(depth)])
        print(f"✅ {depth}层DiT blocks构建完成")
        
        self.final_layer = FinalLayer(hidden_size, output_dim)
        
        # FLARE组件
        print("🏗️  构建FLARE组件...")
        
        # # 1. VL Token生成器
        # self.vl_token_generator = VLTokenGenerator(
        #     vision_model_name=self.future_vision_path,     # SigLIP2-256
        #     text_model_name=self.future_text_path,         # SigLIP2-256
        #     hidden_size=hidden_size,
        #     num_fusion_layers=num_vl_fusion_layers,
        #     num_heads=num_heads,
        #     max_text_length=max_lang_cond_len,             # 32
        #     image_size=self.future_vision_image_size,      # 256
        # )
        
        # # 2. 🎯 2层Q-Former目标生成器
        # print(f"🏗️  构建{num_qformer_layers}层Q-Former...")
        # self.target_generator = QFormerTargetGenerator(
        #     hidden_size=hidden_size,
        #     num_query_tokens=num_future_tokens,
        #     num_layers=num_qformer_layers,  # 使用2层
        #     num_heads=num_heads
        # )
        # print(f"✅ {num_qformer_layers}层Q-Former构建完成")
        # ===========================================
        # 🎯 FLARE核心：SigLIP2目标生成器
        # ===========================================
        print("🏗️  初始化SigLIP2目标生成器...")
        self.siglip2_model = AutoModel.from_pretrained(
            siglip2_model_name, 
            local_files_only=True
        )
        self.siglip2_processor = AutoImageProcessor.from_pretrained(
            siglip2_model_name,
            local_files_only=True
        )
        
        # 冻结SigLIP2模型
        self.siglip2_model.requires_grad_(False)
        print("✅ SigLIP2模型已冻结")
        
        # SigLIP2特征维度适配
        siglip2_dim = self.siglip2_model.config.hidden_size  # 通常是1024
        self.siglip2_adapter = nn.Linear(siglip2_dim, hidden_size, bias=False)
        print(f"   SigLIP2维度: {siglip2_dim} → {hidden_size}")
        
        
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
        
        print("✅ FLARE组件构建完成")
        
        self.initialize_weights()
        self._ensure_bf16_consistency()
        
        # 模型规模统计
        total_params = sum(p.numel() for p in self.parameters())
        dit_params = sum(p.numel() for p in self.blocks.parameters())
        siglip2_params = sum(p.numel() for p in self.siglip2_model.parameters())
        
        print(f"📊 模型参数统计:")
        print(f"   总参数: {total_params:,}")
        print(f"   DiT参数: {dit_params:,} ({dit_params/total_params:.1%})")
        print(f"   SigLIP2参数: {siglip2_params:,} (冻结)")
    def _compute_sequence_indices(self):
        """预计算序列中各部分的索引位置"""
        self.indices = {}
        start_idx = 0
        for key, length in self.seq_structure.items():
            self.indices[key] = (start_idx, start_idx + length)
            start_idx += length
    def _generate_siglip2_targets(self, future_obs_images):
        """
        使用SigLIP2生成目标tokens
        
        Args:
            future_obs_images: (B, 3, H, W) 未来观测图像
            
        Returns:
            target_tokens: (B, target_tokens, hidden_size) 目标tokens
        """
        batch_size = future_obs_images.shape[0]
        device = future_obs_images.device
        
        with torch.no_grad():
            # 1. 通过SigLIP2提取特征
            try:
                # 确保图像尺寸正确 (SigLIP2期望256x256)
                if future_obs_images.shape[-1] != 256:
                    future_obs_images = F.interpolate(
                        future_obs_images, size=(256, 256), 
                        mode='bilinear', align_corners=False
                    )
                
                # SigLIP2前向传播
                siglip2_outputs = self.siglip2_model(future_obs_images)
                siglip2_features = siglip2_outputs.last_hidden_state  # (B, 256, 1024)
                
                print(f"🔍 SigLIP2原始特征: {siglip2_features.shape}")
                
                # 2. 可选的2x2池化
                if self.use_pooling:
                    # 重塑为2D: (B, 16, 16, 1024) -> (B, 1024, 16, 16)
                    B, L, D = siglip2_features.shape
                    H = W = int(L ** 0.5)  # 256 tokens -> 16x16
                    assert H * W == L, f"Expected square tokens, got {L}"
                    
                    siglip2_features = siglip2_features.transpose(1, 2).reshape(B, D, H, W)
                    
                    # 2x2平均池化: 16x16 -> 8x8 = 64 tokens
                    pooled_features = F.avg_pool2d(siglip2_features, kernel_size=2)  # (B, 1024, 8, 8)
                    
                    # 重塑回token序列: (B, 1024, 8, 8) -> (B, 64, 1024)
                    _, D, H_new, W_new = pooled_features.shape
                    siglip2_features = pooled_features.reshape(B, D, H_new * W_new).transpose(1, 2)
                    
                    print(f"🔍 池化后特征: {siglip2_features.shape}")
                
            except Exception as e:
                print(f"❌ SigLIP2特征提取失败: {e}")
                # 回退到零特征
                target_len = self.target_tokens if self.use_pooling else 256
                siglip2_features = torch.zeros(
                    batch_size, target_len, self.siglip2_model.config.hidden_size,
                    device=device, dtype=self.dtype
                )
        
        # 3. 维度适配：SigLIP2维度 -> DiT隐藏维度
        target_tokens = self.siglip2_adapter(siglip2_features.to(self.dtype))
        
        print(f"🎯 最终目标tokens: {target_tokens.shape}")
        
        return target_tokens
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
        # # L2归一化
        # pred_norm = F.normalize(pred_future_tokens, p=2, dim=-1)  # (B, M, D)
        # target_norm = F.normalize(target_future_tokens, p=2, dim=-1)  # (B, M, D)
        
        # batch_size, num_tokens, hidden_dim = pred_norm.shape
        
        # # 计算相似度矩阵 (B, M, M)
        # similarity = torch.bmm(pred_norm, target_norm.transpose(1, 2)) / temperature
        
        # # 对角线元素是正样本对
        # labels = torch.arange(num_tokens, device=similarity.device).unsqueeze(0).expand(batch_size, -1)
        
        # # 计算对比损失
        # loss = F.cross_entropy(similarity.reshape(-1, num_tokens), labels.reshape(-1))
        
        # return loss
         # 计算余弦相似度
        cosine_sim = F.cosine_similarity(pred_future_tokens, target_future_tokens, dim=-1)
        
        # 返回负余弦相似度（最大化相似度）
        loss = 1 - cosine_sim.mean()

        return loss

#     def forward(self, x, freq, t, lang_c, img_c, lang_mask=None, img_mask=None, 
#                 future_vision_tokens=None, text_instructions=None, future_obs_image=None, return_alignment_loss=False):
#         """
#         FLARE模型前向传播
        
#         Args:
#             x: (B, T, D) 状态和动作序列
#             freq: (B,) 控制频率
#             t: (B,) 时间步
#             lang_c: (B, L, D) 语言条件
#             img_c: (B, I, D) 图像条件
#             lang_mask: (B, L) 语言掩码
#             img_mask: (B, I) 图像掩码
#             future_vision_tokens: (B, V, D) 未来观测的视觉token
#             text_instructions: 文本指令（用于VL生成）
#             return_alignment_loss: 是否返回对齐损失
#         """
#         # 🎯 确保输入数据类型一致
#         # 统一数据类型处理
#         target_dtype = self.dtype
#         device = x.device

#         # 确保所有输入张量使用相同的数据类型
#         x = x.to(dtype=target_dtype, device=device)
#         if isinstance(freq, torch.Tensor):
#             freq = freq.to(dtype=target_dtype, device=device)
#         if isinstance(t, torch.Tensor):
#             t = t.to(dtype=target_dtype, device=device)
#         lang_c = lang_c.to(dtype=target_dtype, device=device)
#         img_c = img_c.to(dtype=target_dtype, device=device)
#         if future_vision_tokens is not None:
#             future_vision_tokens = future_vision_tokens.to(dtype=target_dtype, device=device)
        
#         # batch_size = x.shape[0]
        
#         # # 编码时间步和频率
#         # t = self.t_embedder(t).unsqueeze(1)  # (B, 1, D)
#         # freq = self.freq_embedder(freq).unsqueeze(1)  # (B, 1, D)
        
#         # # 初始化未来观测token
#         # future_obs_tokens = self.future_obs_tokens.expand(batch_size, -1, -1)  # (B, M, D)
#         # future_obs_tokens = self.future_obs_mlp(future_obs_tokens)
        
#         # # 扩展时间步到batch
#         # if t.shape[0] == 1:
#         #     t = t.expand(batch_size, -1, -1)
        
#         # # 拼接所有token: [timestep, freq, state+action, future_obs]
#         # #x = torch.cat([t, freq, x, future_obs_tokens], dim=1)  # (B, T+3+M, D)
#         # # 添加位置编码
#         # x = x + self.x_pos_embed
#         # lang_c = lang_c + self.lang_cond_pos_embed[:, :lang_c.shape[1]]
#         batch_size = x.shape[0]
#         if t.shape[0] == 1 and batch_size != 1:
#             # 推理/采样时如果t是单元素，扩展成和batch_size一致
#             t = t.expand(batch_size)
#         t_embed = self.t_embedder(t).unsqueeze(1)  # (B, 1, D)
#         device = x.device
#         target_dtype = self.dtype
        
#         # 1. 编码时间步和频率
#         t_embed = self.t_embedder(t).unsqueeze(1)  # (B, 1, D)
#         freq_embed = self.freq_embedder(freq).unsqueeze(1)  # (B, 1, D)
        
#         # 2. 分离状态和动作 (关键修复)
#         # x 输入应该是 [state_tokens, action_tokens]
#         state_start, state_end = 0, self.state_token_len
#         action_start, action_end = self.state_token_len, x.shape[1]
        
#         state_tokens = x[:, state_start:state_end]  # (B, state_len, D)
#         action_tokens = x[:, action_start:action_end]  # (B, action_len, D)
        
#         # 3. 处理未来观测token (FLARE核心修复)
#         future_obs_tokens = self._process_future_obs_tokens(
#             batch_size, future_obs_image, device, target_dtype
#         )
        
#         # 4. 正确的序列拼接 (按照定义的结构)
#         sequence_parts = [
#             t_embed,           # timestep
#             freq_embed,        # freq
#             state_tokens,      # state
#             action_tokens,     # action  
#             future_obs_tokens  # future_obs
#         ]
#         sequence_parts = [part for part in sequence_parts if part is not None]
#         if not sequence_parts:
#             # 如果所有部分都是None，创建dummy tensor
#             sequence_parts = [torch.zeros(batch_size, 1, 2048, device=device, dtype=dtype)] 
#         for i, p in enumerate(sequence_parts):
#             assert p.shape[0] == batch_size, f"part {i} batch size {p.shape[0]} != {batch_size}"  
#         sequence = torch.cat(sequence_parts, dim=1)  # (B, total_seq_len, D)
        
#         # 5. 验证序列长度
#         expected_len = sum(self.seq_structure.values())
#         assert sequence.shape[1] == expected_len, \
#             f"序列长度不匹配: {sequence.shape[1]} vs {expected_len}"
        
#         # 6. 添加位置编码
#         sequence = sequence + self.x_pos_embed
        
#         # 7. 准备条件
#         conds = [lang_c, img_c]
#         masks = [lang_mask, img_mask]
#         # 8. 通过transformer blocks (改进条件使用策略)
#         target_future_tokens = None
#         alignment_loss = None
        

#         # FLARE: 计算目标tokens（如果需要）
#         # target_future_tokens = None
#         # if future_vision_tokens is not None and return_alignment_loss:
#         #     try:
#         #         # 1. 生成VL tokens
#         #         vl_tokens, vl_mask = self.vl_token_generator(
#         #             future_obs_image, text_instructions
#         #         )

                
#         #         # 2. 生成目标tokens
#         #         target_future_tokens = self.target_generator(vl_tokens, vl_mask)
#         #     except Exception as e:
#         #         raise  # 直接让程序崩溃，打印完整Traceback
#         # FLARE: 计算目标tokens（在transformer处理前）
#         if return_alignment_loss and future_obs_image is not None:
#             import torch.nn.functional as F
#             # 把图像从 (B,3,384,384) resize 到 (B,3,256,256)
#             future_obs_image = F.interpolate(
#                 future_obs_image,
#                 size=(256, 256),
#                 mode='bilinear',
#                 align_corners=False
# )
#             try:
#                 vl_tokens, vl_mask = self.vl_token_generator(
#                     future_obs_image, text_instructions
#                 )
#                 vl_mask = vl_mask.bool()
#                 target_future_tokens = self.target_generator(vl_tokens, vl_mask)
#             except Exception as e:
#                 print(f"🔴 Target token generation failed: {repr(e)}")
#                 traceback.print_exc()
#                 target_future_tokens = None

#         # 通过transformer blocks
#         # conds = [lang_c, img_c]
#         # masks = [lang_mask, img_mask]
#         # alignment_loss = None
        
#         # Transformer处理

#         for i, block in enumerate(self.blocks):
#             # 交替注入语言/视觉条件
#             condition_idx = i % len(conds)     # i 为偶数时 0 → 语言；i 为奇数时 1 → 视觉
#             c, mask = conds[condition_idx], masks[condition_idx]
#             sequence = block(sequence, c, mask)

#         # 最终层处理
#         sequence = self.final_layer(sequence)

#         # 只返回动作token，去除时间步、频率和未来观测token
#         #action_tokens = x[:, 2:2+self.horizon]  # (B, horizon, out_channels)
#         action_start_idx, action_end_idx = self.indices['action']
#         action_tokens = sequence[:, action_start_idx:action_end_idx]
        
#         # 11. 计算对齐损失 (使用正确的索引)
#         if return_alignment_loss and target_future_tokens is not None:
#             try:
#                 self._initialize_activation_aligner()
                
#                 # 使用正确的未来token位置
#                 future_start_idx, future_end_idx = self.indices['future_obs']
#                 alignment_loss, _ = self.activation_aligner.compute_precise_alignment_loss(
#                     target_future_tokens, 
#                     horizon=self.horizon,
#                     future_token_indices=(future_start_idx, future_end_idx)
#                 )
#             except Exception as e:
#                 print(f"Warning: Alignment loss computation failed: {e}")
#                 alignment_loss = torch.tensor(0.0, device=action_tokens.device)
        
#         if return_alignment_loss:
#             return action_tokens, alignment_loss
#         else:
#             return action_tokens
    def forward(self, x, freq, t, lang_c, img_c, lang_mask=None, img_mask=None, 
                future_obs_image=None, return_alignment_loss=False, **kwargs):
        """
        FLARE模型前向传播
        """
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
        
        # ===========================================
        # 🎯 FLARE: 生成SigLIP2目标tokens
        # ===========================================
        target_future_tokens = None
        if return_alignment_loss and future_obs_image is not None:
            try:
                target_future_tokens = self._generate_siglip2_targets(future_obs_image)
                print(f"✅ 成功生成目标tokens: {target_future_tokens.shape}")
            except Exception as e:
                print(f"🔴 目标token生成失败: {repr(e)}")
                target_future_tokens = None

        # 7. 通过transformer blocks
        for i, block in enumerate(self.blocks):
            condition_idx = i % len(conds)
            c, mask = conds[condition_idx], masks[condition_idx]
            sequence = block(sequence, c, mask)

        # 8. 最终层处理
        sequence = self.final_layer(sequence)

        # 9. 只返回动作token
        action_start_idx, action_end_idx = self.indices['action']
        action_tokens = sequence[:, action_start_idx:action_end_idx]
        
        # 10. 计算对齐损失
        alignment_loss = None
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
                print(f"✅ 对齐损失计算成功: {alignment_loss.item():.4f}")
            except Exception as e:
                print(f"⚠️ 对齐损失计算失败: {e}")
                alignment_loss = torch.tensor(0.0, device=action_tokens.device)
        
        if return_alignment_loss:
            return action_tokens, alignment_loss
        else:
            return action_tokens
        
    def _process_future_obs_tokens(self, batch_size, device, target_dtype):
        """处理未来观测tokens - DiT序列中的占位符"""
        # 确保future_obs_tokens参数存在且维度正确
        if not hasattr(self, 'future_obs_tokens') or self.future_obs_tokens.shape[-1] != self.hidden_size:
            print(f"🔧 创建可学习future_obs_tokens: {self.num_future_tokens} x {self.hidden_size}")
            self.future_obs_tokens = nn.Parameter(
                torch.randn(1, self.num_future_tokens, self.hidden_size) * 0.02
            ).to(device=device, dtype=target_dtype)
        
        # 扩展到batch size
        future_obs_tokens = self.future_obs_tokens.expand(batch_size, -1, -1)
        
        # 通过MLP处理
        future_obs_tokens = self.future_obs_mlp(future_obs_tokens)
        
        return future_obs_tokens.to(device=device, dtype=target_dtype)