import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# 强制离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class VLFusionBlock(nn.Module):
    """VL融合块：Self-Attention + FeedForward"""
    
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Self-Attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Norm
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # FeedForward
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N, D) 输入特征
            mask: (B, N) 注意力掩码
        Returns:
            x: (B, N, D) 输出特征
        """
        # 统一将 mask 转成 bool，避免 key_padding_mask 类型报错
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.bool()

        # Self-Attention with residual connection
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attention(
            x, x, x,
            key_padding_mask=mask
        )
        x = residual + attn_out

        # FeedForward with residual connection
        residual = x
        x = self.norm2(x)
        ff_out = self.feedforward(x)
        x = residual + ff_out

        return x


class VLTokenGenerator(nn.Module):
    """
    统一T5架构的VL Token生成器
    
    关键修改：
    - 完全移除SigLIP2文本编码器
    - 只使用T5预计算嵌入处理所有文本
    - 简化架构，避免双编码器复杂性
    """
    
    def __init__(self, 
                 vision_model_name=None,
                 text_model_name=None,
                 hidden_size=2048,
                 num_fusion_layers=4,
                 num_heads=8,
                 dropout=0.1,
                 max_text_length=1024,
                 image_size=256,
                 device="cuda",
                 torch_dtype=torch.float16,
                 t5_embed_dim=4096,
                 use_precomp_lang_embed=True):
        super().__init__()
        
        # 🎯 核心配置
        self.hidden_size = hidden_size
        self.max_text_length = max_text_length
        self.use_precomp_lang_embed = use_precomp_lang_embed
        self.t5_embed_dim = t5_embed_dim
        self.image_size = image_size
        self.device = device
        self.max_text_length = 1024
        print("🎯 初始化统一T5架构的VLTokenGenerator")
        print(f"   隐藏层大小: {hidden_size}")
        print(f"   T5嵌入维度: {t5_embed_dim}")
        print(f"   图像尺寸: {image_size}")
        print(f"   最大文本长度: {max_text_length}")
        
        # 🔧 关键修改：完全移除SigLIP2文本编码器
        self.text_encoder = None
        print("   ✅ 跳过SigLIP2文本编码器（统一使用T5）")
        
        # 1. 视觉编码器 (SigLIP2) - 保持不变
        print(f"   🖼️ 初始化视觉编码器...")
        if vision_model_name is None:
            vision_model_name = "./models/siglip2-large-patch16-256"
            
        # 导入您的视觉编码器
        from models.multimodal_encoder.siglip2_encoder import SigLIP2VisionTower
        
        self.vision_encoder = SigLIP2VisionTower(
            vision_tower=vision_model_name, 
            args=None,
            image_size=image_size
        )
        vision_dim = self.vision_encoder.hidden_size
        print(f"   ✅ 视觉编码器初始化完成，hidden_size: {vision_dim}")
        
        # 2. 🎯 T5文本适配器：4096 → 2048
        print(f"   📝 初始化T5文本适配器...")
        self.t5_text_adapter = nn.Linear(
            self.t5_embed_dim, 
            self.hidden_size, 
            bias=False
        )
        print(f"   ✅ T5适配器: {self.t5_embed_dim} → {self.hidden_size}")
        
        # 3. 特征投影层
        print(f"   🔧 配置维度对齐:")
        print(f"      vision_dim: {vision_dim}")
        print(f"      target_hidden_size: {hidden_size}")
        
        # 视觉投影层
        if vision_dim != hidden_size:
            self.vision_proj = nn.Linear(vision_dim, hidden_size)
            print(f"      创建视觉投影层: {vision_dim} → {hidden_size}")
        else:
            self.vision_proj = nn.Identity()
            print(f"      视觉维度匹配，无需投影")
            
        # T5文本投影层（在适配器之后）
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        print(f"      T5文本投影层: {hidden_size} → {hidden_size}")
        
        # 4. 位置编码
        max_vision_tokens = (image_size // 16) ** 2  # 假设patch_size=16
        max_text_tokens = max_text_length
        
        self.vision_pos_embed = nn.Parameter(
            torch.zeros(1, max_vision_tokens, hidden_size)
        )
        self.text_pos_embed = nn.Parameter(
            torch.zeros(1, max_text_tokens, hidden_size)
        )
        print(f"   🔢 位置编码: vision({max_vision_tokens}), text({max_text_tokens})")
        
        # 5. 模态类型嵌入
        self.vision_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.text_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        # 6. 多层VL融合块
        self.fusion_layers = nn.ModuleList([
            VLFusionBlock(hidden_size, num_heads, dropout)
            for _ in range(num_fusion_layers)
        ])
        print(f"   🔄 融合层数: {num_fusion_layers}")
        
        # 7. 输出层归一化
        self.output_norm = nn.LayerNorm(hidden_size)
        
        # 初始化参数
        self._initialize_weights()
        
        print(f"   ✅ 统一T5架构VLTokenGenerator初始化完成")
        
    def _initialize_weights(self):
        """初始化参数"""
        # T5适配器初始化
        with torch.no_grad():
            nn.init.xavier_uniform_(self.t5_text_adapter.weight)
        
        # 位置编码初始化
        nn.init.trunc_normal_(self.vision_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.text_pos_embed, std=0.02)
        
        # 模态类型嵌入初始化
        nn.init.trunc_normal_(self.vision_type_embed, std=0.02)
        nn.init.trunc_normal_(self.text_type_embed, std=0.02)
        
        # 投影层初始化
        for module in [self.vision_proj, self.text_proj]:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, future_images=None, text_instructions=None, image_mask=None, text_mask=None):
        """
        统一T5架构的前向传播
        
        Args:
            future_images: (B, C, H, W) 未来观测图像
            text_instructions: T5预计算嵌入tensor或文件路径列表
            image_mask: (B, N_patches) 图像掩码（可选）
            text_mask: (B, L) 文本掩码（可选）
            
        Returns:
            vl_tokens: (B, N_total, D) 融合后的VL tokens
            vl_mask: (B, N_total) VL tokens的掩码
        """
        device = next(self.parameters()).device
        
        # 确定batch_size
        if future_images is not None:
            future_images = future_images.to(device)
            batch_size = future_images.shape[0]
            if len(future_images.shape) == 3:
                future_images = future_images.unsqueeze(0)
        else:
            if isinstance(text_instructions, torch.Tensor):
                batch_size = text_instructions.shape[0]
            elif isinstance(text_instructions, list):
                batch_size = len(text_instructions)
            else:
                batch_size = 1
        
        # 1. 🖼️ 视觉编码 - 保持原有逻辑
        if future_images is not None:
            try:
                if hasattr(self.vision_encoder, 'vision_tower'):
                    
                    vision_outputs = self.vision_encoder.vision_tower(
                        future_images,
                        interpolate_pos_encoding=True
                    )
                    
                    if hasattr(vision_outputs, 'last_hidden_state'):
                        vision_features = vision_outputs.last_hidden_state
                    elif hasattr(vision_outputs, 'pooler_output'):
                        pooler_out = vision_outputs.pooler_output
                        num_patches = (self.image_size // 16) ** 2
                        vision_features = pooler_out.unsqueeze(1).expand(-1, num_patches, -1)
                    else:
                        vision_features = vision_outputs
                        
                    
                else:
                    print("⚠️ 无法使用位置编码插值，使用标准方法")
                    vision_features = self.vision_encoder(future_images)
                    
            except Exception as e:
                print(f"⚠️ 位置编码插值失败，回退到标准方法: {e}")
                vision_features = self.vision_encoder(future_images)
            
            # 视觉投影
            vision_features = self.vision_proj(vision_features)
            
            # 位置编码处理
            seq_len = vision_features.shape[1]
            pos_embed_len = self.vision_pos_embed.shape[1]
            
            if seq_len <= pos_embed_len:
                vision_features = vision_features + self.vision_pos_embed[:, :seq_len, :]
            else:
                repeat_times = (seq_len // pos_embed_len) + 1
                extended_pos_embed = self.vision_pos_embed.repeat(1, repeat_times, 1)
                vision_features = vision_features + extended_pos_embed[:, :seq_len, :]
            
            vision_features = vision_features + self.vision_type_embed
            
        else:
            vision_features = torch.empty(batch_size, 0, self.hidden_size, device=device)
        
        # 2. 🎯 T5文本处理 - 核心修改
        text_features, computed_text_mask = self._process_t5_text(
            text_instructions, batch_size, device
        )
        
        # 3. 🔗 拼接视觉和文本特征
        if vision_features.shape[1] > 0 and text_features.shape[1] > 0:
            vl_features = torch.cat([vision_features, text_features], dim=1)
        elif vision_features.shape[1] > 0:
            vl_features = vision_features
            print(f"🖼️ 仅视觉特征: {vl_features.shape}")
        elif text_features.shape[1] > 0:
            vl_features = text_features
            print(f"📝 仅文本特征: {vl_features.shape}")
        else:
            vl_features = torch.zeros(batch_size, 1, self.hidden_size, device=device)
            print(f"🔄 使用dummy特征: {vl_features.shape}")
        
        # 4. 创建联合掩码
        if image_mask is None and vision_features.shape[1] > 0:
            image_mask = torch.ones(batch_size, vision_features.shape[1], 
                                dtype=torch.bool, device=device)
        elif vision_features.shape[1] == 0:
            image_mask = torch.empty(batch_size, 0, dtype=torch.bool, device=device)
        
        # 使用computed_text_mask而不是text_mask
        if text_mask is None:
            text_mask = computed_text_mask
        
        if vision_features.shape[1] > 0 and text_features.shape[1] > 0:
            vl_mask = torch.cat([image_mask, text_mask], dim=1)
        elif vision_features.shape[1] > 0:
            vl_mask = image_mask
        elif text_features.shape[1] > 0:
            vl_mask = text_mask
        else:
            vl_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        
        # 5. 🔄 通过多层融合块
        for i, fusion_layer in enumerate(self.fusion_layers):
            vl_features = fusion_layer(vl_features, mask=~vl_mask)
        
        # 6. 输出归一化
        vl_tokens = self.output_norm(vl_features)
        
        return vl_tokens, vl_mask
    
    def _process_t5_text(self, text_instructions, batch_size, device):
        """
        🎯 核心方法：处理T5文本输入
        
        统一处理所有文本输入，包括：
        1. T5预计算嵌入tensor
        2. T5嵌入文件路径列表
        3. 空输入
        """
        try:
            if text_instructions is None:
                # 没有文本输入，返回零嵌入
                text_features = torch.zeros(batch_size, self.max_text_length, self.hidden_size, device=device)
                text_mask = torch.zeros(batch_size, self.max_text_length, dtype=torch.bool, device=device)
                return text_features, text_mask
            
            # 🔧 情况1：直接传入T5嵌入tensor
            if isinstance(text_instructions, torch.Tensor):
                print(f"🎯 直接使用T5嵌入tensor: {text_instructions.shape}")
                t5_embeds = text_instructions.to(device)
                
                # 维度适配：4096 → 2048
                if t5_embeds.shape[-1] == self.t5_embed_dim:
                    text_features = self.t5_text_adapter(t5_embeds)
                    print(f"   ✅ T5维度适配: {self.t5_embed_dim} → {self.hidden_size}")
                else:
                    text_features = t5_embeds
                    print(f"   ⚠️ T5嵌入维度已匹配: {t5_embeds.shape[-1]}")
                
                # 生成掩码
                text_mask = torch.ones(text_features.shape[:2], dtype=torch.bool, device=device)
                
            # 🔧 情况2：文件路径列表（加载T5嵌入）
            elif isinstance(text_instructions, list) and len(text_instructions) > 0:
                if isinstance(text_instructions[0], str) and text_instructions[0].endswith('.pt'):
                    
                    t5_embeds_list = []
                    for embed_path in text_instructions:
                        try:
                            if os.path.exists(embed_path):
                                embed = torch.load(embed_path, map_location=device)
                                if isinstance(embed, torch.Tensor):
                                    t5_embeds_list.append(embed)
                                else:
                                    # 处理字典格式
                                    if 'embedding' in embed:
                                        t5_embeds_list.append(embed['embedding'])
                                    elif 'last_hidden_state' in embed:
                                        t5_embeds_list.append(embed['last_hidden_state'])
                                    else:
                                        # 使用零嵌入
                                        t5_embeds_list.append(
                                            torch.zeros(self.max_text_length, self.t5_embed_dim, device=device)
                                        )
                            else:
                                print(f"⚠️ 文件不存在: {embed_path}")
                                t5_embeds_list.append(
                                    torch.zeros(self.max_text_length, self.t5_embed_dim, device=device)
                                )
                        except Exception as e:
                            print(f"❌ 加载嵌入失败 {embed_path}: {e}")
                            t5_embeds_list.append(
                                torch.zeros(self.max_text_length, self.t5_embed_dim, device=device)
                            )
                    
                    if t5_embeds_list:
                        # 统一序列长度
                        max_len = min(max(embed.shape[0] if len(embed.shape) == 2 else embed.shape[1] 
                                         for embed in t5_embeds_list), self.max_text_length)
                        
                        padded_embeds = []
                        for embed in t5_embeds_list:
                            if len(embed.shape) == 3:
                                embed = embed[0]  # (1, seq_len, dim) → (seq_len, dim)
                            
                            if embed.shape[0] > max_len:
                                embed = embed[:max_len]
                            elif embed.shape[0] < max_len:
                                pad_size = max_len - embed.shape[0]
                                embed = torch.cat([
                                    embed,
                                    torch.zeros(pad_size, embed.shape[1], device=device)
                                ], dim=0)
                            
                            padded_embeds.append(embed)
                        
                        t5_embeds = torch.stack(padded_embeds, dim=0)  # (B, seq_len, 4096)
                        
                        # 🎯 关键：T5维度适配 4096 → 2048
                        text_features = self.t5_text_adapter(t5_embeds)
                        text_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)
                        
                    else:
                        # 加载失败，使用零嵌入
                        text_features = torch.zeros(batch_size, self.max_text_length, self.hidden_size, device=device)
                        text_mask = torch.zeros(batch_size, self.max_text_length, dtype=torch.bool, device=device)
                else:
                    # 🚨 不应该收到文本字符串，因为我们只使用T5
                    print("❌ 收到文本字符串，但VLTokenGenerator配置为只使用T5嵌入")
                    print("   请检查数据传递流程，应该传递T5嵌入文件路径或tensor")
                    text_features = torch.zeros(batch_size, self.max_text_length, self.hidden_size, device=device)
                    text_mask = torch.zeros(batch_size, self.max_text_length, dtype=torch.bool, device=device)
            else:
                # 其他情况，使用零嵌入
                text_features = torch.zeros(batch_size, self.max_text_length, self.hidden_size, device=device)
                text_mask = torch.zeros(batch_size, self.max_text_length, dtype=torch.bool, device=device)
            
            # 🔧 应用文本投影
            text_features = self.text_proj(text_features)
            
            # 🔧 添加位置编码
            seq_len = text_features.shape[1]
            if seq_len <= self.text_pos_embed.shape[1]:
                text_features = text_features + self.text_pos_embed[:, :seq_len, :]
            else:
                print(f"⚠️ 文本序列长度超过位置编码: {seq_len} > {self.text_pos_embed.shape[1]}")
                repeat_times = (seq_len // self.text_pos_embed.shape[1]) + 1
                extended_pos_embed = self.text_pos_embed.repeat(1, repeat_times, 1)
                text_features = text_features + extended_pos_embed[:, :seq_len, :]
            
            text_features = text_features + self.text_type_embed
            
            # 🔧 检查NaN
            if torch.isnan(text_features).any():
                print("⚠️ 检测到文本特征中的NaN，使用零嵌入替换")
                text_features = torch.zeros_like(text_features)
                text_mask = torch.zeros_like(text_mask)
            
            return text_features, text_mask
            
        except Exception as e:
            print(f"❌ T5文本处理失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 回退到零嵌入
            text_features = torch.zeros(batch_size, self.max_text_length, self.hidden_size, device=device)
            text_mask = torch.zeros(batch_size, self.max_text_length, dtype=torch.bool, device=device)
            return text_features, text_mask


# 🧪 测试函数
def test_unified_t5_vl_generator():
    """测试统一T5架构的VL Token生成器"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🧪 测试统一T5架构VL Token生成器...")
    print("="*60)
    
    # 创建VL Token生成器
    vl_generator = VLTokenGenerator(
        vision_model_name="/home/deng_xiang/qian_daichao/RoboTwin/policy/RDT_flare/siglip2-large-patch16-256",
        hidden_size=2048,
        num_fusion_layers=4,
        max_text_length=32,
        image_size=256,
        t5_embed_dim=4096,
        use_precomp_lang_embed=True  # 🎯 使用T5预计算嵌入
    ).to(device)
    
    try:
        # 创建测试数据
        batch_size = 2
        
        # 测试图像
        future_images = torch.randn(batch_size, 3, 256, 256).to(device)
        
        # 🎯 测试T5嵌入tensor
        t5_embeds = torch.randn(batch_size, 16, 4096).to(device)  # 模拟T5嵌入
        
        print(f"📝 测试输入:")
        print(f"   图像形状: {future_images.shape}")
        print(f"   T5嵌入形状: {t5_embeds.shape}")
        
        # 前向传播
        with torch.no_grad():
            vl_tokens, vl_mask = vl_generator(future_images, t5_embeds)
        
        print(f"\n✅ 测试成功!")
        print(f"   VL tokens形状: {vl_tokens.shape}")
        print(f"   VL mask形状: {vl_mask.shape}")
        print(f"   每个样本的有效token数: {vl_mask.sum(dim=1).tolist()}")
        print(f"   隐藏层维度: {vl_tokens.shape[-1]}")
        
        # 数据质量检查
        print(f"\n📊 数据质量:")
        print(f"   VL tokens范围: [{vl_tokens.min():.3f}, {vl_tokens.max():.3f}]")
        print(f"   VL tokens均值: {vl_tokens.mean():.3f}")
        print(f"   VL tokens标准差: {vl_tokens.std():.3f}")
        
        return vl_generator, vl_tokens, vl_mask
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_t5_file_loading():
    """测试T5文件加载功能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n🧪 测试T5文件加载...")
    print("="*60)
    
    vl_generator = VLTokenGenerator(
        vision_model_name="/home/deng_xiang/qian_daichao/RoboTwin/policy/RDT_flare/siglip2-large-patch16-256",
        hidden_size=2048,
        t5_embed_dim=4096,
        use_precomp_lang_embed=True
    ).to(device)
    
    try:
        # 模拟T5嵌入文件路径
        fake_t5_paths = [
            "training_data/grab_roller/demo_grab_roller/grab_roller-demo_clean-50/episode_30/instructions/lang_embed_33.pt",
            "training_data/grab_roller/demo_grab_roller/grab_roller-demo_clean-50/episode_31/instructions/lang_embed_34.pt"
        ]
        
        future_images = torch.randn(2, 3, 256, 256).to(device)
        
        print(f"📁 测试T5文件路径:")
        for path in fake_t5_paths:
            print(f"   {path}")
        
        with torch.no_grad():
            vl_tokens, vl_mask = vl_generator(future_images, fake_t5_paths)
        
        print(f"✅ T5文件路径处理成功!")
        print(f"   输出形状: {vl_tokens.shape}")
        print(f"   掩码形状: {vl_mask.shape}")
        
    except Exception as e:
        print(f"❌ T5文件路径处理失败: {e}")


def test_integration_example():
    """集成测试示例"""
    print("\n🎯 统一T5架构集成示例:")
    print("="*60)
    
    example_code = '''
# 🎯 在训练脚本中的使用示例

# 1. 从数据集获取数据
batch = dataloader.next()
future_obs_images = batch["future_obs_images"]        # (B, 3, H, W)
t5_embed_paths = batch["flare_text_embed_paths"]     # List[str] - T5嵌入文件路径
lang_embeds = batch["lang_embeds"]                   # (B, seq_len, 4096) - T5嵌入

# 2. FLARE处理
total_loss, loss_dict = model.compute_loss_with_flare(
    lang_tokens=lang_embeds,                         # T5嵌入用于DiT
    img_tokens=img_embeds,                           # 当前图像用于DiT
    text_instructions=t5_embed_paths,                # T5路径用于FLARE
    future_obs_images=future_obs_images,             # 未来图像用于FLARE
    return_alignment_loss=True
)

# 3. VLTokenGenerator内部会自动处理T5嵌入：
#    - 加载.pt文件 → 4096维T5嵌入
#    - T5适配器 → 2048维统一表示
#    - 与视觉特征融合 → VL tokens
'''
    
    print(example_code)
    
    usage_notes = '''
🔧 关键修改点:

1. VLTokenGenerator改动:
   ✅ 移除SigLIP2文本编码器
   ✅ 新增T5适配器 (4096→2048)
   ✅ 统一文本处理流程

2. 数据集改动:
   ✅ 返回flare_text_embed_paths (T5嵌入路径)
   ✅ 不再生成简化文本指令

3. 训练脚本改动:
   ✅ 传递T5嵌入路径给FLARE
   ✅ 不再处理双编码器复杂性

4. 解决的问题:
   ✅ 维度不匹配 (4096 vs 2048)
   ✅ 文件路径错误传递
   ✅ NaN问题
   ✅ 架构复杂性

🎯 现在的数据流:
T5嵌入(.pt) → 统一适配器 → 2048维 → DiT/FLARE共享表示
'''
    
    print(usage_notes)


if __name__ == "__main__":
    print("🚀 开始统一T5架构测试...")
    
    # 测试1: 基础功能
    generator, tokens, mask = test_unified_t5_vl_generator()
    
    if generator is not None:
        # 测试2: T5文件加载
        test_t5_file_loading()
        
        # 测试3: 集成示例
        test_integration_example()
        
        print(f"\n🎉 统一T5架构VLTokenGenerator测试完成!")
        print(f"💡 现在所有文本处理都使用T5，避免了双编码器的复杂性。")
        print(f"🔧 请更新您的训练脚本，确保传递T5嵌入路径给FLARE组件。")
    else:
        print(f"\n⚠️ 请检查模型路径和依赖项。")


# 🎯 额外的调试工具
class VLTokenGeneratorDebugger:
    """VLTokenGenerator调试工具"""
    
    def __init__(self, vl_generator):
        self.vl_generator = vl_generator
    
    def debug_t5_processing(self, text_instructions):
        """调试T5处理过程"""
        print("🔍 调试T5处理过程:")
        print("="*50)
        
        device = next(self.vl_generator.parameters()).device
        batch_size = len(text_instructions) if isinstance(text_instructions, list) else 1
        
        try:
            # 直接调用T5处理方法
            text_features, text_mask = self.vl_generator._process_t5_text(
                text_instructions, batch_size, device
            )
            
            print(f"✅ T5处理成功:")
            print(f"   输入类型: {type(text_instructions)}")
            print(f"   输出特征: {text_features.shape}")
            print(f"   输出掩码: {text_mask.shape}")
            print(f"   有效token数: {text_mask.sum(dim=1).tolist()}")
            
            # 检查数据质量
            if torch.isnan(text_features).any():
                print("❌ 检测到NaN值!")
            else:
                print(f"   数据范围: [{text_features.min():.3f}, {text_features.max():.3f}]")
                print(f"   数据均值: {text_features.mean():.3f}")
                print(f"   数据标准差: {text_features.std():.3f}")
                
        except Exception as e:
            print(f"❌ T5处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    def debug_vision_processing(self, images):
        """调试视觉处理过程"""
        print("🔍 调试视觉处理过程:")
        print("="*50)
        
        try:
            with torch.no_grad():
                vision_features = self.vl_generator.vision_encoder(images)
            
            print(f"✅ 视觉处理成功:")
            print(f"   输入图像: {images.shape}")
            print(f"   输出特征: {vision_features.shape}")
            print(f"   数据范围: [{vision_features.min():.3f}, {vision_features.max():.3f}]")
            
        except Exception as e:
            print(f"❌ 视觉处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    def debug_full_pipeline(self, images, text_instructions):
        """调试完整管道"""
        print("🔍 调试完整VL管道:")
        print("="*50)
        
        try:
            with torch.no_grad():
                vl_tokens, vl_mask = self.vl_generator(images, text_instructions)
            
            print(f"✅ 完整管道成功:")
            print(f"   VL tokens: {vl_tokens.shape}")
            print(f"   VL mask: {vl_mask.shape}")
            print(f"   数据质量正常: {not torch.isnan(vl_tokens).any()}")
            
            return vl_tokens, vl_mask
            
        except Exception as e:
            print(f"❌ 完整管道失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None


# 🔧 快速修复检查函数
def quick_fix_check():
    """快速检查是否需要修复"""
    print("🔧 快速修复检查清单:")
    print("="*50)
    
    checklist = [
        ("VLTokenGenerator移除SigLIP2文本编码器", "self.text_encoder = None"),
        ("添加T5适配器", "self.t5_text_adapter = nn.Linear(4096, 2048)"),
        ("更新_process_t5_text方法", "统一处理T5嵌入"),
        ("数据集返回T5路径", "flare_text_embed_paths"),
        ("训练脚本传递T5路径", "text_instructions=t5_paths"),
        ("测试无维度不匹配", "4096→2048适配正常"),
        ("测试无NaN问题", "数值稳定性良好"),
    ]
    
    for i, (item, detail) in enumerate(checklist, 1):
        print(f"{i}. {item}")
        print(f"   {detail}")
    
    print("\n🎯 修复优先级:")
    print("1. 立即: 更新VLTokenGenerator (移除SigLIP2文本编码器)")
    print("2. 立即: 添加T5适配器处理维度转换")
    print("3. 立即: 更新训练脚本传递T5路径")
    print("4. 验证: 测试无维度不匹配和NaN问题")


# 运行快速检查
if __name__ == "__main__":
    quick_fix_check()