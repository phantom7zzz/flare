import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
import os

# 强制离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

class SigLIP2TextEncoder(nn.Module):
    """SigLIP2文本编码器，用于处理任务指令 - 离线版本"""
    
    def __init__(self, text_model_name="google/siglip-so400m-patch14-384", max_length=77, device="cuda", torch_dtype=torch.float16):
        super().__init__()
        self.max_length = max_length
        self.device = device
        self.torch_dtype = torch_dtype
        
        print(f"🔧 初始化SigLIP2TextEncoder (离线模式)")
        print(f"   模型路径: {text_model_name}")
        
        # 强制使用本地模型，避免网络下载
        try:
            print(f"   尝试从本地加载SigLIP模型...")
            from transformers import SiglipTextModel, SiglipTokenizer
            
            # 强制本地加载
            self.tokenizer = SiglipTokenizer.from_pretrained(
                text_model_name, 
                local_files_only=True,
                trust_remote_code=False
            )
            self.text_model = SiglipTextModel.from_pretrained(
                text_model_name,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=torch_dtype
            )
            print(f"   ✅ 成功加载本地SigLIP模型")
            
        except Exception as e:
            print(f"   ❌ 本地SigLIP加载失败: {e}")
            print(f"   🔄 尝试使用CLIP作为fallback...")
            
            try:
                from transformers import CLIPTextModel, CLIPTokenizer
                self.tokenizer = CLIPTokenizer.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    local_files_only=True
                )
                self.text_model = CLIPTextModel.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    local_files_only=True,
                    torch_dtype=torch_dtype
                )
                print(f"   ✅ 成功加载本地CLIP模型作为fallback")
                
            except Exception as e2:
                print(f"   ❌ CLIP fallback也失败: {e2}")
                print(f"   🆘 创建dummy文本编码器")
                
                # 创建dummy tokenizer和模型
                self.tokenizer = None
                self.text_model = None
                self.hidden_size = 768  # 默认隐藏层大小
                return
        
        if self.text_model is not None:
            self.text_model.eval()
            self.hidden_size = self.text_model.config.hidden_size
        
    def forward(self, text_inputs):
        """
        编码文本指令
        Args:
            text_inputs: 文本指令列表或已编码的token ids
        Returns:
            text_embeddings: (B, L, D) 文本嵌入
            text_mask: (B, L) 文本掩码
        """
        # 如果没有可用的模型，返回dummy结果
        if self.text_model is None or self.tokenizer is None:
            if isinstance(text_inputs, list):
                batch_size = len(text_inputs)
                seq_length = self.max_length
            else:
                batch_size, seq_length = text_inputs.shape[:2]
            
            # 返回零嵌入和全True掩码
            device = next(self.parameters()).device if list(self.parameters()) else torch.device('cuda')
            text_embeddings = torch.zeros(batch_size, seq_length, self.hidden_size, 
                                        device=device, dtype=self.torch_dtype)
            text_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.bool)
            
            print(f"⚠️ 返回dummy文本嵌入: {text_embeddings.shape}")
            return text_embeddings, text_mask

        device = next(self.text_model.parameters()).device

        if isinstance(text_inputs, list):
            tokens = self.tokenizer(
                text_inputs,
                max_length=self.max_length,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True
            )
            input_ids = tokens.input_ids.to(device)
            attention_mask = tokens.attention_mask.to(device)
        else:
            input_ids = text_inputs.to(device)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(device)

        # 获取文本嵌入
        with torch.no_grad():
            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            text_embeddings = text_outputs.last_hidden_state

        return text_embeddings, attention_mask.bool()


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
        # Self-Attention with residual connection
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attention(
            x, x, x, 
            key_padding_mask=mask if mask is not None else None
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
    完整的VL Token生成模块 - 离线版本
    
    流程：
    1. 未来观测图像 → SigLIP视觉编码 → patch tokens
    2. 任务指令 → SigLIP2文本编码 → 语言tokens
    3. patch tokens + 语言tokens → 多层self-attention融合 → VL tokens
    """
    
    def __init__(self, 
                 vision_model_name="google/siglip-so400m-patch14-384",
                 text_model_name="google/siglip-so400m-patch14-384",
                 hidden_size=1152,
                 num_fusion_layers=4,
                 num_heads=8,
                 dropout=0.1,
                 max_text_length=77,
                 device="cuda",
                 torch_dtype=torch.float16,
                 use_precomp_lang_embed=False):
        super().__init__()
        
        print(f"🔧 初始化VLTokenGenerator (离线模式)")
        print(f"   vision_model: {vision_model_name}")
        print(f"   text_model: {text_model_name}")
        print(f"   use_precomp_lang_embed: {use_precomp_lang_embed}")
        
        self.hidden_size = hidden_size
        self.num_fusion_layers = num_fusion_layers
        self.use_precomp_lang_embed = use_precomp_lang_embed
        
        # 1. 视觉编码器 (SigLIP) - 强制本地加载
        print(f"   🖼️ 初始化视觉编码器...")
        self.vision_encoder = SiglipVisionTower(
            vision_tower=vision_model_name, 
            args=None
        )
        print(f"   ✅ 视觉编码器初始化完成，hidden_size: {self.vision_encoder.hidden_size}")
        
        # 2. 文本编码器 (SigLIP2) - 根据是否使用预计算嵌入决定
        if use_precomp_lang_embed:
            print(f"   ⚠️ 跳过文本编码器初始化（使用预计算语言嵌入）")
            self.text_encoder = None
            # 为预计算嵌入设置默认文本维度
            text_dim = hidden_size  # 假设预计算嵌入已经是目标维度
        else:
            print(f"   📝 初始化文本编码器...")
            self.text_encoder = SigLIP2TextEncoder(
                text_model_name=text_model_name,
                max_length=max_text_length,
                device=device,
                torch_dtype=torch_dtype
            )
            text_dim = self.text_encoder.hidden_size if self.text_encoder.text_model is not None else hidden_size
            print(f"   ✅ 文本编码器初始化完成，hidden_size: {text_dim}")
        
        # 3. 特征维度对齐
        vision_dim = self.vision_encoder.hidden_size
        
        print(f"   🔧 配置维度对齐:")
        print(f"      vision_dim: {vision_dim}")
        print(f"      text_dim: {text_dim}")
        print(f"      target_hidden_size: {hidden_size}")
        
        # 如果维度不匹配，添加投影层
        if vision_dim != hidden_size:
            self.vision_proj = nn.Linear(vision_dim, hidden_size)
            print(f"      创建视觉投影层: {vision_dim} -> {hidden_size}")
        else:
            self.vision_proj = nn.Identity()
            print(f"      视觉维度匹配，无需投影")
            
        if text_dim != hidden_size:
            self.text_proj = nn.Linear(text_dim, hidden_size)
            print(f"      创建文本投影层: {text_dim} -> {hidden_size}")
        else:
            self.text_proj = nn.Identity()
            print(f"      文本维度匹配，无需投影")
        
        # 4. 位置编码
        self.vision_pos_embed = nn.Parameter(
            torch.zeros(1, self.vision_encoder.num_patches, hidden_size)
        )
        self.text_pos_embed = nn.Parameter(
            torch.zeros(1, max_text_length, hidden_size)
        )
        
        # 5. 模态类型嵌入
        self.vision_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.text_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        # 6. 多层VL融合块
        self.fusion_layers = nn.ModuleList([
            VLFusionBlock(hidden_size, num_heads, dropout)
            for _ in range(num_fusion_layers)
        ])
        
        # 7. 输出层归一化
        self.output_norm = nn.LayerNorm(hidden_size)
        
        # 初始化参数
        self._initialize_weights()
        
        print(f"   ✅ VLTokenGenerator初始化完成")
        
    def _initialize_weights(self):
        """初始化参数"""
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
                nn.init.constant_(module.bias, 0)
    
    def forward(self, future_images, text_instructions, image_mask=None, text_mask=None):
        """
        生成VL tokens
        
        Args:
            future_images: (B, C, H, W) 未来观测图像
            text_instructions: (B, L) 文本指令token ids 或文本列表 或预计算嵌入
            image_mask: (B, N_patches) 图像掩码（可选）
            text_mask: (B, L) 文本掩码（可选）
            
        Returns:
            vl_tokens: (B, N_total, D) 融合后的VL tokens
            vl_mask: (B, N_total) VL tokens的掩码
        """
        # 确保所有输入数据在同一设备上
        device = next(self.parameters()).device
        
        if future_images is not None:
            future_images = future_images.to(device)
            batch_size = future_images.shape[0]
            
            # 处理3D图像输入
            if len(future_images.shape) == 3:
                future_images = future_images.unsqueeze(0)
        else:
            # 如果没有图像输入，从文本获取batch_size
            if isinstance(text_instructions, torch.Tensor):
                batch_size = text_instructions.shape[0]
            elif isinstance(text_instructions, list):
                batch_size = len(text_instructions)
            else:
                batch_size = 1
        
        # 1. 视觉编码
        if future_images is not None:
            vision_features = self.vision_encoder(future_images)  # (B, N_patches, D_vision)
            vision_features = self.vision_proj(vision_features)   # (B, N_patches, hidden_size)
            
            # 添加视觉位置编码和模态类型嵌入
            vision_features = vision_features + self.vision_pos_embed[:, :vision_features.shape[1], :]
            vision_features = vision_features + self.vision_type_embed
        else:
            # 如果没有图像，创建空的视觉特征
            vision_features = torch.empty(batch_size, 0, self.hidden_size, device=device)
        
        # 2. 文本编码
        if self.use_precomp_lang_embed:
            # 使用预计算的语言嵌入
            if isinstance(text_instructions, torch.Tensor):
                text_features = text_instructions.to(device)
                if text_mask is None:
                    text_mask = torch.ones(text_features.shape[:2], dtype=torch.bool, device=device)
            else:
                # 如果没有预计算嵌入，创建零嵌入
                seq_len = 77  # 默认长度
                text_features = torch.zeros(batch_size, seq_len, self.hidden_size, device=device)
                text_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
                
            computed_text_mask = text_mask
        else:
            # 使用文本编码器
            if self.text_encoder is not None:
                text_features, computed_text_mask = self.text_encoder(text_instructions)  # (B, L, D_text)
                text_features = text_features.to(device)
                computed_text_mask = computed_text_mask.to(device)
            else:
                # 如果文本编码器不可用，创建dummy特征
                seq_len = 77
                text_features = torch.zeros(batch_size, seq_len, self.hidden_size, device=device)
                computed_text_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        text_features = self.text_proj(text_features)  # (B, L, hidden_size)
        
        # 使用计算得到的掩码或提供的掩码
        if text_mask is None:
            text_mask = computed_text_mask
        
        # 添加文本位置编码和模态类型嵌入
        if text_features.shape[1] > 0:
            text_features = text_features + self.text_pos_embed[:, :text_features.shape[1], :]
            text_features = text_features + self.text_type_embed
        
        # 3. 拼接视觉和文本特征
        if vision_features.shape[1] > 0 and text_features.shape[1] > 0:
            vl_features = torch.cat([vision_features, text_features], dim=1)
        elif vision_features.shape[1] > 0:
            vl_features = vision_features
        elif text_features.shape[1] > 0:
            vl_features = text_features
        else:
            # 如果都为空，创建一个dummy特征
            vl_features = torch.zeros(batch_size, 1, self.hidden_size, device=device)
        
        # 4. 创建联合掩码
        if image_mask is None and vision_features.shape[1] > 0:
            image_mask = torch.ones(batch_size, vision_features.shape[1], 
                                  dtype=torch.bool, device=device)
        elif vision_features.shape[1] == 0:
            image_mask = torch.empty(batch_size, 0, dtype=torch.bool, device=device)
        
        if vision_features.shape[1] > 0 and text_features.shape[1] > 0:
            vl_mask = torch.cat([image_mask, text_mask], dim=1)
        elif vision_features.shape[1] > 0:
            vl_mask = image_mask
        elif text_features.shape[1] > 0:
            vl_mask = text_mask
        else:
            vl_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        
        # 5. 通过多层融合块
        for fusion_layer in self.fusion_layers:
            vl_features = fusion_layer(vl_features, mask=~vl_mask)  # 注意：mask需要反转
        
        # 6. 输出归一化
        vl_tokens = self.output_norm(vl_features)
        
        return vl_tokens, vl_mask
    
    def get_vision_features(self, images):
        """单独获取视觉特征（用于调试）"""
        return self.vision_encoder(images)
    
    def get_text_features(self, text_instructions):
        """单独获取文本特征（用于调试）"""
        if self.text_encoder is not None:
            return self.text_encoder(text_instructions)
        else:
            print("⚠️ 文本编码器不可用")
            return None, None


# 使用示例和测试函数
def test_vl_token_generator():
    """测试VL Token生成器"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建VL Token生成器
    vl_generator = VLTokenGenerator(
        vision_model_name="../weights/RDT/siglip-so400m-patch14-384",  # 使用本地路径
        text_model_name="../weights/RDT/siglip-so400m-patch14-384",    # 使用本地路径
        hidden_size=1152,
        num_fusion_layers=4,
        use_precomp_lang_embed=True  # 使用预计算嵌入
    ).to(device)
    
    # 创建测试数据
    batch_size = 2
    future_images = torch.randn(batch_size, 3, 384, 384).to(device)
    
    # 使用预计算嵌入而不是文本
    text_embeddings = torch.randn(batch_size, 77, 1152).to(device)  # 预计算嵌入
    
    # 前向传播
    with torch.no_grad():
        vl_tokens, vl_mask = vl_generator(future_images, text_embeddings)
    
    print(f"VL tokens shape: {vl_tokens.shape}")
    print(f"VL mask shape: {vl_mask.shape}")
    print(f"Valid tokens per sample: {vl_mask.sum(dim=1)}")
    
    return vl_tokens, vl_mask


if __name__ == "__main__":
    # 运行测试
    test_vl_token_generator()