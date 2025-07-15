

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower


class SigLIP2TextEncoder(nn.Module):
    """SigLIP2文本编码器，用于处理任务指令"""
    
    def __init__(self, text_model_name="google/siglip-so400m-patch14-384", max_length=77):
        super().__init__()
        self.max_length = max_length
        
        # 加载SigLIP2的文本编码器
        try:
            from transformers import SiglipTextModel, SiglipTokenizer
            self.tokenizer = SiglipTokenizer.from_pretrained(text_model_name)
            self.text_model = SiglipTextModel.from_pretrained(text_model_name)
        except ImportError:
            # 如果SigLIP2不可用，使用CLIP文本编码器作为替代
            print("Warning: SigLIP2 not available, using CLIP text encoder as fallback")
            from transformers import CLIPTextModel, CLIPTokenizer
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        
        self.text_model.eval()
        
    def forward(self, text_inputs):
        """
        编码文本指令
        Args:
            text_inputs: 文本指令列表或已编码的token ids
        Returns:
            text_embeddings: (B, L, D) 文本嵌入
            text_mask: (B, L) 文本掩码
        """
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
    完整的VL Token生成模块
    
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
                 max_text_length=77):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_fusion_layers = num_fusion_layers
        
        # 1. 视觉编码器 (SigLIP)
        self.vision_encoder = SiglipVisionTower(
            vision_tower=vision_model_name, 
            args=None
        )
        
        # 2. 文本编码器 (SigLIP2)
        self.text_encoder = SigLIP2TextEncoder(
            text_model_name=text_model_name,
            max_length=max_text_length
        )
        
        # 3. 特征维度对齐
        vision_dim = self.vision_encoder.hidden_size
        text_dim = self.text_encoder.text_model.config.hidden_size
        
        # 如果维度不匹配，添加投影层
        if vision_dim != hidden_size:
            self.vision_proj = nn.Linear(vision_dim, hidden_size)
        else:
            self.vision_proj = nn.Identity()
            
        if text_dim != hidden_size:
            self.text_proj = nn.Linear(text_dim, hidden_size)
        else:
            self.text_proj = nn.Identity()
        
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
            text_instructions: (B, L) 文本指令token ids 或文本列表
            image_mask: (B, N_patches) 图像掩码（可选）
            text_mask: (B, L) 文本掩码（可选）
            
        Returns:
            vl_tokens: (B, N_total, D) 融合后的VL tokens
            vl_mask: (B, N_total) VL tokens的掩码
        """
        # 🔧 确保所有输入数据在同一设备上
        device = next(self.parameters()).device  # 获取模型所在设备
        
        if future_images is not None:
            future_images = future_images.to(device)
        
        if text_instructions is not None:
            if isinstance(text_instructions, torch.Tensor):
                text_instructions = text_instructions.to(device)
            elif isinstance(text_instructions, list):
                # 如果是字符串列表，保持不变
                pass
        batch_size = future_images.shape[0]
        device = future_images.device
        if future_images is not None and len(future_images.shape) == 3:
            future_images = future_images.unsqueeze(0)
        # 1. 视觉编码
        vision_features = self.vision_encoder(future_images)  # (B, N_patches, D_vision)
        vision_features = self.vision_proj(vision_features)   # (B, N_patches, hidden_size)
        
        # 添加视觉位置编码和模态类型嵌入
        vision_features = vision_features + self.vision_pos_embed[:, :vision_features.shape[1], :]
        vision_features = vision_features + self.vision_type_embed
        
        # 2. 文本编码
        text_features, computed_text_mask = self.text_encoder(text_instructions)  # (B, L, D_text)
        text_features = self.text_proj(text_features)  # (B, L, hidden_size)
        
        # 使用计算得到的掩码或提供的掩码
        if text_mask is None:
            text_mask = computed_text_mask
        
        # 添加文本位置编码和模态类型嵌入
        text_features = text_features + self.text_pos_embed[:, :text_features.shape[1], :]
        text_features = text_features + self.text_type_embed
        
        # 3. 拼接视觉和文本特征
        vl_features = torch.cat([vision_features, text_features], dim=1)  # (B, N_patches+L, hidden_size)
        
        # 4. 创建联合掩码
        if image_mask is None:
            image_mask = torch.ones(batch_size, vision_features.shape[1], 
                                  dtype=torch.bool, device=device)
        
        vl_mask = torch.cat([image_mask, text_mask], dim=1)  # (B, N_patches+L)
        
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
        return self.text_encoder(text_instructions)


# 使用示例和测试函数
def test_vl_token_generator():
    """测试VL Token生成器"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建VL Token生成器
    vl_generator = VLTokenGenerator(
        vision_model_name="google/siglip-so400m-patch14-384",
        text_model_name="google/siglip-so400m-patch14-384",
        hidden_size=1152,
        num_fusion_layers=4
    ).to(device)
    
    # 创建测试数据
    batch_size = 2
    future_images = torch.randn(batch_size, 3, 384, 384).to(device)
    text_instructions = [
        "Pick up the red cup and place it on the table",
        "Move the robot arm to the left side"
    ]
    
    # 前向传播
    with torch.no_grad():
        vl_tokens, vl_mask = vl_generator(future_images, text_instructions)
    
    print(f"VL tokens shape: {vl_tokens.shape}")
    print(f"VL mask shape: {vl_mask.shape}")
    print(f"Valid tokens per sample: {vl_mask.sum(dim=1)}")
    
    return vl_tokens, vl_mask


if __name__ == "__main__":
    # 运行测试
    test_vl_token_generator()