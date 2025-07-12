
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class QFormerLayer(nn.Module):
    """
    Q-Former层：Self-Attention + Cross-Attention + FeedForward
    参考BLIP-2的Q-Former设计
    """
    
    def __init__(self, hidden_size, num_heads=8, dropout=0.1, cross_attention_freq=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.cross_attention_freq = cross_attention_freq
        
        # 1. Self-Attention for query tokens
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 2. Cross-Attention: queries attend to VL tokens
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 3. Layer Normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        
        # 4. FeedForward Network
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        # 5. Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query_tokens, vl_tokens, vl_mask=None, layer_idx=0):
        """
        Q-Former层前向传播
        
        Args:
            query_tokens: (B, num_queries, D) 查询tokens
            vl_tokens: (B, N_vl, D) VL tokens (key & value)
            vl_mask: (B, N_vl) VL tokens掩码
            layer_idx: 当前层索引，用于决定是否使用cross-attention
            
        Returns:
            query_tokens: (B, num_queries, D) 更新后的查询tokens
            attn_weights: 注意力权重（可选，用于可视化）
        """
        batch_size, num_queries, hidden_size = query_tokens.shape
        
        # 1. Self-Attention among query tokens
        residual = query_tokens
        query_tokens = self.norm1(query_tokens)
        
        self_attn_out, self_attn_weights = self.self_attention(
            query_tokens, query_tokens, query_tokens
        )
        query_tokens = residual + self.dropout(self_attn_out)
        
        # 2. Cross-Attention: queries attend to VL tokens
        # 只在特定层使用cross-attention来控制计算复杂度
        if layer_idx % self.cross_attention_freq == 0:
            residual = query_tokens
            query_tokens = self.norm2(query_tokens)
            
            # Convert mask for cross-attention (mask out invalid VL tokens)
            key_padding_mask = ~vl_mask if vl_mask is not None else None
            
            cross_attn_out, cross_attn_weights = self.cross_attention(
                query_tokens, vl_tokens, vl_tokens,
                key_padding_mask=key_padding_mask
            )
            query_tokens = residual + self.dropout(cross_attn_out)
        else:
            cross_attn_weights = None
        
        # 3. FeedForward
        residual = query_tokens
        query_tokens = self.norm3(query_tokens)
        ff_out = self.feedforward(query_tokens)
        query_tokens = residual + ff_out
        
        return query_tokens, {
            'self_attn_weights': self_attn_weights,
            'cross_attn_weights': cross_attn_weights
        }


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, hidden_size, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                           (-math.log(10000.0) / hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: (B, seq_len, hidden_size)
        """
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class QFormerTargetGenerator(nn.Module):
    """
    Q-Former风格的目标Token生成器
    
    架构：
    32个可学习queries → 多层Q-Former → 目标tokens
    每层包含：Self-Attention + Cross-Attention + FeedForward
    """
    
    def __init__(self,
                 hidden_size=1152,
                 num_query_tokens=32,
                 num_layers=6,
                 num_heads=8,
                 dropout=0.1,
                 cross_attention_freq=2,
                 use_positional_encoding=True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_query_tokens = num_query_tokens
        self.num_layers = num_layers
        self.use_positional_encoding = use_positional_encoding
        
        # 1. 可学习的query tokens (关键组件)
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, hidden_size) * 0.02
        )
        
        # 2. 位置编码（可选）
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(hidden_size)
        
        # 3. 多层Q-Former
        self.qformer_layers = nn.ModuleList([
            QFormerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                cross_attention_freq=cross_attention_freq
            ) for _ in range(num_layers)
        ])
        
        # 4. 输出投影层
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # 5. 初始化参数
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型参数"""
        # Query tokens初始化
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        
        # 线性层初始化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, vl_tokens, vl_mask=None, return_attention_weights=False):
        """
        生成目标tokens
        
        Args:
            vl_tokens: (B, N_vl, D) VL tokens
            vl_mask: (B, N_vl) VL tokens的掩码
            return_attention_weights: 是否返回注意力权重
            
        Returns:
            target_tokens: (B, num_query_tokens, D) 目标tokens
            attention_weights: 注意力权重字典（可选）
        """
        batch_size = vl_tokens.shape[0]
        device = vl_tokens.device
        
        # 1. 扩展query tokens到batch size
        query_tokens = self.query_tokens.expand(batch_size, -1, -1).clone()
        
        # 2. 添加位置编码（可选）
        if self.use_positional_encoding:
            query_tokens = self.pos_encoding(query_tokens)
        
        # 3. 存储注意力权重
        all_attention_weights = [] if return_attention_weights else None
        
        # 4. 通过多层Q-Former
        for layer_idx, qformer_layer in enumerate(self.qformer_layers):
            query_tokens, attn_weights = qformer_layer(
                query_tokens=query_tokens,
                vl_tokens=vl_tokens,
                vl_mask=vl_mask,
                layer_idx=layer_idx
            )
            
            if return_attention_weights:
                all_attention_weights.append(attn_weights)
        
        # 5. 输出投影
        target_tokens = self.output_projection(query_tokens)
        
        if return_attention_weights:
            return target_tokens, all_attention_weights
        else:
            return target_tokens
    
    def get_query_tokens(self):
        """获取原始query tokens（用于可视化）"""
        return self.query_tokens.clone()
    
    def freeze_query_tokens(self):
        """冻结query tokens（用于特定训练策略）"""
        self.query_tokens.requires_grad = False
    
    def unfreeze_query_tokens(self):
        """解冻query tokens"""
        self.query_tokens.requires_grad = True


class AdaptiveQFormerTargetGenerator(QFormerTargetGenerator):
    """
    自适应Q-Former目标生成器
    支持动态调整query tokens数量和注意力机制
    """
    
    def __init__(self, *args, **kwargs):
        # 额外参数
        self.adaptive_query_selection = kwargs.pop('adaptive_query_selection', False)
        self.min_query_tokens = kwargs.pop('min_query_tokens', 16)
        self.max_query_tokens = kwargs.pop('max_query_tokens', 64)
        
        super().__init__(*args, **kwargs)
        
        # 自适应查询选择器
        if self.adaptive_query_selection:
            self.query_selector = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 2, self.num_query_tokens),
                nn.Sigmoid()
            )
    
    def forward(self, vl_tokens, vl_mask=None, return_attention_weights=False):
        """
        自适应前向传播
        """
        if self.adaptive_query_selection and self.training:
            # 计算VL tokens的全局表示
            vl_global = vl_tokens.mean(dim=1)  # (B, D)
            
            # 生成query tokens的重要性分数
            query_importance = self.query_selector(vl_global)  # (B, num_query_tokens)
            
            # 根据重要性分数选择活跃的query tokens
            batch_size = vl_tokens.shape[0]
            active_queries = []
            
            for b in range(batch_size):
                importance_scores = query_importance[b]
                # 选择top-k个重要的query tokens
                k = torch.randint(self.min_query_tokens, self.max_query_tokens + 1, (1,)).item()
                _, top_indices = torch.topk(importance_scores, k)
                active_queries.append(top_indices)
            
            # 这里为了简化，我们还是使用全部query tokens
            # 在实际实现中，可以根据active_queries动态调整
        
        return super().forward(vl_tokens, vl_mask, return_attention_weights)


# 集成到FLARE模型的示例
class FLAREWithQFormer(nn.Module):
    """
    集成Q-Former的FLARE模型示例
    """
    
    def __init__(self, hidden_size=1152, num_query_tokens=32):
        super().__init__()
        
        # 导入已实现的VL Token生成器
        from models.multimodal_encoder.vl_token_generator import VLTokenGenerator
        
        # VL Token生成器
        self.vl_token_generator = VLTokenGenerator(hidden_size=hidden_size)
        
        # Q-Former目标生成器
        self.target_generator = QFormerTargetGenerator(
            hidden_size=hidden_size,
            num_query_tokens=num_query_tokens,
            num_layers=6,
            num_heads=8
        )
    
    def generate_target_tokens(self, future_images, text_instructions):
        """
        生成目标tokens的完整流程
        
        Args:
            future_images: (B, C, H, W) 未来观测图像
            text_instructions: 文本指令
            
        Returns:
            target_tokens: (B, num_query_tokens, D) 目标tokens
        """
        # 1. 生成VL tokens
        vl_tokens, vl_mask = self.vl_token_generator(future_images, text_instructions)
        
        # 2. 通过Q-Former生成目标tokens
        target_tokens = self.target_generator(vl_tokens, vl_mask)
        
        return target_tokens


# 测试函数
def test_qformer_target_generator():
    """测试Q-Former目标生成器"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建测试数据
    batch_size = 2
    num_vl_tokens = 100  # VL tokens数量
    hidden_size = 1152
    num_query_tokens = 32
    
    # 模拟VL tokens
    vl_tokens = torch.randn(batch_size, num_vl_tokens, hidden_size).to(device)
    vl_mask = torch.ones(batch_size, num_vl_tokens, dtype=torch.bool).to(device)
    
    # 创建Q-Former目标生成器
    target_generator = QFormerTargetGenerator(
        hidden_size=hidden_size,
        num_query_tokens=num_query_tokens,
        num_layers=6,
        num_heads=8
    ).to(device)
    
    # 前向传播
    with torch.no_grad():
        target_tokens, attention_weights = target_generator(
            vl_tokens, vl_mask, return_attention_weights=True
        )
    
    print(f"Input VL tokens shape: {vl_tokens.shape}")
    print(f"Output target tokens shape: {target_tokens.shape}")
    print(f"Query tokens shape: {target_generator.get_query_tokens().shape}")
    print(f"Number of Q-Former layers: {len(attention_weights)}")
    
    return target_tokens, attention_weights


if __name__ == "__main__":
    test_qformer_target_generator()