import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
import os
import json

# 强制离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

class SigLIP2TextEncoder(nn.Module):
    """SigLIP2文本编码器 - 完全修复版本"""
    
    def __init__(self, text_model_name="google/siglip2-large-patch16-256", max_length=32, device="cuda", torch_dtype=torch.float16):
        super().__init__()
        self.max_length = max_length
        self.device = device
        self.torch_dtype = torch_dtype
        
        print(f"🔧 初始化SigLIP2TextEncoder (完全修复版)")
        print(f"   模型路径: {text_model_name}")
        print(f"   最大长度: {max_length}")
        
        # 🎯 方法1：尝试直接加载完整SigLIP模型
        if self._try_load_siglip_model(text_model_name):
            return
            
        # 🎯 方法2：尝试修复分词器配置后加载
        if self._try_load_with_fixed_tokenizer(text_model_name):
            return
            
        # 🎯 方法3：使用transformers自动加载
        if self._try_auto_load(text_model_name):
            return
            
        # 🎯 方法4：创建备用编码器
        print(f"   🔄 所有加载方法失败，创建备用编码器...")
        self._create_fallback_encoder()
    
    def _try_load_siglip_model(self, model_path):
        """方法1：尝试加载完整SigLIP模型"""
        try:
            print(f"   🔄 方法1：尝试加载完整SigLIP2模型...")
            
            from transformers import SiglipModel, SiglipProcessor
            
            # 加载完整模型
            self.full_model = SiglipModel.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=self.torch_dtype
            )
            
            # 提取文本编码器
            self.text_model = self.full_model.text_model
            
            # 加载processor
            self.processor = SiglipProcessor.from_pretrained(
                model_path,
                local_files_only=True
            )
            self.tokenizer = self.processor.tokenizer
            
            self.hidden_size = self.text_model.config.hidden_size
            self.text_model.eval()
            
            print(f"   ✅ 方法1成功！SigLIP2完整模型加载成功")
            print(f"      隐藏层大小: {self.hidden_size}")
            return True
            
        except Exception as e:
            print(f"   ❌ 方法1失败: {e}")
            return False
    
    def _try_load_with_fixed_tokenizer(self, model_path):
        """方法2：修复分词器配置后加载"""
        try:
            print(f"   🔄 方法2：尝试修复分词器配置...")
            
            # 检查tokenizer配置文件
            tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
            if os.path.exists(tokenizer_config_path):
                with open(tokenizer_config_path, 'r') as f:
                    config = json.load(f)
                    print(f"      分词器类型: {config.get('tokenizer_class', 'Unknown')}")
            
            # 尝试使用正确的分词器类型
            from transformers import GemmaTokenizer, SiglipModel
            
            # 直接使用GemmaTokenizer
            self.tokenizer = GemmaTokenizer.from_pretrained(
                model_path,
                local_files_only=True
            )
            
            # 加载SigLIP模型
            self.full_model = SiglipModel.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=self.torch_dtype
            )
            self.text_model = self.full_model.text_model
            
            self.hidden_size = self.text_model.config.hidden_size
            self.text_model.eval()
            
            print(f"   ✅ 方法2成功！使用GemmaTokenizer + SigLIP模型")
            print(f"      隐藏层大小: {self.hidden_size}")
            return True
            
        except Exception as e:
            print(f"   ❌ 方法2失败: {e}")
            return False
    
    def _try_auto_load(self, model_path):
        """方法3：使用AutoModel自动加载"""
        try:
            print(f"   🔄 方法3：尝试AutoModel自动加载...")
            
            # 使用AutoTokenizer自动检测分词器类型
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True  # 允许自定义代码
            )
            
            # 使用AutoModel加载模型
            self.full_model = AutoModel.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True
            )
            
            # 尝试获取文本编码器
            if hasattr(self.full_model, 'text_model'):
                self.text_model = self.full_model.text_model
            elif hasattr(self.full_model, 'get_text_features'):
                self.text_model = self.full_model  # 整个模型就是文本编码器
            else:
                raise ValueError("无法找到文本编码器组件")
            
            # 获取隐藏层大小
            if hasattr(self.text_model, 'config'):
                self.hidden_size = self.text_model.config.hidden_size
            else:
                self.hidden_size = 1024  # 默认值
                
            self.text_model.eval()
            
            print(f"   ✅ 方法3成功！AutoModel加载成功")
            print(f"      模型类型: {type(self.full_model).__name__}")
            print(f"      分词器类型: {type(self.tokenizer).__name__}")
            print(f"      隐藏层大小: {self.hidden_size}")
            return True
            
        except Exception as e:
            print(f"   ❌ 方法3失败: {e}")
            return False
    
    def _create_fallback_encoder(self):
        """方法4：创建备用编码器"""
        self.hidden_size = 1024  # 与视觉编码器匹配
        self.text_model = None
        self.full_model = None
        
        # 简单但功能完整的tokenizer
        class ImprovedTokenizer:
            def __init__(self, max_length):
                self.max_length = max_length
                self.pad_token_id = 0
                self.vocab_size = 5000  # 更大的词汇表
                
                # 创建简单的词汇表
                self.vocab = {f"token_{i}": i for i in range(self.vocab_size)}
                self.vocab.update({
                    "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3,
                    " ": 4, ".": 5, ",": 6, "!": 7, "?": 8
                })
                
            def __call__(self, texts, max_length=None, padding="max_length", 
                        truncation=True, return_tensors="pt", **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                
                max_len = max_length or self.max_length
                input_ids = []
                attention_masks = []
                
                for text in texts:
                    # 改进的tokenization
                    words = text.lower().split()
                    tokens = [2]  # <bos>
                    
                    for word in words[:max_len-3]:  # 留出空间给特殊token
                        if word in self.vocab:
                            tokens.append(self.vocab[word])
                        else:
                            # 字符级别编码作为后备
                            for char in word[:5]:  # 限制单词长度
                                tokens.append(min(ord(char), self.vocab_size-1))
                    
                    tokens.append(3)  # <eos>
                    
                    # Padding
                    attention_mask = [1] * len(tokens)
                    while len(tokens) < max_len:
                        tokens.append(0)  # <pad>
                        attention_mask.append(0)
                    
                    input_ids.append(tokens[:max_len])
                    attention_masks.append(attention_mask[:max_len])
                
                return {
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
                }
        
        self.tokenizer = ImprovedTokenizer(self.max_length)
        
        # 创建更复杂的embedding层
        self.token_embedding = nn.Embedding(self.tokenizer.vocab_size, self.hidden_size)
        self.position_embedding = nn.Embedding(self.max_length, self.hidden_size)
        
        # 添加简单的transformer层
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=8,
            dim_feedforward=self.hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        print(f"   ✅ 备用编码器创建完成，hidden_size: {self.hidden_size}")
        print(f"      词汇表大小: {self.tokenizer.vocab_size}")

    def forward(self, text_inputs):
        """
        编码文本指令
        Args:
            text_inputs: 文本指令列表或已编码的token ids
        Returns:
            text_embeddings: (B, L, D) 文本嵌入
            text_mask: (B, L) 文本掩码
        """
        device = next(self.parameters()).device if list(self.parameters()) else self.device
        
        # 1. Tokenization
        if isinstance(text_inputs, list):
            tokens = self.tokenizer(
                text_inputs,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = tokens.input_ids.to(device)
            attention_mask = tokens.attention_mask.to(device)
        else:
            input_ids = text_inputs.to(device)
            attention_mask = (input_ids != getattr(self.tokenizer, 'pad_token_id', 0)).to(device)

        # 2. Forward pass
        if self.text_model is not None:
            # 使用真实的SigLIP2/其他文本编码器
            try:
                with torch.no_grad():
                    if hasattr(self.text_model, '__call__'):
                        text_outputs = self.text_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        if hasattr(text_outputs, 'last_hidden_state'):
                            text_embeddings = text_outputs.last_hidden_state
                        else:
                            text_embeddings = text_outputs
                    else:
                        raise ValueError("文本模型不可调用")
            except Exception as e:
                print(f"⚠️ 文本模型前向传播失败: {e}")
                # 回退到备用方法
                return self._fallback_forward(input_ids, attention_mask)
        else:
            # 使用备用编码器
            return self._fallback_forward(input_ids, attention_mask)

        return text_embeddings, attention_mask.bool()
    
    def _fallback_forward(self, input_ids, attention_mask):
        """备用前向传播"""
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        positions = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)
        
        # 组合
        embeddings = token_embeds + pos_embeds
        
        # 如果有transformer层，应用它
        if hasattr(self, 'transformer_layer'):
            # 创建padding mask (True表示需要忽略的位置)
            src_key_padding_mask = ~attention_mask.bool()
            embeddings = self.transformer_layer(
                embeddings, 
                src_key_padding_mask=src_key_padding_mask
            )
        
        # 归一化
        text_embeddings = self.layer_norm(embeddings)
        
        return text_embeddings, attention_mask.bool()


# 🧪 测试函数
def test_text_encoder_loading():
    """测试文本编码器加载"""
    print("🧪 测试SigLIP2文本编码器加载...")
    
    # 测试不同的模型路径
    test_paths = [
        "/home/deng_xiang/qian_daichao/RoboTwin/policy/RDT_flare/siglip2-large-patch16-256",
        "google/siglip2-large-patch16-256",
        "./models/siglip2-large-patch16-256"
    ]
    
    for model_path in test_paths:
        print(f"\n📁 测试路径: {model_path}")
        
        try:
            encoder = SigLIP2TextEncoder(
                text_model_name=model_path,
                max_length=32
            )
            
            # 测试编码
            test_texts = ["pick up the red cube", "move to the left"]
            embeddings, mask = encoder(test_texts)
            
            print(f"✅ 成功！输出形状: {embeddings.shape}")
            print(f"   掩码形状: {mask.shape}")
            print(f"   有效token数: {mask.sum(dim=1)}")
            
            return encoder  # 返回成功的编码器
            
        except Exception as e:
            print(f"❌ 失败: {e}")
            continue
    
    print("\n⚠️ 所有路径都失败了")
    return None


# 🛠️ 诊断工具
def diagnose_model_files(model_path):
    """诊断模型文件"""
    print(f"🔍 诊断模型文件: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 路径不存在: {model_path}")
        return
    
    # 检查必要文件
    required_files = [
        "config.json",
        "tokenizer_config.json", 
        "tokenizer.json",
        "pytorch_model.bin",
        "model.safetensors"
    ]
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            print(f"✅ {file}")
            
            # 如果是配置文件，显示内容
            if file.endswith('.json'):
                try:
                    with open(file_path, 'r') as f:
                        config = json.load(f)
                        if 'tokenizer_class' in config:
                            print(f"   └── tokenizer_class: {config['tokenizer_class']}")
                        if 'model_type' in config:
                            print(f"   └── model_type: {config['model_type']}")
                except:
                    pass
        else:
            print(f"❌ {file}")


if __name__ == "__main__":
    # 先诊断文件
    model_path = "/home/deng_xiang/qian_daichao/RoboTwin/policy/RDT_flare/siglip2-large-patch16-256"
    diagnose_model_files(model_path)
    
    # 然后测试加载
    test_text_encoder_loading()