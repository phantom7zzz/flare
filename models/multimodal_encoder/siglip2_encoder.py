import torch
import torch.nn as nn
from transformers import AutoConfig, AutoImageProcessor, AutoModel


class SigLIP2VisionTower(nn.Module):
    """SigLIP2视觉编码器 - large-patch16-256版本"""

    def __init__(self, vision_tower, args, delay_load=False, image_size=256):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.image_size = image_size  # 🔧 存储图像尺寸
        print(f"🔧 初始化SigLIP2VisionTower（未来观测专用）")
        print(f"   模型路径: {self.vision_tower_name}")
        print(f"   图像尺寸: {self.image_size}")
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.vision_tower_name} already loaded, skipping.')
            return

        print(f"🔄 加载SigLIP2模型（未来观测）: {self.vision_tower_name}")
        
        try:
            from transformers import SiglipImageProcessor, SiglipVisionModel
            
            self.image_processor = SiglipImageProcessor.from_pretrained(
                self.vision_tower_name,
                local_files_only=True
            )
            
            # 🔧 确保processor使用正确的图像尺寸
            if hasattr(self.image_processor, 'size'):
                if isinstance(self.image_processor.size, dict):
                    self.image_processor.size['height'] = self.image_size
                    self.image_processor.size['width'] = self.image_size
                else:
                    self.image_processor.size = self.image_size
            
            self.vision_tower = SiglipVisionModel.from_pretrained(
                self.vision_tower_name, 
                device_map=device_map,
                local_files_only=True
            )
            print("✅ 使用SigLIP类加载SigLIP2成功（未来观测）")
            
        except Exception as e:
            print(f"SigLIP类加载失败: {e}")
            print("🔄 尝试使用Auto类...")
            
            # 备选方案
            from transformers import AutoImageProcessor, AutoModel
            
            self.image_processor = AutoImageProcessor.from_pretrained(
                self.vision_tower_name,
                local_files_only=True
            )
            self.vision_tower = AutoModel.from_pretrained(
                self.vision_tower_name, 
                device_map=device_map,
                local_files_only=True
            )
            print("✅ 使用Auto类加载成功（未来观测）")
            
        self.vision_tower.eval()
        self.is_loaded = True
        
        print(f"✅ SigLIP2加载完成（未来观测）:")
        print(f"   Hidden size: {self.vision_tower.config.hidden_size}")
        print(f"   Config image size: {self.vision_tower.config.image_size}")
        print(f"   Processor image size: {self.image_size}")
        print(f"   Patch size: {self.vision_tower.config.patch_size}")

    def feature_select(self, image_forward_outs):
        if self.select_feature == 'patch':
            # SigLIP2的patch特征提取
            if hasattr(image_forward_outs, 'last_hidden_state'):
                image_features = image_forward_outs.last_hidden_state
            else:
                image_features = image_forward_outs.pooler_output
        elif self.select_feature == 'cls_patch':
            if hasattr(image_forward_outs, 'pooler_output'):
                image_features = image_forward_outs.pooler_output
            else:
                image_features = image_forward_outs.last_hidden_state
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    # 其余方法保持与原SiglipVisionTower相同的结构
    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size)**2