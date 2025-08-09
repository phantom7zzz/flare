import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from models.hub_mixin import CompatiblePyTorchModelHubMixin
from models.rdt.model import RDTWithFLARE


class RDTRunnerWithFLARE(nn.Module, CompatiblePyTorchModelHubMixin):
    """
    完整集成的FLARE增强RDT Runner
    """

    def __init__(self,
                 *,
                 action_dim,
                 pred_horizon,
                 config,
                 lang_token_dim,
                 img_token_dim,
                 state_token_dim,
                 max_lang_cond_len=1024,
                 img_cond_len,
                 lang_pos_embed_config=None,
                 img_pos_embed_config=None,
                 dtype=torch.bfloat16,
                 # FLARE参数
                 num_future_tokens=32,
                 activation_layer=21,
                 alignment_loss_weight=0.2,
                 num_vl_fusion_layers=4,
                 num_qformer_layers=2,
                 alignment_temperature=0.07,
                # 🔧 区分两个编码器的参数
                 future_vision_model_name=None,  # 未来观测编码器（SigLIP2-256）
                 future_text_model_name=None,    # 未来观测文本编码器
                 current_vision_image_size=384,  # 当前图像尺寸
                 future_vision_image_size=256,   # 未来图像尺寸
                 enable_flare=True):
        super().__init__()
        # 🔧 路径和配置处理
        self.future_vision_path = future_vision_model_name or "/home/deng_xiang/qian_daichao/RoboTwin/policy/RDT_flare/siglip2-large-patch16-256"
        self.future_text_path = future_text_model_name or self.future_vision_path
        self.current_vision_image_size = current_vision_image_size
        self.future_vision_image_size = future_vision_image_size
        self.max_lang_cond_len = 1024
        print(f"🔧 RDTRunnerWithFLARE 双编码器配置:")
        print(f"   未来观测视觉模型: {self.future_vision_path}")
        print(f"   未来观测文本模型: {self.future_text_path}")
        print(f"   当前图像尺寸: {self.current_vision_image_size}")
        print(f"   未来图像尺寸: {self.future_vision_image_size}")
        print(f"   文本最大长度: {max_lang_cond_len}")
        print(f"   FLARE enabled: {enable_flare}")
        self.alignment_loss_weight = alignment_loss_weight
        self.enable_flare = enable_flare
        self.num_future_tokens = num_future_tokens
        self.activation_layer = activation_layer
        
        # 创建FLARE增强的扩散模型
        hidden_size = config['rdt']['hidden_size']
        self.model = RDTWithFLARE(
            output_dim=action_dim,
            horizon=pred_horizon,
            hidden_size=hidden_size,
            depth=config['rdt']['depth'],
            num_heads=config['rdt']['num_heads'],
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=img_cond_len,
            lang_pos_embed_config=lang_pos_embed_config,
            img_pos_embed_config=img_pos_embed_config,
            dtype=dtype,
            num_future_tokens=num_future_tokens,
            activation_layer=activation_layer,
            num_vl_fusion_layers=num_vl_fusion_layers,
            num_qformer_layers=num_qformer_layers,
            alignment_temperature=alignment_temperature,
            # 🔧 只传递未来观测相关的路径
            future_vision_model_name=self.future_vision_path,
            future_text_model_name=self.future_text_path,
            future_vision_image_size=self.future_vision_image_size,
        )

        # 创建条件适配器
        self.lang_adaptor = self.build_condition_adapter(config['lang_adaptor'],
                                                         in_features=lang_token_dim,
                                                         out_features=hidden_size).to(dtype)
        self.img_adaptor = self.build_condition_adapter(config['img_adaptor'],
                                                        in_features=img_token_dim,
                                                        out_features=hidden_size).to(dtype)
        self.state_adaptor = self.build_condition_adapter(
            config['state_adaptor'],
            in_features=state_token_dim * 2,
            out_features=hidden_size).to(dtype)

        # FLARE: 未来观测视觉token适配器
        self.future_vision_adaptor = self.build_condition_adapter(
            config.get('future_vision_adaptor', 'linear'),
            in_features=img_token_dim,
            out_features=hidden_size).to(dtype)
        self._ensure_bf16_consistency()

        # 噪声调度器
        noise_scheduler_config = config['noise_scheduler']
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
            clip_sample=noise_scheduler_config['clip_sample'],
        )
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
        )

        self.num_train_timesteps = noise_scheduler_config['num_train_timesteps']
        self.num_inference_timesteps = noise_scheduler_config['num_inference_timesteps']
        self.prediction_type = noise_scheduler_config['prediction_type']

        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        print("FLARE Diffusion params: %e" %
              sum([p.numel() for p in self.model.parameters()] + 
                  [p.numel() for p in self.lang_adaptor.parameters()] +
                  [p.numel() for p in self.img_adaptor.parameters()] + 
                  [p.numel() for p in self.state_adaptor.parameters()] +
                  [p.numel() for p in self.future_vision_adaptor.parameters()]))
    def _ensure_bf16_consistency(self):
        """确保所有组件使用BF16"""
        target_dtype = torch.bfloat16
        
        self.model = self.model.to(target_dtype)
        self.lang_adaptor = self.lang_adaptor.to(target_dtype)
        self.img_adaptor = self.img_adaptor.to(target_dtype)
        self.state_adaptor = self.state_adaptor.to(target_dtype)
        self.future_vision_adaptor = self.future_vision_adaptor.to(target_dtype)
        
        print(f"✅ RDT Runner统一使用: {target_dtype}")
        
    def build_condition_adapter(self, projector_type, in_features, out_features):
        """构建条件适配器"""
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector

    
    def adapt_conditions(self, lang_tokens, img_tokens, state_action_traj, future_vision_tokens=None):
        # 获取模型期望的数据类型
        target_dtype = next(self.lang_adaptor.parameters()).dtype
        
        # 确保所有输入张量与模型参数使用相同的数据类型
        if lang_tokens is not None:
            lang_tokens = lang_tokens.to(dtype=target_dtype)
        if img_tokens is not None:
            img_tokens = img_tokens.to(dtype=target_dtype)
        if state_action_traj is not None:
            state_action_traj = state_action_traj.to(dtype=target_dtype)
        if future_vision_tokens is not None:
            future_vision_tokens = future_vision_tokens.to(dtype=target_dtype)
        
        # 在autocast范围内执行适配
        with torch.autocast(device_type='cuda', dtype=target_dtype, 
                        enabled=(target_dtype in [torch.float16, torch.bfloat16])):
            adapted_lang = self.lang_adaptor(lang_tokens) if lang_tokens is not None else None
            adapted_img = self.img_adaptor(img_tokens) if img_tokens is not None else None
            adapted_state = self.state_adaptor(state_action_traj) if state_action_traj is not None else None
            adapted_future_vision = self.future_vision_adaptor(future_vision_tokens) if future_vision_tokens is not None else None
        
        return adapted_lang, adapted_img, adapted_state, adapted_future_vision

    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond, state_traj, action_mask, ctrl_freqs):
        """条件采样（推理时不需要未来观测）"""
        device = state_traj.device
        dtype = state_traj.dtype
        noisy_action = torch.randn(size=(state_traj.shape[0], self.pred_horizon, self.action_dim),
                                   dtype=dtype,
                                   device=device)
        action_mask = action_mask.expand(-1, self.pred_horizon, -1)

        # 设置采样步数
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)

        for t in self.noise_scheduler_sample.timesteps:
            # 准备状态-动作轨迹
            action_traj = torch.cat([noisy_action, action_mask], dim=2)
            action_traj = self.state_adaptor(action_traj)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)

            # 模型预测（推理时不使用未来观测）
            model_output = self.model(state_action_traj,
                                      ctrl_freqs,
                                      t.unsqueeze(-1).to(device),
                                      lang_cond,
                                      img_cond,
                                      lang_mask=lang_attn_mask,
                                      future_vision_tokens=None,
                                      text_instructions=None,
                                      return_alignment_loss=False)

            # 计算前一步动作: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)

        # 应用动作掩码
        noisy_action = noisy_action * action_mask

        return noisy_action

    def compute_loss_with_flare(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, 
                            action_gt, action_mask, ctrl_freqs, future_vision_tokens=None, 
                            text_instructions=None, has_future_obs=None,
                            future_obs_images=None):
        """
        计算FLARE增强的损失 - 优化版本
        """
        # 🔧 统一设备和数据类型处理
        device = next(self.model.parameters()).device  # 获取模型设备
        target_dtype = torch.bfloat16  # 明确使用BF16
        batch_size = lang_tokens.shape[0]
        
        # 🔧 统一将所有输入移动到正确设备和数据类型
        def to_device_dtype(tensor, device, dtype):
            """统一的设备和数据类型转换"""
            if tensor is not None:
                return tensor.to(dtype=dtype, device=device)
            return tensor
        
        # 转换所有张量输入
        lang_tokens = to_device_dtype(lang_tokens, device, target_dtype)
        img_tokens = to_device_dtype(img_tokens, device, target_dtype)
        state_tokens = to_device_dtype(state_tokens, device, target_dtype)
        action_gt = to_device_dtype(action_gt, device, target_dtype)
        action_mask = to_device_dtype(action_mask, device, target_dtype)
        future_vision_tokens = to_device_dtype(future_vision_tokens, device, target_dtype)
        future_obs_images = to_device_dtype(future_obs_images, device, target_dtype)
        
        # 处理text_instructions（可能是字符串列表或张量）
        if text_instructions is not None:
            if isinstance(text_instructions, torch.Tensor):
                text_instructions = text_instructions.to(device)
            # 如果是字符串列表，保持不变
            
        
        # 🔧 确保ctrl_freqs也在正确设备上
        if isinstance(ctrl_freqs, torch.Tensor):
            ctrl_freqs = ctrl_freqs.to(dtype=target_dtype, device=device)
        
        # 在autocast范围内进行计算
        with torch.autocast(device_type='cuda', dtype=target_dtype):
            # 采样噪声和时间步
            noise = torch.randn(action_gt.shape, dtype=target_dtype, device=device)
            timesteps = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device).long()
            noisy_action = self.noise_scheduler.add_noise(action_gt, noise, timesteps)

            # 拼接状态和动作token
            state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
            action_mask_expanded = action_mask.expand(-1, state_action_traj.shape[1], -1)
            state_action_traj = torch.cat([state_action_traj, action_mask_expanded], dim=2)
            
            # 适配条件
            adapted_results = self.adapt_conditions(lang_tokens, img_tokens, state_action_traj, future_vision_tokens)
            lang_cond, img_cond, state_action_traj = adapted_results[:3]
            adapted_future_vision = adapted_results[3] if len(adapted_results) > 3 else None
            
            # 🔧 改进的FLARE使用判断逻辑
            use_flare = (
                self.enable_flare and 
                future_obs_images is not None and  # 使用原始图像而不是vision tokens
                text_instructions is not None
            )
            
            # 进一步检查has_future_obs
            if use_flare and has_future_obs is not None:
                valid_indices = has_future_obs.bool()
                if valid_indices.sum() == 0:
                    use_flare = False
            
            # 🔧 模型前向传播 - 简化逻辑
            try:
                if use_flare:
                    # 使用FLARE增强的模型
                    pred, alignment_loss = self.model(
                        state_action_traj, 
                        ctrl_freqs, 
                        timesteps, 
                        lang_cond, 
                        img_cond, 
                        lang_mask=lang_attn_mask, 
                        img_mask=None,
                        future_vision_tokens=adapted_future_vision,
                        text_instructions=text_instructions, 
                        future_obs_image=future_obs_images,
                        return_alignment_loss=True
                    )
                else:
                    # 标准扩散模型
                    pred = self.model(
                        state_action_traj, 
                        ctrl_freqs, 
                        timesteps, 
                        lang_cond, 
                        img_cond,
                        lang_mask=lang_attn_mask,
                        img_mask=None,
                        future_vision_tokens=None,
                        text_instructions=None,
                        future_obs_image=None,  # 🔧 标准模式不使用未来观测
                        return_alignment_loss=False
                    )
                    alignment_loss = torch.tensor(0.0, device=device, dtype=target_dtype)
                    
            except Exception as e:
                print(f"❌ 模型前向传播失败: {e}")
                print(f"   use_flare: {use_flare}")
                print(f"   future_obs_images: {future_obs_images.shape if future_obs_images is not None else None}")
                print(f"   text_instructions: {type(text_instructions)}")
                raise e

            # 🔧 对齐损失的处理
            if use_flare and alignment_loss is not None and has_future_obs is not None:
                valid_count = has_future_obs.sum().float()
                if valid_count > 0 and valid_count < batch_size:
                    # 只对有效样本进行归一化
                    alignment_loss = alignment_loss * (batch_size / valid_count)
                elif valid_count == 0:
                    alignment_loss = torch.tensor(0.0, device=device, dtype=target_dtype)

            # 计算目标
            if self.prediction_type == 'epsilon':
                target = noise
            elif self.prediction_type == 'sample':
                target = action_gt
            else:
                raise ValueError(f"Unsupported prediction type {self.prediction_type}")
                
            # 扩散损失
            diffusion_loss = F.mse_loss(pred, target)
            
            # 总损失
            total_loss = diffusion_loss
            if use_flare and alignment_loss is not None:
                total_loss = total_loss + self.alignment_loss_weight * alignment_loss
                
        # 构建损失字典
        loss_dict = {
            'diffusion_loss': diffusion_loss.item(),
            'alignment_loss': alignment_loss.item() if alignment_loss is not None else 0.0,
            'total_loss': total_loss.item(),
            'alignment_loss_weight': self.alignment_loss_weight,
            'used_flare': use_flare,
            'batch_size': batch_size,
            'valid_future_obs': has_future_obs.sum().item() if has_future_obs is not None else 0,
        }
        
        return total_loss, loss_dict

    def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_gt, action_mask,
                     ctrl_freqs, future_vision_tokens=None, text_instructions=None, has_future_obs=None):
        """
        兼容性接口：计算损失
        """
        total_loss, loss_dict = self.compute_loss_with_flare(
            lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_gt, action_mask,
            ctrl_freqs, future_vision_tokens, text_instructions, has_future_obs
        )
        return total_loss

    def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_mask, ctrl_freqs):
        """预测动作（推理时不使用未来观测）"""
        # 准备状态和条件
        state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        lang_cond, img_cond, state_traj, _ = self.adapt_conditions(lang_tokens, img_tokens, state_tokens)

        # 运行采样
        action_pred = self.conditional_sample(
            lang_cond,
            lang_attn_mask,
            img_cond,
            state_traj,
            action_mask,
            ctrl_freqs,
        )

        return action_pred

    def get_alignment_metrics(self):
        """获取对齐相关的指标"""
        if hasattr(self.model, 'activation_aligner') and self.model.activation_aligner is not None:
            return self.model.activation_aligner.get_alignment_metrics()
        return {}

    def set_alignment_loss_weight(self, weight):
        """动态设置对齐损失权重"""
        self.alignment_loss_weight = weight

    def enable_flare_mode(self, enable=True):
        """启用/禁用FLARE模式"""
        self.enable_flare = enable

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """前向传播接口"""
        return self.compute_loss(*args, **kwargs)