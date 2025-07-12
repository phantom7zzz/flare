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
                 max_lang_cond_len,
                 img_cond_len,
                 lang_pos_embed_config=None,
                 img_pos_embed_config=None,
                 dtype=torch.bfloat16,
                 # FLARE参数
                 num_future_tokens=32,
                 activation_layer=6,
                 alignment_loss_weight=0.1,
                 num_vl_fusion_layers=4,
                 num_qformer_layers=6,
                 alignment_temperature=0.07,
                 vision_model_name="google/siglip-so400m-patch14-384",
                 text_model_name="google/siglip-so400m-patch14-384"):
        super().__init__()
        
        self.alignment_loss_weight = alignment_loss_weight
        
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
            vision_model_name=vision_model_name,
            text_model_name=text_model_name,
        )

        # 创建条件适配器
        self.lang_adaptor = self.build_condition_adapter(config['lang_adaptor'],
                                                         in_features=lang_token_dim,
                                                         out_features=hidden_size)
        self.img_adaptor = self.build_condition_adapter(config['img_adaptor'],
                                                        in_features=img_token_dim,
                                                        out_features=hidden_size)
        self.state_adaptor = self.build_condition_adapter(
            config['state_adaptor'],
            in_features=state_token_dim * 2,
            out_features=hidden_size)

        # FLARE: 未来观测视觉token适配器
        self.future_vision_adaptor = self.build_condition_adapter(
            config.get('future_vision_adaptor', 'linear'),
            in_features=img_token_dim,
            out_features=hidden_size)

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

    def adapt_conditions(self, lang_tokens, img_tokens, state_tokens, future_vision_tokens=None):
        """适配条件输入"""
        adapted_lang = self.lang_adaptor(lang_tokens)
        adapted_img = self.img_adaptor(img_tokens)
        adapted_state = self.state_adaptor(state_tokens)
        
        adapted_future_vision = None
        if future_vision_tokens is not None:
            adapted_future_vision = self.future_vision_adaptor(future_vision_tokens)

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

    def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_gt, action_mask,
                     ctrl_freqs, future_vision_tokens=None, text_instructions=None) -> tuple:
        """
        计算损失，包含FLARE对齐损失
        
        Returns:
            total_loss: 总损失
            diffusion_loss: 扩散损失
            alignment_loss: 对齐损失（如果使用FLARE）
        """
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        
        # 采样噪声和时间步
        noise = torch.randn(action_gt.shape, dtype=action_gt.dtype, device=device)
        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device).long()
        noisy_action = self.noise_scheduler.add_noise(action_gt, noise, timesteps)

        # 拼接状态和动作token
        state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
        action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
        state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)
        
        # 适配条件
        adapted_results = self.adapt_conditions(lang_tokens, img_tokens, state_action_traj, future_vision_tokens)
        lang_cond, img_cond, state_action_traj = adapted_results[:3]
        adapted_future_vision = adapted_results[3] if len(adapted_results) > 3 else None
        
        # 预测去噪结果
        return_alignment = adapted_future_vision is not None
        if return_alignment:
            pred, alignment_loss = self.model(
                state_action_traj, ctrl_freqs, timesteps, lang_cond, img_cond, 
                lang_mask=lang_attn_mask, future_vision_tokens=adapted_future_vision,
                text_instructions=text_instructions, return_alignment_loss=True
            )
        else:
            pred = self.model(
                state_action_traj, ctrl_freqs, timesteps, lang_cond, img_cond,
                lang_mask=lang_attn_mask, future_vision_tokens=None,
                text_instructions=None, return_alignment_loss=False
            )
            alignment_loss = None

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
        if alignment_loss is not None:
            total_loss = total_loss + self.alignment_loss_weight * alignment_loss
            
        return total_loss, diffusion_loss, alignment_loss

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

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)