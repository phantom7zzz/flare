import re, sys, os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from pathlib import Path
# get current workspace
current_file = Path(__file__)
sys.path.append(os.path.join(current_file.parent))
from hub_mixin import CompatiblePyTorchModelHubMixin
# 修改导入以使用FLARE版本的RDT
from rdt.model import RDTWithFLARE


class RDTRunnerWithFLARE(nn.Module,
                         CompatiblePyTorchModelHubMixin,
                         repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b"):

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
                 # FLARE相关参数
                 num_future_tokens=32,
                 activation_layer=6,
                 alignment_loss_weight=0.1):
        super(RDTRunnerWithFLARE, self).__init__()
        
        # FLARE相关属性
        self.alignment_loss_weight = alignment_loss_weight
        
        # Create diffusion model with FLARE
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
            num_future_tokens=num_future_tokens,  # 新增
            activation_layer=activation_layer,    # 新增
        )

        # Create adapters for various conditional inputs (保持不变)
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

        # 新增：未来观测视觉token的适配器
        self.future_vision_adaptor = self.build_condition_adapter(
            config.get('future_vision_adaptor', 'linear'),
            in_features=img_token_dim,  # 与普通图像token相同
            out_features=hidden_size)

        # Create the noise scheduler (保持不变)
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

        print("Diffusion params: %e" %
              sum([p.numel() for p in self.model.parameters()] + 
                  [p.numel() for p in self.lang_adaptor.parameters()] +
                  [p.numel() for p in self.img_adaptor.parameters()] + 
                  [p.numel() for p in self.state_adaptor.parameters()] +
                  [p.numel() for p in self.future_vision_adaptor.parameters()]))  # 新增

    def build_condition_adapter(self, projector_type, in_features, out_features):
        # 保持原有实现不变
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
        '''
        修改以支持未来观测token的适配
        '''
        adapted_lang = self.lang_adaptor(lang_tokens)
        adapted_img = self.img_adaptor(img_tokens)
        adapted_state = self.state_adaptor(state_tokens)
        
        adapted_future_vision = None
        if future_vision_tokens is not None:
            adapted_future_vision = self.future_vision_adaptor(future_vision_tokens)

        return adapted_lang, adapted_img, adapted_state, adapted_future_vision

    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond, state_traj, action_mask, ctrl_freqs):
        '''
        推理时的条件采样（保持不变，因为推理时不需要未来观测）
        '''
        device = state_traj.device
        dtype = state_traj.dtype
        noisy_action = torch.randn(size=(state_traj.shape[0], self.pred_horizon, self.action_dim),
                                   dtype=dtype,
                                   device=device)
        action_mask = action_mask.expand(-1, self.pred_horizon, -1)

        # Set step values
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)

        for t in self.noise_scheduler_sample.timesteps:
            # Prepare state-action trajectory
            action_traj = torch.cat([noisy_action, action_mask], dim=2)
            action_traj = self.state_adaptor(action_traj)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)

            # Predict the model output (推理时不使用未来观测)
            model_output = self.model(state_action_traj,
                                      ctrl_freqs,
                                      t.unsqueeze(-1).to(device),
                                      lang_cond,
                                      img_cond,
                                      lang_mask=lang_attn_mask,
                                      future_vision_tokens=None,
                                      return_alignment_loss=False)

            # Compute previous actions: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)

        # Finally apply the action mask to mask invalid action dimensions
        noisy_action = noisy_action * action_mask

        return noisy_action

    def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_gt, action_mask,
                     ctrl_freqs, future_vision_tokens=None) -> torch.Tensor:
        '''
        修改损失计算以包含对齐损失
        
        新增参数:
            future_vision_tokens: (batch_size, future_vision_len, vision_token_dim) 未来观测视觉token
        '''
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        
        # Sample noise that we'll add to the actions
        noise = torch.randn(action_gt.shape, dtype=action_gt.dtype, device=device)
        # Sample random diffusion timesteps
        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size, ), device=device).long()
        # Add noise to the clean actions according to the noise magnitude at each timestep
        noisy_action = self.noise_scheduler.add_noise(action_gt, noise, timesteps)

        # Concatenate the state and action tokens to form the input sequence
        state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
        # Append the action mask to the input sequence
        action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
        state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)
        
        # Align the dimension with the hidden size
        adapted_results = self.adapt_conditions(lang_tokens, img_tokens, state_action_traj, future_vision_tokens)
        lang_cond, img_cond, state_action_traj = adapted_results[:3]
        adapted_future_vision = adapted_results[3] if len(adapted_results) > 3 else None
        
        # Predict the denoised result with alignment loss
        return_alignment = adapted_future_vision is not None
        if return_alignment:
            pred, alignment_loss = self.model(
                state_action_traj, ctrl_freqs, timesteps, lang_cond, img_cond, 
                lang_mask=lang_attn_mask, future_vision_tokens=adapted_future_vision,
                return_alignment_loss=True
            )
        else:
            pred = self.model(
                state_action_traj, ctrl_freqs, timesteps, lang_cond, img_cond,
                lang_mask=lang_attn_mask, future_vision_tokens=None,
                return_alignment_loss=False
            )
            alignment_loss = None

        pred_type = self.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = action_gt
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
            
        # 主要的扩散损失
        diffusion_loss = F.mse_loss(pred, target)
        
        # 总损失 = 扩散损失 + 对齐损失
        total_loss = diffusion_loss
        if alignment_loss is not None:
            total_loss = total_loss + self.alignment_loss_weight * alignment_loss
            
        return total_loss, diffusion_loss, alignment_loss

    def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_mask, ctrl_freqs):
        '''
        预测动作（推理时保持不变）
        '''
        # Prepare the state and conditions
        state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        lang_cond, img_cond, state_traj, _ = self.adapt_conditions(lang_tokens, img_tokens, state_tokens)

        # Run sampling
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