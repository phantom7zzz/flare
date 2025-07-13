#!/usr/bin/env python
# coding=utf-8
# FLARE增强的训练脚本

import copy
import logging
import math
import os
from pathlib import Path

import diffusers
import torch
import torch.utils.checkpoint
import transformers
import yaml
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from safetensors.torch import load_model

from models.ema_model import EMAModel
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.rdt_runner import RDTRunnerWithFLARE
from train.dataset import DataCollatorForVLAConsumerDatasetWithFLARE, VLAConsumerDatasetWithFLARE
from train.sample import log_sample_res

if is_wandb_available():
    import wandb


def save_model_card(repo_id: str, base_model=str, repo_folder=None):
    yaml_content = f"""
---
license: mit
base_model: {base_model}
language:
- en
pipeline_tag: robotics
library_name: transformers
tags:
- robotics
- pytorch
- multimodal
- pretraining
- vla
- diffusion
- rdt
- flare
---
    """
    model_card = f"""
# RDT-FLARE - {repo_id}

This is a FLARE-enhanced RDT model derived from {base_model}. The weights were trained using [RDT](https://rdt-robotics.github.io/rdt-robotics/) with FLARE (Future-conditioned Language-guided Action REpresentation) enhancement.

## FLARE Features
- Future observation alignment
- Vision-Language token fusion
- Q-Former target generation
- DiT activation alignment

The model includes future observation prediction capabilities for improved action planning.
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml_content + model_card)


def train(args, logger):
    # Read the config
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)

    with open(args.model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    
    args.output_dir = model_config["checkpoint_path"]
    logging_dir = Path(args.output_dir, args.logging_dir)

    # FLARE参数从模型配置中读取
    enable_flare = getattr(args, 'enable_flare', True)
    num_future_tokens = getattr(args, 'num_future_tokens', 32)
    activation_layer = getattr(args, 'activation_layer', 6)
    alignment_loss_weight = getattr(args, 'alignment_loss_weight', 0.1)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        deepspeed_plugin=(DeepSpeedPlugin(hf_ds_config=args.deepspeed) if args.deepspeed is not None else None),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 初始化编码器
    if args.precomp_lang_embed:
        tokenizer, text_encoder = None, None
    else:
        text_embedder = T5Embedder(
            from_pretrained=args.pretrained_text_encoder_name_or_path,
            model_max_length=config["dataset"]["tokenizer_max_length"],
            device=accelerator.device,
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    vision_encoder = SiglipVisionTower(vision_tower=args.pretrained_vision_encoder_name_or_path, args=None)
    image_processor = vision_encoder.image_processor

    # 构建FLARE增强的RDT模型
    if args.pretrained_model_name_or_path is not None and not os.path.isfile(args.pretrained_model_name_or_path):
        logger.info("Constructing FLARE model from pretrained checkpoint.")
        # 这里需要特殊处理，因为预训练模型可能不包含FLARE组件
        logger.warning("Loading from standard RDT checkpoint - FLARE components will be randomly initialized")
        rdt = create_flare_model_from_standard_rdt(args, config, vision_encoder, weight_dtype)
    else:
        logger.info("Constructing FLARE model from provided config.")
        img_cond_len = (config["common"]["img_history_size"] * config["common"]["num_cameras"] *
                        vision_encoder.num_patches)
        rdt = RDTRunnerWithFLARE(
            action_dim=config["common"]["state_dim"],
            pred_horizon=config["common"]["action_chunk_size"],
            config=config["model"],
            lang_token_dim=config["model"]["lang_token_dim"],
            img_token_dim=config["model"]["img_token_dim"],
            state_token_dim=config["model"]["state_token_dim"],
            max_lang_cond_len=config["dataset"]["tokenizer_max_length"],
            img_cond_len=img_cond_len,
            img_pos_embed_config=[
                ("image", (
                    config["common"]["img_history_size"],
                    config["common"]["num_cameras"],
                    -vision_encoder.num_patches,
                )),
            ],
            lang_pos_embed_config=[
                ("lang", -config["dataset"]["tokenizer_max_length"]),
            ],
            dtype=weight_dtype,
            # FLARE特定参数
            num_future_tokens=num_future_tokens,
            activation_layer=activation_layer,
            alignment_loss_weight=alignment_loss_weight,
            enable_flare=enable_flare,
        )
        # 确保模型参数的数据类型正确
        rdt = rdt.to(dtype=weight_dtype)

    # EMA模型
    ema_rdt = copy.deepcopy(rdt)
    ema_model = EMAModel(
        ema_rdt,
        update_after_step=config["model"]["ema"]["update_after_step"],
        inv_gamma=config["model"]["ema"]["inv_gamma"],
        power=config["model"]["ema"]["power"],
        min_value=config["model"]["ema"]["min_value"],
        max_value=config["model"]["ema"]["max_value"],
    )

    # create custom saving & loading hooks
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                model_to_save = model.module if hasattr(model, "module") else model
                if isinstance(model_to_save, type(accelerator.unwrap_model(rdt))):
                    model_to_save.save_pretrained(output_dir)

    accelerator.register_save_state_pre_hook(save_model_hook)

    if args.gradient_checkpointing:
        raise NotImplementedError("Gradient checkpointing is not yet implemented for FLARE.")

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size *
                              accelerator.num_processes)

    # Optimizer creation
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = rdt.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 创建FLARE增强的数据集
    train_dataset = VLAConsumerDatasetWithFLARE(
        model_config_path=args.model_config_path,
        config=config["dataset"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=args.dataset_type,
        image_aug=args.image_aug,
        cond_mask_prob=args.cond_mask_prob,
        cam_ext_mask_prob=args.cam_ext_mask_prob,
        state_noise_snr=args.state_noise_snr,
        use_hdf5=args.load_from_hdf5,
        use_precomp_lang_embed=args.precomp_lang_embed,
        # FLARE参数
        enable_future_obs=enable_flare,
        future_obs_prob=0.8,
        action_chunk_size=config["common"]["action_chunk_size"],
    )
    
    sample_dataset = VLAConsumerDatasetWithFLARE(
        model_config_path=args.model_config_path,
        config=config["dataset"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=args.dataset_type,
        image_aug=False,
        cond_mask_prob=0,
        cam_ext_mask_prob=-1,
        state_noise_snr=None,
        use_hdf5=args.load_from_hdf5,
        use_precomp_lang_embed=args.precomp_lang_embed,
        # FLARE参数
        enable_future_obs=False,  # 采样时不使用未来观测
        future_obs_prob=0.0,
        action_chunk_size=config["common"]["action_chunk_size"],
    )

    data_collator = DataCollatorForVLAConsumerDatasetWithFLARE(tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    sample_dataloader = torch.utils.data.DataLoader(
        sample_dataset,
        batch_size=args.sample_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # Scheduler and math around the number of training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with accelerator
    rdt, optimizer, train_dataloader, sample_dataloader, lr_scheduler = (accelerator.prepare(
        rdt, optimizer, train_dataloader, sample_dataloader, lr_scheduler))

    # 将模型移动到正确的设备和数据类型
    rdt.to(accelerator.device, dtype=weight_dtype)
    ema_rdt.to(accelerator.device, dtype=weight_dtype)

    if text_encoder is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if vision_encoder is not None:
        vision_encoder.vision_tower.to(accelerator.device, dtype=weight_dtype)

    # Recalculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Initialize trackers
    if accelerator.is_main_process:
        tracker_config = vars(args)
        tracker_config.update({
            "enable_flare": enable_flare,
            "num_future_tokens": num_future_tokens,
            "activation_layer": activation_layer,
            "alignment_loss_weight": alignment_loss_weight,
        })
        
        accelerator.init_trackers(
            "VLA_FLARE",
            config=tracker_config,
            init_kwargs={"wandb": {
                "name": f"RDT_FLARE_{args.CONFIG_NAME}",
                "tags": ["rdt", "flare", "multimodal", "robotics"],
            }},
        )

    # Training info
    total_batch_size = (args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps)

    logger.info("***** Running FLARE training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  FLARE enabled = {enable_flare}")
    logger.info(f"  Future tokens = {num_future_tokens}")
    logger.info(f"  Activation layer = {activation_layer}")
    logger.info(f"  Alignment loss weight = {alignment_loss_weight}")
    
    global_step = 0
    first_epoch = 0

    # Load from pretrained checkpoint
    if (args.resume_from_checkpoint is None and args.pretrained_model_name_or_path is not None
            and os.path.isfile(args.pretrained_model_name_or_path)):
        logger.info("Loading from a pretrained checkpoint.")
        checkpoint = torch.load(args.pretrained_model_name_or_path)
        unwrapped_rdt = accelerator.unwrap_model(rdt)
        unwrapped_rdt.load_state_dict(checkpoint["module"], strict=False)  # strict=False for FLARE components

    # Resume from checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            try:
                accelerator.load_state(os.path.join(args.output_dir, path))
            except:
                logger.info("Resuming training state failed. Attempting to only load from model checkpoint.")
                checkpoint = torch.load(
                    os.path.join(
                        args.output_dir,
                        path,
                        "pytorch_model",
                        "mp_rank_00_model_states.pt",
                    ))
                unwrapped_rdt = accelerator.unwrap_model(rdt)
                unwrapped_rdt.load_state_dict(checkpoint["module"], strict=False)

            load_model(ema_rdt, os.path.join(args.output_dir, path, "ema", "model.safetensors"))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Progress bar
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("FLARE Training Steps")

    # Training metrics
    loss_for_log = {}
    alignment_metrics_log = {}
    
    for epoch in range(first_epoch, args.num_train_epochs):
        rdt.train()

        # Set progress bar to correct position
        if args.resume_from_checkpoint and epoch == first_epoch:
            progress_bar.update(resume_step // args.gradient_accumulation_steps)

        # FLARE训练循环
        for batch in train_dataloader:
            with accelerator.accumulate(rdt):
                # 基础数据准备
                images = batch["images"].to(dtype=weight_dtype)
                states = batch["states"].to(dtype=weight_dtype)
                states = states[:, -1:, :]  # 只使用最后一个状态
                actions = batch["actions"].to(dtype=weight_dtype)
                state_elem_mask = batch["state_elem_mask"].to(dtype=weight_dtype)
                ctrl_freqs = batch["ctrl_freqs"]

                # FLARE特定数据
                future_obs_images = batch.get("future_obs_images")
                text_instructions = batch.get("text_instructions", [""] * images.shape[0])
                has_future_obs = batch.get("has_future_obs")

                with torch.no_grad():
                    # 编码图像
                    batch_size, _, C, H, W = images.shape
                    image_embeds = vision_encoder(images.reshape(-1, C, H, W)).detach()
                    image_embeds = image_embeds.reshape((batch_size, -1, vision_encoder.hidden_size))

                    # 编码语言
                    lang_attn_mask = batch["lang_attn_mask"]
                    text_embeds = (batch["lang_embeds"].to(dtype=weight_dtype) 
                                  if args.precomp_lang_embed 
                                  else text_encoder(input_ids=batch["input_ids"], 
                                                   attention_mask=lang_attn_mask)["last_hidden_state"].detach())

                    # 编码未来观测图像（如果存在）
                    future_vision_embeds = None
                    if future_obs_images is not None and enable_flare:
                        future_vision_embeds = vision_encoder(future_obs_images.to(dtype=weight_dtype)).detach()

                state_elem_mask = state_elem_mask.unsqueeze(1)
                
                # 计算FLARE增强的损失
                # 使用 accelerator.unwrap_model 来访问原始模型
                unwrapped_rdt = accelerator.unwrap_model(rdt)
                total_loss, loss_dict = unwrapped_rdt.compute_loss_with_flare(
                    lang_tokens=text_embeds,
                    lang_attn_mask=lang_attn_mask,
                    img_tokens=image_embeds,
                    state_tokens=states,
                    action_gt=actions,
                    action_mask=state_elem_mask,
                    ctrl_freqs=ctrl_freqs,
                    future_vision_tokens=future_vision_embeds,
                    text_instructions=text_instructions,
                    has_future_obs=has_future_obs,
                )

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    params_to_clip = rdt.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            ema_model.step(accelerator.unwrap_model(rdt))

            # Check if optimization step occurred
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Checkpointing
                if global_step % args.checkpointing_period == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    ema_save_path = os.path.join(save_path, f"ema")
                    accelerator.save_model(ema_rdt, ema_save_path)
                    logger.info(f"Saved state to {save_path}")

                # Sampling and evaluation
                if args.sample_period > 0 and global_step % args.sample_period == 0:
                    sample_loss_for_log = log_sample_res(
                        text_encoder,
                        vision_encoder,
                        rdt,
                        args,
                        accelerator,
                        weight_dtype,
                        sample_dataset.get_dataset_id2name(),
                        sample_dataloader,
                        logger,
                    )
                    logger.info(sample_loss_for_log)
                    accelerator.log(sample_loss_for_log, step=global_step)

                # 获取对齐指标
                if enable_flare and global_step % 100 == 0:
                    try:
                        unwrapped_rdt = accelerator.unwrap_model(rdt)
                        alignment_metrics = unwrapped_rdt.get_alignment_metrics()
                        alignment_metrics_log.update({f"alignment_{k}": v for k, v in alignment_metrics.items()})
                    except Exception as e:
                        logger.debug(f"Failed to get alignment metrics: {e}")

            # 记录损失
            logs = {
                "loss": total_loss.detach().item(), 
                "lr": lr_scheduler.get_last_lr()[0],
                "diffusion_loss": loss_dict.get('diffusion_loss', 0.0),
                "alignment_loss": loss_dict.get('alignment_loss', 0.0),
                "used_flare": loss_dict.get('used_flare', False),
            }
            
            progress_bar.set_postfix(**logs)
            logs.update(loss_for_log)
            logs.update(alignment_metrics_log)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(rdt).save_pretrained(args.output_dir)
        ema_save_path = os.path.join(args.output_dir, f"ema")
        accelerator.save_model(ema_rdt, ema_save_path)

        logger.info(f"Saved FLARE Model to {args.output_dir}")

        if args.push_to_hub:
            save_model_card(
                repo_id,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of FLARE training",
                token=args.hub_token,
                allow_patterns=["pytorch_model.bin", "*.json", "*.md"],
            )

    accelerator.end_training()


def create_flare_model_from_standard_rdt(args, config, vision_encoder, weight_dtype):
    """
    创建随机初始化的FLARE模型（从头训练）
    """
    from models.rdt_runner import RDTRunnerWithFLARE
    import logging
    
    logger = logging.getLogger(__name__)
    
    # 计算图像条件长度
    img_cond_len = (config["common"]["img_history_size"] * config["common"]["num_cameras"] *
                    vision_encoder.num_patches)
    
    logger.info("正在创建随机初始化的FLARE模型...")
    
    # 直接创建FLARE模型，随机初始化所有参数
    flare_rdt = RDTRunnerWithFLARE(
        action_dim=config["common"]["state_dim"],
        pred_horizon=config["common"]["action_chunk_size"],
        config=config["model"],
        lang_token_dim=config["model"]["lang_token_dim"],
        img_token_dim=config["model"]["img_token_dim"],
        state_token_dim=config["model"]["state_token_dim"],
        max_lang_cond_len=config["dataset"]["tokenizer_max_length"],
        img_cond_len=img_cond_len,
        img_pos_embed_config=[
            ("image", (
                config["common"]["img_history_size"],
                config["common"]["num_cameras"],
                -vision_encoder.num_patches,
            )),
        ],
        lang_pos_embed_config=[
            ("lang", -config["dataset"]["tokenizer_max_length"]),
        ],
        dtype=weight_dtype,
        # FLARE参数
        num_future_tokens=getattr(args, 'num_future_tokens', 32),
        activation_layer=getattr(args, 'activation_layer', 6),
        alignment_loss_weight=getattr(args, 'alignment_loss_weight', 0.1),
        enable_flare=getattr(args, 'enable_flare', True),
    )
    
    # 确保模型参数的数据类型正确
    flare_rdt = flare_rdt.to(dtype=weight_dtype)
    
    logger.info(f"FLARE模型创建成功，参数总数: {sum(p.numel() for p in flare_rdt.parameters())}")
    logger.info(f"模型数据类型: {weight_dtype}")
    logger.info("所有参数已随机初始化，准备开始从头训练")
    
    return flare_rdt