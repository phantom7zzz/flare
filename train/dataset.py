# train/dataset.py - 简化版本

import traceback
import time
import os
import json
import math
import random
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import transformers

from data.filelock import FileLock
from data.hdf5_vla_dataset import HDF5VLADataset
from train.image_corrupt import image_corrupt


class VLAConsumerDatasetWithFLARE(Dataset):
    """
    支持FLARE功能的VLA消费者数据集
    核心理念：动态使用轨迹最后一帧作为未来观测，无需预处理
    """

    def __init__(
        self,
        model_config_path,
        config,
        tokenizer,
        image_processor,
        num_cameras,
        img_history_size,
        image_size=None,
        auto_adjust_image_brightness=False,
        image_aug=False,
        dataset_type="pretrain",
        cond_mask_prob=0.1,
        cam_ext_mask_prob=-1.0,
        state_noise_snr=None,
        use_hdf5=False,
        use_precomp_lang_embed=False,
        # FLARE特定参数
        enable_future_obs=True,
        future_obs_prob=0.8,  # 使用未来观测的概率
        action_chunk_size=32,  # 动作块大小
    ):
        super(VLAConsumerDatasetWithFLARE, self).__init__()

        # ... 保持原有的初始化代码 ...
        # 加载控制频率
        with open("configs/dataset_control_freq.json", "r") as fp:
            self.control_freq = json.load(fp)
        
        # 加载数据集名称
        dataset_names_cfg = ("configs/pretrain_datasets.json"
                             if dataset_type == "pretrain" else "configs/finetune_datasets.json")
        with open(dataset_names_cfg, "r") as file:
            DATASET_NAMES = json.load(file)
        
        # 创建数据集名称和ID之间的映射
        self.dataset_name2id = {name: i for i, name in enumerate(DATASET_NAMES)}
        self.dataset_id2name = {i: name for i, name in enumerate(DATASET_NAMES)}

        self.image_processor = image_processor
        self.model_config_path = model_config_path
        self.buffer_dir = config["buf_path"]
        self.num_chunks = config["buf_num_chunks"]
        self.chunk_size = config["buf_chunk_size"]
        self.tokenizer_max_length = config["tokenizer_max_length"]
        self.image_aspect_ratio = config["image_aspect_ratio"]
        self.state_noise_snr = state_noise_snr
        self.num_cameras = num_cameras
        self.img_history_size = img_history_size
        self.cond_mask_prob = cond_mask_prob
        self.cam_ext_mask_prob = cam_ext_mask_prob
        self.use_hdf5 = use_hdf5
        self.hdf5_dataset = None
        
        # FLARE特定属性
        self.enable_future_obs = enable_future_obs
        self.future_obs_prob = future_obs_prob
        self.action_chunk_size = action_chunk_size
        
        if use_hdf5:
            self.hdf5_dataset = HDF5VLADataset(self.model_config_path)
        self.use_precomp_lang_embed = use_precomp_lang_embed
        if use_precomp_lang_embed:
            self.empty_lang_embed = torch.load("data/empty_lang_embed.pt")

        # 加载数据集统计信息
        with open("configs/dataset_stat.json", "r") as f:
            dataset_stat = json.load(f)
        self.dataset_stat = dataset_stat

        self.tokenizer = tokenizer
        self.image_size = image_size
        self.auto_adjust_image_brightness = auto_adjust_image_brightness
        self.image_aug = image_aug

        self.last_content = None
        self.last_meta = None

    def get_dataset_name2id(self):
        return self.dataset_name2id

    def get_dataset_id2name(self):
        return self.dataset_id2name

    @staticmethod
    def pairwise(iterable):
        a = iter(iterable)
        return zip(a, a)

    def _load_data_from_chunk(self, chunk_dir, chunk_item_idx):
        """从chunk加载数据"""
        time_stmp = time.time()
        while time.time() - time_stmp < 10.0:
            try:
                locks = []
                file_path = os.path.join(chunk_dir, f"json_content_{chunk_item_idx}.json")
                lock = FileLock(file_path)
                locks.append(lock)
                lock.acquire_read_lock()
                with open(file_path, "r") as file:
                    json_content = json.load(file)
                lock.release_lock()
                
                file_path = os.path.join(chunk_dir, f"sample_{chunk_item_idx}.npz")
                lock = FileLock(file_path)
                locks.append(lock)
                lock.acquire_read_lock()
                with open(file_path, "rb") as file:
                    sample_dict = np.load(file)
                    meta = tuple(sample_dict.values())
                lock.release_lock()
                return json_content, meta
            except KeyboardInterrupt:
                for lock in locks:
                    lock.release_lock()
                raise KeyboardInterrupt
            except BaseException:
                for lock in locks:
                    lock.release_lock()
                continue
        raise RuntimeError("Failed to load sample.")

    def _safe_load(self, index):
        """安全加载数据"""
        from data.producer import get_dirty_item, read_dirty_bit, save_dirty_bit
        
        read_chunk_item_indices = []
        read_chunk_idx = index // self.chunk_size
        
        while len(read_chunk_item_indices) == 0:
            read_chunk_dir = os.path.join(self.buffer_dir, f"chunk_{read_chunk_idx}")
            try:
                read_chunk_item_indices = get_dirty_item(read_chunk_dir)
            except BaseException as e:
                print("Error catched when searching a clean chunk:", e)
                traceback.print_exc()
                read_chunk_item_indices = []
            read_chunk_idx = (read_chunk_idx + 1) % self.num_chunks

        random_item_index = index % len(read_chunk_item_indices)
        read_chunk_item_index = read_chunk_item_indices[random_item_index]

        # 修改dirty bit
        try:
            dirty_bit = read_dirty_bit(read_chunk_dir)
            dirty_bit[read_chunk_item_index] = 1
            save_dirty_bit(read_chunk_dir, dirty_bit)
        except BaseException as e:
            print("Error catched when modifying the dirty bit:", e)
            traceback.print_exc()

        # 加载样本
        try:
            content, meta = self._load_data_from_chunk(read_chunk_dir, read_chunk_item_index)
            self.last_content, self.last_meta = content, meta
        except BaseException as e:
            print("Error catched when loading sample:", e)
            traceback.print_exc()
            content, meta = self.last_content, self.last_meta

        return (content, *meta)

    def _extract_future_obs_from_chunk_data(self, chunk_data):
        """
        从chunk数据中提取未来观测
        核心：直接从现有数据中动态计算未来观测
        """
        try:
            # 检查是否有预计算的未来观测数据
            if len(chunk_data) >= 16:  # 原有14个字段 + 2个未来观测字段
                future_obs_frame = chunk_data[14] if len(chunk_data) > 14 else None
                future_obs_mask = chunk_data[15] if len(chunk_data) > 15 else False
                return future_obs_frame, future_obs_mask
            
            # 如果没有预计算的数据，返回None
            return None, False
            
        except Exception as e:
            print(f"Error extracting future obs from chunk data: {e}")
            return None, False

    def _compute_future_obs_from_episode_data(self, content, image_metas, current_step_id):
        """
        从episode数据动态计算未来观测
        这是FLARE的核心：使用action chunk结束后的观测作为未来观测
        """
        try:
            # 获取当前时间步
            step_id = current_step_id
            
            # 计算未来观测的时间步：当前步 + action_chunk_size - 1
            future_step = step_id + self.action_chunk_size - 1
            
            # 从image_metas中获取主摄像头（通常是第一个摄像头）的所有帧
            # image_metas结构：[cam0_images, cam0_mask, cam1_images, cam1_mask, ...]
            main_camera_images = image_metas[0]  # 主摄像头的图像序列
            
            # 检查未来步是否在有效范围内
            if future_step < len(main_camera_images):
                future_obs_frame = main_camera_images[future_step]
                future_obs_mask = True
            else:
                # 如果超出范围，使用最后一帧
                future_obs_frame = main_camera_images[-1]
                future_obs_mask = False
                
            return future_obs_frame, future_obs_mask
            
        except Exception as e:
            print(f"Error computing future obs from episode data: {e}")
            return None, False

    def _process_future_obs_image(self, future_obs_frame):
        """处理未来观测图像"""
        try:
            if future_obs_frame is None:
                return None
            
            # 检查图像是否为空（形状为0或全零）
            if hasattr(future_obs_frame, 'shape'):
                if any(dim == 0 for dim in future_obs_frame.shape):
                    print("⚠️  未来观测图像形状包含0维度")
                    return None
                
            # 检查是否为全零图像
                if hasattr(future_obs_frame, 'sum') and future_obs_frame.sum() == 0:
                    print("⚠️  未来观测图像为全零")
                    return None
                
            # 转换为PIL图像
            future_image = Image.fromarray(future_obs_frame)
            
            # 应用相同的图像处理流程
            if self.image_size is not None:
                future_image = transforms.Resize(self.image_size)(future_image)
            
            if self.auto_adjust_image_brightness:
                pixel_values = list(future_image.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                if average_brightness <= 0.15:
                    future_image = transforms.ColorJitter(brightness=(1.75, 1.75))(future_image)
            
            # 轻微的图像增强（避免影响目标生成质量）
            if self.image_aug and random.random() > 0.9:  # 只有10%概率
                future_image = transforms.ColorJitter(brightness=0.05, contrast=0.05)(future_image)
            
            # 图像填充
            if self.image_aspect_ratio == "pad":
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                        
                future_image = expand2square(
                    future_image, 
                    tuple(int(x * 255) for x in self.image_processor.image_mean)
                )
            
            # 最终预处理
            future_obs_tensor = self.image_processor.preprocess(
                future_image, 
                return_tensors="pt"
            )["pixel_values"][0]
            
            return future_obs_tensor
            
        except Exception as e:
            print(f"Error processing future observation image: {e}")
            return None

    def __len__(self) -> int:
        if self.use_hdf5:
            return len(self.hdf5_dataset)
        else:
            return self.num_chunks * self.chunk_size

    # train/dataset.py - 关键修复部分

    def __getitem__(self, index):
        """获取数据项,包含FLARE的动态未来观测"""
        while True:
            data_dict = None
            try:
                if self.use_hdf5:
                    res = self.hdf5_dataset.get_item()
                    content = res["meta"]
                    states = res["state"]
                    actions = res["actions"]
                    state_elem_mask = res["state_indicator"]
                    image_metas = [
                        res["cam_high"],
                        res["cam_high_mask"],
                        res["cam_right_wrist"],
                        res["cam_right_wrist_mask"],
                        res["cam_left_wrist"],
                        res["cam_left_wrist_mask"],
                    ]
                    state_std = res["state_std"]
                    state_mean = res["state_mean"]
                    state_norm = res["state_norm"]
                    
                    # 🔥 修复：从HDF5正确获取未来观测数据
                    future_obs_frame = res.get("future_obs_frame")
                    future_obs_mask = res.get("future_obs_mask", False)
                    future_step_id = res.get("future_step_id", -1)
                    # 只要有图像数据就认为有效，不再严格要求在action chunk范围内
                    if future_obs_frame is not None:
                        # 检查图像是否有有效内容
                        if hasattr(future_obs_frame, 'shape') and future_obs_frame.shape[0] > 0:
                            future_obs_mask = True  # 强制设为有效
                            #print(f"🔥 重新评估未来观测为有效: shape={future_obs_frame.shape}")
                        else:
                            print(f"⚠️  未来观测图像无效: shape={getattr(future_obs_frame, 'shape', 'None')}")
                    
                    
                else:
                    # 从buffer加载数据
                    loaded_data = self._safe_load(index)
                    
                    # 基础数据解析
                    base_data_count = 14  # 原有的数据字段数
                    (
                        content,
                        step_id,  # 这个很重要，用于计算未来观测
                        states,
                        _,  # state_chunk_time_mask
                        actions,
                        _,  # action_chunk_time_mask
                        state_elem_mask,
                        *image_metas,  # 图像相关数据
                        state_std,
                        state_mean,
                        state_norm,
                    ) = loaded_data[:base_data_count]
                    
                    # 提取未来观测：优先从预计算数据，否则动态计算
                    future_obs_frame, future_obs_mask = self._extract_future_obs_from_chunk_data(loaded_data)
                    
                    # 如果没有预计算的未来观测，则动态计算
                    if future_obs_frame is None and self.enable_future_obs:
                        future_obs_frame, future_obs_mask = self._compute_future_obs_from_episode_data(
                            content, image_metas, step_id
                        )

                # 构建数据字典
                data_dict = {}
                data_dict["dataset_name"] = content["dataset_name"]
                data_dict["data_idx"] = self.dataset_name2id[data_dict["dataset_name"]]
                data_dict["ctrl_freq"] = (self.control_freq[data_dict["dataset_name"]]
                                        if random.random() > self.cond_mask_prob else 0)

                # 状态噪声处理（保持原逻辑）
                if self.state_noise_snr is not None:
                    states += np.random.normal(
                        0.0,
                        state_std / np.sqrt(10**(self.state_noise_snr / 10)),
                        states.shape,
                    )
                    
                ds_state_mean = np.array(self.dataset_stat[data_dict["dataset_name"]]["state_mean"])
                ds_state_mean = np.tile(ds_state_mean[None], (states.shape[0], 1))
                data_dict["states"] = (states if random.random() > self.cond_mask_prob else ds_state_mean)
                data_dict["actions"] = actions
                data_dict["state_elem_mask"] = (state_elem_mask if random.random() > self.cond_mask_prob else
                                                np.zeros_like(state_elem_mask))
                data_dict["state_norm"] = state_norm

                # 处理历史图像（保持原有逻辑）
                background_color = np.array(
                    [int(x * 255) for x in self.image_processor.image_mean],
                    dtype=np.uint8,
                ).reshape(1, 1, 3)
                background_image = (np.ones(
                    (
                        self.image_processor.size["height"],
                        self.image_processor.size["width"],
                        3,
                    ),
                    dtype=np.uint8,
                ) * background_color)

                image_metas = list(self.pairwise(image_metas))
                mask_probs = [self.cond_mask_prob] * self.num_cameras
                if self.cam_ext_mask_prob >= 0.0:
                    mask_probs[0] = self.cam_ext_mask_prob
                    
                rearranged_images = []
                for i in range(self.img_history_size):
                    for j in range(self.num_cameras):
                        images, image_mask = image_metas[j]
                        image, valid = images[i], image_mask[i]
                        if (valid and (math.prod(image.shape) > 0) and (random.random() > mask_probs[j])):
                            rearranged_images.append((image, True))
                        else:
                            rearranged_images.append((background_image.copy(), False))

                preprocessed_images = []
                processor = self.image_processor
                for image, valid in rearranged_images:
                    image = Image.fromarray(image)
                    if self.image_size is not None:
                        image = transforms.Resize(self.image_size)(image)

                    if valid and self.auto_adjust_image_brightness:
                        pixel_values = list(image.getdata())
                        average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                        if average_brightness <= 0.15:
                            image = transforms.ColorJitter(brightness=(1.75, 1.75))(image)

                    # 图像增强
                    if valid and self.image_aug and (random.random() > 0.5):
                        aug_type = random.choice(["corrput_only", "color_only", "both"])
                        if aug_type != "corrput_only":
                            image = transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5,
                                                        hue=0.03)(image)
                        if aug_type != "color_only":
                            image = image_corrupt(image)

                    # 图像填充
                    if self.image_aspect_ratio == "pad":
                        def expand2square(pil_img, background_color):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result

                        image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                    preprocessed_images.append(image)
                data_dict["images"] = preprocessed_images

                # 🔥 FLARE核心：处理未来观测图像
                future_obs_image = None
                use_future_obs = (self.enable_future_obs and 
                                future_obs_frame is not None and 
                                future_obs_mask)
                
                if use_future_obs:
                    future_obs_image = self._process_future_obs_image(future_obs_frame)
                    

                data_dict["future_obs_image"] = future_obs_image
                data_dict["has_future_obs"] = use_future_obs

                # 处理文本指令
                text_instruction = content.get("instruction", "")
                if isinstance(text_instruction, bytes):
                    text_instruction = text_instruction.decode('utf-8')
                data_dict["text_instruction"] = text_instruction

                # 处理语言指令（原有逻辑）
                if self.use_precomp_lang_embed:
                    if content["instruction"][-1] == ".":
                        content["instruction"] = content["instruction"][:-1]
                    data_dict["lang_embed"] = (torch.load(content["instruction"])
                                            if random.random() > self.cond_mask_prob else self.empty_lang_embed)
                else:
                    instruction = (text_instruction if random.random() > self.cond_mask_prob else "")
                    data_dict["input_ids"] = self.tokenizer(
                        instruction,
                        return_tensors="pt",
                        padding="longest",
                        truncation=False,
                    ).input_ids[0]

                    assert (
                        len(data_dict["input_ids"]) <= self.tokenizer_max_length
                    ), f"Instruction length {len(data_dict['input_ids'])} exceeds the maximum length {self.tokenizer_max_length}."

                # 转换numpy数组为tensor
                for k, v in data_dict.items():
                    if isinstance(v, np.ndarray):
                        data_dict[k] = torch.from_numpy(v)

                

                return data_dict
                
            except BaseException as e:
                if data_dict is not None:
                    print(
                        f"Error catched when processing sample from {data_dict.get('dataset_name')}:",
                        e,
                    )
                else:
                    print(f"Error catched when processing sample:", e)
                traceback.print_exc()
                index = (index + 1) % len(self)


class DataCollatorForVLAConsumerDatasetWithFLARE(object):
    """支持FLARE功能的数据收集器"""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, instances):
        batch = {
            "states": [],
            "actions": [],
            "state_elem_mask": [],
            "state_norm": [],
            "images": [],
            "data_indices": [],
            "ctrl_freqs": [],
            "future_obs_images": [],  # 未来观测图像
            "has_future_obs": [],     # 是否有有效的未来观测
            "text_instructions": [],  # 文本指令
        }
        input_ids = []
        lang_embeds = []
        lang_embed_lens = []

        for instance in instances:
            # 转换numpy数组为tensor
            keys_to_check = [
                "states",
                "actions", 
                "state_elem_mask",
                "state_norm",
            ]
            for key in keys_to_check:
                item = instance[key]
                if not isinstance(item, torch.Tensor):
                    item = torch.from_numpy(item)
                batch[key].append(item)

            if "input_ids" in instance:
                input_ids.append(instance["input_ids"])
            else:
                lang_embeds.append(instance["lang_embed"])
                lang_embed_lens.append(instance["lang_embed"].shape[0])

            batch["images"].append(torch.stack(instance["images"], dim=0))
            batch["data_indices"].append(instance["data_idx"])
            batch["ctrl_freqs"].append(instance["ctrl_freq"])
            batch["text_instructions"].append(instance.get("text_instruction", ""))
            batch["has_future_obs"].append(instance.get("has_future_obs", False))
            
            # 处理未来观测图像
            future_obs_image = instance.get("future_obs_image")
            # 保证 future_obs_image 一定是 [3, H, W]
            if future_obs_image is not None:
                if isinstance(future_obs_image, torch.Tensor):
                    if future_obs_image.ndim == 4:
                        # 偶尔多 unsqueeze 了一下
                        future_obs_image = future_obs_image.squeeze(0)
                    assert future_obs_image.ndim == 3, f"future_obs_image shape 错误: {future_obs_image.shape}"
                    batch["future_obs_images"].append(future_obs_image)
                else:
                    # 万一是 numpy
                    batch["future_obs_images"].append(torch.from_numpy(future_obs_image))
            else:
                dummy_shape = instance["images"][0].shape  # [3, H, W]
                batch["future_obs_images"].append(torch.zeros(dummy_shape, dtype=batch["images"][0].dtype))

        # images: [B, img_history_size, 3, H, W]
        batch["images"] = torch.stack(batch["images"], dim=0)
        # future_obs_images: [B, 3, H, W]
        batch["future_obs_images"] = torch.stack(batch["future_obs_images"], dim=0)
        assert batch["future_obs_images"].ndim == 4, f"final future_obs_images shape: {batch['future_obs_images'].shape}"

        # 其余字段
        for key in ["states", "actions", "state_elem_mask", "state_norm"]:
            batch[key] = torch.stack(batch[key], dim=0)
        batch["ctrl_freqs"] = torch.tensor(batch["ctrl_freqs"])
        batch["has_future_obs"] = torch.tensor(batch["has_future_obs"], dtype=torch.bool)

        # 语言
        if len(input_ids) > 0:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            )
            batch["input_ids"] = input_ids
            batch["lang_attn_mask"] = input_ids.ne(self.tokenizer.pad_token_id)
        else:
            lang_embeds = torch.nn.utils.rnn.pad_sequence(lang_embeds, batch_first=True, padding_value=0)
            input_lang_attn_mask = torch.zeros(lang_embeds.shape[0], lang_embeds.shape[1], dtype=torch.bool)
            for i, l in enumerate(lang_embed_lens):
                input_lang_attn_mask[i, :l] = True
            batch["lang_embeds"] = lang_embeds
            batch["lang_attn_mask"] = input_lang_attn_mask

        return batch