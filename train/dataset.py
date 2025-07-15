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
        future_obs_prob=1,  # 使用未来观测的概率
        action_chunk_size=32,  # 动作块大小
        future_obs_consistency_check=True
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

        # 添加一致性检查参数
        self.future_obs_consistency_check = future_obs_consistency_check
        # 记录统计信息
        self.future_obs_stats = {
            'total_samples': 0,
            'valid_future_obs': 0,
            'invalid_future_obs': 0,
            'synthetic_future_obs': 0
        }
        
        
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
    def _get_buffer_sample_data(self, index):
        """从buffer获取数据 - 简单修复"""
        loaded_data = self._safe_load(index)
        
        # 基础数据解析
        (
            content,
            step_id,
            states,
            _,  # state_chunk_time_mask
            actions,
            _,  # action_chunk_time_mask
            state_elem_mask,
            *image_metas,
            state_std,
            state_mean,
            state_norm,
        ) = loaded_data[:14]  # 取前14个字段

        # 提取未来观测
        future_obs_frame, future_obs_mask = self._extract_future_obs_from_chunk_data(loaded_data)
        
        # 如果没有预计算的未来观测，动态计算
        if future_obs_frame is None and self.enable_future_obs:
            future_obs_frame, future_obs_mask = self._compute_future_obs_from_episode_data(
                content, image_metas, step_id
            )

        # 验证未来观测质量
        valid_future_obs = self._validate_future_obs(
            future_obs_frame, future_obs_mask, step_id, content
        )
        
        if not valid_future_obs and self.enable_future_obs:
            # 尝试使用最后一帧
            try:
                main_camera_images = image_metas[0]
                if len(main_camera_images) > 0:
                    future_obs_frame = main_camera_images[-1]
                    valid_future_obs = self._validate_future_obs(future_obs_frame, True, -1, content)
                    if valid_future_obs:
                        self.future_obs_stats['synthetic_future_obs'] += 1
            except:
                valid_future_obs = False

        # 更新统计
        if valid_future_obs:
            self.future_obs_stats['valid_future_obs'] += 1
        else:
            self.future_obs_stats['invalid_future_obs'] += 1

        # 构建数据字典
        data_dict = self._build_data_dict(
            content, states, actions, state_elem_mask, image_metas,
            state_std, state_mean, state_norm,
            future_obs_frame, valid_future_obs
        )
        
        return data_dict
    def _build_data_dict(self, content, states, actions, state_elem_mask, image_metas,
                     state_std, state_mean, state_norm, future_obs_frame, has_future_obs):
    """构建数据字典 - 简单实现"""
    
        data_dict = {}
        data_dict["dataset_name"] = content["dataset_name"]
        data_dict["data_idx"] = self.dataset_name2id.get(data_dict["dataset_name"], 0)
        data_dict["ctrl_freq"] = self.control_freq.get(data_dict["dataset_name"], 25)

        # 基础数据
        data_dict["states"] = states
        data_dict["actions"] = actions
        data_dict["state_elem_mask"] = state_elem_mask
        data_dict["state_norm"] = state_norm

        # 处理历史图像（保持您的原有逻辑）
        background_color = np.array([int(x * 255) for x in self.image_processor.image_mean], dtype=np.uint8).reshape(1, 1, 3)
        background_image = np.ones((self.image_processor.size["height"], self.image_processor.size["width"], 3), dtype=np.uint8) * background_color

        image_metas = list(self.pairwise(image_metas))
        mask_probs = [self.cond_mask_prob] * self.num_cameras
        if self.cam_ext_mask_prob >= 0.0:
            mask_probs[0] = self.cam_ext_mask_prob

        rearranged_images = []
        for i in range(self.img_history_size):
            for j in range(self.num_cameras):
                if j < len(image_metas):
                    images, image_mask = image_metas[j]
                    if i < len(images) and i < len(image_mask):
                        image, valid = images[i], image_mask[i]
                        if valid and math.prod(image.shape) > 0 and random.random() > mask_probs[j]:
                            rearranged_images.append(image)
                        else:
                            rearranged_images.append(background_image.copy())
                    else:
                        rearranged_images.append(background_image.copy())
                else:
                    rearranged_images.append(background_image.copy())

        # 预处理图像
        preprocessed_images = []
        for image in rearranged_images:
            processed_image = self._preprocess_single_image(image)
            preprocessed_images.append(processed_image)
        data_dict["images"] = preprocessed_images

        # 处理未来观测图像
        future_obs_image = None
        if has_future_obs and future_obs_frame is not None:
            future_obs_image = self._process_future_obs_image(future_obs_frame)
            if future_obs_image is None:
                has_future_obs = False

        data_dict["future_obs_image"] = future_obs_image
        data_dict["has_future_obs"] = has_future_obs

        # 处理文本指令
        text_instruction = content.get("instruction", "")
        if isinstance(text_instruction, bytes):
            text_instruction = text_instruction.decode('utf-8')
        data_dict["text_instruction"] = text_instruction

        # 语言嵌入处理
        if self.use_precomp_lang_embed:
            try:
                lang_embed = torch.load(content["instruction"]) if random.random() > self.cond_mask_prob else self.empty_lang_embed
                data_dict["lang_embed"] = lang_embed
            except:
                data_dict["lang_embed"] = self.empty_lang_embed
        else:
            instruction = text_instruction if random.random() > self.cond_mask_prob else ""
            tokenized = self.tokenizer(instruction, return_tensors="pt", padding="longest", truncation=True, max_length=self.tokenizer_max_length)
            data_dict["input_ids"] = tokenized.input_ids[0]

        # 转换为tensor
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                data_dict[k] = torch.from_numpy(v)

        return data_dict
    def _preprocess_single_image(self, image):
        """预处理单个图像 - 简单实现"""
        try:
            image = Image.fromarray(image)
            
            if self.image_size is not None:
                image = transforms.Resize(self.image_size)(image)

            if self.image_aspect_ratio == "pad":
                # 简单的正方形填充
                background_color = tuple(int(x * 255) for x in self.image_processor.image_mean)
                width, height = image.size
                if width != height:
                    size = max(width, height)
                    result = Image.new(image.mode, (size, size), background_color)
                    result.paste(image, ((size - width) // 2, (size - height) // 2))
                    image = result

            image = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            return image
        except:
            # 失败时返回零张量
            return torch.zeros(3, 224, 224)
    
    def __getitem__(self, index):
        """
        增强的数据获取，确保未来观测质量和一致性
        """
        while True:
            try:
                data_dict = self._get_sample_data(index)
                
                # 验证数据完整性
                if self._validate_sample_data(data_dict):
                    self.future_obs_stats['total_samples'] += 1
                    return data_dict
                else:
                    # 数据无效，尝试下一个样本
                    index = (index + 1) % len(self)
                    continue
                    
            except Exception as e:
                print(f"Error in __getitem__: {e}")
                index = (index + 1) % len(self)
                continue
    
    def _get_sample_data(self, index):
        """获取样本数据的核心逻辑"""
        if self.use_hdf5:
            return self._get_hdf5_sample_data(index)
        else:
            return self._get_buffer_sample_data(index)
    
    def _get_hdf5_sample_data(self):
        """从HDF5获取数据 - 修复版本"""
        res = self.hdf5_dataset.get_item()
        
        # 基础数据
        content = res["meta"]
        states = res["state"] 
        actions = res["actions"]
        state_elem_mask = res["state_indicator"]
        
        # 完整的图像数据
        image_metas = [
            res["cam_high"], res["cam_high_mask"],
            res["cam_left_wrist"], res["cam_left_wrist_mask"],
            res["cam_right_wrist"], res["cam_right_wrist_mask"],
        ]
        
        state_std = res["state_std"]
        state_mean = res["state_mean"]
        state_norm = res["state_norm"]
        
        # 未来观测处理
        future_obs_frame = res.get("future_obs_frame")
        future_obs_mask = res.get("future_obs_mask", False)
        future_step_id = res.get("future_step_id", -1)
        
        # 验证未来观测
        valid_future_obs = self._validate_future_obs(future_obs_frame, future_obs_mask, future_step_id, content)
        
        if not valid_future_obs and self.enable_future_obs:
            # 尝试使用最后一帧
            try:
                main_camera_images = image_metas[0]
                if len(main_camera_images) > 0:
                    future_obs_frame = main_camera_images[-1]
                    valid_future_obs = self._validate_future_obs(future_obs_frame, True, -1, content)
                    if valid_future_obs:
                        self.future_obs_stats['synthetic_future_obs'] += 1
            except:
                valid_future_obs = False
        
        # 更新统计
        if valid_future_obs:
            self.future_obs_stats['valid_future_obs'] += 1
        else:
            self.future_obs_stats['invalid_future_obs'] += 1
        
        # 构建数据字典
        data_dict = self._build_data_dict(
            content, states, actions, state_elem_mask, image_metas,
            state_std, state_mean, state_norm,
            future_obs_frame, valid_future_obs
        )
        
        return data_dict
    
    def _validate_future_obs(self, future_obs_frame, future_obs_mask, future_step_id, content):
        """验证未来观测的质量"""
        if future_obs_frame is None:
            return False
        
        # 检查图像有效性
        if hasattr(future_obs_frame, 'shape'):
            # 检查形状
            if any(dim <= 0 for dim in future_obs_frame.shape):
                return False
            
            # 检查是否为空图像
            if hasattr(future_obs_frame, 'sum') and future_obs_frame.sum() == 0:
                return False
            
            # 检查图像内容是否合理 (避免全白或全黑)
            if hasattr(future_obs_frame, 'std'):
                std_val = future_obs_frame.std()
                if std_val < 1.0:  # 图像变化太小，可能是无效图像
                    return False
        
        # 如果启用一致性检查
        if self.future_obs_consistency_check:
            # 检查未来步骤ID是否合理
            current_step = content.get("#steps", 0)
            if future_step_id >= current_step:
                return False
        
        return True
    
    def _generate_synthetic_future_obs(self, res, content):
        """生成合成的未来观测"""
        try:
            # 使用最后一帧作为未来观测
            image_metas = [
                res["cam_high"],
                res["cam_high_mask"],
                # ... 其他摄像头
            ]
            
            main_camera_images = image_metas[0]
            if len(main_camera_images) > 0:
                # 使用最后一帧
                synthetic_frame = main_camera_images[-1]
                
                # 验证合成帧
                if self._validate_future_obs(synthetic_frame, True, -1, content):
                    return synthetic_frame, True
            
            return None, False
            
        except Exception as e:
            print(f"Failed to generate synthetic future obs: {e}")
            return None, False
    
    def _validate_sample_data(self, data_dict):
        """验证样本数据的完整性"""
        required_keys = [
            "states", "actions", "images", "text_instruction"
        ]
        
        # 检查必需字段
        for key in required_keys:
            if key not in data_dict:
                return False
        
        # 检查tensor形状
        try:
            states = data_dict["states"]
            actions = data_dict["actions"] 
            images = data_dict["images"]
            
            if states.shape[0] != 1:  # 状态应该是1个token
                return False
                
            if len(images) == 0:  # 必须有图像
                return False
                
        except Exception:
            return False
        
        return True
    
    def get_stats(self):
        """获取数据集统计信息"""
        total = self.future_obs_stats['total_samples']
        if total > 0:
            stats = {
                'total_samples': total,
                'valid_future_obs_ratio': self.future_obs_stats['valid_future_obs'] / total,
                'invalid_future_obs_ratio': self.future_obs_stats['invalid_future_obs'] / total,
                'synthetic_future_obs_ratio': self.future_obs_stats['synthetic_future_obs'] / total,
            }
        else:
            stats = self.future_obs_stats.copy()
        
        return stats
    

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
        valid_future_obs_count = 0
        total_count = len(instances)
        
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
            
            # 处理未来观测 (关键修复)
            future_obs_image = instance.get("future_obs_image")
            has_future_obs = instance.get("has_future_obs", False)
            
            # 验证未来观测质量
            if future_obs_image is not None and has_future_obs:
                # 双重验证
                if self._validate_future_obs_tensor(future_obs_image):
                    batch["future_obs_images"].append(future_obs_image)
                    batch["has_future_obs"].append(True)
                    valid_future_obs_count += 1
                else:
                    # 使用零填充
                    dummy_shape = instance["images"][0].shape
                    batch["future_obs_images"].append(torch.zeros(dummy_shape))
                    batch["has_future_obs"].append(False)
            else:
                # 使用零填充
                dummy_shape = instance["images"][0].shape
                batch["future_obs_images"].append(torch.zeros(dummy_shape))
                batch["has_future_obs"].append(False)

        # 批次质量检查
        batch["future_obs_images"] = torch.stack(batch["future_obs_images"], dim=0)
        batch["has_future_obs"] = torch.tensor(batch["has_future_obs"], dtype=torch.bool)
        
        # 记录批次统计
        batch["future_obs_ratio"] = valid_future_obs_count / total_count
        
        # 如果批次中未来观测太少，发出警告
        if valid_future_obs_count < total_count * 0.5:  # 少于50%
            print(f"Warning: Low future obs ratio in batch: {valid_future_obs_count}/{total_count}")
        
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
    def _validate_future_obs_tensor(self, tensor):
        """验证未来观测tensor的质量"""
        if tensor is None:
            return False
        
        try:
            # 检查形状
            if tensor.ndim != 3:  # 应该是 [C, H, W]
                return False
            
            # 检查数值范围 (假设是0-1或0-255)
            if tensor.min() < 0 or tensor.max() > 255:
                return False
                
            # 检查是否全零
            if tensor.sum() == 0:
                return False
                
            # 检查变化程度
            if tensor.std() < 0.01:  # 变化太小
                return False
                
            return True
            
        except Exception:
            return False
        