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


# 更新VLAConsumerDataset以支持未来观测
class VLAConsumerDatasetWithFLARE(Dataset):
    """
    支持FLARE功能的VLA消费者数据集，包含动态未来观测采样
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
    ):
        super(VLAConsumerDatasetWithFLARE, self).__init__()

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

    def _load_data_from_chunk_with_future_obs(self, chunk_dir, chunk_item_idx):
        """
        从chunk加载数据，包含未来观测信息
        """
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

    def __len__(self) -> int:
        if self.use_hdf5:
            return len(self.hdf5_dataset)
        else:
            return self.num_chunks * self.chunk_size

    def _safe_load_with_future_obs(self, index):
        """
        安全加载数据，包含未来观测处理
        """
        read_chunk_item_indices = []
        read_chunk_idx = index // self.chunk_size
        
        while len(read_chunk_item_indices) == 0:
            read_chunk_dir = os.path.join(self.buffer_dir, f"chunk_{read_chunk_idx}")
            try:
                from train.dataset import get_clean_item
                read_chunk_item_indices = get_clean_item(read_chunk_dir)
            except BaseException as e:
                print("Error catched when searching a clean chunk:", e)
                traceback.print_exc()
                read_chunk_item_indices = []
            read_chunk_idx = (read_chunk_idx + 1) % self.num_chunks

        random_item_index = index % len(read_chunk_item_indices)
        read_chunk_item_index = read_chunk_item_indices[random_item_index]

        # 修改dirty bit
        try:
            from train.dataset import read_dirty_bit, save_dirty_bit
            dirty_bit = read_dirty_bit(read_chunk_dir)
            dirty_bit[read_chunk_item_index] = 1
            save_dirty_bit(read_chunk_dir, dirty_bit)
        except BaseException as e:
            print("Error catched when modifying the dirty bit:", e)
            traceback.print_exc()

        # 加载样本
        try:
            content, meta = self._load_data_from_chunk_with_future_obs(read_chunk_dir, read_chunk_item_index)
            self.last_content, self.last_meta = content, meta
        except BaseException as e:
            print("Error catched when loading sample:", e)
            traceback.print_exc()
            content, meta = self.last_content, self.last_meta

        return (content, *meta)

    def __getitem__(self, index):
        """
        获取数据项，包含FLARE的未来观测处理
        """
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
                    # HDF5暂时不支持未来观测，设为None
                    future_obs_frame = None
                    future_obs_mask = False
                else:
                    # 从buffer加载，包含未来观测数据
                    loaded_data = self._safe_load_with_future_obs(index)
                    (
                        content,
                        _,
                        states,
                        _,
                        actions,
                        _,
                        state_elem_mask,
                        *image_metas,
                        state_std,
                        state_mean,
                        state_norm,
                    ) = loaded_data[:14]  # 原有数据
                    
                    # 检查是否有未来观测数据
                    if len(loaded_data) > 14:
                        future_obs_frame = loaded_data[14]
                        future_obs_mask = loaded_data[15] if len(loaded_data) > 15 else True
                    else:
                        future_obs_frame = None
                        future_obs_mask = False

                data_dict = {}
                data_dict["dataset_name"] = content["dataset_name"]
                data_dict["data_idx"] = self.dataset_name2id[data_dict["dataset_name"]]
                data_dict["ctrl_freq"] = (self.control_freq[data_dict["dataset_name"]]
                                          if random.random() > self.cond_mask_prob else 0)

                # 状态噪声处理
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

                # 处理图像（保持原有逻辑）
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

                # 处理未来观测图像
                future_obs_image = None
                if (self.enable_future_obs and future_obs_frame is not None and 
                    future_obs_mask and random.random() < self.future_obs_prob):
                    try:
                        # 处理未来观测帧
                        if math.prod(future_obs_frame.shape) > 0:
                            future_image = Image.fromarray(future_obs_frame)
                            if self.image_size is not None:
                                future_image = transforms.Resize(self.image_size)(future_image)
                                
                            # 应用相同的图像处理流程
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
                                future_image = expand2square(future_image, tuple(int(x * 255) for x in processor.image_mean))
                                
                            future_obs_image = processor.preprocess(future_image, return_tensors="pt")["pixel_values"][0]
                    except Exception as e:
                        print(f"Error processing future observation: {e}")
                        future_obs_image = None

                data_dict["future_obs_image"] = future_obs_image

                # 处理语言指令
                if self.use_precomp_lang_embed:
                    if content["instruction"][-1] == ".":
                        content["instruction"] = content["instruction"][:-1]
                    data_dict["lang_embed"] = (torch.load(content["instruction"])
                                               if random.random() > self.cond_mask_prob else self.empty_lang_embed)
                else:
                    instruction = (content["instruction"] if random.random() > self.cond_mask_prob else "")
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

                for k, v in data_dict.items():
                    assert not isinstance(v, np.ndarray), f"key: {k}, value: {v}"

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
    """
    支持FLARE功能的数据收集器
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = {
            "states": [],
            "actions": [],
            "state_elem_mask": [],
            "state_norm": [],
            "images": [],
            "data_indices": [],
            "ctrl_freqs": [],
            "future_obs_images": [],  # 新增
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
                if isinstance(instance[key], torch.Tensor):
                    item = instance[key]
                else:
                    item = torch.from_numpy(instance[key])
                batch[key].append(item)

            if "input_ids" in instance:
                input_ids.append(instance["input_ids"])
            else:
                lang_embeds.append(instance["lang_embed"])
                lang_embed_lens.append(instance["lang_embed"].shape[0])

            batch["images"].append(torch.stack(instance["images"], dim=0))
            batch["data_indices"].append(instance["data_idx"])
            batch["ctrl_freqs"].append(instance["ctrl_freq"])
            
            # 处理未来观测图像
            future_obs_image = instance.get("future_obs_image")
            if future_obs_image is not None:
                batch["future_obs_images"].append(future_obs_image)
            else:
                # 使用零张量作为占位符
                batch["future_obs_images"].append(torch.zeros_like(instance["images"][0]))

        # 堆叠数据
        keys_to_stack = ["states", "actions", "state_elem_mask", "state_norm", "images", "future_obs_images"]
        for key in keys_to_stack:
            batch[key] = torch.stack(batch[key], dim=0)

        batch["ctrl_freqs"] = torch.tensor(batch["ctrl_freqs"])

        # 处理语言数据
        if len(input_ids) > 0:
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                        batch_first=True,
                                                        padding_value=self.tokenizer.pad_token_id)
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