import json
import random
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import yaml

from data.episode_transform import (
    process_episode_with_future_obs,  # 使用新的处理函数
    flatten_episode_with_future_obs,  # 使用新的扁平化函数
    flatten_episode_agilex_with_future_obs,  # 使用新的agilex扁平化函数
    bgr_to_rgb,
)
from data.utils import dataset_to_path
from data.preprocess_scripts import *

# Producer does not need GPU
tf.config.set_visible_devices([], "GPU")

OPENX_EMBOD_DIR = "data/datasets/openx_embod"

DATASET_NAMES_NOOPENX = [
    "aloha_mobile",
    "aloha_static", 
    "roboset",
    "agilex",
    "rh20t",
    "calvin",
    "bridgev2",
]

# Read the config
with open("configs/base.yaml", "r") as file:
    config = yaml.safe_load(file)
EPSD_LEN_THRESH_LOW = config["dataset"]["epsd_len_thresh_low"]
EPSD_LEN_THRESH_HIGH = config["dataset"]["epsd_len_thresh_high"]
with open("configs/dataset_img_keys.json", "r") as file:
    IMAGE_KEYS = json.load(file)


class VLADatasetWithFLARE:
    """
    支持FLARE的VLA数据集类，包含未来观测采样功能
    """

    def __init__(self, seed, dataset_type, repeat=True, enable_future_obs=True):
        """
        Args:
            seed: 随机种子
            dataset_type: 'pretrain' 或 'finetune'
            repeat: 是否重复到无限长度
            enable_future_obs: 是否启用未来观测功能
        """
        dataset_names_cfg = ("configs/pretrain_datasets.json"
                             if dataset_type == "pretrain" else "configs/finetune_datasets.json")
        with open(dataset_names_cfg, "r") as file:
            DATASET_NAMES = json.load(file)
        self.dataset_names = DATASET_NAMES
        sample_weights_cfg = ("configs/pretrain_sample_weights.json"
                              if dataset_type == "pretrain" else "configs/finetune_sample_weights.json")
        with open(sample_weights_cfg, "r") as file:
            SAMPLE_WEIGHTS = json.load(file)
            
        self.openx_dir = OPENX_EMBOD_DIR
        self.epsd_len_thresh_low = EPSD_LEN_THRESH_LOW
        self.epsd_len_thresh_high = EPSD_LEN_THRESH_HIGH
        self.repeat = repeat
        self.enable_future_obs = enable_future_obs

        # Set the random seed
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # Weights of the each dataset in the collection to sample from
        sample_weights = []

        self.name2dataset = {}
        for dataset_name in self.dataset_names:
            if dataset_name in DATASET_NAMES_NOOPENX:
                dataset = globals()[dataset_name].load_dataset(seed)
            else:
                dataset_path = dataset_to_path(dataset_name, self.openx_dir)
                dataset = tfds.builder_from_directory(builder_dir=dataset_path)
                dataset = dataset.as_dataset(split="all", shuffle_files=True)

                # 数据集过滤（保持原有逻辑）
                if dataset_name == "kuka":
                    dataset = dataset.filter(lambda x: x["success"])
                elif dataset_name == "bc_z":
                    dataset = dataset.filter(lambda x: tf.math.greater(
                        next(iter(x["steps"]))["observation"]["episode_success"],
                        0.5,
                    ))
                elif (dataset_name == "ucsd_pick_and_place_dataset_converted_externally_to_rlds"):
                    dataset = dataset.filter(lambda x: x["episode_metadata"]["success"])
                elif (dataset_name == "utokyo_xarm_bimanual_converted_externally_to_rlds"):
                    dataset = dataset.filter(lambda x: tf.math.equal(
                        next(iter(x["steps"]))["language_instruction"],
                        tf.constant("Unfold a wrinkled towel."),
                    ))

            # 使用支持未来观测的处理函数
            if self.enable_future_obs:
                dataset = dataset.map(lambda x: process_episode_with_future_obs(
                    x,
                    dataset_name,
                    IMAGE_KEYS[dataset_name]["image_keys"],
                    IMAGE_KEYS[dataset_name]["image_mask"],
                ))
            else:
                # 保持原有处理逻辑
                from data.episode_transform import process_episode
                dataset = dataset.map(lambda x: process_episode(
                    x,
                    dataset_name,
                    IMAGE_KEYS[dataset_name]["image_keys"],
                    IMAGE_KEYS[dataset_name]["image_mask"],
                ))

            # Change BGR to RGB if needed
            if dataset_name == "fmb":
                dataset = dataset.map(bgr_to_rgb)

            if self.repeat:
                dataset = dataset.repeat()
            self.name2dataset[dataset_name] = iter(dataset)
            sample_weights.append(SAMPLE_WEIGHTS[dataset_name])
            
        # Normalize the sample weights
        sample_weights = np.array(sample_weights)
        self.sample_weights = sample_weights / np.sum(sample_weights)

    def __iter__(self):
        """
        为epoch采样episode批次
        """
        while True:
            dataset_name = np.random.choice(self.dataset_names, p=self.sample_weights)
            episode = next(self.name2dataset[dataset_name])
            
            # 根据数据集类型和是否启用未来观测选择处理函数
            if self.enable_future_obs:
                if dataset_name == "agilex":
                    episode_steps = flatten_episode_agilex_with_future_obs(episode)
                else:
                    episode_steps = flatten_episode_with_future_obs(episode)
            else:
                # 使用原有处理逻辑
                from data.episode_transform import flatten_episode, flatten_episode_agilex
                if dataset_name == "agilex":
                    episode_steps = flatten_episode_agilex(episode)
                else:
                    episode_steps = flatten_episode(episode)
                    
            # Filter too short
            if len(episode_steps) < self.epsd_len_thresh_low:
                continue
            # Randomly sample too long
            if len(episode_steps) > self.epsd_len_thresh_high:
                episode_steps = random.sample(episode_steps, self.epsd_len_thresh_high)

            yield episode_steps