
import os
import fnmatch
import json

import h5py
import yaml
import cv2
import numpy as np

from configs.state_vec import STATE_VEC_IDX_MAPPING


class HDF5VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """

    def __init__(self, model_config_path, verbose=False) -> None:
        # [Modify] The path to the HDF5 dataset directory
        # Each HDF5 file contains one episode
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)
        HDF5_DIR = model_config["data_path"]
        self.DATASET_NAME = "agilex"
        self.verbose = verbose  # 控制调试输出

        self.file_paths = []
        for root, _, files in os.walk(HDF5_DIR):
            for filename in fnmatch.filter(files, "*.hdf5"):
                file_path = os.path.join(root, filename)
                self.file_paths.append(file_path)

        if self.verbose:
            print(f"📁 找到 {len(self.file_paths)} 个HDF5文件")

        # Load the config
        with open("configs/base.yaml", "r") as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config["common"]["action_chunk_size"]
        self.IMG_HISORY_SIZE = config["common"]["img_history_size"]
        self.STATE_DIM = config["common"]["state_dim"]

        # Get each episode's len
        episode_lens = []
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res["state"].shape[0] if valid else 0
            episode_lens.append(_len)
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)

        if self.verbose:
            print(f"⚙️ 配置: CHUNK_SIZE={self.CHUNK_SIZE}, IMG_HISTORY_SIZE={self.IMG_HISORY_SIZE}")

    def __len__(self):
        return len(self.file_paths)

    def get_dataset_name(self):
        return self.DATASET_NAME

    def get_item(self, index: int = None, state_only=False):
        """Get a training sample at a random timestep."""
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            
            try:
                if not state_only:
                    result = self.parse_hdf5_file(file_path)
                else:
                    result = self.parse_hdf5_file_state_only(file_path)
                
                if result is None:
                    if self.verbose:
                        print(f"❌ 文件解析失败: {os.path.basename(file_path)}")
                    index = np.random.randint(0, len(self.file_paths))
                    continue
                    
                valid, sample = result
                if valid:
                    return sample
                else:
                    if self.verbose:
                        print(f"⚠️ 样本无效: {os.path.basename(file_path)}")
                    index = np.random.randint(0, len(self.file_paths))
            except Exception as e:
                if self.verbose:
                    print(f"❌ 处理异常: {os.path.basename(file_path)} - {str(e)[:50]}")
                index = np.random.randint(0, len(self.file_paths))

    def _get_arm_dimension(self, dim_data):
        """安全地获取臂维度信息"""
        if isinstance(dim_data, np.ndarray):
            if dim_data.ndim == 0:  # 0维数组（标量）
                return int(dim_data)
            else:  # 多维数组，取第一个值
                return int(dim_data.flat[0])
        else:
            return int(dim_data)

    def parse_hdf5_file(self, file_path):
        """Parse a hdf5 file to generate a training sample at a random timestep."""
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                if self.verbose:
                    print(f"❌ 文件不存在: {file_path}")
                return False, None
                
            with h5py.File(file_path, "r") as f:
                # 检查必要的数据是否存在
                required_keys = ["observations", "action"]
                for key in required_keys:
                    if key not in f:
                        if self.verbose:
                            print(f"❌ 缺少必要的键: {key}")
                        return False, None
                
                if "qpos" not in f["observations"]:
                    if self.verbose:
                        print(f"❌ 缺少 observations/qpos")
                    return False, None
                    
                qpos = f["observations"]["qpos"][:]
                
                # 修复：安全地读取维度信息
                left_arm_dim = self._get_arm_dimension(f["observations"]["left_arm_dim"][:])
                right_arm_dim = self._get_arm_dimension(f["observations"]["right_arm_dim"][:])
                
                num_steps = qpos.shape[0]
                
                # [Optional] We drop too-short episode
                if num_steps < 10:
                    if self.verbose:
                        print(f"❌ Episode 太短: {num_steps}")
                    return False, None

                # [Optional] We skip the first few still steps
                EPS = 1e-2
                qpos_delta = np.abs(qpos - qpos[0:1])
                indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
                if len(indices) > 0:
                    first_idx = indices[0]
                else:
                    if self.verbose:
                        print(f"❌ 找不到超过阈值的 qpos")
                    return False, None

                # We randomly sample a timestep
                if first_idx - 1 >= num_steps:
                    if self.verbose:
                        print(f"❌ first_idx ({first_idx}) 超出范围")
                    return False, None
                    
                step_id = np.random.randint(max(0, first_idx - 1), num_steps)

                # Load the instruction
                dir_path = os.path.dirname(file_path)
                instructions_path = os.path.join(dir_path, "instructions")
                
                if not os.path.exists(instructions_path):
                    if self.verbose:
                        print(f"❌ Instructions 目录不存在: {instructions_path}")
                    return False, None
                    
                instructions_names = []
                for filename in os.listdir(instructions_path):
                    if filename.endswith(".pt"):
                        instructions_names.append(os.path.join(instructions_path, filename))
                
                if not instructions_names:
                    if self.verbose:
                        print(f"❌ 没有找到 .pt 指令文件")
                    return False, None
                    
                instruction = np.random.choice(instructions_names)
                
                # Assemble the meta
                meta = {
                    "dataset_name": self.DATASET_NAME,
                    "#steps": num_steps,
                    "step_id": step_id,
                    "instruction": instruction,
                }

                # Rescale gripper to [0, 1]
                total_dim = left_arm_dim + 1 + right_arm_dim + 1
                qpos = qpos / np.array([[1 for i in range(total_dim)]])
                
                available_actions = f["action"][step_id:step_id + self.CHUNK_SIZE]
                action_chunk_end_step = step_id + available_actions.shape[0] - 1  # 实际动作序列的最后一步

                # 计算未来观测应该对应的步骤
                if available_actions.shape[0] == self.CHUNK_SIZE:
                    # 完整的action chunk，未来观测是chunk的最后一步
                    future_step_id = step_id + self.CHUNK_SIZE - 1
                    has_valid_future = True
                else:
                    # 不完整的action chunk，未来观测是episode的最后一步
                    future_step_id = num_steps - 1
                    has_valid_future = True  # 仍然有效，只是使用最后一帧
                    
                target_qpos = available_actions / np.array([[1 for i in range(total_dim)]])

                # Parse the state and action
                state = qpos[step_id:step_id + 1]
                state_std = np.std(qpos, axis=0)
                state_mean = np.mean(qpos, axis=0)
                state_norm = np.sqrt(np.mean(qpos**2, axis=0))
                actions = target_qpos
                
                if actions.shape[0] < self.CHUNK_SIZE:
                    # Pad the actions using the last action
                    actions = np.concatenate(
                        [
                            actions,
                            np.tile(actions[-1:], (self.CHUNK_SIZE - actions.shape[0], 1)),
                        ],
                        axis=0,
                    )

                # Fill the state/action into the unified vector
                def fill_in_state(values):
                    UNI_STATE_INDICES = (
                        [STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"]
                         for i in range(left_arm_dim)] + [STATE_VEC_IDX_MAPPING["left_gripper_open"]] +
                        [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"]
                         for i in range(right_arm_dim)] + [STATE_VEC_IDX_MAPPING["right_gripper_open"]])
                    uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM, ))
                    uni_vec[..., UNI_STATE_INDICES] = values
                    return uni_vec

                state = fill_in_state(state)
                state_indicator = fill_in_state(np.ones_like(state_std))
                state_std = fill_in_state(state_std)
                state_mean = fill_in_state(state_mean)
                state_norm = fill_in_state(state_norm)
                actions = fill_in_state(actions)

                # 修复图像解析函数 - 支持指定步骤（移除调试输出）
                def parse_img(key, target_step=None):
                    """解析图像，支持指定步骤或历史序列"""
                    try:
                        if key not in f["observations"]["images"]:
                            # 返回默认零图像，不输出警告
                            if target_step is not None:
                                return np.zeros((480, 640, 3), dtype=np.uint8)
                            else:
                                return np.zeros((self.IMG_HISORY_SIZE, 480, 640, 3), dtype=np.uint8)
                        
                        if target_step is not None:
                            # 解析单个步骤的图像
                            if 0 <= target_step < num_steps:
                                img_bits = f["observations"]["images"][key][target_step]
                            else:
                                # 如果步骤超出范围，返回最后一帧
                                img_bits = f["observations"]["images"][key][num_steps - 1]
                            
                            img = cv2.imdecode(np.frombuffer(img_bits, np.uint8), cv2.IMREAD_COLOR)
                            if img is None:
                                return np.zeros((480, 640, 3), dtype=np.uint8)
                            return img
                        else:
                            # 解析历史图像序列
                            imgs = []
                            for i in range(max(step_id - self.IMG_HISORY_SIZE + 1, 0), step_id + 1):
                                img_bits = f["observations"]["images"][key][i]
                                img = cv2.imdecode(np.frombuffer(img_bits, np.uint8), cv2.IMREAD_COLOR)
                                if img is None:
                                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                                imgs.append(img)
                            
                            imgs = np.stack(imgs)
                            if imgs.shape[0] < self.IMG_HISORY_SIZE:
                                # Pad the images using the first image
                                imgs = np.concatenate(
                                    [
                                        np.tile(
                                            imgs[:1],
                                            (self.IMG_HISORY_SIZE - imgs.shape[0], 1, 1, 1),
                                        ),
                                        imgs,
                                    ],
                                    axis=0,
                                )
                            return imgs
                    except Exception:
                        # 静默处理异常
                        if target_step is not None:
                            return np.zeros((480, 640, 3), dtype=np.uint8)
                        else:
                            return np.zeros((self.IMG_HISORY_SIZE, 480, 640, 3), dtype=np.uint8)

                # 计算未来观测帧的步骤ID（移除调试输出）
                future_obs_frame = parse_img("cam_high", target_step=future_step_id)

                # Parse the images (历史图像)
                cam_high = parse_img("cam_high")
                # For step_id = first_idx - 1, the valid_len should be one
                valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
                cam_high_mask = np.array([False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len)
                cam_left_wrist = parse_img("cam_left_wrist")
                cam_left_wrist_mask = cam_high_mask.copy()
                cam_right_wrist = parse_img("cam_right_wrist")
                cam_right_wrist_mask = cam_high_mask.copy()

                # Return the resulting sample
                result = {
                    "meta": meta,
                    "state": state,
                    "state_std": state_std,
                    "state_mean": state_mean,
                    "state_norm": state_norm,
                    "actions": actions,
                    "state_indicator": state_indicator,
                    "cam_high": cam_high,
                    "cam_high_mask": cam_high_mask,
                    "cam_left_wrist": cam_left_wrist,
                    "cam_left_wrist_mask": cam_left_wrist_mask,
                    "cam_right_wrist": cam_right_wrist,
                    "cam_right_wrist_mask": cam_right_wrist_mask,
                    # FLARE新增字段
                    "future_obs_frame": future_obs_frame,
                    "future_obs_mask": has_valid_future,
                    "future_step_id": future_step_id,
                }
                
                return True, result
                
        except Exception as e:
            if self.verbose:
                print(f"❌ 解析文件异常: {os.path.basename(file_path)} - {str(e)[:50]}")
            return False, None

    def parse_hdf5_file_state_only(self, file_path):
        """Parse a hdf5 file to generate a state trajectory."""
        try:
            with h5py.File(file_path, "r") as f:
                qpos = f["observations"]["qpos"][:]
                left_arm_dim = self._get_arm_dimension(f["observations"]["left_arm_dim"][:])
                right_arm_dim = self._get_arm_dimension(f["observations"]["right_arm_dim"][:])

                num_steps = qpos.shape[0]

                # [Optional] We skip the first few still steps
                EPS = 1e-2
                qpos_delta = np.abs(qpos - qpos[0:1])
                indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
                if len(indices) > 0:
                    first_idx = indices[0]
                else:
                    raise ValueError("Found no qpos that exceeds the threshold.")

                # Rescale gripper to [0, 1]
                qpos = qpos / np.array([[1 for i in range(left_arm_dim + right_arm_dim + 2)]])
                target_qpos = f["action"][:] / np.array([[1 for i in range(left_arm_dim + right_arm_dim + 2)]])

                # Parse the state and action
                state = qpos[first_idx - 1:]
                action = target_qpos[first_idx - 1:]

                # Fill the state/action into the unified vector
                def fill_in_state(values):
                    UNI_STATE_INDICES = (
                        [STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"]
                         for i in range(left_arm_dim)] + [STATE_VEC_IDX_MAPPING["left_gripper_open"]] +
                        [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"]
                         for i in range(right_arm_dim)] + [STATE_VEC_IDX_MAPPING["right_gripper_open"]])
                    uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM, ))
                    uni_vec[..., UNI_STATE_INDICES] = values
                    return uni_vec

                state = fill_in_state(state)
                action = fill_in_state(action)

                return True, {"state": state, "action": action}
        except Exception:
            return False, None


if __name__ == "__main__":
    # 测试时开启详细输出
    ds = HDF5VLADataset("model_config/grab_roller_flare.yml", verbose=False)
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)