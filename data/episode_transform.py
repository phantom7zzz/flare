import numpy as np
import tensorflow as tf
import yaml

from data.preprocess import generate_json_state
from configs.state_vec import STATE_VEC_IDX_MAPPING

# Read the config
with open("configs/base.yaml", "r") as file:
    config = yaml.safe_load(file)
# Load some constants from the config
IMG_HISTORY_SIZE = config["common"]["img_history_size"]
if IMG_HISTORY_SIZE < 1:
    raise ValueError("Config `img_history_size` must be at least 1.")
ACTION_CHUNK_SIZE = config["common"]["action_chunk_size"]
if ACTION_CHUNK_SIZE < 1:
    raise ValueError("Config `action_chunk_size` must be at least 1.")


@tf.function
def process_episode_with_future_obs(epsd: dict, dataset_name: str, image_keys: list, image_mask: list) -> dict:
    """
    处理episode以提取frames和json内容，同时支持未来观测
    """
    # Frames of each camera (保持原有逻辑)
    frames_0 = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)
    frames_1 = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)
    frames_2 = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)
    frames_3 = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)
    
    # 新增：存储所有时刻的主摄像头图像（用于未来观测采样）
    all_main_camera_frames = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)
    
    # Traverse the episode to collect...
    for step in iter(epsd["steps"]):
        # Parse the image
        frames_0 = frames_0.write(
            frames_0.size(),
            tf.cond(
                tf.equal(image_mask[0], 1),
                lambda: step["observation"][image_keys[0]],
                lambda: tf.zeros([0, 0, 0], dtype=tf.uint8),
            ),
        )
        frames_1 = frames_1.write(
            frames_1.size(),
            tf.cond(
                tf.equal(image_mask[1], 1),
                lambda: step["observation"][image_keys[1]],
                lambda: tf.zeros([0, 0, 0], dtype=tf.uint8),
            ),
        )
        frames_2 = frames_2.write(
            frames_2.size(),
            tf.cond(
                tf.equal(image_mask[2], 1),
                lambda: step["observation"][image_keys[2]],
                lambda: tf.zeros([0, 0, 0], dtype=tf.uint8),
            ),
        )
        frames_3 = frames_3.write(
            frames_3.size(),
            tf.cond(
                tf.equal(image_mask[3], 1),
                lambda: step["observation"][image_keys[3]],
                lambda: tf.zeros([0, 0, 0], dtype=tf.uint8),
            ),
        )
        
        # 新增：收集主摄像头图像（通常是第一个摄像头）
        main_camera_frame = tf.cond(
            tf.equal(image_mask[0], 1),
            lambda: step["observation"][image_keys[0]],
            lambda: tf.zeros([0, 0, 0], dtype=tf.uint8),
        )
        all_main_camera_frames = all_main_camera_frames.write(
            all_main_camera_frames.size(),
            main_camera_frame
        )

    # Stack all main camera frames
    all_main_frames = all_main_camera_frames.stack()

    # Calculate the past_frames_0 for each step (保持原有逻辑)
    frames_0 = frames_0.stack()
    first_frame = tf.expand_dims(frames_0[0], axis=0)
    first_frame = tf.repeat(first_frame, IMG_HISTORY_SIZE - 1, axis=0)
    padded_frames_0 = tf.concat([first_frame, frames_0], axis=0)
    indices = tf.range(IMG_HISTORY_SIZE, tf.shape(frames_0)[0] + IMG_HISTORY_SIZE)
    past_frames_0 = tf.map_fn(lambda i: padded_frames_0[i - IMG_HISTORY_SIZE:i], indices, dtype=tf.uint8)
    frames_0_time_mask = tf.ones([tf.shape(frames_0)[0]], dtype=tf.bool)
    padded_frames_0_time_mask = tf.pad(
        frames_0_time_mask,
        [[IMG_HISTORY_SIZE - 1, 0]],
        "CONSTANT",
        constant_values=False,
    )
    past_frames_0_time_mask = tf.map_fn(
        lambda i: padded_frames_0_time_mask[i - IMG_HISTORY_SIZE:i],
        indices,
        dtype=tf.bool,
    )

    # For past_frames_1 (保持原有逻辑)
    frames_1 = frames_1.stack()
    first_frame = tf.expand_dims(frames_1[0], axis=0)
    first_frame = tf.repeat(first_frame, IMG_HISTORY_SIZE - 1, axis=0)
    padded_frames_1 = tf.concat([first_frame, frames_1], axis=0)
    past_frames_1 = tf.map_fn(lambda i: padded_frames_1[i - IMG_HISTORY_SIZE:i], indices, dtype=tf.uint8)
    frames_1_time_mask = tf.ones([tf.shape(frames_1)[0]], dtype=tf.bool)
    padded_frames_1_time_mask = tf.pad(
        frames_1_time_mask,
        [[IMG_HISTORY_SIZE - 1, 0]],
        "CONSTANT",
        constant_values=False,
    )
    past_frames_1_time_mask = tf.map_fn(
        lambda i: padded_frames_1_time_mask[i - IMG_HISTORY_SIZE:i],
        indices,
        dtype=tf.bool,
    )

    # For past_frames_2 (保持原有逻辑)
    frames_2 = frames_2.stack()
    first_frame = tf.expand_dims(frames_2[0], axis=0)
    first_frame = tf.repeat(first_frame, IMG_HISTORY_SIZE - 1, axis=0)
    padded_frames_2 = tf.concat([first_frame, frames_2], axis=0)
    past_frames_2 = tf.map_fn(lambda i: padded_frames_2[i - IMG_HISTORY_SIZE:i], indices, dtype=tf.uint8)
    frames_2_time_mask = tf.ones([tf.shape(frames_2)[0]], dtype=tf.bool)
    padded_frames_2_time_mask = tf.pad(
        frames_2_time_mask,
        [[IMG_HISTORY_SIZE - 1, 0]],
        "CONSTANT",
        constant_values=False,
    )
    past_frames_2_time_mask = tf.map_fn(
        lambda i: padded_frames_2_time_mask[i - IMG_HISTORY_SIZE:i],
        indices,
        dtype=tf.bool,
    )

    # For past_frames_3 (保持原有逻辑)
    frames_3 = frames_3.stack()
    first_frame = tf.expand_dims(frames_3[0], axis=0)
    first_frame = tf.repeat(first_frame, IMG_HISTORY_SIZE - 1, axis=0)
    padded_frames_3 = tf.concat([first_frame, frames_3], axis=0)
    past_frames_3 = tf.map_fn(lambda i: padded_frames_3[i - IMG_HISTORY_SIZE:i], indices, dtype=tf.uint8)
    frames_3_time_mask = tf.ones([tf.shape(frames_3)[0]], dtype=tf.bool)
    padded_frames_3_time_mask = tf.pad(
        frames_3_time_mask,
        [[IMG_HISTORY_SIZE - 1, 0]],
        "CONSTANT",
        constant_values=False,
    )
    past_frames_3_time_mask = tf.map_fn(
        lambda i: padded_frames_3_time_mask[i - IMG_HISTORY_SIZE:i],
        indices,
        dtype=tf.bool,
    )

    # Create the ids for each step
    step_id = tf.range(0, tf.shape(frames_0)[0])

    return {
        "dataset_name": dataset_name,
        "episode_dict": epsd,
        "step_id": step_id,
        "past_frames_0": past_frames_0,
        "past_frames_0_time_mask": past_frames_0_time_mask,
        "past_frames_1": past_frames_1,
        "past_frames_1_time_mask": past_frames_1_time_mask,
        "past_frames_2": past_frames_2,
        "past_frames_2_time_mask": past_frames_2_time_mask,
        "past_frames_3": past_frames_3,
        "past_frames_3_time_mask": past_frames_3_time_mask,
        # 新增：存储所有主摄像头帧，用于未来观测采样
        "all_main_camera_frames": all_main_frames,
    }


def flatten_episode_with_future_obs(episode: dict) -> tf.data.Dataset:
    """
    将episode扁平化为步骤列表，同时支持未来观测的动态采样
    """
    episode_dict = episode["episode_dict"]
    dataset_name = episode["dataset_name"]
    all_main_frames = episode["all_main_camera_frames"]

    json_content, states, masks = generate_json_state(episode_dict, dataset_name)

    # Calculate the past_states for each step (保持原有逻辑)
    first_state = tf.expand_dims(states[0], axis=0)
    first_state = tf.repeat(first_state, ACTION_CHUNK_SIZE - 1, axis=0)
    padded_states = tf.concat([first_state, states], axis=0)
    indices = tf.range(ACTION_CHUNK_SIZE, tf.shape(states)[0] + ACTION_CHUNK_SIZE)
    past_states = tf.map_fn(lambda i: padded_states[i - ACTION_CHUNK_SIZE:i], indices, dtype=tf.float32)
    states_time_mask = tf.ones([tf.shape(states)[0]], dtype=tf.bool)
    padded_states_time_mask = tf.pad(
        states_time_mask,
        [[ACTION_CHUNK_SIZE - 1, 0]],
        "CONSTANT",
        constant_values=False,
    )
    past_states_time_mask = tf.map_fn(
        lambda i: padded_states_time_mask[i - ACTION_CHUNK_SIZE:i],
        indices,
        dtype=tf.bool,
    )

    # Calculate the future_states for each step (保持原有逻辑)
    last_state = tf.expand_dims(states[-1], axis=0)
    last_state = tf.repeat(last_state, ACTION_CHUNK_SIZE, axis=0)
    padded_states = tf.concat([states, last_state], axis=0)
    indices = tf.range(1, tf.shape(states)[0] + 1)
    future_states = tf.map_fn(lambda i: padded_states[i:i + ACTION_CHUNK_SIZE], indices, dtype=tf.float32)
    states_time_mask = tf.ones([tf.shape(states)[0]], dtype=tf.bool)
    padded_states_time_mask = tf.pad(states_time_mask, [[0, ACTION_CHUNK_SIZE]], "CONSTANT", constant_values=False)
    future_states_time_mask = tf.map_fn(
        lambda i: padded_states_time_mask[i:i + ACTION_CHUNK_SIZE],
        indices,
        dtype=tf.bool,
    )

    # 新增：计算每个步骤对应的未来观测帧（action chunk的最后一帧）
    total_frames = tf.shape(all_main_frames)[0]
    
    # 为每个时间步计算对应的未来观测帧索引
    def get_future_obs_index(current_step):
        """计算当前步骤对应的未来观测帧索引"""
        # 未来观测是当前步骤开始的action chunk的最后一个观测
        future_step = current_step + ACTION_CHUNK_SIZE - 1
        # 确保索引不超出范围
        future_step = tf.minimum(future_step, total_frames - 1)
        return future_step
    
    # 计算所有步骤的未来观测索引
    future_obs_indices = tf.map_fn(
        get_future_obs_index,
        tf.range(tf.shape(states)[0]),
        dtype=tf.int32
    )
    
    # 提取未来观测帧
    future_obs_frames = tf.gather(all_main_frames, future_obs_indices)
    
    # 创建未来观测的有效性掩码
    future_obs_mask = tf.map_fn(
        lambda i: tf.less(i + ACTION_CHUNK_SIZE - 1, total_frames),
        tf.range(tf.shape(states)[0]),
        dtype=tf.bool
    )

    # Calculate the mean and std for state (保持原有逻辑)
    state_std = tf.math.reduce_std(states, axis=0, keepdims=True)
    state_std = tf.repeat(state_std, tf.shape(states)[0], axis=0)
    state_mean = tf.math.reduce_mean(states, axis=0, keepdims=True)
    state_mean = tf.repeat(state_mean, tf.shape(states)[0], axis=0)

    state_norm = tf.math.reduce_mean(tf.math.square(states), axis=0, keepdims=True)
    state_norm = tf.math.sqrt(state_norm)
    state_norm = tf.repeat(state_norm, tf.shape(states)[0], axis=0)

    # Create a list of steps
    step_data = []
    for i in range(tf.shape(states)[0]):
        step_data.append({
            "step_id": episode["step_id"][i],
            "json_content": json_content,
            "state_chunk": past_states[i],
            "state_chunk_time_mask": past_states_time_mask[i],
            "action_chunk": future_states[i],
            "action_chunk_time_mask": future_states_time_mask[i],
            "state_vec_mask": masks[i],
            "past_frames_0": episode["past_frames_0"][i],
            "past_frames_0_time_mask": episode["past_frames_0_time_mask"][i],
            "past_frames_1": episode["past_frames_1"][i],
            "past_frames_1_time_mask": episode["past_frames_1_time_mask"][i],
            "past_frames_2": episode["past_frames_2"][i],
            "past_frames_2_time_mask": episode["past_frames_2_time_mask"][i],
            "past_frames_3": episode["past_frames_3"][i],
            "past_frames_3_time_mask": episode["past_frames_3_time_mask"][i],
            "state_std": state_std[i],
            "state_mean": state_mean[i],
            "state_norm": state_norm[i],
            # 新增：未来观测相关数据
            "future_obs_frame": future_obs_frames[i],
            "future_obs_mask": future_obs_mask[i],
            "future_obs_index": future_obs_indices[i],
        })

    return step_data


def flatten_episode_agilex_with_future_obs(episode: dict) -> tf.data.Dataset:
    """
    为agilex数据集扁平化episode，支持未来观测
    """
    episode_dict = episode["episode_dict"]
    dataset_name = episode["dataset_name"]
    all_main_frames = episode["all_main_camera_frames"]

    json_content, states, masks, acts = generate_json_state(episode_dict, dataset_name)

    # Calculate the past_states for each step (保持原有逻辑)
    first_state = tf.expand_dims(states[0], axis=0)
    first_state = tf.repeat(first_state, ACTION_CHUNK_SIZE - 1, axis=0)
    padded_states = tf.concat([first_state, states], axis=0)
    indices = tf.range(ACTION_CHUNK_SIZE, tf.shape(states)[0] + ACTION_CHUNK_SIZE)
    past_states = tf.map_fn(lambda i: padded_states[i - ACTION_CHUNK_SIZE:i], indices, dtype=tf.float32)
    states_time_mask = tf.ones([tf.shape(states)[0]], dtype=tf.bool)
    padded_states_time_mask = tf.pad(
        states_time_mask,
        [[ACTION_CHUNK_SIZE - 1, 0]],
        "CONSTANT",
        constant_values=False,
    )
    past_states_time_mask = tf.map_fn(
        lambda i: padded_states_time_mask[i - ACTION_CHUNK_SIZE:i],
        indices,
        dtype=tf.bool,
    )

    # NOTE: 未来状态应该是动作
    last_act = tf.expand_dims(acts[-1], axis=0)
    last_act = tf.repeat(last_act, ACTION_CHUNK_SIZE, axis=0)
    padded_states = tf.concat([acts, last_act], axis=0)
    indices = tf.range(0, tf.shape(acts)[0])
    future_states = tf.map_fn(lambda i: padded_states[i:i + ACTION_CHUNK_SIZE], indices, dtype=tf.float32)
    states_time_mask = tf.ones([tf.shape(acts)[0]], dtype=tf.bool)
    padded_states_time_mask = tf.pad(states_time_mask, [[0, ACTION_CHUNK_SIZE]], "CONSTANT", constant_values=False)
    future_states_time_mask = tf.map_fn(
        lambda i: padded_states_time_mask[i:i + ACTION_CHUNK_SIZE],
        indices,
        dtype=tf.bool,
    )

    # 新增：计算未来观测（与上面逻辑相同）
    total_frames = tf.shape(all_main_frames)[0]
    
    def get_future_obs_index(current_step):
        future_step = current_step + ACTION_CHUNK_SIZE - 1
        future_step = tf.minimum(future_step, total_frames - 1)
        return future_step
    
    future_obs_indices = tf.map_fn(
        get_future_obs_index,
        tf.range(tf.shape(states)[0]),
        dtype=tf.int32
    )
    
    future_obs_frames = tf.gather(all_main_frames, future_obs_indices)
    
    future_obs_mask = tf.map_fn(
        lambda i: tf.less(i + ACTION_CHUNK_SIZE - 1, total_frames),
        tf.range(tf.shape(states)[0]),
        dtype=tf.bool
    )

    # Calculate the std and mean for state (保持原有逻辑)
    state_std = tf.math.reduce_std(states, axis=0, keepdims=True)
    state_std = tf.repeat(state_std, tf.shape(states)[0], axis=0)
    state_mean = tf.math.reduce_mean(states, axis=0, keepdims=True)
    state_mean = tf.repeat(state_mean, tf.shape(states)[0], axis=0)

    state_norm = tf.math.reduce_mean(tf.math.square(acts), axis=0, keepdims=True)
    state_norm = tf.math.sqrt(state_norm)
    state_norm = tf.repeat(state_norm, tf.shape(states)[0], axis=0)

    # Create a list of steps
    step_data = []
    for i in range(tf.shape(states)[0]):
        step_data.append({
            "step_id": episode["step_id"][i],
            "json_content": json_content,
            "state_chunk": past_states[i],
            "state_chunk_time_mask": past_states_time_mask[i],
            "action_chunk": future_states[i],
            "action_chunk_time_mask": future_states_time_mask[i],
            "state_vec_mask": masks[i],
            "past_frames_0": episode["past_frames_0"][i],
            "past_frames_0_time_mask": episode["past_frames_0_time_mask"][i],
            "past_frames_1": episode["past_frames_1"][i],
            "past_frames_1_time_mask": episode["past_frames_1_time_mask"][i],
            "past_frames_2": episode["past_frames_2"][i],
            "past_frames_2_time_mask": episode["past_frames_2_time_mask"][i],
            "past_frames_3": episode["past_frames_3"][i],
            "past_frames_3_time_mask": episode["past_frames_3_time_mask"][i],
            "state_std": state_std[i],
            "state_mean": state_mean[i],
            "state_norm": state_norm[i],
            # 新增：未来观测相关数据
            "future_obs_frame": future_obs_frames[i],
            "future_obs_mask": future_obs_mask[i],
            "future_obs_index": future_obs_indices[i],
        })

    return step_data