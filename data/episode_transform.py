# data/episode_transform.py - 简化版本

import numpy as np
import tensorflow as tf
import yaml

from data.preprocess import generate_json_state
from configs.state_vec import STATE_VEC_IDX_MAPPING

# Read the config
with open("configs/base.yaml", "r") as file:
    config = yaml.safe_load(file)
IMG_HISTORY_SIZE = config["common"]["img_history_size"]
ACTION_CHUNK_SIZE = config["common"]["action_chunk_size"]

@tf.function
def process_episode_with_flare(epsd: dict, dataset_name: str, image_keys: list, image_mask: list) -> dict:
    """
    处理episode以提取frames和json内容，支持FLARE的简化未来观测采样
    """
    # 收集所有摄像头的所有帧 - 这是核心！
    all_frames = {}
    total_frames = 0
    
    for cam_idx in range(len(image_keys)):
        if image_mask[cam_idx] == 1:
            frames = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)
            for step in iter(epsd["steps"]):
                frame = step["observation"][image_keys[cam_idx]]
                frames = frames.write(frames.size(), frame)
            all_frames[f"camera_{cam_idx}"] = frames.stack()
            total_frames = frames.size()  # 所有摄像头帧数应该相同
        else:
            # 创建空帧占位符
            dummy_frame = tf.zeros([0, 0, 0], dtype=tf.uint8)
            all_frames[f"camera_{cam_idx}"] = tf.expand_dims(dummy_frame, 0)

    # 处理历史frames（保持原逻辑）
    frames_0 = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)
    frames_1 = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)
    frames_2 = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)
    frames_3 = tf.TensorArray(dtype=tf.uint8, size=0, dynamic_size=True)
    
    for step in iter(epsd["steps"]):
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

    # 处理历史frames（保持原逻辑）
    frames_0 = frames_0.stack()
    
    # 计算past frames（保持原有逻辑）
    first_frame = tf.expand_dims(frames_0[0], axis=0)
    first_frame = tf.repeat(first_frame, IMG_HISTORY_SIZE - 1, axis=0)
    padded_frames_0 = tf.concat([first_frame, frames_0], axis=0)
    indices = tf.range(IMG_HISTORY_SIZE, tf.shape(frames_0)[0] + IMG_HISTORY_SIZE)
    past_frames_0 = tf.map_fn(lambda i: padded_frames_0[i - IMG_HISTORY_SIZE:i], indices, dtype=tf.uint8)
    
    # 类似处理其他摄像头...
    frames_1 = frames_1.stack()
    first_frame = tf.expand_dims(frames_1[0], axis=0)
    first_frame = tf.repeat(first_frame, IMG_HISTORY_SIZE - 1, axis=0)
    padded_frames_1 = tf.concat([first_frame, frames_1], axis=0)
    past_frames_1 = tf.map_fn(lambda i: padded_frames_1[i - IMG_HISTORY_SIZE:i], indices, dtype=tf.uint8)
    
    frames_2 = frames_2.stack()
    first_frame = tf.expand_dims(frames_2[0], axis=0)
    first_frame = tf.repeat(first_frame, IMG_HISTORY_SIZE - 1, axis=0)
    padded_frames_2 = tf.concat([first_frame, frames_2], axis=0)
    past_frames_2 = tf.map_fn(lambda i: padded_frames_2[i - IMG_HISTORY_SIZE:i], indices, dtype=tf.uint8)
    
    frames_3 = frames_3.stack()
    first_frame = tf.expand_dims(frames_3[0], axis=0)
    first_frame = tf.repeat(first_frame, IMG_HISTORY_SIZE - 1, axis=0)
    padded_frames_3 = tf.concat([first_frame, frames_3], axis=0)
    past_frames_3 = tf.map_fn(lambda i: padded_frames_3[i - IMG_HISTORY_SIZE:i], indices, dtype=tf.uint8)
    
    # 生成time masks（保持原逻辑）
    frames_time_mask = tf.ones([tf.shape(frames_0)[0]], dtype=tf.bool)
    padded_frames_time_mask = tf.pad(
        frames_time_mask,
        [[IMG_HISTORY_SIZE - 1, 0]],
        "CONSTANT",
        constant_values=False,
    )
    past_frames_0_time_mask = tf.map_fn(
        lambda i: padded_frames_time_mask[i - IMG_HISTORY_SIZE:i],
        indices,
        dtype=tf.bool,
    )
    past_frames_1_time_mask = past_frames_0_time_mask
    past_frames_2_time_mask = past_frames_0_time_mask
    past_frames_3_time_mask = past_frames_0_time_mask

    # 创建step IDs
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
        # 存储所有摄像头的完整帧序列，用于FLARE动态采样
        "all_camera_frames": all_frames,
        "total_frame_count": total_frames,
    }


def flatten_episode_with_flare(episode: dict) -> list:
    """
    将episode扁平化为步骤列表，支持FLARE的动态未来观测采样
    """
    episode_dict = episode["episode_dict"]
    dataset_name = episode["dataset_name"]
    all_camera_frames = episode["all_camera_frames"]
    total_frames = episode["total_frame_count"]

    json_content, states, masks = generate_json_state(episode_dict, dataset_name)

    # 计算past_states（保持原逻辑）
    first_state = tf.expand_dims(states[0], axis=0)
    first_state = tf.repeat(first_state, ACTION_CHUNK_SIZE - 1, axis=0)
    padded_states = tf.concat([first_state, states], axis=0)
    indices = tf.range(ACTION_CHUNK_SIZE, tf.shape(states)[0] + ACTION_CHUNK_SIZE)
    past_states = tf.map_fn(lambda i: padded_states[i - ACTION_CHUNK_SIZE:i], indices, dtype=tf.float32)
    
    # 计算future_states（保持原逻辑）
    last_state = tf.expand_dims(states[-1], axis=0)
    last_state = tf.repeat(last_state, ACTION_CHUNK_SIZE, axis=0)
    padded_states = tf.concat([states, last_state], axis=0)
    indices = tf.range(1, tf.shape(states)[0] + 1)
    future_states = tf.map_fn(lambda i: padded_states[i:i + ACTION_CHUNK_SIZE], indices, dtype=tf.float32)
    
    # 计算time masks
    states_time_mask = tf.ones([tf.shape(states)[0]], dtype=tf.bool)
    padded_states_time_mask = tf.pad(
        states_time_mask,
        [[ACTION_CHUNK_SIZE - 1, 0]],
        "CONSTANT",
        constant_values=False,
    )
    past_states_time_mask = tf.map_fn(
        lambda i: padded_states_time_mask[i - ACTION_CHUNK_SIZE:i],
        tf.range(ACTION_CHUNK_SIZE, tf.shape(states)[0] + ACTION_CHUNK_SIZE),
        dtype=tf.bool,
    )
    
    padded_states_time_mask = tf.pad(states_time_mask, [[0, ACTION_CHUNK_SIZE]], "CONSTANT", constant_values=False)
    future_states_time_mask = tf.map_fn(
        lambda i: padded_states_time_mask[i:i + ACTION_CHUNK_SIZE],
        indices,
        dtype=tf.bool,
    )

    # FLARE核心：计算每个时间步的未来观测帧
    # 使用主摄像头（camera_0）作为未来观测
    main_camera_frames = all_camera_frames["camera_0"]
    
    def get_future_obs_frame(current_step):
    """
    获取当前步骤对应的未来观测帧
    核心逻辑：
    1. 如果 current_step + ACTION_CHUNK_SIZE - 1 < total_frames，使用该帧
    2. 否则使用最后一帧，这样与动作填充逻辑保持一致
    """
        ideal_future_step = current_step + ACTION_CHUNK_SIZE - 1
        
        # 如果理想步骤在范围内，使用它；否则使用最后一帧
        actual_future_step = tf.minimum(ideal_future_step, total_frames - 1)
        
        # 获取未来观测帧
        future_frame = tf.gather(main_camera_frames, actual_future_step)
        
        # 所有情况都认为是有效的，因为我们总是能提供一个合理的未来观测
        is_valid = tf.constant(True)
        
        return future_frame, is_valid
    
    # 为每个时间步计算未来观测
    future_obs_frames = []
    future_obs_masks = []
    
    for i in range(tf.shape(states)[0]):
        future_frame, is_valid = get_future_obs_frame(i)
        future_obs_frames.append(future_frame)
        future_obs_masks.append(is_valid)
    
    future_obs_frames = tf.stack(future_obs_frames)
    future_obs_masks = tf.stack(future_obs_masks)

    # 计算统计信息（保持原逻辑）
    state_std = tf.math.reduce_std(states, axis=0, keepdims=True)
    state_std = tf.repeat(state_std, tf.shape(states)[0], axis=0)
    state_mean = tf.math.reduce_mean(states, axis=0, keepdims=True)
    state_mean = tf.repeat(state_mean, tf.shape(states)[0], axis=0)
    state_norm = tf.math.reduce_mean(tf.math.square(states), axis=0, keepdims=True)
    state_norm = tf.math.sqrt(state_norm)
    state_norm = tf.repeat(state_norm, tf.shape(states)[0], axis=0)

    # 创建步骤数据列表
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
            # FLARE: 添加动态计算的未来观测数据
            "future_obs_frame": future_obs_frames[i],
            "future_obs_mask": future_obs_masks[i],
        })

    return step_data