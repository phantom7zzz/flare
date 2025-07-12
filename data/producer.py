def save_sample_with_future_obs(step_dict, chunk_dir, chunk_item_idx):
    """
    保存样本到chunk目录，包含未来观测数据
    """
    time_stmp = time.time()
    while time.time() - time_stmp < 10.0:
        try:
            locks = []
            json_content = step_dict["json_content"]
            file_path = os.path.join(chunk_dir, f"json_content_{chunk_item_idx}.json")
            lock = FileLock(file_path)
            locks.append(lock)
            lock.acquire_write_lock()
            with open(file_path, "w") as file:
                json.dump(json_content, file, indent=4)
            lock.release_lock()
            
            # 保存所有其他tensor到npz文件
            file_path = os.path.join(chunk_dir, f"sample_{chunk_item_idx}.npz")
            lock = FileLock(file_path)
            locks.append(lock)
            lock.acquire_write_lock()
            
            # 构建保存数据字典
            save_data = {
                "step_id": step_dict["step_id"].numpy(),
                "state_chunk": step_dict["state_chunk"].numpy(),
                "state_chunk_time_mask": step_dict["state_chunk_time_mask"].numpy(),
                "action_chunk": step_dict["action_chunk"].numpy(),
                "action_chunk_time_mask": step_dict["action_chunk_time_mask"].numpy(),
                "state_vec_mask": step_dict["state_vec_mask"].numpy(),
                "past_frames_0": step_dict["past_frames_0"].numpy(),
                "past_frames_0_time_mask": step_dict["past_frames_0_time_mask"].numpy(),
                "past_frames_1": step_dict["past_frames_1"].numpy(),
                "past_frames_1_time_mask": step_dict["past_frames_1_time_mask"].numpy(),
                "past_frames_2": step_dict["past_frames_2"].numpy(),
                "past_frames_2_time_mask": step_dict["past_frames_2_time_mask"].numpy(),
                "past_frames_3": step_dict["past_frames_3"].numpy(),
                "past_frames_3_time_mask": step_dict["past_frames_3_time_mask"].numpy(),
                "state_std": step_dict["state_std"].numpy(),
                "state_mean": step_dict["state_mean"].numpy(),
                "state_norm": step_dict["state_norm"].numpy(),
            }
            
            # 添加未来观测数据（如果存在）
            if "future_obs_frame" in step_dict:
                save_data["future_obs_frame"] = step_dict["future_obs_frame"].numpy()
            if "future_obs_mask" in step_dict:
                save_data["future_obs_mask"] = step_dict["future_obs_mask"].numpy()
            if "future_obs_index" in step_dict:
                save_data["future_obs_index"] = step_dict["future_obs_index"].numpy()
                
            with open(file_path, "wb") as file:
                np.savez(file, **save_data)
            lock.release_lock()
            return
        except KeyboardInterrupt:
            for lock in locks:
                lock.release_lock()
            raise KeyboardInterrupt
        except BaseException:
            for lock in locks:
                lock.release_lock()
            continue
    print("Failed to save sample with future obs.")


def run_producer_with_flare(seed, num_workers, worker_id, fill_up, clean_dirty, dataset_type, enable_future_obs=True):
    """
    运行支持FLARE的producer
    """
    from data.vla_dataset import VLADatasetWithFLARE
    vla_dataset = VLADatasetWithFLARE(seed=seed, dataset_type=dataset_type, enable_future_obs=enable_future_obs)
    
    # 其余逻辑与原producer相同，但使用新的保存函数
    from data.producer import (BUF_PATH, BUF_NUM_CHUNKS, BUF_CHUNK_SIZE, 
                               get_dirty_item, save_dirty_bit, read_dirty_bit)
    
    chunk_start_idx = worker_id * BUF_NUM_CHUNKS // num_workers
    chunk_end_idx = (worker_id + 1) * BUF_NUM_CHUNKS // num_workers
    
    if fill_up:
        print(f"Worker {worker_id}: Start filling up the buffer with FLARE support...")
    elif clean_dirty:
        print(f"Worker {worker_id}: Start refreshing the dirty bits...")
        for chunk_idx in range(chunk_start_idx, chunk_end_idx):
            chunk_dir = os.path.join(BUF_PATH, f"chunk_{chunk_idx}")
            dirty_bit = np.zeros(BUF_CHUNK_SIZE, dtype=np.uint8)
            save_dirty_bit(chunk_dir, dirty_bit)
        print(f"Worker {worker_id}: Refreshed the dirty bits.")

    fill_chunk_idx = chunk_start_idx
    fill_chunk_item_idx = 0
    dirty_chunk_idx = chunk_start_idx
    dirty_chunk_item_idxs = []
    time_stmp = time.time()
    
    for episode_steps in vla_dataset:
        for step in episode_steps:
            if fill_up and fill_chunk_idx < chunk_end_idx:
                # 填充缓冲区
                chunk_dir = os.path.join(BUF_PATH, f"chunk_{fill_chunk_idx}")
                if fill_chunk_item_idx == 0:
                    os.makedirs(chunk_dir, exist_ok=True)
                    dirty_bit = np.zeros(BUF_CHUNK_SIZE, dtype=np.uint8)
                    save_dirty_bit(chunk_dir, dirty_bit)

                # 使用支持未来观测的保存函数
                save_sample_with_future_obs(step, chunk_dir, fill_chunk_item_idx)

                local_fill_chunk_idx = fill_chunk_idx - chunk_start_idx
                local_num_chunks = chunk_end_idx - chunk_start_idx
                if (local_fill_chunk_idx % 10 == 0
                        or local_fill_chunk_idx == local_num_chunks - 1) and fill_chunk_item_idx == 0:
                    print(f"Worker {worker_id}: Filled up chunk {local_fill_chunk_idx+1}/{local_num_chunks}")
                    
                fill_chunk_item_idx += 1
                if fill_chunk_item_idx == BUF_CHUNK_SIZE:
                    fill_chunk_idx += 1
                    fill_chunk_item_idx = 0
                if fill_chunk_idx == BUF_NUM_CHUNKS:
                    print(f"Worker {worker_id}: Buffer filled up. Start replacing dirty samples...")

            else:
                # 搜索dirty chunk进行替换
                while len(dirty_chunk_item_idxs) == 0:
                    dirty_chunk_dir = os.path.join(BUF_PATH, f"chunk_{dirty_chunk_idx}")
                    dirty_chunk_item_idxs = get_dirty_item(dirty_chunk_dir)
                    
                    if time.time() - time_stmp > 2.0:
                        dirty_ratio = len(dirty_chunk_item_idxs) / BUF_CHUNK_SIZE
                        print(f"Worker {worker_id}: Dirty Ratio for Chunk {dirty_chunk_idx}: {dirty_ratio:.2f}")
                        time_stmp = time.time()

                    if len(dirty_chunk_item_idxs) > 0:
                        dirty_bit = np.ones(BUF_CHUNK_SIZE, dtype=np.uint8)
                        save_dirty_bit(dirty_chunk_dir, dirty_bit)

                    dirty_chunk_idx += 1
                    if dirty_chunk_idx == chunk_end_idx:
                        dirty_chunk_idx = chunk_start_idx

                # 替换dirty item
                dirty_item_idx = dirty_chunk_item_idxs.pop()
                chunk_dir = os.path.join(BUF_PATH, f"chunk_{dirty_chunk_idx}")
                save_sample_with_future_obs(step, chunk_dir, dirty_item_idx)

                if len(dirty_chunk_item_idxs) == 0:
                    dirty_bit = np.zeros(BUF_CHUNK_SIZE, dtype=np.uint8)
                    save_dirty_bit(dirty_chunk_dir, dirty_bit)
                    print(f"Worker {worker_id}: Replaced dirty chunk {dirty_chunk_idx}.")