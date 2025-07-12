class HDF5VLADatasetWithFLARE:
    """
    支持FLARE功能的HDF5 VLA数据集，包含未来观测采样
    """

    def __init__(self, model_config_path, enable_future_obs=True) -> None:
        # 加载模型配置
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)
        HDF5_DIR = model_config["data_path"]
        self.DATASET_NAME = "agilex"
        self.enable_future_obs = enable_future_obs

        self.file_paths = []
        for root, _, files in os.walk(HDF5_DIR):
            for filename in fnmatch.filter(files, "*.hdf5"):
                file_path = os.path.join(root, filename)
                self.file_paths.append(file_path)

        # 加载配置
        with open("configs/base.yaml", "r") as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config["common"]["action_chunk_size"]
        self.IMG_HISORY_SIZE = config["common"]["img_history_size"]
        self.STATE_DIM = config["common"]["state_dim"]

        # 获取每个episode的长度
        episode_lens = []
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res["state"].shape[0] if valid else 0
            episode_lens.append(_len)
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)

    def __len__(self):
        return len(self.file_paths)

    def get_dataset_name(self):
        return self.DATASET_NAME

    def get_item(self, index: int = None, state_only=False):
        """
        获取训练样本，支持未来观测
        
        Args:
            index: episode索引
            state_only: 是否只返回状态数据
        """
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = (self.parse_hdf5_file_with_future_obs(file_path)
                             if not state_only else self.parse_hdf5_file_state_only(file_path))
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))

    def parse_hdf5_file_with_future_obs(self, file_path):
        """
        解析HDF5文件生成包含未来观测的训练样本
        """
        with h5py.File(file_path, "r") as f:
            qpos = f["observations"]["qpos"][:]
            left_arm_dim = f["observations"]["left_arm_dim"][:]
            right_arm_dim = f["observations"]["right_arm_dim"][:]
            num_steps = qpos.shape[0]

            # 跳过前几个静止步骤
            EPS = 1e-2
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")

            # 随机采样一个时间步
            step_id = np.random.randint(first_idx - 1, num_steps)

            # 加载指令
            dir_path = os.path.dirname(file_path)
            instructions_path = os.path.join(dir_path, "instructions")
            instructions_names = []

            for filename in os.listdir(instructions_path):
                if filename.endswith(".pt"):
                    instructions_names.append(os.path.join(instructions_path, filename))
            instruction = np.random.choice(instructions_names)

            # 组装meta
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction,
            }

            # 重新缩放gripper到[0, 1]
            qpos = qpos / np.array([[1 for i in range(left_arm_dim[0] + 1 + right_arm_dim[0] + 1)]])
            target_qpos = f["action"][step_id:step_id + self.CHUNK_SIZE] / np.array(
                [[1 for i in range(left_arm_dim[0] + 1 + right_arm_dim[0] + 1)]])

            # 解析状态和动作
            state = qpos[step_id:step_id + 1]
            state_std = np.std(qpos, axis=0)
            state_mean = np.mean(qpos, axis=0)
            state_norm = np.sqrt(np.mean(qpos**2, axis=0))
            actions = target_qpos
            if actions.shape[0] < self.CHUNK_SIZE:
                actions = np.concatenate(
                    [
                        actions,
                        np.tile(actions[-1:], (self.CHUNK_SIZE - actions.shape[0], 1)),
                    ],
                    axis=0,
                )

            # 填充状态/动作到统一向量
            def fill_in_state(values):
                from configs.state_vec import STATE_VEC_IDX_MAPPING
                UNI_STATE_INDICES = (
                    [STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"]
                     for i in range(left_arm_dim[0])] + [STATE_VEC_IDX_MAPPING["left_gripper_open"]] +
                    [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"]
                     for i in range(right_arm_dim[0])] + [STATE_VEC_IDX_MAPPING["right_gripper_open"]])
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM, ))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec

            state = fill_in_state(state)
            state_indicator = fill_in_state(np.ones_like(state_std))
            state_std = fill_in_state(state_std)
            state_mean = fill_in_state(state_mean)
            state_norm = fill_in_state(state_norm)
            actions = fill_in_state(actions)

            # 解析图像
            def parse_img(key):
                imgs = []
                for i in range(max(step_id - self.IMG_HISORY_SIZE + 1, 0), step_id + 1):
                    img_bits = f["observations"]["images"][key][i]
                    img = cv2.imdecode(np.frombuffer(img_bits, np.uint8), cv2.IMREAD_COLOR)
                    imgs.append(img)
                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
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

            # 外部摄像头图像
            cam_high = parse_img("cam_high")
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
            cam_high_mask = np.array([False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len)
            cam_left_wrist = parse_img("cam_left_wrist")
            cam_left_wrist_mask = cam_high_mask.copy()
            cam_right_wrist = parse_img("cam_right_wrist")
            cam_right_wrist_mask = cam_high_mask.copy()

            # 处理未来观测
            future_obs_frame = None
            future_obs_mask = False
            if self.enable_future_obs:
                try:
                    # 计算未来观测的时间步（action chunk的最后一帧）
                    future_step = step_id + self.CHUNK_SIZE - 1
                    if future_step < num_steps:
                        # 使用主摄像头（cam_high）作为未来观测
                        future_img_bits = f["observations"]["images"]["cam_high"][future_step]
                        future_obs_frame = cv2.imdecode(np.frombuffer(future_img_bits, np.uint8), cv2.IMREAD_COLOR)
                        future_obs_mask = True
                except Exception as e:
                    print(f"Error loading future observation: {e}")
                    future_obs_frame = None
                    future_obs_mask = False

            # 返回结果样本
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
            }
            
            # 添加未来观测数据
            if self.enable_future_obs:
                result["future_obs_frame"] = future_obs_frame
                result["future_obs_mask"] = future_obs_mask

            return True, result

    def parse_hdf5_file_state_only(self, file_path):
        """
        解析HDF5文件生成状态轨迹（用于统计）
        """
        with h5py.File(file_path, "r") as f:
            qpos = f["observations"]["qpos"][:]
            left_arm_dim = f["observations"]["left_arm_dim"][:]
            right_arm_dim = f["observations"]["right_arm_dim"][:]

            num_steps = qpos.shape[0]

            # 跳过前几个静止步骤
            EPS = 1e-2
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")

            # 重新缩放gripper到[0, 1]
            qpos = qpos / np.array([[1 for i in range(left_arm_dim[0] + right_arm_dim[0] + 2)]])
            target_qpos = f["action"][:] / np.array([[1 for i in range(left_arm_dim[0] + right_arm_dim[0] + 2)]])

            # 解析状态和动作
            state = qpos[first_idx - 1:]
            action = target_qpos[first_idx - 1:]

            # 填充状态/动作到统一向量
            def fill_in_state(values):
                from configs.state_vec import STATE_VEC_IDX_MAPPING
                UNI_STATE_INDICES = (
                    [STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"]
                     for i in range(left_arm_dim[0])] + [STATE_VEC_IDX_MAPPING["left_gripper_open"]] +
                    [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"]
                     for i in range(right_arm_dim[0])] + [STATE_VEC_IDX_MAPPING["right_gripper_open"]])
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM, ))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec

            state = fill_in_state(state)
            action = fill_in_state(action)

            return True, {"state": state, "action": action}