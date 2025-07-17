import sys, os

# 获取当前目录路径，不依赖 __file__
try:
    current_file_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(current_file_path)
except NameError:
    # 如果 __file__ 不存在，使用当前工作目录
    parent_directory = os.path.dirname(os.path.abspath('.'))

def encode_obs(observation):  # Post-Process Observation
    observation["agent_pos"] = observation["joint_action"]["vector"]
    return observation

def get_model(usr_args):  # keep
    # 延迟导入，避免启动时的依赖问题
    import sys
    import os
    
    # 获取当前模块所在目录
    try:
        current_file_path = os.path.abspath(__file__)
        parent_directory = os.path.dirname(current_file_path)
    except NameError:
        # 如果在 exec 环境中，尝试其他方法获取路径
        import inspect
        frame = inspect.currentframe()
        try:
            caller_file = frame.f_back.f_globals.get('__file__')
            if caller_file:
                parent_directory = os.path.dirname(os.path.abspath(caller_file))
            else:
                parent_directory = os.getcwd()
        finally:
            del frame
    
    sys.path.append(parent_directory)
    from model import RDT
    
    model_name = usr_args["ckpt_setting"]
    checkpoint_id = usr_args["checkpoint_id"]
    left_arm_dim, right_arm_dim, rdt_step = (
        usr_args["left_arm_dim"],
        usr_args["right_arm_dim"],
        usr_args["rdt_step"],
    )
    rdt = RDT(
        os.path.join(
            parent_directory,
            f"checkpoints/{model_name}/checkpoint-{checkpoint_id}/pytorch_model.bin",
        ),
        usr_args["task_name"],
        left_arm_dim,
        right_arm_dim,
        rdt_step,
    )
    return rdt

def eval(TASK_ENV, model, observation):
    """
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    obs = encode_obs(observation)  # Post-Process Observation
    instruction = TASK_ENV.get_instruction()
    input_rgb_arr, input_state = [
        obs["observation"]["head_camera"]["rgb"],
        obs["observation"]["right_camera"]["rgb"],
        obs["observation"]["left_camera"]["rgb"],
    ], obs["agent_pos"]  # TODO

    if (model.observation_window
            is None):  # Force an update of the observation at the first frame to avoid an empty observation window
        model.set_language_instruction(instruction)
        model.update_observation_window(input_rgb_arr, input_state)

    actions = model.get_action()  # Get Action according to observation chunk

    for action in actions:  # Execute each step of the action
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        input_rgb_arr, input_state = [
            obs["observation"]["head_camera"]["rgb"],
            obs["observation"]["right_camera"]["rgb"],
            obs["observation"]["left_camera"]["rgb"],
        ], obs["agent_pos"]  # TODO
        model.update_observation_window(input_rgb_arr, input_state)  # Update Observation

def reset_model(
        model):  # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.reset_obsrvationwindows()