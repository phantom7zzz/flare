import torch

# 检查几个您的T5文件
t5_file = "/home/deng_xiang/qian_daichao/RoboTwin/policy/RDT_flare/training_data/grab_roller/demo_grab_roller/grab_roller-demo_clean-50/episode_0/instructions/lang_embed_5.pt"
embed = torch.load(t5_file)

if len(embed.shape) == 3:  # (1, seq_len, 4096)
    actual_length = embed.shape[1]
elif len(embed.shape) == 2:  # (seq_len, 4096)  
    actual_length = embed.shape[0]

print(f"实际T5长度: {actual_length} tokens")