import os
import torch
import warnings

import clip
from clip.model import AttentionPool2d

input_resolution = 224
# spacial_dim=7
embed_dim=2048
# embed_dim=256
heads=32
output_dim=1024
device = "cuda" if torch.cuda.is_available() else "cpu"
jit = False
model_path = os.path.join(os.environ['HOME'], '.cache/clip/RN50.pt')
print(model_path)
if not os.path.exists(model_path):
    clip.load("RN50", device=device)

attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

def update_parameters(state_dict):
    # 使用 state_dict 进行初始化
    model_dict = attnpool.state_dict()
    for key, val in model_dict.items():
        key_ori = "visual.attnpool." + key 
        if key_ori in state_dict:
            model_dict[key] = state_dict[key_ori]

    attnpool.load_state_dict(model_dict, strict=False)
    attnpool.to(device)

    for para in attnpool.parameters(): #lzk: freeze model parameters
        para.requires_grad = False

# 加载 checkpoint
with open(model_path, 'rb') as opened_file:
    try:
        # loading JIT archive
        model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(opened_file, map_location="cpu")

update_parameters(state_dict or model.state_dict())

