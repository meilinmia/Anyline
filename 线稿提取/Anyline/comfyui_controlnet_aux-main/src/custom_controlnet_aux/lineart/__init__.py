import os
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image

from custom_controlnet_aux.util import HWC3, resize_image_with_pad, common_input_validate, custom_hf_download, HF_MODEL_NAME

norm_layer = nn.InstanceNorm2d


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features)
                        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        # Initial convolution block
        model0 = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    norm_layer(64),
                    nn.ReLU(inplace=True) ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model1 += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features//2
        for _ in range(2):
            model3 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [  nn.ReflectionPad2d(3),
                        nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out


class LineartDetector:
    def __init__(self, model, coarse_model):
        self.model = model
        self.model_coarse = coarse_model
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, local_model_path=None, local_coarse_model_path=None):
        """
        仅从本地加载 lineart 权重
        默认位置：
        comfyui_controlnet_aux-main/ckpts/Illyasviel/Annotators/sk_model.pth
        comfyui_controlnet_aux-main/ckpts/Illyasviel/Annotators/sk_model2.pth
        """
        # 计算到 comfyui_controlnet_aux-main 的根目录
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        ckpt_dir = os.path.join(base_dir, "ckpts", "Illyasviel", "Annotators")

        # 默认权重路径
        if local_model_path is None:
            local_model_path = os.path.join(ckpt_dir, "sk_model.pth")
        if local_coarse_model_path is None:
            # 优先 sk_model2.pth，没有就回退到 sk_model.pth
            coarse_candidate = os.path.join(ckpt_dir, "sk_model2.pth")
            local_coarse_model_path = coarse_candidate if os.path.isfile(coarse_candidate) else local_model_path

        # 存在性检查
        if not os.path.isfile(local_model_path):
            raise FileNotFoundError(f"未找到 lineart 主权重：{local_model_path}")
        if not os.path.isfile(local_coarse_model_path):
            raise FileNotFoundError(f"未找到 lineart coarse 权重：{local_coarse_model_path}")

        # 加载权重到 CPU（之后可 .to("cuda")）
        state_main = torch.load(local_model_path, map_location="cpu")
        state_coarse = torch.load(local_coarse_model_path, map_location="cpu")

        # 兼容 {"state_dict": ...} 的存储格式
        if isinstance(state_main, dict) and "state_dict" in state_main and isinstance(state_main["state_dict"], dict):
            state_main = state_main["state_dict"]
        if isinstance(state_coarse, dict) and "state_dict" in state_coarse and isinstance(state_coarse["state_dict"], dict):
            state_coarse = state_coarse["state_dict"]

        model = Generator(3, 1, 3)
        model.load_state_dict(state_main, strict=True)
        model.eval()

        coarse_model = Generator(3, 1, 3)
        coarse_model.load_state_dict(state_coarse, strict=True)
        coarse_model.eval()

        return cls(model, coarse_model)


    
    def to(self, device):
        self.model.to(device)
        self.model_coarse.to(device)
        self.device = device
        return self
    
    def __call__(self, input_image, coarse=False, detect_resolution=512, output_type="pil", upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        detected_map, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)

        model = self.model_coarse if coarse else self.model
        assert detected_map.ndim == 3
        with torch.no_grad():
            image = torch.from_numpy(detected_map).float().to(self.device)
            image = image / 255.0
            image = rearrange(image, 'h w c -> 1 c h w')
            line = model(image)[0][0]

            line = line.cpu().numpy()
            line = (line * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = HWC3(line)
        detected_map = remove_pad(255 - detected_map)
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map
