# -*- coding: utf-8 -*-
"""
用法示例（PowerShell/终端）：
  python run.py --input .\inputs\1.jpg --resolution 1280 --outdir .\outputs
  python run.py --input .\inputs\1.jpg --resolution 1280 --use_mteed --outdir .\outputs
  # 只用CPU（更稳）
  python run.py --input .\inputs\1.jpg --resolution 640 --use_mteed --force_cpu
"""

import os, sys, argparse
import numpy as np
from PIL import Image
from skimage import morphology
import torch

# ===================== 路径设置（请确认） =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# controlnet-aux 源码路径（本代码在 ANYLINE/comfyui_controlnet_aux-main）
CN_AUX_SRC = os.path.join(BASE_DIR, "comfyui_controlnet_aux-main", "src")

# 本地 Annotators 权重所在目录（本代码下载在 comfyui_controlnet_aux-main\ckpt\Illyasviel\Annotators）
ANN_DIR_A = os.path.join(BASE_DIR, "comfyui_controlnet_aux-main", "ckpts", "Illyasviel", "Annotators")

# MTEED 权重（本代码下载在 ANYLINE/ckpts/TheMistoAI/MistoLine/Anyline/MTEED.pth）
MTEED_CKPT_DEFAULT = os.path.join(BASE_DIR, "comfyui_controlnet_aux-main", "ckpts", "TheMistoAI", "MistoLine", "Anyline", "MTEED.pth")

# 把 controlnet_aux 加入 sys.path
if CN_AUX_SRC not in sys.path:
    sys.path.append(CN_AUX_SRC)
# =============================================================

# ------------------- 工具函数 -------------------
def to_float01(x):
    arr = np.array(x) if isinstance(x, Image.Image) else x
    if arr.dtype != np.float32 and arr.max() > 1:
        arr = arr.astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], -1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, -1)
    return arr

def to_pil01(arr):
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def get_intensity_mask(image_array, lower_bound, upper_bound):
    ch = image_array[:, :, 0]
    mask = np.where((ch >= lower_bound) & (ch <= upper_bound), ch, 0.0)
    mask = np.expand_dims(mask, 2).repeat(3, axis=2)
    return mask

def combine_layers(base_layer, top_layer):
    mask = top_layer.astype(bool)
    temp = 1 - (1 - top_layer) * (1 - base_layer)
    result = base_layer * (~mask) + temp * mask
    return result

#处理input
def common_annotator_call(model, tensor_image, input_batch=False, show_pbar=False, **kwargs):
    """
    纯本地版：与 ComfyUI 的 common_annotator_call 行为保持一致
    - tensor_image: torch.Tensor, 形状 [B,H,W,C] 且取值 [0,1]
    - model(np_uint8_image, output_type="np", detect_resolution=..., **kwargs) -> np.uint8 或 [0,255] np
    - 返回 torch.Tensor，形状 [B,H,W,C] 且取值 [0,1]
    """
    # 与原版一致：避免同时存在 detect_resolution
    if "detect_resolution" in kwargs:
        del kwargs["detect_resolution"]

    if "resolution" in kwargs:
        res = kwargs["resolution"]
        detect_resolution = res if isinstance(res, int) and res >= 64 else 512
        del kwargs["resolution"]
    else:
        detect_resolution = 512

    # 批输入的快速路径（模型支持一次性 batch）
    if input_batch:
        # tensor -> uint8 numpy
        np_images = np.asarray(tensor_image * 255.0, dtype=np.uint8)
        np_results = model(np_images, output_type="np", detect_resolution=detect_resolution, **kwargs)
        return torch.from_numpy(np.asarray(np_results, dtype=np.float32) / 255.0)

    # 单张循环
    batch_size = tensor_image.shape[0]
    out_tensor = None
    for i, image in enumerate(tensor_image):
        np_image = np.asarray(image.detach().cpu() * 255.0, dtype=np.uint8)  # HWC, uint8
        np_result = model(np_image, output_type="np", detect_resolution=detect_resolution, **kwargs)
        out = torch.from_numpy(np.asarray(np_result, dtype=np.float32) / 255.0)  # HWC, [0,1]
        if out_tensor is None:
            out_tensor = torch.zeros((batch_size, *out.shape), dtype=torch.float32)
        out_tensor[i] = out
    return out_tensor

# -------------------------------------------------------------

def run_mteed(img01, resolution, force_cpu=False):

    from custom_controlnet_aux.teed import TEDDetector

    model = TEDDetector.from_pretrained()
    device = "cuda" if (not force_cpu and torch.cuda.is_available()) else "cpu"
    model.to(device)

    img_t = torch.from_numpy(img01).float().unsqueeze(0)  # [1,H,W,C], [0,1]
    out_t = common_annotator_call(model, img_t, resolution=resolution, show_pbar=False)
    return out_t[0].cpu().numpy()



def run_lineart_realistic(img01, resolution, force_cpu=False):

    from custom_controlnet_aux.lineart import LineartDetector

    device = "cuda" if (not force_cpu and torch.cuda.is_available()) else "cpu"

    # 直接用你本地化后的 from_pretrained()
    det = LineartDetector.from_pretrained().to(device)

    img_t = torch.from_numpy(img01).float().unsqueeze(0)  # [1,H,W,C], [0,1]
    out_t = common_annotator_call(det, img_t, resolution=resolution, show_pbar=False)
    return out_t[0].cpu().numpy()


def invert_colors(arr: np.ndarray) -> np.ndarray:
    # 输入为 [0,1] 浮点 HWC，返回同尺寸反色结果
    return 1.0 - np.clip(arr, 0.0, 1.0)


def process_one(args):
    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[-2]

    img = Image.open(args.input).convert("RGB")
    img01 = to_float01(img)

    # 右路：realistic
    line = run_lineart_realistic(img01, args.resolution, force_cpu=args.force_cpu)
    # 白底黑线
    line_inv = invert_colors(line)
    #to_pil01(line_inv).save(os.path.join(args.outdir, f"{base}_lineart_realistic.png"))

    if not args.use_mteed:
        print("✅ 输出：", os.path.join(args.outdir, f"{base}_lineart_realistic.png"))
        return

    # 左路：MTEED
    mteed = run_mteed(img01, args.resolution, force_cpu=args.force_cpu)
    mteed_inv = invert_colors(mteed)
    #to_pil01(mteed_inv).save(os.path.join(args.outdir, f"{base}_mteed.png"))

    # 融合（与节点一致）
    masked = get_intensity_mask(line, args.lower, args.upper)
    cleaned = morphology.remove_small_objects(masked.astype(bool), min_size=args.min_size, connectivity=args.connectivity)
    masked = masked * cleaned
    masked_inv = invert_colors((masked > 0).astype(np.float32))
    #to_pil01(masked_inv).save(os.path.join(args.outdir, f"{base}_mask.png"))

    fused = combine_layers(mteed, masked)
    fused_inv = invert_colors(fused)
    to_pil01(fused_inv).save(os.path.join(args.outdir, f"{base}_anyline.png"))

    print("✅ 输出：")
    print("  ", os.path.join(args.outdir, f"{base}_anyline.png"))


def main():
    ap = argparse.ArgumentParser("AnyLine (realistic) offline")
    ap.add_argument("--input", required=True, help="输入图片路径")
    ap.add_argument("--outdir", default=os.path.join(BASE_DIR, "out"))
    ap.add_argument("--resolution", type=int, default=1280)
    ap.add_argument("--use_mteed", action="store_true",default=True, help="默认开启 MTEED")
    ap.add_argument("--lower", type=float, default=0.0)
    ap.add_argument("--upper", type=float, default=1.0)
    ap.add_argument("--min_size", type=int, default=36)
    ap.add_argument("--connectivity", type=int, default=1)
    ap.add_argument("--force_cpu", action="store_true")
    args = ap.parse_args()

    assert os.path.isdir(CN_AUX_SRC), f"找不到 controlnet-aux 源码目录（【改路径1】）：{CN_AUX_SRC}"
    assert os.path.isdir(ANN_DIR_A), f"找不到 Annotators 目录（【改路径2】）：{ANN_DIR_A}"

    process_one(args)

if __name__ == "__main__":
    main()
