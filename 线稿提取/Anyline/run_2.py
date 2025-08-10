# -*- coding: utf-8 -*-
"""
AnyLine（保留“自定义 PIL/NumPy 处理路径”版本）
- 使用 resize_long_edge 控制分辨率（长边等于 --resolution）
- 直接给 controlnet-aux 的 Detector 传入 uint8 NumPy / PIL，不使用 common_annotator_call
- 可选仅 realistic 线稿，或与 MTEED 融合

示例：
  python run_2.py --input ./inputs/1.jpg --resolution 1280 --outdir ./outputs
  python run_2.py --input ./inputs/1.jpg --resolution 1280 --use_mteed --outdir ./outputs
  python run_2.py --input ./inputs/1.jpg --resolution 640 --use_mteed --force_cpu
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

# 本地 Annotators 权重所在目录（本代码下载在 comfyui_controlnet_aux-main\\ckpts\\Illyasviel\\Annotators）
ANN_DIR_A = os.path.join(BASE_DIR, "comfyui_controlnet_aux-main", "ckpts", "Illyasviel", "Annotators")

# 把 controlnet_aux 加入 sys.path
if CN_AUX_SRC not in sys.path:
    sys.path.append(CN_AUX_SRC)
# =============================================================

# ---------------------- 你想保留的处理函数 ----------------------
def resize_long_edge(pil, target):
    if target is None: return pil
    w, h = pil.size
    if max(w, h) == target: return pil
    if w >= h:
        return pil.resize((target, int(h * target / w)), Image.BICUBIC)
    else:
        return pil.resize((int(w * target / h), target), Image.BICUBIC)

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
# --------------------------------------------------------------

def to_float01(x):
    """x: np.uint8(HWC) or PIL -> np.float32(HWC) in [0,1]"""
    if isinstance(x, Image.Image):
        x = np.array(x)
    if x.dtype != np.float32:
        x = x.astype(np.float32) / 255.0
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    return x

def to_uint8(x01):
    """[0,1] -> uint8"""
    x01 = np.clip(x01, 0.0, 1.0)
    return (x01 * 255.0).astype(np.uint8)

def to_pil01(x01):
    x01 = np.clip(x01, 0.0, 1.0)
    return Image.fromarray((x01 * 255.0).astype(np.uint8))

def invert_colors(arr01: np.ndarray) -> np.ndarray:
    return 1.0 - np.clip(arr01, 0.0, 1.0)

# ---------------------- Detector 调用（PIL/NumPy 路径） ----------------------
def run_lineart_realistic_pil(img_pil, resolution, force_cpu=False):
    """
    输入：PIL 图（已按长边 resize），输出：np.float32[0,1] HWC
    """
    from custom_controlnet_aux.lineart import LineartDetector
    device = "cuda" if (not force_cpu and torch.cuda.is_available()) else "cpu"
    det = LineartDetector.from_pretrained().to(device)

    # 直接传 uint8 NumPy，保持你的风格
    np_in = np.array(img_pil).astype(np.uint8)
    # detect_resolution 可传也可不传；这里已经预先 resize，这里不再改分辨率
    np_out = det(np_in, output_type="np")  # HWC, uint8
    return to_float01(np_out)

def run_mteed_pil(img_pil, resolution, force_cpu=False):
    """
    输入：PIL 图（已按长边 resize），输出：np.float32[0,1] HWC
    """
    from custom_controlnet_aux.teed import TEDDetector
    device = "cuda" if (not force_cpu and torch.cuda.is_available()) else "cpu"
    det = TEDDetector.from_pretrained().to(device)

    np_in = np.array(img_pil).astype(np.uint8)
    np_out = det(np_in, output_type="np")  # HWC, uint8
    return to_float01(np_out)

# ----------------------------- 主处理流 -----------------------------
def process_one(args):
    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]

    # 读图并按“长边=resolution”缩放（你的方式）
    pil = Image.open(args.input).convert("RGB")
    pil_resized = resize_long_edge(pil, args.resolution)

    # 右路：realistic
    line01 = run_lineart_realistic_pil(pil_resized, args.resolution, force_cpu=args.force_cpu)   # [0,1]
    # 白底黑线
    line_inv01 = invert_colors(line01)
    to_pil01(line_inv01).save(os.path.join(args.outdir, f"{base}_lineart_realistic.png"))

    if not args.use_mteed:
        print("✅ 输出：", os.path.join(args.outdir, f"{base}_lineart_realistic.png"))
        return

    # 左路：MTEED
    mteed01 = run_mteed_pil(pil_resized, args.resolution, force_cpu=args.force_cpu)
    mteed_inv01 = invert_colors(mteed01)
    to_pil01(mteed_inv01).save(os.path.join(args.outdir, f"{base}_mteed.png"))

    # 融合（按你的掩膜逻辑；阈值在 [0,1] 空间）
    masked = get_intensity_mask(line01, args.lower, args.upper)         # 取线稿通道做强度阈值
    cleaned = morphology.remove_small_objects(masked.astype(bool),
                                              min_size=args.min_size,
                                              connectivity=args.connectivity)
    masked = masked * cleaned
    masked_inv = invert_colors((masked > 0).astype(np.float32))
    to_pil01(masked_inv).save(os.path.join(args.outdir, f"{base}_mask.png"))

    fused = combine_layers(mteed01, masked)
    fused_inv = invert_colors(fused)
    to_pil01(fused_inv).save(os.path.join(args.outdir, f"{base}_anyline.png"))

    print("✅ 输出：")
    print("  ", os.path.join(args.outdir, f"{base}_mteed.png"))
    print("  ", os.path.join(args.outdir, f"{base}_lineart_realistic.png"))
    print("  ", os.path.join(args.outdir, f"{base}_mask.png"))
    print("  ", os.path.join(args.outdir, f"{base}_anyline.png"))

def main():
    ap = argparse.ArgumentParser("AnyLine (自定义 PIL/NumPy 路径)")
    ap.add_argument("--input", required=True, help="输入图片路径")
    ap.add_argument("--outdir", default=os.path.join(BASE_DIR, "outputs"))
    ap.add_argument("--resolution", type=int, default=1280, help="按长边缩放到此分辨率")
    ap.add_argument("--use_mteed", action="store_true", default=True, help="默认开启 MTEED")
    ap.add_argument("--lower", type=float, default=0.0)
    ap.add_argument("--upper", type=float, default=1.0)
    ap.add_argument("--min_size", type=int, default=36)
    ap.add_argument("--connectivity", type=int, default=1)
    ap.add_argument("--force_cpu", action="store_true")
    args = ap.parse_args()

    # 目录存在性校验（和之前相同）
    assert os.path.isdir(CN_AUX_SRC), f"找不到 controlnet-aux 源码目录（【改路径1】）：{CN_AUX_SRC}"
    assert os.path.isdir(ANN_DIR_A), f"找不到 Annotators 目录（【改路径2】）：{ANN_DIR_A}"

    process_one(args)

if __name__ == "__main__":
    main()
