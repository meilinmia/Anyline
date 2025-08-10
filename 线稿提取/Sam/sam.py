# -*- coding: utf-8 -*-
import os
import sys
import argparse
import copy
import numpy as np
from PIL import Image
import torch
from pathlib import Path

# === 基准目录（脚本所在目录） ===
BASE_DIR = Path(__file__).resolve().parent
ROOT = str(BASE_DIR)
if ROOT not in sys.path:
    sys.path.append(ROOT)   # 能看到 sam_hq / local_groundingdino

# === 默认模型/配置路径（相对脚本目录） ===
DEFAULT_SAM_CKPT = {
    "sam_vit_h": BASE_DIR / "models" / "sam" / "sam_vit_h_4b8939.pth",
    # 需要的话可继续加：
    # "sam_vit_b": BASE_DIR / "models" / "sam" / "sam_vit_b_01ec64.pth",
    # "sam_vit_l": BASE_DIR / "models" / "sam" / "sam_vit_l_0b3195.pth",
}

DEFAULT_DINO_CFG  = BASE_DIR / "models" / "groundingdino" / "GroundingDINO_SwinT_OGC.cfg.py"
DEFAULT_DINO_CKPT = BASE_DIR / "models" / "groundingdino" / "groundingdino_swint_ogc.pth"
DEFAULT_BERT_DIR  = BASE_DIR / "models" / "bert-base-uncased"   # 目录

from sam_hq.build_sam_hq import sam_model_registry
from sam_hq.predictor import SamPredictorHQ

from local_groundingdino.datasets import transforms as T
from local_groundingdino.util.utils import clean_state_dict as gd_clean_state_dict
from local_groundingdino.util.slconfig import SLConfig as GD_SLConfig
from local_groundingdino.models import build_model as gd_build_model


# ====== 2) GroundingDINO：与 node.py 同步的工具函数 ======
# —— DINO：与 node.py 一致 ——
def _normalize_prompt(prompt: str) -> str:
    cap = (prompt or "").lower().strip()
    if not cap.endswith("."):
        cap = cap + "."
    return cap

def _gd_preprocess_pil(image_pil):
    transform = T.Compose(
        [T.RandomResize([800], max_size=1333),
         T.ToTensor(),
         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    image, _ = transform(image_pil.convert("RGB"), None)
    return image

@torch.no_grad()
def dino_boxes(dino_model, image_pil, prompt, threshold, device):
    img = _gd_preprocess_pil(image_pil).to(device)
    outputs = dino_model(img[None], captions=[_normalize_prompt(prompt)])
    logits = outputs["pred_logits"].sigmoid()[0]   # (nq,256)
    boxes  = outputs["pred_boxes"][0]              # (nq,4) cxcywh
    keep = logits.max(dim=1)[0] > threshold
    boxes = boxes[keep]
    if boxes.numel() == 0:
        return torch.zeros((0,4))
    W, H = image_pil.size
    boxes = boxes * torch.tensor([W,H,W,H], device=boxes.device)
    boxes[:, :2] = boxes[:, :2] - boxes[:, 2:]/2
    boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
    return boxes.cpu()

# —— SAM：与 node.py 一致（multimask_output=False） ——
@torch.no_grad()
def sam_masks_one_per_box(sam_model, image_pil, boxes_xyxy, device):
    if boxes_xyxy.shape[0] == 0:
        return [], []
    sam_is_hq = hasattr(sam_model, "model_name") and ("hq" in sam_model.model_name.lower())
    predictor = SamPredictorHQ(sam_model, sam_is_hq)

    np_rgba = np.array(image_pil.convert("RGBA"))
    predictor.set_image(np_rgba[..., :3])

    tb = predictor.transform.apply_boxes_torch(boxes_xyxy, np_rgba.shape[:2]).to(device)
    masks, _, _ = predictor.predict_torch(
        point_coords=None, point_labels=None, boxes=tb, multimask_output=False
    )  # (B,1,Hm,Wm)
    masks = masks.permute(1,0,2,3).cpu().numpy()

    out_imgs, out_msks = [], []
    for m in masks:
        rgba = np.array(np_rgba)  # copy
        rgba[~np.any(m, axis=0)] = np.array([0,0,0,0])
        img = torch.from_numpy((rgba[...,:3].astype(np.float32)/255.0))[None,...]  # (1,H,W,3)
        msk = torch.from_numpy((rgba[...,3].astype(np.float32)/255.0))[None,...]   # (1,H,W)
        out_imgs.append(img); out_msks.append(msk)
    return out_imgs, out_msks


# ====== 4) 模型加载 ======
def load_sam_local(checkpoint_path: str, model_type: str = "sam_vit_h", device: torch.device = None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device).eval()
    sam.model_name = os.path.basename(checkpoint_path)
    return sam, device

def load_groundingdino_local(config_path: str, weights_path: str, bert_path: str = None, device: torch.device = None):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"DINO config not found: {config_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"DINO weights not found: {weights_path}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = GD_SLConfig.fromfile(config_path)

    # 兼容本地 BERT（可选）
    if bert_path:
        args.text_encoder_type = bert_path

    dino = gd_build_model(args)
    ckpt = torch.load(weights_path, map_location="cpu")
    dino.load_state_dict(gd_clean_state_dict(ckpt["model"]), strict=False)
    dino.to(device=device).eval()
    return dino, device


# ====== 5) 主流程（批量/单图），prompt 允许为空 ======
def save_results(outdir: str, stem: str, images, masks):
    os.makedirs(outdir, exist_ok=True)
    saved = []
    for i, (img_t, m_t) in enumerate(zip(images, masks), start=1):
        # img_t: (1,H,W,3) 0..1 ; m_t: (1,H,W) 0..1
        img  = (img_t[0].numpy() * 255.0).clip(0, 255).astype(np.uint8)   # (H,W,3)
        mask = (m_t[0].numpy() * 255.0).clip(0, 255).astype(np.uint8)     # (H,W)

        # —— 关键：把背景填白（白底）——
        bg = mask < 128                  # 背景位置（False/True）
        img[bg] = 255                    # (H,W,3) 广播赋值为白色

        img_p = os.path.join(outdir, f"{stem}_cut_{i}.png")
        msk_p = os.path.join(outdir, f"{stem}_mask_{i}.png")
        Image.fromarray(img).save(img_p)                 # 白底的抠图
        Image.fromarray(mask).save(msk_p)                # 原样的二值mask（白前景/黑背景）
        saved.append((img_p, msk_p))
    return saved

    # --- 路径解析与默认补齐 ---
def _resolve(p):
    if p is None:
        return None
    p = Path(os.path.expandvars(os.path.expanduser(str(p))))
    if p.is_absolute():
        return str(p)
    cand1 = (Path.cwd() / p).resolve()
    if cand1.exists():
        return str(cand1)
    return str((BASE_DIR / p).resolve())

def _default_sam_ckpt_for(model_type: str) -> str:
    if model_type not in DEFAULT_SAM_CKPT:
        raise ValueError(f"未配置 {model_type} 的默认 SAM 权重，请传 --sam-ckpt 或在 DEFAULT_SAM_CKPT 中补上。")
    return str(DEFAULT_SAM_CKPT[model_type])



def run_once(
    sam_ckpt: str,
    dino_cfg: str,
    dino_ckpt: str,
    image_path: str,
    outdir: str,
    prompt: str = "",
    threshold: float = 0.3,
    model_type: str = "sam_vit_h",
    bert_path: str = None,
):
    # 1) 加载模型
    sam, device = load_sam_local(sam_ckpt, model_type=model_type)
    dino, _      = load_groundingdino_local(dino_cfg, dino_ckpt, bert_path=bert_path, device=device)

    # 2) 读图
    pil = Image.open(image_path).convert("RGBA")

    # 3) DINO 产框（空 prompt 会被补 "."）
    boxes = dino_boxes(dino, pil, prompt, threshold, device)   # ← 用 dino_boxes

    if boxes.shape[0] == 0:
        print(f"[INFO] No boxes found (prompt='{prompt}', thr={threshold}).")
        return

    # 4) SAM 按框分割、保存
    images, masks = sam_masks_one_per_box(sam, pil, boxes, device)   # ← 用 sam_masks_one_per_box

    stem = os.path.splitext(os.path.basename(image_path))[0]
    saved = save_results(outdir, stem, images, masks)
    for i, (img_p, msk_p) in enumerate(saved, start=1):
        print(f"[OK] {stem}: cut_{i} -> {img_p} | mask_{i} -> {msk_p}")


def main():
    ap = argparse.ArgumentParser("GroundingDINO -> SAM (no prompt needed)")
    # 改为“可选”，给出默认
    ap.add_argument("--sam-ckpt", default=None, help="SAM 权重；不填则按 --model-type 自动选择")
    ap.add_argument("--dino-cfg",  default=DEFAULT_DINO_CFG,  help="DINO 配置 .py")
    ap.add_argument("--dino-ckpt", default=DEFAULT_DINO_CKPT, help="DINO 权重 .pth")
    ap.add_argument("--image", required=True, help="输入图像路径")
    ap.add_argument("--outdir", default=BASE_DIR / "out", help="输出目录")
    ap.add_argument("--prompt", default="", help="留空即可（内部会补 '.'）")
    ap.add_argument("--thr", type=float, default=0.3, help="DINO 置信度阈值")
    ap.add_argument("--model-type", default="sam_vit_h", help="sam_vit_h / sam_vit_l / sam_vit_b / sam_hq_vit_h ...")
    ap.add_argument("--bert-path", default=DEFAULT_BERT_DIR, help="本地 bert-base-uncased 目录")
    args = ap.parse_args()

    # 自动补齐 SAM 权重
    if args.sam_ckpt is None:
        args.sam_ckpt = _default_sam_ckpt_for(args.model_type)

    # 统一解析为绝对路径
    args.sam_ckpt  = _resolve(args.sam_ckpt)
    args.dino_cfg  = _resolve(args.dino_cfg)
    args.dino_ckpt = _resolve(args.dino_ckpt)
    args.image     = _resolve(args.image)
    args.outdir    = _resolve(args.outdir)
    args.bert_path = _resolve(args.bert_path)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    run_once(
        sam_ckpt=args.sam_ckpt,
        dino_cfg=args.dino_cfg,
        dino_ckpt=args.dino_ckpt,
        image_path=args.image,
        outdir=args.outdir,
        prompt=args.prompt,
        threshold=args.thr,
        model_type=args.model_type,
        bert_path=args.bert_path,
    )

if __name__ == "__main__":
    main()