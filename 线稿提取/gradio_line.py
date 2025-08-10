# pip install gradio opencv-python pillow numpy
import gradio as gr
from PIL import Image
import cv2
import sys, subprocess, uuid
from pathlib import Path
from typing import Optional
import shutil

# 路径设定
BASE_DIR = Path(__file__).resolve().parent
SAM_DIR = BASE_DIR / "Sam"
SAM_IN_DIR = SAM_DIR / "inputs"
SAM_OUT_DIR = SAM_DIR / "out"
ANYLINE_DIR = BASE_DIR / "Anyline"
ANYLINE_IN_DIR = ANYLINE_DIR / "inputs"
ANYLINE_OUT_DIR = ANYLINE_DIR / "out"
for d in [SAM_IN_DIR, SAM_OUT_DIR, ANYLINE_IN_DIR, ANYLINE_OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def _clear_dir(dir_path: Path):
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    

def _save_img(pil_img: Image.Image, target_dir: Path, name: str) -> Path:
    path = target_dir / f"{name}.png"
    pil_img.save(path)
    return path

def _run_cmd(args, cwd: Path, timeout=600):
    proc = subprocess.run(
        args, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=timeout
    )
    return proc.returncode, proc.stdout

def _first_image(dir_path: Path) -> Optional[Path]:
    for pat in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        files = list(dir_path.glob(pat))
        if files:
            return files[0]
    return None

def process_image(pil_img: Image.Image):
    if pil_img is None:
        raise gr.Error("请先上传图片。")

    # 每次运行前清空目录
    _clear_dir(SAM_IN_DIR)
    _clear_dir(SAM_OUT_DIR)
    _clear_dir(ANYLINE_IN_DIR)
    _clear_dir(ANYLINE_OUT_DIR)

    req_id = uuid.uuid4().hex[:8]

    # 1) 保存原图给 SAM
    img_path = _save_img(pil_img, SAM_IN_DIR, req_id)

    # 2) SAM 分割（输出即为 cut 后结果）
    sam_out_subdir = SAM_OUT_DIR / req_id
    sam_out_subdir.mkdir(parents=True, exist_ok=True)
    rc, sam_log = _run_cmd(
        [sys.executable, "sam.py", "--image", str(img_path), "--outdir", str(sam_out_subdir)],
        cwd=SAM_DIR
    )
    # 可选：调试打印
    # print(sam_log)

    sam_result = _first_image(sam_out_subdir)
    if sam_result is None:
        raise gr.Error("SAM 未生成输出，请检查 Sam/out/ 和脚本参数。")

    # 3) 直接把 SAM 的结果送入 Anyline（无需掩膜计算）
    rc, any_log = _run_cmd(
        [sys.executable, "run.py", "--input", str(sam_result), "--outdir", str(ANYLINE_OUT_DIR)],
        cwd=ANYLINE_DIR
    )
    # 可选：调试打印
    # print(any_log)

    final_img_path = _first_image(ANYLINE_OUT_DIR)
    if final_img_path is None:
        raise gr.Error("Anyline 未生成输出，请检查 Anyline/out/ 和脚本参数。")

    return Image.open(final_img_path)

# Gradio UI
with gr.Blocks(title="线稿提取") as demo:
    gr.Markdown("## 线稿提取")

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(label="上传图片", type="pil", image_mode="RGB", height=480)
        with gr.Column(scale=1):
            out = gr.Image(label="提取结果", height=480)

    with gr.Row():
        btn = gr.Button("运行", variant="primary")
        clr = gr.ClearButton(components=[inp, out], value="清空")  # ← 清空页面按钮

    # 运行
    btn.click(process_image, inputs=inp, outputs=out)

    # 只要更换/重新上传输入图，自动清空右侧结果
    inp.change(lambda _ : None, inputs=inp, outputs=out)



if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
