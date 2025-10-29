#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline submission using OpenGVLab/InternVL3_5-8B with:
- FlashAttention2 (nếu có)
- bitsandbytes 8-bit quantization (LLM int8)
- Video tiling pipeline đúng theo hướng dẫn tác giả (decord + dynamic_preprocess)
- Đọc public_test/public_test.json, xuất submission.csv (id,answer)

Chạy:
python baseline.py  --json public_test/public_test.json  --videos_root .  --output submission.csv  --model OpenGVLab/InternVL3_5-8B  --dtype bfloat16  --num_segments 8 --max_num 1 --input_size 448 --no_flash_attn
"""

import os
import re
import csv
import json
import argparse
from typing import List, Dict

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoModel, AutoTokenizer

from decord import VideoReader, cpu

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            # tie-break theo area
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    w, h = image.size
    aspect_ratio = w / h
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, w, h, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized = image.resize((target_width, target_height))
    tiles = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        tiles.append(resized.crop(box))
    assert len(tiles) == blocks
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path: str, bound=None, input_size=448, max_num=1, num_segments=8):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    transform = build_transform(input_size=input_size)

    pixel_values_list, num_patches_list = [], []
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)

    for fi in frame_indices:
        img = Image.fromarray(vr[fi].asnumpy()).convert('RGB')
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pv = [transform(t) for t in tiles]
        pv = torch.stack(pv)
        num_patches_list.append(pv.shape[0])
        pixel_values_list.append(pv)

    pixel_values = torch.cat(pixel_values_list)  # [sum_tiles, 3, H, W]
    return pixel_values, num_patches_list

LETTER_RE = re.compile(r"\b([ABCD])\b", flags=re.IGNORECASE)

def ensure_letter_choices(choices: List[str]) -> List[str]:
    letters = ["A", "B", "C", "D"]
    if all(c.strip().upper().startswith(tuple(l + "." for l in letters)) for c in choices):
        return choices
    out = []
    for i, c in enumerate(choices):
        lab = letters[i]
        c = c.strip()
        out.append(c if c.upper().startswith(lab + ".") else f"{lab}. {c}")
    return out

def build_prompt(question: str, choices: List[str]) -> str:
    guide = (
        "Bạn là trợ lý giao thông Việt Nam. Chỉ dựa vào VIDEO và câu hỏi, "
        "Hãy chọn DUY NHẤT một đáp án trong các lựa chọn. "
        "CHỈ TRẢ LỜI BẰNG 1 KÝ TỰ: A, B, C hoặc D (không giải thích)."
    )
    return f"{guide}\n\nCâu hỏi: {question}\nCác lựa chọn:\n" + "\n".join(choices) + \
           "\n\nTrả lời DUY NHẤT: A, B, C hoặc D."

def parse_answer(text: str, choices: List[str]) -> str:
    if not text:
        return "A"
    m = LETTER_RE.search(text.strip())
    if m:
        return m.group(1).upper()
    # fallback: trùng nội dung
    norm = text.lower()
    best = ("A", -1)
    for ch in ["A", "B", "C", "D"][:len(choices)]:
        body = next((c.split(".", 1)[-1].strip().lower() for c in choices
                     if c.strip().upper().startswith(ch + ".")), "")
        overlap = len(set(norm.split()) & set(body.split()))
        if overlap > best[1]:
            best = (ch, overlap)
    return best[0]

class InternVL35Baseline:
    def __init__(self, model_id: str, dtype: str = "bfloat16", use_flash_attn: bool = True):
        """
        - BNB 8-bit: giảm VRAM cho LLM weights
        - dtype (bf16/fp16) cho inputs & một số projection (vision)
        - use_flash_attn: bật FlashAttention nếu model hỗ trợ (trust_remote_code=True)
        """
        torch_dtype = getattr(torch, dtype) if dtype != "auto" else None
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            load_in_8bit=True,          # BNB 8-bit
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash_attn,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        self.dtype = dtype

        # Thiết lập deterministic để chấm (không sampling)
        self.generation_config = dict(max_new_tokens=16, do_sample=False, temperature=0.0)

    @torch.inference_mode()
    def answer_mcq(self, video_path: str, question: str, choices: List[str],
                   num_segments: int = 8, max_num: int = 1, input_size: int = 448) -> str:
        # 1) Load + tile video
        pixel_values, num_patches_list = load_video(
            video_path, num_segments=num_segments, max_num=max_num, input_size=input_size
        )
        # Đưa inputs về dtype + device của model (khi 8-bit, forward vẫn nhận fp16/bf16 cho inputs)
        pv_dtype = getattr(torch, self.dtype) if self.dtype != "auto" else torch.float32
        pixel_values = pixel_values.to(pv_dtype).to(self.model.device)

        # 2) Format prompt: Frame1..K: <image> + câu hỏi
        k = len(num_patches_list)
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(k)])
        prompt = build_prompt(question, choices)
        full_q = video_prefix + prompt

        # 3) Gọi chat theo API repo tác giả
        try:
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                full_q,
                self.generation_config,
                num_patches_list=num_patches_list,
            )
        except TypeError:
            # 1 vài version cần return_history
            response, _ = self.model.chat(
                self.tokenizer,
                pixel_values,
                full_q,
                self.generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True
            )
        return parse_answer(response, choices)

# --------- JSON loader ----------
def load_public_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        jd = json.load(f)
    return jd["data"] if isinstance(jd, dict) and "data" in jd else jd

# --------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="public_test/public_test.json")
    ap.add_argument("--videos_root", default=".")
    ap.add_argument("--output", default="submission.csv")
    ap.add_argument("--model", default="OpenGVLab/InternVL3_5-8B")
    ap.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--num_segments", type=int, default=8)
    ap.add_argument("--max_num", type=int, default=1)
    ap.add_argument("--input_size", type=int, default=448)
    ap.add_argument("--no_flash_attn", action="store_true")
    args = ap.parse_args()

    # Log môi trường
    print("=" * 60)
    print("ENVIRONMENT INFO")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: cuda:{torch.cuda.current_device()}")
    else:
        print("Running on: CPU")
    print(f"Model: {args.model}")
    print(f"Dtype: {args.dtype}")
    print(f"8-bit quantization: Enabled")
    print(f"Flash Attention: {'Disabled' if args.no_flash_attn else 'Enabled'}")
    print("=" * 60)
    print()

    data = load_public_json(args.json)
    vlm = InternVL35Baseline(
        model_id=args.model,
        dtype=args.dtype,
        use_flash_attn=(not args.no_flash_attn)
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "answer"])

        for item in data:
            sid = item["id"]
            q = item["question"]
            choices = ensure_letter_choices(item["choices"])
            rel = item["video_path"]
            vpath = rel if os.path.isabs(rel) else os.path.join(args.videos_root, rel)

            try:
                ans = vlm.answer_mcq(
                    vpath, q, choices,
                    num_segments=args.num_segments, max_num=args.max_num, input_size=args.input_size
                )
            except Exception as e:
                print(f"[WARN] {sid} failed: {e}. Fallback 'A'.")
                ans = "A"

            writer.writerow([sid, ans])
            print(f"{sid}: {ans}")

    print(f"Done. Saved to {args.output}")

if __name__ == "__main__":
    main()
