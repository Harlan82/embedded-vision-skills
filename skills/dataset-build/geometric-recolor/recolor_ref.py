"""
几何体训练数据集 —— 批量换色增强
=====================================================
用途：把中性背景上的单色/多色几何体，批量换到指定的目标色，用于训练数据增强。

关键做法：按目标色显式设定 H/S/V 三个分量。
  - 只改 H、让 S/V 继承原图，是高亮度色（黄/青）发暗发闷的根因；
  - V 用「亮度比例」保留 3D 光影，S 用「比例」保留材质质感，
    这样每种颜色都统一、不发灰。
输出：原始形态 + 拉伸形态，按目标色均衡分布。
"""
import cv2
import numpy as np
import os
import shutil
import random
from pathlib import Path

# ============================================================
# 配置
# ============================================================
SRC_DIR = r"./work/"
OUT_DIR = r"./work/output_images"

# 目标色：直接用 HSV 精确控制（H:0-179, S:0-255, V:0-255）
# 选择鲜艳且符合"赤橙黄绿青蓝紫粉"常识认知的取值
TARGET_HSV = {
    "chi":   (0,   240, 215),   # 赤  纯红
    "cheng": (16,  240, 235),   # 橙  橙黄
    "huang": (28,  240, 242),   # 黄  亮黄
    "lv":    (60,  230, 210),   # 绿  草绿
    "qing":  (90,  235, 230),   # 青  青蓝
    "lan":   (117, 225, 215),   # 蓝  宝蓝
    "zi":    (140, 205, 185),   # 紫  深紫
    "fen":   (165, 120, 240),   # 粉  浅粉
}

TARGET_COUNT = 1500
SEED = 42
random.seed(SEED)

STRETCH_PARAMS = [
    (1.35, 1.00, "_hx"),
    (1.00, 1.25, "_vy"),
    (1.20, 1.15, "_hv"),
]

VARIANTS = [
    (1.00, "_"),      # 基准
    (1.08, "_b"),     # 微亮
    (0.92, "_d"),     # 微暗
]                     # (bright_factor, suffix)

DIST_TARGET = {"orig": 0.50, "stretch": 0.50}


# ============================================================
# 工具函数
# ============================================================
def hue_distance(h1, h2):
    d = np.abs(h1 - h2)
    return np.minimum(d, 180 - d)

def build_foreground_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    bgr_f = img.astype(np.float32)
    b, g, r = cv2.split(bgr_f)
    cf = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
    sat_mask = (s > 28).astype(np.uint8) * 255
    cf_mask  = (cf > 22).astype(np.uint8) * 255
    combined = cv2.bitwise_or(sat_mask, cf_mask)
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    final_mask = np.zeros_like(cleaned)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 200:
            final_mask[labels == i] = 255
    return final_mask

def recolor_accurate(img, mask, t_h, t_s, t_v, bright_factor=1.0):
    """
    精确换色：
      H  = 目标色相（精确）
      V  = 目标平均亮度 × 原图亮度比例（保留3D光影）
      S  = 目标平均饱和度 × 原图饱和度比例（保留材质质感）
    这样每种颜色都鲜艳、统一、且保留立体感
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h = hsv[:, :, 0].copy()
    s = hsv[:, :, 1].copy()
    v = hsv[:, :, 2].copy()
    fg = mask > 128

    # 1. 色相
    h[fg] = t_h

    # 2. 亮度：保留光影比例，归一化到目标均值
    ov = v[fg]
    mv = ov.mean()
    if mv > 5:
        scale = (t_v / mv) * bright_factor
        nv = np.clip(ov * scale, 35, 255)
    else:
        nv = np.full_like(ov, t_v)
    v[fg] = nv

    # 3. 饱和度：保留材质纹理比例，归一化到目标均值
    os = s[fg]
    ms = os.mean()
    if ms > 5:
        ss = t_s / ms
        ns = np.clip(os * ss, 55, 255)
    else:
        ns = np.full_like(os, t_s)
    s[fg] = ns

    out = np.stack([h, s, v], axis=-1)
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_HSV2BGR)

def validate_output(output_img, target_h):
    out_hsv = cv2.cvtColor(output_img, cv2.COLOR_BGR2HSV)
    out_s = out_hsv[:, :, 1].astype(np.float32)
    out_h = out_hsv[:, :, 0].astype(np.float32)
    bgr = output_img.astype(np.float32)
    b, g, r = cv2.split(bgr)
    cf_out = (np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b]))
    out_fg = ((out_s > 28) | (cf_out > 22)).astype(bool)
    fg_pixels = int(out_fg.sum())
    total_pixels = out_fg.shape[0] * out_fg.shape[1]
    if fg_pixels == 0:
        return False, "无前景"
    fg_ratio = fg_pixels / total_pixels
    if fg_ratio < 0.008 or fg_ratio > 0.82:
        return False, f"前景比例异常 {fg_ratio:.3f}"
    fg_hues = out_h[out_fg]
    if np.sum(hue_distance(fg_hues, target_h) <= 20) / len(fg_hues) < 0.70:
        return False, f"目标色覆盖率低"
    target_near_magenta = (target_h >= 130 and target_h <= 178)
    if not target_near_magenta:
        residual = np.sum(((fg_hues >= 138) & (fg_hues <= 172)) |
                         (hue_distance(fg_hues, 154) <= 22))
        if residual / len(fg_hues) > 0.15:
            return False, f"原色残留高"
    bg_sat = out_s[~out_fg]
    if len(bg_sat) > 0 and np.sum(bg_sat > 45) / max(len(bg_sat), 1) > 0.02:
        return False, f"背景污染"
    return True, "OK"

def stretch_image(img, sx, sy):
    h, w = img.shape[:2]
    return cv2.resize(img, (max(int(w * sx), w), max(int(h * sy), h)),
                     interpolation=cv2.INTER_LINEAR)

def detect_box_candidates(img_files, src_path):
    candidates = []
    for f in img_files:
        img = cv2.imread(str(src_path / f))
        if img is None:
            continue
        mask = build_foreground_mask(img)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels < 2:
            continue
        main_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x, y, w, h, area = stats[main_idx]
        aspect = w / h if h > 0 else 0
        img_area = mask.shape[0] * mask.shape[1]
        area_ratio = area / img_area
        score = 0.0
        score += (1.0 - min(abs(aspect - 1.0), 0.5)) * 5.0
        if 0.03 < area_ratio < 0.60:
            score += 3.0
        elif area_ratio > 0.01:
            score += 1.0
        n_comps = num_labels - 1
        if n_comps == 1:
            score += 4.0
        elif n_comps <= 3:
            score += 2.0
        roi = mask[y:y+h, x:x+w]
        if roi.sum() / (w * h * 255) > 0.5:
            score += 2.0
        candidates.append({"file": str(f), "score": round(score, 2)})
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


# ============================================================
# 主流程
# ============================================================
def main():
    src_path = Path(SRC_DIR)
    out_path = Path(OUT_DIR)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    img_files = sorted([f for f in src_path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    print(f"源图片数: {len(img_files)}")

    candidates = detect_box_candidates(img_files, src_path)
    BOX_COUNT = min(35, len(candidates))
    box_set = set(c["file"] for c in candidates[:BOX_COUNT])
    print(f"形状H候选数: {BOX_COUNT}")

    all_results = {"orig": [], "stretch": []}
    total_gen = 0

    for idx, img_file in enumerate(img_files):
        fname = str(img_file)
        img = cv2.imread(fname)
        if img is None:
            continue
        stem = Path(fname).stem
        is_box = fname in box_set
        mask = build_foreground_mask(img)

        versions = [("orig", img, mask, "")]
        if is_box:
            for sx, sy, ssuffix in STRETCH_PARAMS:
                simg = stretch_image(img, sx, sy)
                smask = build_foreground_mask(simg)
                versions.append((f"stretch{ssuffix}", simg, smask, ssuffix))

        for ver_tag, ver_img, ver_mask, ssuffix in versions:
            bucket = "orig" if ver_tag == "orig" else "stretch"
            for color_key, (th, ts, tv) in TARGET_HSV.items():
                for bf, vsuffix in VARIANTS:
                    prefix = stem if ver_tag == "orig" else f"{stem}_stretch{ssuffix}"
                    out_fname = f"{prefix}_{color_key}{vsuffix}.jpg"
                    out_file = out_path / out_fname
                    recolored = recolor_accurate(ver_img, ver_mask, th, ts, tv, bf)
                    cv2.imwrite(str(out_file), recolored, [cv2.IMWRITE_JPEG_QUALITY, 93])
                    total_gen += 1
                    ok, _ = validate_output(recolored, th)
                    if not ok:
                        os.remove(str(out_file))
                    else:
                        all_results[bucket].append(out_fname)

        if (idx + 1) % 20 == 0 or idx == len(img_files) - 1:
            print(f"  [{idx+1}/{len(img_files)}] 生成:{total_gen} "
                  f"合格 orig:{len(all_results['orig'])} "
                  f"stretch:{len(all_results['stretch'])}")

    # 分布裁剪
    print("\n" + "=" * 60)
    print("=== 最终报告 ===")
    n_orig = len(all_results["orig"])
    n_stretch = len(all_results["stretch"])
    total_avail = n_orig + n_stretch
    print(f"\n可用候选: orig={n_orig} stretch={n_stretch} 共 {total_avail}")

    tgt_orig = int(TARGET_COUNT * DIST_TARGET["orig"])
    tgt_stretch = TARGET_COUNT - tgt_orig
    take_orig = min(tgt_orig, n_orig)
    take_stretch = min(tgt_stretch, n_stretch)
    deficit = TARGET_COUNT - (take_orig + take_stretch)
    if deficit > 0:
        extras = []
        if take_orig < n_orig: extras.extend(["orig"] * (n_orig - take_orig))
        if take_stretch < n_stretch: extras.extend(["stretch"] * (n_stretch - take_stretch))
        random.shuffle(extras)
        for b in extras[:deficit]:
            if b == "orig": take_orig += 1
            else: take_stretch += 1

    random.shuffle(all_results["orig"])
    random.shuffle(all_results["stretch"])
    final_files = all_results["orig"][:take_orig] + all_results["stretch"][:take_stretch]

    selected_set = set(final_files)
    for bucket in all_results:
        for fn in all_results[bucket]:
            if fn not in selected_set:
                fp = out_path / fn
                if fp.exists():
                    os.remove(fp)

    print(f"\n最终构成:")
    print(f"  原始形状: {take_orig} 张 ({take_orig/len(final_files)*100:.1f}%)")
    print(f"  拉伸长方: {take_stretch} 张 ({take_stretch/len(final_files)*100:.1f}%)")
    print(f"  合计: {len(final_files)} 张")

    color_counts = {k: 0 for k in TARGET_HSV}
    for fn in final_files:
        for ck in TARGET_HSV:
            if f"_{ck}" in fn:
                color_counts[ck] += 1
                break
    print(f"\n各颜色分布:")
    name_map = {"chi":"赤","cheng":"橙","huang":"黄","lv":"绿","qing":"青","lan":"蓝","zi":"紫","fen":"粉"}
    for k in TARGET_HSV:
        c = color_counts[k]
        bar = "█" * max(c // 8, 0)
        print(f"  {name_map[k]:>4}({k:>6}): {c:>5} 张  {bar}")

    print(f"\n✅ 完成! 输出目录: {out_path}")
    print(f"最终文件数: {len(list(out_path.iterdir()))}")


if __name__ == "__main__":
    main()
