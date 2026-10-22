# -*- coding: utf-8 -*-
"""
sdpipe —— 「相机照片 -> YOLO 检测数据集 -> 平台可导入 ZIP」共享核心库。

硬约束（都是踩坑换来的，改前先读 SKILL.md 对应条目）：
  A. 不依赖任何硬编码路径，一切从 CLI 参数来。
  B. 绝不做批量删除（不 rmtree、不 os.remove）。需要「重来」就换一个新的唯一输出目录。
  C. 文本一律 utf-8 + LF。本文件内**不出现反斜杠转义**（用 LF = chr(10) 代替），
     因为少数字符串通道会把源码里的换行转义破坏成字面量 /n，使 data.yaml 变成非法 YAML。
     同理：正则里不写 \\d 之类，改用 [0-9]。
  D. 平台目录结构固定为 train/images、train/labels、val/images、val/labels；
     data.yaml **不写 path**（写了会按当前工作目录解析，换机器就找不到图片）。
  E. 图片读取统一走 imread_u()：Windows 中文路径下 cv2.imread 会静默返回 None。

主要可调参数见 DEFAULTS，全部可被 CLI 覆盖。
"""
import os
import io
import re
import json
import math
import random
import zipfile

import numpy as np
import cv2

LF = chr(10)
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
KEEP = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")

DEFAULTS = {
    "sat_lo": 60,               # mask 基础饱和度阈值（白底通常 < 40）
    "val_max": 0,               # 暗部补充规则；0 = 关闭（背景有深色机构件时必须关）
    "cand_s_min": 125,          # 候选级饱和度下限，剔手指/肤色（实测手指 90~118）
    "tighten_k": 0.5,           # 收紧阈值 = k * 目标自身饱和中位数
    "tighten_hi": 130,          # 收紧阈值上限
    "clip_tol": 0.10,           # 收紧后裁剪率超此值 -> 放弃收紧（防咬进目标）
    "fallback_hi": (120, 135, 150),
    "fallback_lo": 55,
    "area_lo": 0.02,
    "area_hi": 0.35,
    "ratio_max": 2.0,
    "fill_min": 0.35,
    "center_box": 0.45,         # |cx-0.5| 与 |cy-0.5| 上限；0 = 关闭
    "center_w": 0.05,           # 择优时中心邻近度的权重
    "edge_margin": 0.0,         # 目标可能被画面边缘裁切时必须为 0
    "val_ratio": 0.15,
    "seed": 42,
}

CLASS_ORDER_HINT = ["class_1", "class_2", "class_3",
                    "class_4", "class_5", "class_6",
                    "class_7", "class_8", "class_9"]
# 示例值。按你的实际类别顺序替换；仅用于给用户建议，不参与判定


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def imread_u(path):
    """中文/非 ASCII 路径下唯一可靠的读图方式。cv2.imread 会静默返回 None。"""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def imencode_u(ext, img, quality=92):
    params = []
    if ext.lower() in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(ext, img, params)
    return buf.tobytes() if ok else None


def sanitize_name(base):
    """文件名清洗成 [A-Za-z0-9._-] —— 平台拒收空格与括号（实测 2186 个中招）。"""
    stem, ext = os.path.splitext(base)
    out, prev_bad = [], False
    for ch in stem:
        if ch in KEEP:
            out.append(ch)
            prev_bad = False
        else:
            if not prev_bad:
                out.append("_")
            prev_bad = True
    s = "".join(out).strip("_")
    return (s or "img") + ext.lower()


def nkey(name):
    """自然排序：cam_9 排在 cam_10 前面。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split("([0-9]+)", name)]


def list_images(src):
    try:
        names = os.listdir(src)
    except OSError:
        return []
    fs = [os.path.join(src, n) for n in names if n.lower().endswith(EXTS)]
    fs.sort(key=lambda p: nkey(os.path.basename(p)))
    return fs


def _morph(fg):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    return fg


def _fg_mask(img, thr, val_max):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    fg = (s > thr)
    if val_max and val_max > 0:
        fg = fg | (hsv[:, :, 2] < val_max)
    return _morph((fg.astype(np.uint8) * 255)), s


def blob_for_box(img, thr, val_max, box_px):
    """在 base 掩膜里取「与给定框重叠最多」的那个连通块（= 目标本体）。

    为什么不能直接取「全局最大块」：画面里常有比目标更大的异物（手指、手掌、
    线缆），此时全局最大块与框零重叠，clip 会恒等于 1.000 —— 这是**指标失真**，
    不是标注错误（实测 7 个数据集四千余张里有 10 张踩到，逐张看图后全是正确标注）。
    """
    m, _ = _fg_mask(img, thr, val_max)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0
    H, W = m.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box_px]
    x0, x1 = max(0, min(W, x0)), max(0, min(W, x1))
    y0, y1 = max(0, min(H, y0)), max(0, min(H, y1))
    best, best_inter = None, 0
    for c in cnts:
        cm = np.zeros((H, W), np.uint8)
        cv2.drawContours(cm, [c], -1, 255, -1)
        inter = int((cm[y0:y1, x0:x1] > 0).sum())
        if inter > best_inter:
            best, best_inter = cm, inter
    if best is None or best_inter == 0:
        return None, 0
    return best, int((best > 0).sum())


def candidates(img, thr, P, cand_s_min=None, excl=None):
    """按几何判据 + 候选级饱和度过滤，返回候选列表。"""
    H, W = img.shape[:2]
    m, S = _fg_mask(img, thr, P["val_max"])
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cand_s_min is None:
        cand_s_min = P["cand_s_min"]

    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a <= 0:
            continue
        ar = a / float(W * H)
        if ar < P["area_lo"] or ar > P["area_hi"]:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw <= 0 or bh <= 0:
            continue
        ratio = bw / float(bh)
        fill = a / float(bw * bh)
        if ratio > P["ratio_max"] or fill < P["fill_min"]:
            continue
        cx, cy = (x + bw / 2.0) / W, (y + bh / 2.0) / H
        cb = P["center_box"]
        if cb > 0 and (abs(cx - 0.5) > cb or abs(cy - 0.5) > cb):
            continue
        em = P["edge_margin"]
        if em > 0 and (x <= W * em or y <= H * em or
                       x + bw >= W * (1 - em) or y + bh >= H * (1 - em)):
            continue
        if excl:
            hit = False
            for (ex, ey, er) in excl:
                if math.hypot(cx - ex, cy - ey) <= er:
                    hit = True
                    break
            if hit:
                continue
        cm = np.zeros((H, W), np.uint8)
        cv2.drawContours(cm, [c], -1, 255, -1)
        sat_med = float(np.median(S[cm > 0])) if (cm > 0).any() else 0.0
        if cand_s_min > 0 and sat_med < cand_s_min:
            continue
        hull = cv2.convexHull(c)
        out.append(dict(c=c, area=float(a), area_ratio=float(ar), x=int(x), y=int(y),
                        w=int(bw), h=int(bh), ratio=float(ratio), fill=float(fill),
                        cx=float(cx), cy=float(cy), sat=sat_med,
                        hull_ratio=float(cv2.contourArea(hull) / a) if a > 0 else 0.0))
    return out


def pick_best(cands, center_w):
    if not cands:
        return None
    def score(c):
        d = math.hypot(c["cx"] - 0.5, c["cy"] - 0.5)
        return c["area_ratio"] - center_w * d
    return max(cands, key=score)


def _clip_ratio(blob_mask, box_px):
    """目标在 base 阈值下的完整掩膜，有多少比例落在给定框之外。"""
    if blob_mask is None:
        return None
    x0, y0, x1, y1 = box_px
    H, W = blob_mask.shape[:2]
    x0 = max(0, min(W, int(round(x0))))
    x1 = max(0, min(W, int(round(x1))))
    y0 = max(0, min(H, int(round(y0))))
    y1 = max(0, min(H, int(round(y1))))
    tot = int((blob_mask > 0).sum())
    if tot <= 0:
        return None
    inside = int((blob_mask[y0:y1, x0:x1] > 0).sum())
    return 1.0 - inside / float(tot)


def _norm_box(c, W, H):
    x0 = c["x"] / float(W)
    y0 = c["y"] / float(H)
    x1 = (c["x"] + c["w"]) / float(W)
    y1 = (c["y"] + c["h"]) / float(H)
    eps = 2e-4
    x0 = max(eps, min(1 - eps, x0))
    y0 = max(eps, min(1 - eps, y0))
    x1 = max(eps, min(1 - eps, x1))
    y1 = max(eps, min(1 - eps, y1))
    if x1 <= x0:
        x1 = min(1 - eps, x0 + eps)
    if y1 <= y0:
        y1 = min(1 - eps, y0 + eps)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bw, bh = (x1 - x0), (y1 - y0)
    return [round(cx, 6), round(cy, 6), round(bw, 6), round(bh, 6)]


# --------------------------------------------------------------------------
# 单图标注（自适应阈值 + 三层回退）
# --------------------------------------------------------------------------
def annotate_image(img, P, excl=None):
    """返回 dict：status / box(归一化 cxcywh) / box_px / thr / sat0 / area_ratio / clip。"""
    H, W = img.shape[:2]
    res = dict(status="miss", ok=False, thr=None, sat0=None, area_ratio=None,
               clip=None, cx=None, cy=None, box=None, box_px=None, path=None)

    base = pick_best(candidates(img, P["sat_lo"], P, excl=excl), P["center_w"])
    chosen = None
    status = None
    thr_used = None
    sat0 = None
    path = None

    if base is not None:
        # ---- 正常路径：先用 base 定位，再按目标自身饱和度决定是否收紧 ----
        sat0 = base["sat"]
        thr = int(round(min(P["tighten_hi"], max(P["sat_lo"], P["tighten_k"] * sat0))))
        chosen, status, thr_used, path = base, "strict", P["sat_lo"], "strict"
        if thr > P["sat_lo"]:
            base_px = (base["x"], base["y"], base["x"] + base["w"], base["y"] + base["h"])
            blob, _ = blob_for_box(img, P["sat_lo"], P["val_max"], base_px)
            b2 = pick_best(candidates(img, thr, P, excl=excl), P["center_w"])
            if b2 is None:
                status = "strict_no_tighten_cand"
            else:
                b2_px = (b2["x"], b2["y"], b2["x"] + b2["w"], b2["y"] + b2["h"])
                cr = _clip_ratio(blob, b2_px)
                if cr is not None and cr <= P["clip_tol"]:
                    chosen, status, thr_used = b2, "strict_tightened", thr
                else:
                    status = "strict_no_tighten"
    else:
        # ---- 回退 1：高阈值，从「手指+目标粘连块」里捞出目标本体 ----
        for hi in P["fallback_hi"]:
            b = pick_best(candidates(img, hi, dict(P, cand_s_min=0), excl=excl), P["center_w"])
            if b is not None:
                chosen, status, thr_used, sat0, path = b, "fallback_hi_%d" % hi, hi, b["sat"], "fallback_hi"
                break
        # ---- 回退 2：低阈值（仍高于阴影），救暗色工件 ----
        if chosen is None:
            b = pick_best(candidates(img, P["fallback_lo"], dict(P, cand_s_min=0), excl=excl),
                          P["center_w"])
            if b is not None:
                chosen = b
                status = "fallback_lo_%d" % P["fallback_lo"]
                thr_used = P["fallback_lo"]
                sat0 = b["sat"]
                path = "fallback_lo"

    if chosen is None:
        return res

    box_px = (chosen["x"], chosen["y"], chosen["x"] + chosen["w"], chosen["y"] + chosen["h"])
    blob, _ = blob_for_box(img, P["sat_lo"], P["val_max"], box_px)
    res.update(status=status, ok=True, thr=thr_used, sat0=sat0,
               area_ratio=chosen["area_ratio"], clip=_clip_ratio(blob, box_px),
               cx=chosen["cx"], cy=chosen["cy"], path=path)
    res["box"] = _norm_box(chosen, W, H)
    res["box_px"] = [int(chosen["x"]), int(chosen["y"]),
                     int(chosen["x"] + chosen["w"]), int(chosen["y"] + chosen["h"])]
    res["ratio"], res["fill"] = chosen["ratio"], chosen["fill"]
    return res


# --------------------------------------------------------------------------
# 固定杂物自动识别（把「人工看标定清单」自动化）
# --------------------------------------------------------------------------
def detect_fixed_artifacts(records, frac=0.10, min_count=8, dist_min=0.30,
                           tol=0.05, grid=0.02):
    """在整批结果里找「重复出现在同一位置、且远离画面中心」的框 -> 固定杂物。

    固定机位的照片里，设备边框/线缆/色卡的位置几乎逐像素重复；
    真实工件的坐标随拍摄抖动。用中心聚类即可把它们分出来。
    """
    ok = [r for r in records if r.get("ok") and r.get("cx") is not None]
    if len(ok) < min_count:
        return []
    buckets = {}
    for r in ok:
        key = (int(round(r["cx"] / grid)), int(round(r["cy"] / grid)))
        buckets.setdefault(key, []).append(r)
    thr = max(min_count, int(len(ok) * frac))
    arts = []
    for key, rs in buckets.items():
        if len(rs) < thr:
            continue
        cx = float(np.mean([r["cx"] for r in rs]))
        cy = float(np.mean([r["cy"] for r in rs]))
        if math.hypot(cx - 0.5, cy - 0.5) < dist_min:
            continue                      # 靠近中心的重复目标不算杂物（可能就是工件）
        spread = float(np.mean([math.hypot(r["cx"] - cx, r["cy"] - cy) for r in rs]))
        if spread > grid * 1.5:
            continue                      # 位置不固定 -> 不是固定杂物
        arts.append(dict(cx=round(cx, 4), cy=round(cy, 4), count=len(rs),
                         radius=tol, area_ratio=round(float(np.mean(
                             [r["area_ratio"] for r in rs])), 4)))
    arts.sort(key=lambda d: -d["count"])
    return arts


# --------------------------------------------------------------------------
# 构建数据集
# --------------------------------------------------------------------------
def _yaml_bytes(specs):
    """平台规范：不写 path；train/val 指向 split 下的 images。"""
    lines = ["train: train/images", "val: val/images", "names:"]
    for (src, cls, cid) in specs:
        lines.append("  " + str(int(cid)) + ": " + cls)
    return (LF.join(lines) + LF).encode("utf-8")


def _label_bytes(cid, box):
    return ("%d %.6f %.6f %.6f %.6f" % (cid, box[0], box[1], box[2], box[3]) + LF).encode("utf-8")


def build_dataset(specs, zip_path, P=None, gray_frac=0.0, gray_val_ratio=0.15,
                  audit_dir=None, audit_n=24, suppress_fixed=True, log=None):
    """specs = [(src_dir, cls_name, cid), ...]  ->  平台可导入 ZIP。

    一次调用即可完成：标注 -> （可选）灰度增广 -> 打包。多个 spec 即等于「多数据集合并」。
    """
    P = dict(DEFAULTS)
    P.update(P or {})
    rng = random.Random(P["seed"])
    out = dict(zip=zip_path, classes=[c for (_, c, _) in specs], per_class={},
               total=0, hit=0, miss=[], artifacts=[], threshold_hist={},
               clip_p50=None, report=None)

    def say(s):
        if log:
            log(s)

    # ---------- pass 1：标注 ----------
    records = []
    for (src, cls, cid) in specs:
        fs = list_images(src)
        say("[%s] 图片 %d 张" % (cls, len(fs)))
        for f in fs:
            img = imread_u(f)
            if img is None:
                records.append(dict(file=f, cls=cls, cid=int(cid), status="read_fail", ok=False))
                continue
            r = annotate_image(img, P)
            r.update(file=f, cls=cls, cid=int(cid))
            records.append(r)

    # ---------- 固定杂物自动抑制 ----------
    arts = detect_fixed_artifacts(records)
    out["artifacts"] = arts
    if arts:
        say("发现可疑固定杂物 %d 处：%s" % (len(arts), [(a["cx"], a["cy"], a["count"]) for a in arts]))
    if suppress_fixed and arts:
        fixed = 0
        for r in records:
            if not r.get("ok") or r.get("cx") is None:
                continue
            if any(math.hypot(r["cx"] - a["cx"], r["cy"] - a["cy"]) <= a["radius"] for a in arts):
                img = imread_u(r["file"])
                if img is None:
                    continue
                r2 = annotate_image(img, P, excl=[(a["cx"], a["cy"], a["radius"]) for a in arts])
                if r2.get("ok"):
                    r2.update(file=r["file"], cls=r["cls"], cid=r["cid"])
                    r.update(r2)
                    fixed += 1
        say("固定杂物抑制：修正 %d 张" % fixed)
        out["fixed_fix"] = fixed

    # ---------- 划分 train/val + 灰度抽取 ----------
    per_cls = {}
    for r in records:
        per_cls.setdefault(r["cls"], []).append(r)
    for cls, rs in per_cls.items():
        ok_rs = [r for r in rs if r.get("ok")]
        rng.shuffle(ok_rs)
        n_val = int(round(len(ok_rs) * P["val_ratio"]))
        for i, r in enumerate(ok_rs):
            r["split"] = "val" if i < n_val else "train"
            r["is_gray"] = False
            r["gray_split"] = None
        n_gray = int(round(len(ok_rs) * gray_frac)) if gray_frac > 0 else 0
        gray_picked = ok_rs[:n_gray]
        n_gtr = int(round(n_gray * (1 - gray_val_ratio)))
        for i, r in enumerate(gray_picked):
            r["is_gray"] = True
            r["gray_split"] = "train" if i < n_gtr else "val"

    # ---------- pass 2：写 ZIP ----------
    used = set()
    stat = {}
    n_color = n_gray = 0
    hits, missed = [], []
    clips = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("data.yaml", _yaml_bytes(specs))
        for r in records:
            if not r.get("ok"):
                missed.append(r)
                continue
            base = sanitize_name(os.path.basename(r["file"]))
            stem = os.path.splitext(base)[0]
            name = "%s__%s" % (r["cls"], stem)
            k = 2
            while ("%s/%s" % (r["split"], name)) in used:
                name = "%s__%s_%d" % (r["cls"], stem, k)
                k += 1
            used.add("%s/%s" % (r["split"], name))
            r["name"] = name
            z.write(r["file"], "%s/images/%s.jpg" % (r["split"], name))
            z.writestr("%s/labels/%s.txt" % (r["split"], name),
                       _label_bytes(r["cid"], r["box"]))
            n_color += 1
            hits.append(r)
            stat[r["status"]] = stat.get(r["status"], 0) + 1
            if r.get("clip") is not None:
                clips.append(r["clip"])
            if r["is_gray"]:
                img = imread_u(r["file"])
                if img is not None:
                    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    g3 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                    buf = imencode_u(".jpg", g3)
                    if buf:
                        gname = name + "_g"
                        z.writestr("%s/images/%s.jpg" % (r["gray_split"], gname), buf)
                        z.writestr("%s/labels/%s.txt" % (r["gray_split"], gname),
                                   _label_bytes(r["cid"], r["box"]))
                        n_gray += 1

    # ---------- 逐图清单（给 render_flagged 与人工复核用） ----------
    recs = [dict(name=r.get("name"), file=os.path.basename(r["file"]), cls=r["cls"],
                 status=r["status"], path=r.get("path"), ok=bool(r.get("ok")),
                 clip=(round(r["clip"], 4) if r.get("clip") is not None else None),
                 area_ratio=(round(r["area_ratio"], 4) if r.get("area_ratio") is not None else None),
                 sat0=(round(r["sat0"], 1) if r.get("sat0") is not None else None),
                 thr=r.get("thr"), split=r.get("split"), gray=bool(r.get("is_gray")))
            for r in records]
    with io.open(os.path.splitext(zip_path)[0] + "_records.json", "w", encoding="utf-8") as fp:
        json.dump(recs, fp, ensure_ascii=False)

    # ---------- 统计 ----------
    for cls in out["classes"]:
        rs = [r for r in records if r["cls"] == cls]
        ok = [r for r in rs if r.get("ok")]
        out["per_class"][cls] = dict(total=len(rs), hit=len(ok),
                                     train=len([r for r in ok if r["split"] == "train"]),
                                     val=len([r for r in ok if r["split"] == "val"]),
                                     gray=len([r for r in ok if r["is_gray"]]))
    out["total"] = len(records)
    out["hit"] = len(hits)
    out["miss"] = [dict(file=os.path.basename(r["file"]), cls=r["cls"], status=r["status"])
                   for r in missed]
    out["threshold_hist"] = stat
    out["n_color"] = n_color
    out["n_gray"] = n_gray
    out["clip_p50"] = float(np.median(clips)) if clips else None
    out["clip_p90"] = float(np.percentile(clips, 90)) if clips else None
    out["zip_mb"] = round(os.path.getsize(zip_path) / 1048576.0, 2)

    # ---------- 审计图 ----------
    if audit_dir:
        try:
            os.makedirs(audit_dir, exist_ok=True)
            out["audit"] = render_audit(zip_path, audit_dir, n=audit_n)
            out["flagged"] = render_flagged(zip_path, os.path.join(audit_dir, "flagged.png"))
        except Exception as e:      # 审计图不该影响主流程
            say("审计图渲染失败（不影响数据集）：%s" % e)

    # ---------- 报告 ----------
    stem = os.path.splitext(zip_path)[0]
    with io.open(stem + "_report.json", "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    lines = ["ZIP: %s  (%.2f MB)" % (zip_path, out["zip_mb"]),
             "总计 %d 张，命中 %d，未命中 %d" % (out["total"], out["hit"], len(out["miss"])),
             "彩色 %d / 灰度 %d" % (n_color, n_gray),
             "阈值路径: %s" % stat,
             "裁剪率 clip p50=%s p90=%s" % (out["clip_p50"], out["clip_p90"]),
             ""]
    lines.append("%-14s %6s %6s %7s %6s %6s" % ("class", "total", "hit", "train", "val", "gray"))
    for cls, d in out["per_class"].items():
        lines.append("%-14s %6d %6d %7d %6d %6d" % (cls, d["total"], d["hit"], d["train"],
                                                    d["val"], d["gray"]))
    if out["miss"]:
        lines.append("")
        lines.append("未命中 %d 张:" % len(out["miss"]))
        for m in out["miss"][:40]:
            lines.append("   %-22s %-12s %s" % (m["file"], m["cls"], m["status"]))
    if arts:
        lines.append("")
        lines.append("可疑固定杂物（已自动抑制）:")
        for a in arts:
            lines.append("   center=(%.3f, %.3f) 出现 %d 次  面积比 %.3f" %
                         (a["cx"], a["cy"], a["count"], a["area_ratio"]))
    txt = LF.join(lines) + LF
    with io.open(stem + "_report.txt", "w", encoding="utf-8") as fp:
        fp.write(txt)
    out["report"] = stem + "_report.txt"
    say(txt)
    return out


# --------------------------------------------------------------------------
# 合并多个已有数据集（原样搬运 + 类别 id 重排 + 可选灰度）
# --------------------------------------------------------------------------
def _zip_entries(zip_path):
    """兼容两种布局：images/train/... 与 train/images/...；返回 (split, kind, stem, member)。"""
    z = zipfile.ZipFile(zip_path)
    out = []
    for n in z.namelist():
        low = n.lower()
        if not low.endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        parts = n.split("/")
        if len(parts) < 3:
            continue
        if parts[0] in ("train", "val", "valid", "test"):
            split, kind = parts[0], parts[1]
        elif parts[1] in ("train", "val", "valid", "test"):
            kind, split = parts[0], parts[1]
        else:
            continue
        if split == "valid":
            split = "val"
        if split not in ("train", "val") or kind != "images":
            continue
        out.append((split, os.path.splitext(os.path.basename(n))[0], n))
    return z, out


def merge_zips(specs, zip_path, P=None, gray_frac=0.0, gray_val_ratio=0.15,
               audit_dir=None, audit_n=24, log=None):
    """specs = [(zip_path, cls_name, cid), ...]。把每张图和标签搬进新 zip，
    类别 id 按给定 cid 重写，文件名加 <cls>__ 前缀防跨数据集撞名。"""
    P = dict(DEFAULTS)
    P.update(P or {})
    rng = random.Random(P["seed"])
    out = dict(zip=zip_path, total=0, n_color=0, n_gray=0, per_class={},
               sources=[], artifacts=[])
    def say(s):
        if log:
            log(s)

    plan = []
    for (zp, cls, cid) in specs:
        z, ents = _zip_entries(zp)
        lbl_map = {}
        for n in z.namelist():
            if n.lower().endswith(".txt"):
                lbl_map[os.path.splitext(os.path.basename(n))[0]] = n
        items = []
        for (split, stem, member) in ents:
            if stem not in lbl_map:
                continue
            items.append((split, stem, member, lbl_map[stem]))
        say("[%s] 来自 %s：%d 张" % (cls, os.path.basename(zp), len(items)))
        rs = []
        for i, (split, stem, member, lbln) in enumerate(items):
            rs.append(dict(split=split, stem=stem, member=member, lbln=lbln,
                           is_gray=False, gray_split=None, zp=zp))
        rng.shuffle(rs)
        n_gray = int(round(len(rs) * gray_frac)) if gray_frac > 0 else 0
        n_gtr = int(round(n_gray * (1 - gray_val_ratio)))
        for i, r in enumerate(rs[:n_gray]):
            r["is_gray"] = True
            r["gray_split"] = "train" if i < n_gtr else "val"
        out["sources"].append(dict(cls=cls, cid=int(cid), zip=zp, n=len(rs)))
        out["per_class"][cls] = dict(total=len(rs))
        plan.append((cls, int(cid), zp, rs))
        out["total"] += len(rs)

    zc = {}
    def zopen(p):
        if p not in zc:
            zc[p] = zipfile.ZipFile(p)
        return zc[p]

    used = set()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("data.yaml", _yaml_bytes(specs))
        for (cls, cid, zp, rs) in plan:
            zs = zopen(zp)
            for r in rs:
                stem = sanitize_name(r["stem"])
                name = "%s__%s" % (cls, stem)
                k = 2
                while ("%s/%s" % (r["split"], name)) in used:
                    name = "%s__%s_%d" % (cls, stem, k)
                    k += 1
                used.add("%s/%s" % (r["split"], name))
                raw = zs.read(r["lbln"]).decode("utf-8", "ignore").split()
                txt = " ".join(raw[1:]) if len(raw) >= 5 else ""
                z.writestr("%s/images/%s.jpg" % (r["split"], name), zs.read(r["member"]))
                z.writestr("%s/labels/%s.txt" % (r["split"], name),
                           ("%d %s" % (cid, txt) + LF).encode("utf-8"))
                out["n_color"] += 1
                if r["is_gray"]:
                    img = cv2.imdecode(np.frombuffer(zs.read(r["member"]), np.uint8),
                                       cv2.IMREAD_COLOR)
                    if img is not None:
                        g3 = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                          cv2.COLOR_GRAY2BGR)
                        buf = imencode_u(".jpg", g3)
                        if buf:
                            gname = name + "_g"
                            z.writestr("%s/images/%s.jpg" % (r["gray_split"], gname), buf)
                            z.writestr("%s/labels/%s.txt" % (r["gray_split"], gname),
                                       ("%d %s" % (cid, txt) + LF).encode("utf-8"))
                            out["n_gray"] += 1
    out["zip_mb"] = round(os.path.getsize(zip_path) / 1048576.0, 2)
    if audit_dir:
        os.makedirs(audit_dir, exist_ok=True)
        out["audit"] = render_audit(zip_path, audit_dir, n=audit_n)
    stem = os.path.splitext(zip_path)[0]
    with io.open(stem + "_report.json", "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    txt = LF.join(["合并 -> %s (%.2f MB)" % (zip_path, out["zip_mb"]),
                   "彩色 %d / 灰度 %d" % (out["n_color"], out["n_gray"])] +
                  ["  %-14s id=%d  %d 张" % (c, i, d["total"])
                   for c, i, zp, rs in plan for d in [out["per_class"][c]]]) + LF
    with io.open(stem + "_report.txt", "w", encoding="utf-8") as fp:
        fp.write(txt)
    say(txt)
    return out


# --------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------
def verify_zip(zip_path, check_ultralytics=True):
    """结构 / 配对 / 标签格式 / 坐标 / yaml / 文件名合规。平台导入前的最后一道闸。"""
    res = dict(zip=zip_path, ok=True, problems=[], stats={})
    z = zipfile.ZipFile(zip_path)
    names = z.namelist()
    res["crc_ok"] = z.testzip() is None
    if not res["crc_ok"]:
        res["problems"].append("CRC 校验失败")
        res["ok"] = False

    tops = sorted(set(n.split("/")[0] for n in names))
    res["top_level"] = tops
    if not set(tops) <= {"data.yaml", "train", "val"}:
        res["problems"].append("顶层条目异常: %s（应为 data.yaml + train/ + val/）" % tops)
        res["ok"] = False
    if "data.yaml" not in tops:
        res["problems"].append("缺少 data.yaml")
        res["ok"] = False

    bad_chars = []
    for n in names:
        if n == "data.yaml":
            continue
        base = os.path.basename(n)
        if any(ch not in KEEP for ch in os.path.splitext(base)[0]):
            bad_chars.append(n)
    res["charset_violations"] = len(bad_chars)
    if bad_chars:
        res["problems"].append("文件名含非法字符 %d 个，例: %s" % (len(bad_chars), bad_chars[:3]))
        res["ok"] = False

    zero = [n for n in names if z.getinfo(n).file_size == 0]
    if zero:
        res["problems"].append("零字节条目 %d 个" % len(zero))
        res["ok"] = False

    yaml_txt = z.read("data.yaml").decode("utf-8") if "data.yaml" in tops else ""
    res["data_yaml"] = yaml_txt
    if "/n " in yaml_txt or yaml_txt.rstrip().endswith(":/n"):
        res["problems"].append("data.yaml 含字面量 /n（非法 YAML）")
        res["ok"] = False
    if yaml_txt.count(LF) < 3:
        res["problems"].append("data.yaml 换行异常")
        res["ok"] = False
    if "\r" in yaml_txt:
        res["problems"].append("data.yaml 含 CR（应为纯 LF）")
    try:
        import yaml as _yaml
        res["yaml_parsed"] = _yaml.safe_load(yaml_txt)
        if "path" in (res["yaml_parsed"] or {}):
            res["problems"].append("data.yaml 不应写 path（会按 cwd 解析）")
            res["ok"] = False
    except Exception as e:
        res["problems"].append("data.yaml 解析失败: %s" % e)
        res["ok"] = False

    for split in ("train", "val"):
        im = set(os.path.splitext(os.path.basename(n))[0]
                 for n in names if n.startswith(split + "/images/"))
        lb = set(os.path.splitext(os.path.basename(n))[0]
                 for n in names if n.startswith(split + "/labels/"))
        res["stats"][split] = dict(images=len(im), labels=len(lb),
                                   missing_label=len(im - lb), extra_label=len(lb - im))
        if im - lb or lb - im:
            res["problems"].append("%s: images/labels 不配对（缺标签 %d，多标签 %d）"
                                   % (split, len(im - lb), len(lb - im)))
            res["ok"] = False

    bad = 0
    coords = []
    cids = {}
    for n in names:
        if not n.startswith(("train/labels/", "val/labels/")):
            continue
        p = z.read(n).decode("utf-8", "ignore").split()
        if len(p) != 5:
            bad += 1
            continue
        try:
            c = int(p[0])
            v = [float(t) for t in p[1:]]
        except Exception:
            bad += 1
            continue
        cids[c] = cids.get(c, 0) + 1
        coords.extend(v)
        if not all(0.0 < t < 1.0 for t in v):
            bad += 1
            continue
        cx, cy, w, h = v
        if w <= 0 or h <= 0:
            bad += 1
    res["label_anomalies"] = bad
    res["class_ids"] = cids
    res["coord_range"] = [round(min(coords), 4), round(max(coords), 4)] if coords else None
    if bad:
        res["problems"].append("标签异常 %d 条" % bad)
        res["ok"] = False

    if check_ultralytics and "data.yaml" in tops:
        try:
            import tempfile, shutil
            from ultralytics.data.utils import check_det_dataset
            d = tempfile.mkdtemp(prefix="sdpipe_check_")
            z.extractall(d)
            r = check_det_dataset(os.path.join(d, "data.yaml"))
            res["ultralytics"] = dict(ok=True, nc=r.get("nc"), names=r.get("names"))
        except ImportError:
            res["ultralytics"] = dict(ok=None, note="未安装 ultralytics，跳过")
        except Exception as e:
            res["ultralytics"] = dict(ok=False, error="%s: %s" % (type(e).__name__, e))
            res["problems"].append("ultralytics check_det_dataset 失败: %s" % e)
            res["ok"] = False
    return res


# --------------------------------------------------------------------------
# 审计图
# --------------------------------------------------------------------------
def _grid(tiles, cols, width, out_path):
    if not tiles:
        return None
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    g = np.vstack(rows)
    sc = width / float(g.shape[1])
    g = cv2.resize(g, (int(width), int(g.shape[0] * sc)))
    buf = imencode_u(".png", g)
    if buf:
        with open(out_path, "wb") as fp:
            fp.write(buf)
        return out_path
    return None


def render_audit(zip_path, out_dir, n=24, cols=6, tag="audit"):
    """从 zip 抽 n 张样本，把标签框画上去 —— 交付前必须肉眼看这张图。"""
    z = zipfile.ZipFile(zip_path)
    imgs = sorted([n2 for n2 in z.namelist() if n2.endswith(".jpg")], key=nkey)
    if not imgs:
        return None
    step = max(1, len(imgs) // n)
    tiles = []
    for m in imgs[::step][:n]:
        split = m.split("/")[0]
        stem = os.path.splitext(os.path.basename(m))[0]
        im = cv2.imdecode(np.frombuffer(z.read(m), np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        H, W = im.shape[:2]
        ln = "%s/labels/%s.txt" % (split, stem)
        if ln in z.namelist():
            p = z.read(ln).decode("utf-8", "ignore").split()
            if len(p) == 5:
                cx, cy, bw, bh = [float(t) for t in p[1:]]
                x0, y0 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
                x1, y1 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
                cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 3)
        t = stem[:20]
        cv2.putText(im, t, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(im, t, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        tiles.append(im)
    return _grid(tiles, cols, 1680, os.path.join(out_dir, "%s.png" % tag))


def render_flagged(zip_path, out_path, cols=6, n=18):
    """优先展示「裁剪率最高 / 面积最大 / 最小」的帧，这些最可能是错标。"""
    rep = os.path.splitext(zip_path)[0] + "_records.json"
    recs = []
    if os.path.exists(rep):
        try:
            recs = json.load(io.open(rep, encoding="utf-8"))
        except Exception:
            recs = []
    z = zipfile.ZipFile(zip_path)
    names = z.namelist()

    def find_img(stem):
        for m in names:
            if os.path.basename(m) == stem + ".jpg" or os.path.basename(m) == stem:
                return m
        for m in names:
            if os.path.splitext(os.path.basename(m))[0] == stem and m.endswith(".jpg"):
                return m
        return None

    picked = []
    if recs:
        rs = [r for r in recs if r.get("ok")]
        rs.sort(key=lambda r: -(r.get("clip") or 0))
        picked = [r["name"] for r in rs[:n // 2]]
        rs2 = sorted([r for r in recs if r.get("ok")], key=lambda r: -(r.get("area_ratio") or 0))
        picked += [r["name"] for r in rs2[:n // 2]]
    if not picked:
        picked = [os.path.splitext(os.path.basename(m))[0]
                  for m in sorted([x for x in names if x.endswith(".jpg")], key=nkey)][:n]

    tiles = []
    for stem in picked[:n]:
        m = find_img(stem)
        if not m:
            continue
        im = cv2.imdecode(np.frombuffer(z.read(m), np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        H, W = im.shape[:2]
        ln = "%s/labels/%s.txt" % (m.split("/")[0], stem)
        if ln in names:
            p = z.read(ln).decode("utf-8", "ignore").split()
            if len(p) == 5:
                cx, cy, bw, bh = [float(t) for t in p[1:]]
                cv2.rectangle(im, (int((cx - bw / 2) * W), int((cy - bh / 2) * H)),
                              (int((cx + bw / 2) * W), int((cy + bh / 2) * H)), (0, 140, 255), 3)
        cv2.putText(im, stem[:22], (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(im, stem[:22], (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(im)
    return _grid(tiles, cols, 1680, out_path)


# --------------------------------------------------------------------------
# 标定
# --------------------------------------------------------------------------
def probe(src, out_dir=None, top=12, max_images=200):
    """一条命令量出定阈值所需要的一切：背景/阴影/目标饱和度、几何分位、命名分组、颜色。"""
    fs = list_images(src)
    if not fs:
        return dict(error="no images", src=src)
    step = max(1, len(fs) // max_images)
    samp = fs[::step]
    bg, sh, obj, areas, ratios, fills, ncand, edge_touch = [], [], [], [], [], [], [], 0
    hue_cnt, cen = [], []
    gray_like = 0
    for f in samp:
        img = imread_u(f)
        if img is None:
            continue
        H, W = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        S = hsv[:, :, 1].astype(np.float32)
        V = hsv[:, :, 2].astype(np.float32)
        r = 0.18
        border = np.zeros((H, W), bool)
        border[:int(H * r), :] = True
        border[int(H * (1 - r)):, :] = True
        border[:, :int(W * r)] = True
        border[:, int(W * (1 - r)):] = True
        inner = ~border
        bg.append(float(np.percentile(S[border], 95)))
        cs = candidates(img, DEFAULTS["sat_lo"], dict(DEFAULTS), cand_s_min=0)
        if cs:
            b = pick_best(cs, DEFAULTS["center_w"])
            areas.append(b["area_ratio"])
            ratios.append(b["ratio"])
            fills.append(b["fill"])
            cen.append((b["cx"], b["cy"]))
            cm = np.zeros((H, W), np.uint8)
            cv2.drawContours(cm, [b["c"]], -1, 255, -1)
            px = S[cm > 0]
            obj.extend(px.tolist()[:8000])
            bgm = np.zeros((H, W), np.uint8)
            cv2.drawContours(bgm, [b["c"]], -1, 255, -1)
            sil = cv2.dilate(bgm, np.ones((9, 9), np.uint8)) - bgm
            if (sil > 0).any():
                sh.extend(S[sil > 0].tolist()[:4000])
            if (px < 60).mean() > 0.5:
                gray_like += 1
            hsv_c = hsv[cm > 0]
            hue_cnt.append(int(np.median(hsv_c[:, 0])))
        ncand.append(len(cs))
        if cs and any(c["x"] <= 3 or c["y"] <= 3 or
                      c["x"] + c["w"] >= W - 3 or c["y"] + c["h"] >= H - 3 for c in cs):
            edge_touch += 1
    prefixes = {}
    for f in fs:
        b = os.path.basename(f)
        key = re.split("[0-9]", b)[0].strip()
        prefixes[key] = prefixes.get(key, 0) + 1
    hue_hist = {}
    for h in hue_cnt:
        if h < 20 or h >= 150:
            hue_hist["红"] = hue_hist.get("红", 0) + 1
        elif h < 45:
            hue_hist["橙/黄"] = hue_hist.get("橙/黄", 0) + 1
        elif h < 90:
            hue_hist["绿"] = hue_hist.get("绿", 0) + 1
        else:
            hue_hist["蓝/紫"] = hue_hist.get("蓝/紫", 0) + 1

    def q(a, p):
        return round(float(np.percentile(a, p)), 1) if a else None
    res = dict(src=src, n_images=len(fs), sampled=len(samp),
               name_prefixes=prefixes,
               background_S_p95=q(bg, 50), shadow_S_p50=q(sh, 50), shadow_S_p95=q(sh, 95),
               object_S_p05=q(obj, 5), object_S_p50=q(obj, 50), object_S_p95=q(obj, 95),
               low_sat_frames=gray_like, hue_histogram=hue_hist,
               candidate_count_hist={str(k): ncand.count(k) for k in sorted(set(ncand))},
               multi_candidate_frames=len([c for c in ncand if c > 1]),
               edge_touching_frames=edge_touch,
               area_ratio=dict(p05=q(areas, 5), p50=q(areas, 50), p95=q(areas, 95)),
               ratio=dict(p50=q(ratios, 50), p95=q(ratios, 95), max=round(max(ratios), 3) if ratios else None),
               fill=dict(p05=q(fills, 5), p50=q(fills, 50)),
               center_cx=(round(min(c[0] for c in cen), 3), round(max(c[0] for c in cen), 3)) if cen else None,
               center_cy=(round(min(c[1] for c in cen), 3), round(max(c[1] for c in cen), 3)) if cen else None)

    # 自动推荐参数
    rec = {}
    rec["val_max"] = 0
    if res["background_S_p95"] and res["object_S_p05"]:
        lo = max(60, (res["background_S_p95"] or 0) + 15)
        if res["object_S_p05"] and lo > res["object_S_p05"] * 0.6:
            lo = 60
        rec["sat_lo"] = int(min(130, max(60, lo)))
    else:
        rec["sat_lo"] = 60
    rec["edge_margin"] = 0.0 if edge_touch else 0.0
    rec["center_box"] = 0.45
    if res["center_cx"]:
        rec["center_box"] = round(min(0.49, max(0.35,
                                  max(abs(res["center_cx"][0] - 0.5), abs(res["center_cx"][1] - 0.5),
                                      abs(res["center_cy"][0] - 0.5), abs(res["center_cy"][1] - 0.5)) + 0.05)), 3)
    res["recommended"] = rec
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
            nm = os.path.basename(os.path.normpath(src)) or "dataset"
            base = os.path.join(out_dir, "probe_%s" % sanitize_name(nm)[:40])
            with io.open(base + ".json", "w", encoding="utf-8") as fp:
                json.dump(res, fp, ensure_ascii=False, indent=2)
            lines = [""]
            for k, v in res.items():
                lines.insert(len(lines), "%s" % (k + ": " + str(v)))
            txt = LF.join(["=== 标定 %s ===" % src, "图片 %d 张（抽样 %d）" % (res["n_images"], res["sampled"])] +
                          ["%-24s %s" % (k, v) for k, v in res.items() if k != "src"]) + LF
            with io.open(base + ".txt", "w", encoding="utf-8") as fp:
                fp.write(txt)
            res["report"] = base + ".txt"
        except Exception:
            pass
    return res
