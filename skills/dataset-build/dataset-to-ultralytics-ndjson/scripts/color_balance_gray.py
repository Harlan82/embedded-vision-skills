#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NDJSON -> 按 (类别, 颜色) 均衡的灰度增强子集（images/ + labels/ + data.yaml + ZIP）

用途：给 detect/OBB 数据集做「按类别×颜色均衡抽取、转黑白」的增强子集。
典型场景：物体是红/绿/蓝塑料几何体，需要等量的黑白版本，避免颜色维度造成类别不均衡。

为什么不能只用 bbox 中心取色：
  物体常放在白色/浅灰平台上，且可能**镂空**（中心是洞）。中心取色两种都会错。
  本脚本用「背景弃权 + 色相投票」：低饱和度=背景，弃权；余下高饱和像素按色相投票。

用法：
  python color_balance_gray.py \
    --ndjson "./work/cam-000207.ndjson" \
    --dest   "./work/demo" \
    --zip    "./work/demo_yolo.zip" \
    --per-group 40 \
    --work-dir "./work"

依赖：numpy, Pillow, requests（managed Python 3.14.3 已预装）
产物：dest/images/*.jpg(mode=L)  dest/labels/*.txt  dest/data.yaml
      dest/sampling_manifest.csv  dest/README.txt  + zip(data.yaml, images/, labels/)
"""
import argparse, collections, csv, hashlib, io, json, os, sys, time, zipfile

S_MIN = 0.28      # 饱和度下限：低于此判为背景/白平台/灰影 -> 弃权
V_MIN = 0.22      # 亮度下限：过暗（阴影/黑）也弃权
MIN_OBJ_PX = 40   # 有效物体像素不足 -> unknown
MAX_PX = 9000     # 每框采样像素硬上限（stride 降采样，避免大框拖慢）
HUE_TARGET = {"red": 0.0, "green": 120.0, "blue": 240.0}
COLORS = ("red", "green", "blue")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndjson", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--zip", default=None)
    ap.add_argument("--per-group", type=int, default=40,
                    help="每个 (类别,颜色) 组合抽多少张。"
                         "取「典型组张数 x 20%%」可兼顾均衡与比例，见 SKILL.md 六-补3")
    ap.add_argument("--work-dir", default=None, help="下载缓存与中间产物目录，默认 ndjson 同级 _cbag_work")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--classes", default=None, help="classes.txt / data.yaml；默认取 ndjson 的 class_names")
    ap.add_argument("--s-min", type=float, default=S_MIN)
    ap.add_argument("--v-min", type=float, default=V_MIN)
    args = ap.parse_args()

    import numpy as np
    import requests
    from PIL import Image, ImageDraw
    from concurrent.futures import ThreadPoolExecutor, as_completed

    nd = os.path.abspath(args.ndjson)
    dest = os.path.abspath(args.dest)
    work = args.work_dir or os.path.join(os.path.dirname(nd), "_cbag_work")
    imgdir = os.path.join(work, "imgs")
    os.makedirs(imgdir, exist_ok=True)
    os.makedirs(os.path.join(dest, "images"), exist_ok=True)
    os.makedirs(os.path.join(dest, "labels"), exist_ok=True)

    # ---------- 1. 读 NDJSON ----------
    header, recs = None, []
    with io.open(nd, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("type") == "dataset":
                header = o
            elif o.get("type") == "image":
                recs.append(o)
    if not recs:
        sys.exit("NDJSON 里没有 image 记录")
    task = (header or {}).get("task", "detect")
    if task not in ("detect", "obb"):
        sys.exit("本脚本只支持 detect / obb，当前 task=%s" % task)
    ann_key = "boxes" if task == "detect" else "obb"
    field_n = 5 if task == "detect" else 9
    cnames = (header or {}).get("class_names") or {}
    CLASSES = [cnames[str(i)] for i in range(len(cnames))] if cnames else []
    print("task=%s  images=%d  classes=%d" % (task, len(recs), len(CLASSES)))

    # ---------- 2. 下载（签名 CDN 约 7 天有效） ----------
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0"})

    def dl(item):
        i, r = item
        dst = os.path.join(imgdir, "%05d.jpg" % i)
        if os.path.exists(dst) and os.path.getsize(dst) > 1024:
            return (i, "cached", os.path.getsize(dst))
        last = "?"
        for a in range(4):
            try:
                resp = sess.get(r["url"], timeout=30)
                if resp.status_code == 200 and len(resp.content) > 1024:
                    with open(dst, "wb") as fh:
                        fh.write(resp.content)
                    return (i, "ok", len(resp.content))
                last = "http%d" % resp.status_code
            except Exception as e:
                last = type(e).__name__
            time.sleep(0.8 * (a + 1))
        return (i, "FAIL:" + last, 0)

    t0 = time.time()
    res = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(dl, (i, r)) for i, r in enumerate(recs)]
        for n, fu in enumerate(as_completed(futs), 1):
            res.append(fu.result())
            if n % 500 == 0:
                print("  dl %d/%d %.1fs" % (n, len(recs), time.time() - t0))
    fails = sorted(i for i, s, _ in res if s.startswith("FAIL"))
    print("下载: ok=%d fail=%d  字节=%d  耗时=%.1fs" % (len(res) - len(fails), len(fails),
                                                    sum(x[2] for x in res), time.time() - t0))
    if fails:
        print("  ⚠ 失败 idx(前20):", fails[:20], "  这些图将被跳过")

    # ---------- 3. 判色 ----------
    def hue_deg(r, g, b):
        mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b)
        d = np.where(mx - mn == 0, 1e-9, mx - mn)
        h = np.zeros_like(mx)
        m = mx == r; h[m] = (60.0 * ((g[m] - b[m]) / d[m])) % 360.0
        m = mx == g; h[m] = 60.0 * ((b[m] - r[m]) / d[m]) + 120.0
        m = mx == b; h[m] = 60.0 * ((r[m] - g[m]) / d[m]) + 240.0
        return h % 360.0

    def circ(h, t):
        dd = np.abs(h - t) % 360.0
        return np.minimum(dd, 360.0 - dd)

    rows = []
    skip_noann = 0
    for i, e in enumerate(recs):
        if i in set(fails):
            continue
        ann = e.get("annotations")
        if ann is None or not ann.get(ann_key):
            skip_noann += 1
            continue
        bx = ann[ann_key][0]
        cid = int(bx[0])
        im = Image.open(os.path.join(imgdir, "%05d.jpg" % i)).convert("RGB")
        W, H = im.size
        if task == "detect":
            cx, cy, bw, bh = bx[1:5]
            x1 = int(max(0, (cx - bw / 2) * W)); x2 = int(min(W, (cx + bw / 2) * W))
            y1 = int(max(0, (cy - bh / 2) * H)); y2 = int(min(H, (cy + bh / 2) * H))
        else:
            xs = bx[1::2]; ys = bx[2::2]
            x1 = int(max(0, min(xs) * W)); x2 = int(min(W, max(xs) * W))
            y1 = int(max(0, min(ys) * H)); y2 = int(min(H, max(ys) * H))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        crop = np.asarray(im, dtype=np.float32)[y1:y2, x1:x2] / 255.0
        st = max(1, int(np.ceil(np.sqrt(crop.shape[0] * crop.shape[1] / float(MAX_PX)))))
        crop = crop[::st, ::st]
        r_, g_, b_ = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
        mx = np.maximum(np.maximum(r_, g_), b_); mn = np.minimum(np.minimum(r_, g_), b_)
        S = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0.0)
        m = (S >= args.s_min) & (mx >= args.v_min)          # 背景弃权
        n_obj = int(m.sum())
        if n_obj < MIN_OBJ_PX:
            rows.append(dict(idx=i, file=e.get("file"), cls=cid, color="unknown",
                             purity=0.0, n_obj=n_obj, boxes=ann[ann_key]))
            continue
        Hh = hue_deg(r_[m], g_[m], b_[m])
        dist = np.stack([circ(Hh, HUE_TARGET[c]) for c in COLORS], axis=1)
        pick = np.argmin(dist, axis=1)
        votes = {COLORS[k]: int((pick == k).sum()) for k in range(len(COLORS))}
        best = max(votes, key=votes.get)
        rows.append(dict(idx=i, file=e.get("file"), cls=cid, color=best,
                         purity=round(votes[best] / float(n_obj), 4), n_obj=n_obj,
                         boxes=ann[ann_key]))

    n_ok = sum(1 for r in rows if r["color"] != "unknown")
    print("判色: 有效=%d  无标注(跳过)=%d  判不出=%d" % (n_ok, skip_noann, len(rows) - n_ok))
    low = [r for r in rows if r["color"] != "unknown" and r["purity"] < 0.85]
    if low:
        print("  ⚠ 纯度<0.85 有 %d 个，建议肉眼抽查（可能混入非红绿蓝物体）" % len(low))

    # ---------- 4. 分组均衡抽样 + 灰度化 ----------
    groups = collections.defaultdict(list)
    for r in rows:
        if r["color"] != "unknown":
            groups[(r["cls"], r["color"])].append(r)

    selected, table = [], []
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda r: hashlib.md5((r["file"] or "").encode("utf-8")).hexdigest())
        take = g[:args.per_group]
        selected.extend(take)
        table.append((key[0], key[1], len(g), len(take)))

    print("\n%-20s %-6s %8s %8s" % ("class", "color", "可选", "抽出"))
    for cid, color, na, tk in table:
        print("%-20s %-6s %8d %8d" % (CLASSES[cid] if cid < len(CLASSES) else cid, color, na, tk))

    man_out = []
    for r in selected:
        e = recs[r["idx"]]
        cn = CLASSES[r["cls"]] if r["cls"] < len(CLASSES) else str(r["cls"])
        stem = "%s_%s_%05d" % (cn, r["color"], r["idx"])
        Image.open(os.path.join(imgdir, "%05d.jpg" % r["idx"])).convert("RGB").convert("L").save(
            os.path.join(dest, "images", stem + ".jpg"), "JPEG",
            quality=args.quality, subsampling=0, optimize=True)
        # 标签原样复制：归一化坐标与分辨率/通道数无关，零换算
        line = " ".join([str(int(r["boxes"][0]))] + ["%.6f" % v for v in r["boxes"][1:field_n]])
        io.open(os.path.join(dest, "labels", stem + ".txt"), "w", encoding="utf-8").write(line + "\n")
        man_out.append(dict(bw_stem=stem, orig_file=e.get("file"), idx=r["idx"], cls_id=r["cls"],
                            cls_name=cn, color=r["color"], purity=r["purity"]))

    # ---------- 5. data.yaml / 清单 / README ----------
    has_val = any((e.get("split") == "val") for e in recs)
    base = (header or {}).get("name", "bw_subset")
    io.open(os.path.join(dest, "data.yaml"), "w", encoding="utf-8").write(
        "# 黑白化增强子集 — 由 %s 按 (类别,颜色) 均衡抽取\n"
        "path: %s\ntrain: images\nval: images\nnc: %d\nnames:\n%s" % (
            os.path.basename(nd), dest.replace("\\", "/"), len(CLASSES),
            "".join("  %d: %s\n" % (i, c) for i, c in enumerate(CLASSES))))
    with io.open(os.path.join(dest, "sampling_manifest.csv"), "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(man_out[0].keys()))
        wr.writeheader()
        for m in man_out:
            wr.writerow(m)
    cnt = collections.Counter((m["cls_name"], m["color"]) for m in man_out)
    io.open(os.path.join(dest, "README.txt"), "w", encoding="utf-8").write(
        "%s 黑白化均衡子集\n组合数=%d  每组抽取=%d  合计=%d\n"
        "抽样依据: md5(原始文件名)升序取前 N, 确定性可复现\n"
        "灰度方式: PIL convert('L') (ITU-R 601-2 luma), 单通道 JPEG, quality=%d\n"
        "标签: 原 NDJSON 标注原样复制, 零换算\n"
        "注意: 本子集全量用于 train, val 仅为满足格式要求; 合并进主数据集后请重切 val\n"
        % (base, len(cnt), args.per_group, len(man_out), args.quality))

    # ---------- 6. ZIP ----------
    if args.zip:
        zp = os.path.abspath(args.zip)
        if os.path.exists(zp):
            os.remove(zp)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(os.path.join(dest, "data.yaml"), "data.yaml")
            for x in sorted(os.listdir(os.path.join(dest, "images"))):
                z.write(os.path.join(dest, "images", x), "images/" + x)
            for x in sorted(os.listdir(os.path.join(dest, "labels"))):
                z.write(os.path.join(dest, "labels", x), "labels/" + x)
        with zipfile.ZipFile(zp) as z:
            bad = z.testzip()
            ns = [i.filename for i in z.infolist()]
        print("\nZIP: %s  %.1fMB  CRC=%s  条目=%d" % (
            zp, os.path.getsize(zp) / 1048576.0, "OK" if bad is None else "FAIL", len(ns)))

    # ---------- 7. 验收 ----------
    si = set(os.path.splitext(x)[0] for x in os.listdir(os.path.join(dest, "images")))
    sl = set(os.path.splitext(x)[0] for x in os.listdir(os.path.join(dest, "labels")))
    print("\n=== 验收 ===")
    print("images=%d labels=%d 差集=%d/%d" % (len(si), len(sl), len(si - sl), len(sl - si)))
    print("组合偏差: max=%d min=%d" % (max(cnt.values()), min(cnt.values())))
    if si != sl:
        print("  ✗ images 与 labels 不配对！")
    if max(cnt.values()) != min(cnt.values()):
        print("  ⚠ 组合不均衡 —— 某组可选项少于 --per-group，请调小该值")
    print("未纳入: 无标注 %d 张, 判色失败 %d 张" % (skip_noann, len(rows) - n_ok))


if __name__ == "__main__":
    main()
