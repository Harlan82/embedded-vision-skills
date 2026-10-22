# -*- coding: utf-8 -*-
"""
中心目标 -> YOLO 检测数据集 -> Ultralytics Platform 可导入 ZIP
================================================================
场景假设：固定机位、背景与目标可分、单目标、目标位于画面中心附近。

用法:
  python build_yolo_zip.py --src <图片目录> --cls <类别名> --zip <输出.zip>
可选:
  --work <工作目录>     默认 <zip所在目录>/_ds
  --val-ratio 0.15
  --sat-min 60 --val-max 90
  --area-lo 0.03 --area-hi 0.15
  --ratio-max 2.0 --fill-min 0.35
  --center-box 0.33 --edge-margin 0.04 --center-w 0.05
  --seed 42
  --keep-miss          同时把未命中的图片名写入报告（默认就写）
"""
import os, io, sys, glob, zipfile, random, shutil, argparse
import numpy as np
import cv2


def imread_u(path):
    """读图，兼容 Windows 中文/非 ASCII 路径。

    cv2.imread 在 Windows 下走 ANSI 代码页，遇到中文路径会**静默返回 None**
    （不抛异常、只打 WARN），是极隐蔽的坑。统一改用 np.fromfile + imdecode。
    """
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def segment_candidates(img, P, sat_min=None, use_s_filter=True):
    """分割前景并返回通过判据的候选。

    sat_min: 覆盖 P['sat_min']（用于"高阈值回退"从粘连块中捞目标核心）
    use_s_filter: 是否启用候选级饱和度剔除（肤色手指）
    """
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    sat = P["sat_min"] if sat_min is None else sat_min
    fg = ((s > sat) | (v < P["val_max"])).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k, 1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, 2)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a <= 0:
            continue
        ar = a / (W * H)
        if not (P["area_lo"] <= ar <= P["area_hi"]):
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw <= 0 or bh <= 0:
            continue
        ratio = bw / float(bh)
        fill = a / float(bw * bh)
        if ratio > P["ratio_max"] or fill < P["fill_min"]:
            continue
        # 候选级饱和度判据：肤色手指 S 中位数明显偏低（实测 90~118），
        # 而塑料工件 S >= 133。用中位数（而非逐像素阈值）剔除干扰物，
        # 好处是 mask 仍在低阈值下生成，目标轮廓完整、不会向内咬。
        mm = np.zeros((H, W), np.uint8)
        cv2.drawContours(mm, [c], -1, 255, -1)
        med_s = float(np.median(s[mm > 0]))
        if use_s_filter and P.get("cand_s_min", 0) > 0 and med_s < P["cand_s_min"]:
            continue
        cx, cy = (x + bw / 2) / W, (y + bh / 2) / H
        # 中心区约束：一刀切掉四周固定杂物（<=0 表示禁用）
        if P["center_box"] > 0 and (abs(cx - 0.5) > P["center_box"]
                                    or abs(cy - 0.5) > P["center_box"]):
            continue
        # 边缘余量：避免归一化坐标触 0/1（<=0 表示禁用）
        # 注意：若真目标本身可能被画面边缘裁切（如工件位于台面边缘），
        # 必须禁用它，否则会误杀真目标；此时应改用饱和度阈值排除干扰物。
        m = P["edge_margin"]
        if m > 0 and (x <= W*m or y <= H*m or
                      x + bw >= W*(1-m) or y + bh >= H*(1-m)):
            continue
        d = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
        out.append(dict(score=ar - P["center_w"]*d, c=c, area=ar,
                        x=x, y=y, w=bw, h=bh, ratio=ratio, fill=fill))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--cls", required=True, help="类别名，如 class_9")
    ap.add_argument("--zip", required=True, dest="zip_out")
    ap.add_argument("--work", default=None)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--sat-min", type=int, default=60)
    ap.add_argument("--val-max", type=int, default=90)
    ap.add_argument("--area-lo", type=float, default=0.030)
    ap.add_argument("--area-hi", type=float, default=0.150)
    ap.add_argument("--ratio-max", type=float, default=2.0)
    ap.add_argument("--fill-min", type=float, default=0.35)
    ap.add_argument("--center-box", type=float, default=0.33)
    ap.add_argument("--edge-margin", type=float, default=0.04)
    ap.add_argument("--center-w", type=float, default=0.05)
    ap.add_argument("--cand-s-min", type=float, default=0.0,
                    help="候选级饱和度下限：剔除肤色手指（实测手指 S 90~118，"
                         "塑料工件 S>=133）。0=禁用")
    ap.add_argument("--fallback-sat", type=int, default=0,
                    help="若严格判定后无候选，用该高饱和度阈值重分割一次"
                         "（用于从'手指+目标粘连块'中捞回目标核心）。0=禁用")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audit-n", type=int, default=16)
    a = ap.parse_args()

    P = dict(sat_min=a.sat_min, val_max=a.val_max, area_lo=a.area_lo,
             area_hi=a.area_hi, ratio_max=a.ratio_max, fill_min=a.fill_min,
             center_box=a.center_box, edge_margin=a.edge_margin, center_w=a.center_w,
             cand_s_min=a.cand_s_min, fallback_sat=a.fallback_sat)

    WORK = a.work or (os.path.splitext(a.zip_out)[0] + "_ds")
    ext = (".jpg", ".jpeg", ".png", ".bmp")
    files = [os.path.join(a.src, n) for n in os.listdir(a.src) if n.lower().endswith(ext)]
    files.sort()

    L = []
    def log(s=""):
        L.append(str(s))

    log("源: %s" % a.src)
    log("图片: %d" % len(files))
    log("类别: %s" % a.cls)

    for sub in ("images/train", "labels/train", "images/val", "labels/val"):
        os.makedirs(os.path.join(WORK, sub.replace("/", os.sep)), exist_ok=True)

    ok, miss = [], []
    areas, ratios, fills, audit = [], [], [], []
    for f in files:
        img = imread_u(f)
        if img is None:
            miss.append((os.path.basename(f), "read_fail")); continue
        cs = segment_candidates(img, P)
        if not cs and P["fallback_sat"] > 0:
            # 回退：严格判据下无候选（多因目标与手指粘连），
            # 用高饱和度阈值重分割，只保留高饱和的"目标核心"
            cs = segment_candidates(img, P, sat_min=P["fallback_sat"],
                                    use_s_filter=False)
        if not cs:
            miss.append((os.path.basename(f), "no_target")); continue
        b = max(cs, key=lambda t: t["score"])
        H, W = img.shape[:2]
        areas.append(b["area"]*100); ratios.append(b["ratio"]); fills.append(b["fill"])
        # 紧贴外接框，不外扩（设计约定：框贴合目标，宁紧勿松）
        x0, y0 = float(b["x"]), float(b["y"])
        x1, y1 = float(b["x"]+b["w"]), float(b["y"]+b["h"])
        cl = lambda t: min(max(t, 1e-4), 1-1e-4)      # 严格落在 (0,1)
        nx, ny = cl((x0+x1)/2/W), cl((y0+y1)/2/H)
        nw, nh = cl((x1-x0)/W), cl((y1-y0)/H)
        ok.append((f, nx, ny, nw, nh))
        if len(audit) < a.audit_n or b["area"]*100 > 8.0:
            vis = img.copy()
            cv2.rectangle(vis, (b["x"], b["y"]),
                          (b["x"]+b["w"], b["y"]+b["h"]), (0, 0, 255), 2)
            cv2.putText(vis, "%.1f%% r=%.2f f=%.2f" % (b["area"]*100, b["ratio"], b["fill"]),
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            audit.append(vis)

    log("命中: %d   未命中: %d" % (len(ok), len(miss)))
    for n, why in miss:
        log("   ! %-24s %s" % (n, why))
    if areas:
        x = np.array(areas); r = np.array(ratios); fl = np.array(fills)
        log("面积%% : min=%.2f p50=%.2f p95=%.2f max=%.2f" % (
            x.min(), np.median(x), np.percentile(x, 95), x.max()))
        log("w/h  : min=%.2f p50=%.2f max=%.2f" % (r.min(), np.median(r), r.max()))
        log("fill : min=%.2f p50=%.2f" % (fl.min(), np.median(fl)))

    random.seed(a.seed); random.shuffle(ok)
    n_val = max(1, int(round(len(ok)*a.val_ratio))) if ok else 0
    val_set, train_set = ok[:n_val], ok[n_val:]
    log("train=%d  val=%d" % (len(train_set), len(val_set)))

    for split, items in (("train", train_set), ("val", val_set)):
        for src, nx, ny, nw, nh in items:
            base = os.path.splitext(os.path.basename(src))[0]
            shutil.copy2(src, os.path.join(WORK, "images", split, base+".jpg"))
            with io.open(os.path.join(WORK, "labels", split, base+".txt"),
                         "w", encoding="utf-8") as fp:
                fp.write("0 %.6f %.6f %.6f %.6f\n" % (nx, ny, nw, nh))

    with io.open(os.path.join(WORK, "data.yaml"), "w", encoding="utf-8") as fp:
        fp.write("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: %s\n" % a.cls)

    # zipfile "w" 模式自动截断重建，无需先删除旧文件
    ni = nl = 0
    with zipfile.ZipFile(a.zip_out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(WORK, "data.yaml"), "data.yaml")
        for split in ("train", "val"):
            for sub in ("images", "labels"):
                d = os.path.join(WORK, sub, split)
                for n in sorted(os.listdir(d)):
                    z.write(os.path.join(d, n), "%s/%s/%s" % (sub, split, n))
                    if sub == "images": ni += 1
                    else: nl += 1

    log("")
    log("ZIP: %s  (%.2f MB)" % (a.zip_out, os.path.getsize(a.zip_out)/1024.0/1024.0))
    log("内含 images=%d labels=%d" % (ni, nl))

    if audit:
        h, w = audit[0].shape[:2]
        while len(audit) % 4:
            audit.append(np.zeros((h, w, 3), np.uint8))
        rws = [np.hstack(audit[i:i+4]) for i in range(0, len(audit), 4)]
        ap_audit = os.path.splitext(a.zip_out)[0] + "_audit.png"
        cv2.imwrite(ap_audit, np.vstack(rws))
        log("审计图: %s" % ap_audit)

    rep = os.path.splitext(a.zip_out)[0] + "_build.txt"
    with io.open(rep, "w", encoding="utf-8") as fp:
        fp.write("\n".join(L))
    print("\n".join(L))
    print("\n-> %s" % rep)


if __name__ == "__main__":
    main()
