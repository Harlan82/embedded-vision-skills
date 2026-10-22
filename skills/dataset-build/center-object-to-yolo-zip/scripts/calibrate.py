# -*- coding: utf-8 -*-
"""
标定脚本：统计候选目标的面积占比 / 宽高比 / 填充率分布，用于确定阈值。
用法: python calibrate.py --src <图片目录> [--top 40]
"""
import os, io, sys, argparse
import numpy as np
import cv2

SAT_MIN, VAL_MAX = 60, 90


def imread_u(path):
    """读图，兼容 Windows 中文/非 ASCII 路径（cv2.imread 会静默返回 None）。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--top", type=int, default=40, help="列出面积最大的前 N 个候选")
    ap.add_argument("--sat-min", type=int, default=SAT_MIN)
    ap.add_argument("--val-max", type=int, default=VAL_MAX)
    a = ap.parse_args()

    ext = (".jpg", ".jpeg", ".png", ".bmp")
    files = [os.path.join(a.src, n) for n in os.listdir(a.src) if n.lower().endswith(ext)]
    files.sort()
    rows = []
    sizes = set()
    nfail = 0
    for f in files:
        img = imread_u(f)
        if img is None:
            nfail += 1; continue
        H, W = img.shape[:2]
        sizes.add((W, H))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)
        fg = ((s > a.sat_min) | (v < a.val_max)).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, 1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, 2)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            ar = cv2.contourArea(c)
            if ar <= 0:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bh <= 0 or bw <= 0:
                continue
            rows.append(dict(name=os.path.basename(f),
                             area=100.0*ar/(W*H), ratio=bw/float(bh),
                             fill=ar/float(bw*bh),
                             cx=(x+bw/2)/W, cy=(y+bh/2)/H,
                             x=x, y=y, w=bw, h=bh))

    L = []
    def w(s=""):
        L.append(str(s))

    w("目录: %s" % a.src)
    w("图片: %d (读取失败 %d)" % (len(files), nfail))
    w("分辨率: %s" % sorted(sizes))
    w("候选总数(未过滤): %d" % len(rows))
    if not rows:
        w("没有候选，检查 SAT_MIN / VAL_MAX"); return

    ar = np.array([r["area"] for r in rows])
    ra = np.array([r["ratio"] for r in rows])
    fi = np.array([r["fill"] for r in rows])
    w("")
    w("=== 全部候选分布 ===")
    for nm, v in (("面积占比%", ar), ("宽高比", ra), ("填充率", fi)):
        w("  %-10s min=%.2f  p50=%.2f  p90=%.2f  p95=%.2f  p99=%.2f  max=%.2f" % (
            nm, v.min(), np.median(v), np.percentile(v, 90),
            np.percentile(v, 95), np.percentile(v, 99), v.max()))

    w("")
    w("=== 面积最大的前 %d 个候选（用于识别大块杂物 / 粘连） ===" % a.top)
    w("%-22s %8s %7s %7s %14s %18s" % ("file", "area%", "w/h", "fill", "center", "box"))
    for r in sorted(rows, key=lambda t: -t["area"])[:a.top]:
        w("%-22s %8.2f %7.2f %7.2f %14s %18s" % (
            r["name"], r["area"], r["ratio"], r["fill"],
            "(%.2f,%.2f)" % (r["cx"], r["cy"]),
            "(%d,%d,%d,%d)" % (r["x"], r["y"], r["w"], r["h"])))

    w("")
    w("=== 面积 2%~4% 的候选（用于识别四周固定杂物） ===")
    w("%-22s %8s %7s %7s %14s %18s" % ("file", "area%", "w/h", "fill", "center", "box"))
    small = [r for r in rows if 2.0 <= r["area"] <= 4.0]
    for r in sorted(small, key=lambda t: -t["area"])[:30]:
        w("%-22s %8.2f %7.2f %7.2f %14s %18s" % (
            r["name"], r["area"], r["ratio"], r["fill"],
            "(%.2f,%.2f)" % (r["cx"], r["cy"]),
            "(%d,%d,%d,%d)" % (r["x"], r["y"], r["w"], r["h"])))

    w("")
    w("提示: 若某组候选的 center / box 在大量图片中高度一致 -> 那是固定杂物，")
    w("      应通过 CENTER_BOX（中心区）排除，而不是调面积阈值。")

    out = os.path.join(os.path.dirname(a.src) or ".", "calib.txt")
    with io.open(out, "w", encoding="utf-8") as fp:
        fp.write("\n".join(L))
    print("\n".join(L))
    print("\n-> %s" % out)


if __name__ == "__main__":
    main()
