# -*- coding: utf-8 -*-
"""校验 YOLO 数据集 ZIP：结构 / 配对 / 标签格式 / 坐标范围 / data.yaml。"""
import os, io, sys, zipfile, argparse
import numpy as np
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--audit", default=None, help="输出抽样标注拼图路径")
    a = ap.parse_args()

    lines = []
    def w(s=""):
        lines.append(str(s))

    z = zipfile.ZipFile(a.zip)
    w("ZIP: %s" % a.zip)
    w("大小: %.2f MB" % (os.path.getsize(a.zip)/1024.0/1024.0))
    w("CRC: %s" % ("OK" if z.testzip() is None else "BAD"))

    names = z.namelist()
    w("条目数: %d" % len(names))
    w("顶层: %s" % sorted(set(n.split("/")[0] for n in names)))

    imgs = sorted(n for n in names if n.startswith("images/"))
    lbls = sorted(n for n in names if n.startswith("labels/"))
    w("images=%d  labels=%d" % (len(imgs), len(lbls)))

    si = set(os.path.splitext(os.path.basename(n))[0] for n in imgs)
    sl = set(os.path.splitext(os.path.basename(n))[0] for n in lbls)
    w("缺失标签=%d  多余标签=%d" % (len(si-sl), len(sl-si)))

    bad, cls_ids, xy = [], set(), []
    for n in lbls:
        p = z.read(n).decode("utf-8").strip().split()
        if len(p) != 5:
            bad.append((n, "field_count=%d" % len(p))); continue
        try:
            c = int(p[0]); v = list(map(float, p[1:]))
        except Exception:
            bad.append((n, "parse")); continue
        cls_ids.add(c); xy.extend(v)
        if not all(0.0 < t < 1.0 for t in v):
            bad.append((n, "range %s" % v))
    w("标签异常: %d" % len(bad))
    for n, r in bad[:10]:
        w("   ! %s  %s" % (n, r))
    w("出现的 class id: %s" % sorted(cls_ids))
    if xy:
        arr = np.array(xy)
        w("归一化坐标范围: %.5f ~ %.5f" % (arr.min(), arr.max()))

    w("")
    w("data.yaml:")
    if "data.yaml" in names:
        w(z.read("data.yaml").decode("utf-8").strip())
    else:
        w("   ! 缺失 data.yaml")

    # 抽样可视化
    if a.audit and imgs:
        import random
        random.seed(3)
        pick = random.sample(imgs, min(12, len(imgs)))
        tiles = []
        for n in pick:
            im = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
            H, W = im.shape[:2]
            stem = os.path.splitext(os.path.basename(n))[0]
            ln = "labels/" + "/".join(n.split("/")[1:]).rsplit("/", 1)[0] + "/" + stem + ".txt"
            if ln in z.namelist():
                _, cx, cy, bw, bh = z.read(ln).decode().split()
                cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
                x0 = int((cx-bw/2)*W); y0 = int((cy-bh/2)*H)
                cv2.rectangle(im, (x0, y0), (int(x0+bw*W), int(y0+bh*H)), (0, 0, 255), 2)
            tiles.append(im)
        while len(tiles) % 4:
            tiles.append(np.zeros((H, W, 3), np.uint8))
        rws = [np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)]
        cv2.imwrite(a.audit, np.vstack(rws))
        w("")
        w("抽样标注拼图: %s" % a.audit)

    out = os.path.splitext(a.zip)[0] + "_verify.txt"
    with io.open(out, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    print("\n".join(lines))
    print("\n-> %s" % out)


if __name__ == "__main__":
    main()
