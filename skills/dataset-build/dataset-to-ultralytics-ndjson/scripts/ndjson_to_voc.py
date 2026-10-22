# -*- coding: utf-8 -*-
"""
Ultralytics NDJSON  ->  Pascal VOC zip

输出: <out-zip> 内含 images/*.jpg 与 xml/*.xml (同名配对)

规范(按平台「从本地导入数据集压缩包」要求):
  - zip 内 images/ 与 xml/ 两个文件夹
  - 一张图片对应一个同名 xml
  - 图片数量上限 3000(超出忽略), 压缩包 500MB 以内
  - VOC 坐标为 1-based 闭区间像素值, 需 clamp 到 [1,W]/[1,H]

坐标换算(唯一需要算的地方):
  ndjson boxes [cls, cx, cy, w, h] 均为归一化值 ->
  xmin = (cx - w/2) * W  ... 四舍五入后 clamp
(ndjson 的 obb 键不适合 VOC 的轴对齐框, 本脚本只处理 detect/boxes)
"""
import argparse
import json
import os
import shutil
import struct
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from xml.dom import minidom
from xml.etree import ElementTree as ET

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def jpeg_info(data: bytes):
    """返回 (w, h, channels)；解析失败返回 (None, None, 3)"""
    try:
        if data[:2] != b"\xff\xd8":
            return None, None, 3
        i, n = 2, len(data)
        while i < n - 9:
            if data[i] != 0xFF:
                i += 1
                continue
            m = data[i + 1]
            if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
                i += 2
                continue
            seglen = struct.unpack(">H", data[i + 2:i + 4])[0]
            if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                h = struct.unpack(">H", data[i + 5:i + 7])[0]
                w = struct.unpack(">H", data[i + 7:i + 9])[0]
                c = data[i + 9]
                return w, h, c
            i += 2 + seglen
    except Exception:  # noqa: BLE001
        pass
    return None, None, 3


def fetch(url: str, tries: int = 3) -> bytes:
    last = None
    for _ in range(tries):
        try:
            req = Request(url, headers=UA)
            with urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.6)
    raise RuntimeError(f"download failed: {last}")


def to_voc_xml(filename, folder, w, h, depth, objects):
    ann = ET.Element("annotation")
    ET.SubElement(ann, "folder").text = folder
    ET.SubElement(ann, "filename").text = filename
    ET.SubElement(ann, "path").text = f"{folder}/{filename}"
    src = ET.SubElement(ann, "source")
    ET.SubElement(src, "database").text = "Unknown"
    ET.SubElement(ann, "segmented").text = "0"
    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(depth)
    for name, xmin, ymin, xmax, ymax in objects:
        ob = ET.SubElement(ann, "object")
        ET.SubElement(ob, "name").text = name
        ET.SubElement(ob, "pose").text = "Unspecified"
        ET.SubElement(ob, "truncated").text = "0"
        ET.SubElement(ob, "difficult").text = "0"
        bb = ET.SubElement(ob, "bndbox")
        ET.SubElement(bb, "xmin").text = str(xmin)
        ET.SubElement(bb, "ymin").text = str(ymin)
        ET.SubElement(bb, "xmax").text = str(xmax)
        ET.SubElement(bb, "ymax").text = str(ymax)
    rough = ET.tostring(ann, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndjson", required=True)
    ap.add_argument("--out-zip", required=True)
    ap.add_argument("--stage", required=True, help="暂存目录")
    ap.add_argument("--max-images", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--fallback-dir", default="")
    ap.add_argument("--keep-stage", action="store_true")
    args = ap.parse_args()

    class_names, records = {}, []
    total = 0
    with open(args.ndjson, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("type") == "dataset":
                class_names = d.get("class_names", {})
                task = d.get("task")
            elif d.get("type") == "image":
                total += 1
                if len(records) < args.max_images:
                    records.append(d)

    print(f"ndjson: task={task}  图片总数={total}  本次处理={len(records)}  "
          f"忽略={max(0, total - args.max_images)}")
    print(f"类别({len(class_names)}): " +
          "  ".join(f"{k}:{v}" for k, v in sorted(class_names.items(), key=lambda x: int(x[0]))))
    if task != "detect":
        print(f"[警告] task={task}，本脚本按 detect 的 boxes 解算；obb 请另用轴对齐外接框逻辑")

    no_ann = [r["file"] for r in records if "annotations" not in r]
    if no_ann:
        print(f"[警告] {len(no_ann)} 条记录缺 annotations 键 -> 将生成【空 object】的 xml: {no_ann}")

    img_dir = os.path.join(args.stage, "images")
    xml_dir = os.path.join(args.stage, "xml")
    if os.path.isdir(args.stage):
        shutil.rmtree(args.stage)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(xml_dir, exist_ok=True)
    fb = args.fallback_dir or None

    print(f"\n== 下载 {len(records)} 张图片 ({args.workers} 线程) ==")
    results, failed = {}, []
    fb_used = []

    def task(rec):
        try:
            return rec["file"], fetch(rec["url"]), "http"
        except Exception:  # noqa: BLE001
            if fb and os.path.isfile(os.path.join(fb, rec["file"])):
                with open(os.path.join(fb, rec["file"]), "rb") as fp:
                    return rec["file"], fp.read(), "fallback"
            raise

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(task, r): r for r in records}
        for fut in as_completed(futs):
            rec = futs[fut]
            try:
                name, blob, how = fut.result()
                results[name] = blob
                if how == "fallback":
                    fb_used.append(name)
            except Exception as e:  # noqa: BLE001
                failed.append((rec["file"], str(e)))
            done += 1
            if done % 200 == 0 or done == len(records):
                print(f"  {done}/{len(records)}", flush=True)
    print(f"下载成功 {len(results)}  失败 {len(failed)}  回退 {len(fb_used)}")
    for n, e in failed[:5]:
        print("  FAIL", n, e)
    if failed:
        raise SystemExit("有图片下载失败，终止以避免产出不完整数据集")

    print("\n== 生成 xml ==")
    written, obj_total = 0, 0
    empty_xml, dim_note = [], []
    from collections import Counter
    per_class = Counter()

    for rec in records:
        fname = rec["file"]
        blob = results[fname]
        stem = os.path.splitext(fname)[0]
        aw, ah, depth = jpeg_info(blob)
        w = aw or int(rec["width"])
        h = ah or int(rec["height"])
        if (w, h) != (int(rec["width"]), int(rec["height"])):
            dim_note.append((fname, (rec["width"], rec["height"]), (w, h)))

        objs = []
        for box in (rec.get("annotations") or {}).get("boxes", []):
            cid = int(box[0])
            cx, cy, bw, bh = (float(v) for v in box[1:5])
            name = class_names.get(str(cid), class_names.get(cid, str(cid)))
            x1, y1 = cx - bw / 2.0, cy - bh / 2.0
            x2, y2 = cx + bw / 2.0, cy + bh / 2.0
            xmin = max(1, min(w, int(round(x1 * w))))
            ymin = max(1, min(h, int(round(y1 * h))))
            xmax = max(1, min(w, int(round(x2 * w))))
            ymax = max(1, min(h, int(round(y2 * h))))
            if xmax < xmin:
                xmin, xmax = xmax, xmin
            if ymax < ymin:
                ymin, ymax = ymax, ymin
            if xmax == xmin or ymax == ymin:
                continue
            objs.append((name, xmin, ymin, xmax, ymax))
            per_class[name] += 1
        obj_total += len(objs)
        if not objs:
            empty_xml.append(fname)

        with open(os.path.join(img_dir, fname), "wb") as fp:
            fp.write(blob)
        with open(os.path.join(xml_dir, stem + ".xml"), "wb") as fp:
            fp.write(to_voc_xml(fname, "images", w, h, depth, objs))
        written += 1

    print(f"  图片 {written}  xml {written}  框总数 {obj_total}")
    if dim_note:
        print(f"  [注意] {len(dim_note)} 张实际尺寸与 ndjson 记录不符: {dim_note[:5]}")
    if empty_xml:
        print(f"  [警告] {len(empty_xml)} 张 xml 无 object: {empty_xml}")
    print("  类别分布:")
    for k in sorted(class_names, key=int):
        print("     %-18s %d" % (class_names[k], per_class.get(class_names[k], 0)))

    print("\n== 打包 ==")
    if os.path.exists(args.out_zip):
        os.remove(args.out_zip)
    with zipfile.ZipFile(args.out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for folder in ("images", "xml"):
            for fn in sorted(os.listdir(os.path.join(args.stage, folder))):
                z.write(os.path.join(args.stage, folder, fn), f"{folder}/{fn}")
    size_mb = os.path.getsize(args.out_zip) / 1024 / 1024
    print(f"  -> {args.out_zip}  {size_mb:.2f} MB  ({'OK' if size_mb < 500 else '超过 500MB!'})")

    if not args.keep_stage:
        shutil.rmtree(args.stage, ignore_errors=True)
        print("  (stage cleaned)")


if __name__ == "__main__":
    main()
