# -*- coding: utf-8 -*-
"""
Ultralytics NDJSON  ->  YOLO 结构 ZIP（平台可导入 / 本地可训练）

反向转换（平台导出 -> 可回传 ZIP）。NDJSON 的标注语义与 YOLO txt 同构，
所以只做「重排 + 落盘」，不做任何坐标换算：
    detect : ndjson boxes [cls,cx,cy,w,h]        -> txt "cls cx cy w h"
    obb    : ndjson obb   [cls,x1,y1,...,x4,y4]  -> txt 同序 9 字段

图片来源：按 url 并发下载（平台签名的 CDN 链接是权威副本，与标签一一对应）。
split 直接沿用 ndjson 里的 split 字段，不重新切分。

无 annotations 键的记录 -> 不写 label 文件（忠实反映「未标注」）。
"""
import argparse
import io
import json
import os
import shutil
import struct
import sys
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def jpeg_wh(b: bytes):
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", b[16:24])
    if b[:2] == b"\xff\xd8":
        i, n = 2, len(b)
        while i < n - 9:
            if b[i] != 0xFF:
                i += 1; continue
            m = b[i + 1]
            if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
                i += 2; continue
            seg = struct.unpack(">H", b[i + 2:i + 4])[0]
            if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", b[i + 5:i + 9])
                return w, h
            i += 2 + seg
    return None, None


def fetch(url: str, dst: Path, tries: int = 3):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=45) as r:
                data = r.read()
            if not data:
                raise ValueError("empty body")
            dst.write_bytes(data)
            return len(data)
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"download failed after {tries}: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndjson", required=True)
    ap.add_argument("--out-zip", required=True)
    ap.add_argument("--work-dir", required=True, help="下载图片的暂存目录")
    ap.add_argument("--fallback-dir", default="", help="下载失败时的本地图片回退目录")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--name", default=None, help="数据集名, 默认取 ndjson 文件名")
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args()

    nd = Path(args.ndjson)
    ds_name = args.name or nd.stem
    work = Path(args.work_dir)
    img_dir = work / "images"
    if work.exists():
        shutil.rmtree(work)
    img_dir.mkdir(parents=True, exist_ok=True)

    raw = io.open(nd, encoding="utf-8").read().splitlines()
    hdr = json.loads(raw[0])
    recs = [json.loads(l) for l in raw[1:] if l.strip()]
    task = hdr["task"]
    key = {"detect": "boxes", "obb": "obb"}.get(task)
    if key is None:
        raise SystemExit(f"unsupported task: {task}")
    ncoord = 4 if task == "detect" else 8

    names = {int(k): v for k, v in hdr.get("class_names", {}).items()}
    print(f"dataset={hdr.get('name')}  task={task}  images={len(recs)}  classes={len(names)}")
    print("  splits:", dict(Counter(r["split"] for r in recs)))
    n_noann = sum(1 for r in recs if "annotations" not in r)
    print(f"  records without annotations key: {n_noann}")

    # ---------- 并发下载 ----------
    print(f"\n== downloading {len(recs)} images ({args.workers} workers) ==")
    ok = fail = 0
    dim_bad = []
    failures = []
    fb = Path(args.fallback_dir) if args.fallback_dir else None

    def job(rec):
        dst = img_dir / rec["file"]
        try:
            n = fetch(rec["url"], dst)
        except Exception as e:  # noqa: BLE001
            if fb and (fb / rec["file"]).is_file():
                shutil.copy2(fb / rec["file"], dst)
                return rec["file"], "fallback", (fb / rec["file"]).stat().st_size
            raise RuntimeError(f"{rec['file']}: {e}")
        return rec["file"], "http", n

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(job, r): r for r in recs}
        for i, fu in enumerate(as_completed(futs), 1):
            r = futs[fu]
            try:
                fn, how, n = fu.result()
                ok += 1
            except Exception as e:  # noqa: BLE001
                fail += 1
                failures.append(str(e))
                continue
            w, h = jpeg_wh((img_dir / r["file"]).read_bytes())
            if (w, h) != (r["width"], r["height"]):
                dim_bad.append((r["file"], (r["width"], r["height"]), (w, h)))
            if i % 200 == 0:
                print(f"  ... {i}/{len(recs)}")

    print(f"  ok={ok} fail={fail} dim_mismatch={len(dim_bad)}")
    for d in dim_bad[:10]:
        print("   dim!", d)
    for x in failures[:10]:
        print("   FAIL", x)
    if fail:
        raise SystemExit("有图片下载失败，终止（避免产出不完整数据集）")

    # ---------- 组装 zip ----------
    print("\n== building zip ==")
    out_zip = Path(args.out_zip)
    if out_zip.exists():
        out_zip.unlink()

    y = ["path: .", "train: images/train", "val: images/val", "names:"]
    y += [f"  {k}: {v}" for k, v in sorted(names.items())]
    yaml_text = "\n".join(y) + "\n"

    box_total = 0
    per_class = Counter()
    wrote_labels = 0
    skipped = []
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("data.yaml", yaml_text)
        for r in recs:
            sp = r["split"]
            z.write(img_dir / r["file"], f"images/{sp}/{r['file']}")
            if "annotations" not in r:
                skipped.append(r["file"])
                continue
            rows = r["annotations"].get(key, [])
            if not rows:
                skipped.append(r["file"])
                continue
            lines = []
            for b in rows:
                assert len(b) == ncoord + 1, (r["file"], b)
                if task == "detect":
                    cls, cx, cy, w, h = b
                    lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                else:
                    cls = b[0]
                    co = " ".join(f"{v:.6f}" for v in b[1:])
                    lines.append(f"{int(cls)} {co}")
                per_class[int(b[0])] += 1
                box_total += 1
            z.writestr(f"labels/{sp}/{Path(r['file']).stem}.txt", "\n".join(lines) + "\n")
            wrote_labels += 1

    print(f"  images written : {len(recs)}")
    print(f"  labels written : {wrote_labels}")
    print(f"  boxes total    : {box_total}")
    if skipped:
        print(f"  [注意] 无标注(未写 label 文件) {len(skipped)} 张: {skipped}")
    print("  类别分布:")
    for k in sorted(names):
        print("     %-18s %d" % (names[k], per_class.get(k, 0)))
    zc = [names[k] for k in names if per_class.get(k, 0) == 0]
    if zc:
        print("  [警告] 零样本类别:", zc)
    print(f"  -> {out_zip} ({out_zip.stat().st_size/1024/1024:.2f} MB)")

    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)
        print("  (work dir cleaned)")


if __name__ == "__main__":
    main()
