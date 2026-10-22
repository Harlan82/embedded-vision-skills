# -*- coding: utf-8 -*-
"""
本地标注数据集 -> Ultralytics NDJSON + 平台可导入 YOLO 结构 ZIP

支持:
  - 布局 A(多会话): <root>/images/<session>/<name>.jpg + <name>.txt
  - 布局 B(单层):   <root>/images/<name>.jpg + <name>.txt
  - 平铺标签目录:   <root>/labels_obb/<name>.txt  或  <root>/labels/<name>.txt
  - 类别名来源:     classes.txt ("<id> <name>") 或 data.yaml 的 names 段
  - task: obb (9 字段/行) 与 detect (5 字段/行) 自动按字段数判定

标签解析顺序: 图片同级 txt 优先 -> 平铺目录回退 (后者按裸文件名索引, 跨会话会歧义)
"""
import argparse
import hashlib
import json
import struct
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# 衍生图/重复项标记：「.preview」是采集程序自己的缩略图；「- 副本」是 Windows 在
# 同名文件冲突时自动加的后缀（多次拷贝采集包必然产生）。后者内容往往是【另一个会话
# 的正式图】，不去掉会让同名图被计入两次、张数与框数双双虚高。
DERIVED_MARKERS = (".preview", "_preview", ".thumb", " - 副本", "-副本")


# ---------------- 基础工具 ----------------
def image_wh(p: Path):
    """读真实宽高。JPEG 走 SOF, PNG 走 IHDR。"""
    b = p.read_bytes()
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", b[16:24])
        return w, h
    if b[:2] == b"\xff\xd8":
        i, n = 2, len(b)
        while i < n - 9:
            if b[i] != 0xFF:
                i += 1
                continue
            m = b[i + 1]
            if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
                i += 2
                continue
            seg = struct.unpack(">H", b[i + 2:i + 4])[0]
            if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", b[i + 5:i + 9])
                return w, h
            i += 2 + seg
    raise ValueError(f"unreadable image: {p}")


def load_class_names(root: Path, explicit=None):
    """classes.txt 优先, 否则 data.yaml。返回 {int: str}"""
    cand = [Path(explicit)] if explicit else [root / "classes.txt", root / "classes.yaml"]
    cand += [root / "data.yaml", root / "dataset.yaml"]
    for p in cand:
        if not p.is_file():
            continue
        if p.suffix in (".yaml", ".yml"):
            names, in_block = {}, False
            for line in p.read_text("utf-8").splitlines():
                if line.startswith("names:"):
                    in_block = True
                    continue
                if in_block:
                    if not line.strip():
                        break
                    s = line.strip()
                    if ":" in s:
                        k, v = s.split(":", 1)
                        k = k.strip().strip("'\"")
                        if k.isdigit():
                            names[int(k)] = v.strip().strip("'\"")
            if names:
                return names
        else:
            names = {}
            for line in p.read_text("utf-8").splitlines():
                f = line.split()
                if len(f) >= 2:
                    names[int(f[0])] = " ".join(f[1:])
            if names:
                return names
    return {}


def parse_label_rows(text: str, ncoord: int):
    """按字段数判定, 不做任何坐标换算。返回 [[cls, *coords], ...]"""
    rows = []
    for line in text.splitlines():
        f = line.split()
        if len(f) == ncoord + 1:
            rows.append([int(f[0])] + [round(float(x), 6) for x in f[1:]])
    return rows


def split_of(name: str, val_every: int) -> str:
    if val_every <= 0:
        return "train"
    return "val" if int(hashlib.md5(name.encode()).hexdigest(), 16) % val_every == 0 else "train"


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--task", default="obb", choices=["obb", "detect"])
    ap.add_argument("--out-ndjson", default=None)
    ap.add_argument("--out-zip", default=None)
    ap.add_argument("--label-dir", default=None, help="平铺标签目录名, 默认探测 labels_obb/labels")
    ap.add_argument("--classes", default=None)
    ap.add_argument("--val-every", type=int, default=10, help="每 N 张取 1 张作 val; 0=全 train")
    ap.add_argument("--url-prefix", default="", help="空则 url=images/<file> 占位")
    ap.add_argument("--sessions", default=None,
                    help="只保留这些会话(逗号分隔), 如 20260915_033054; 默认全部")
    ap.add_argument("--exclude-sessions", default=None,
                    help="排除这些会话(逗号分隔), 优先级低于 --sessions")
    ap.add_argument("--name", default=None, help="数据集名, 默认取 root 目录名")
    ap.add_argument("--images-only", action="store_true",
                    help="只打包图片(无标注): 不读也不写 labels/, ZIP 仅含 images/ + data.yaml, "
                         "NDJSON 省略 annotations 字段。供「上传平台后在平台内手工标注」用。"
                         "注意: 不要用空 txt 冒充无标注——空 txt 在 YOLO 里等于【负样本】")
    args = ap.parse_args()

    root = Path(args.root)
    ncoord = 8 if args.task == "obb" else 4
    key = "obb" if args.task == "obb" else "boxes"
    ds_name = args.name or root.name
    out_ndjson = Path(args.out_ndjson) if args.out_ndjson else root / f"{ds_name}.ndjson"
    out_zip = Path(args.out_zip) if args.out_zip else root / f"{ds_name}_yolo.zip"

    names = load_class_names(root, args.classes)
    if not names:
        print("[警告] 未找到类别名 (classes.txt / data.yaml), class_names 将为空")

    flat_dir = None
    for d in ([root / args.label_dir] if args.label_dir else [root / "labels_obb", root / "labels"]):
        if d.is_dir():
            flat_dir = d
            break

    imgs = [p for p in sorted((root / "images").rglob("*"))
            if p.suffix.lower() in IMG_EXT
            and not any(m in p.stem.lower() for m in DERIVED_MARKERS)]

    # ---- 会话过滤 (多轮采集时用来剔除某一批) ----
    keep_only = {s.strip() for s in args.sessions.split(",")} if args.sessions else None
    drop = {s.strip() for s in args.exclude_sessions.split(",")} if args.exclude_sessions else None
    if keep_only or drop:
        def sess_of(p):
            return p.parent.name if p.parent != root / "images" else ""
        before = len(imgs)
        all_sess = sorted({sess_of(p) for p in imgs})
        if keep_only:
            unknown = keep_only - set(all_sess)
            if unknown:
                raise SystemExit(f"--sessions 指定的会话不存在: {sorted(unknown)} (现有: {all_sess})")
            imgs = [p for p in imgs if sess_of(p) in keep_only]
        if drop:
            imgs = [p for p in imgs if sess_of(p) not in drop]
        kept_sessions = sorted({sess_of(p) for p in imgs})
        print(f"  会话过滤: {before} -> {len(imgs)} 张  保留会话={kept_sessions}")

    if not imgs:
        raise SystemExit(f"no images under {root/'images'}")

    records, dup, mismatch, src_kind = [], set(), [], Counter()
    box_total = byte_total = 0
    per_class, per_img = Counter(), Counter()

    for img in imgs:
        if args.images_only:
            lf, kind, text, rows = None, "none", "", []
        else:
            side = img.with_suffix(".txt")
            flat = flat_dir / f"{img.stem}.txt" if flat_dir else None
            if side.is_file():
                lf, kind = side, "side"
            elif flat and flat.is_file():
                lf, kind = flat, "flat"
            else:
                lf, kind = None, "none"
            text = lf.read_text("utf-8") if lf else ""
            rows = parse_label_rows(text, ncoord)
            if kind == "side" and flat and flat.is_file():
                if parse_label_rows(flat.read_text("utf-8"), ncoord) != rows:
                    mismatch.append(str(side.relative_to(root)))
        src_kind[kind] += 1

        session = img.parent.name if img.parent != root / "images" else ""
        fname = f"{session}_{img.stem}{img.suffix.lower()}" if session else img.name
        if fname in dup:
            raise SystemExit(f"duplicate file key: {fname} (需加会话前缀)")
        dup.add(fname)

        w, h = image_wh(img)
        pre = args.url_prefix.rstrip("/") + "/" if args.url_prefix else ""
        rec = {"type": "image", "file": fname, "url": f"{pre}images/{fname}",
               "width": w, "height": h, "split": split_of(fname, args.val_every),
               "_src": img, "_text": text}
        if not args.images_only:
            rec["annotations"] = {key: rows}
        records.append(rec)
        box_total += len(rows)
        byte_total += img.stat().st_size
        per_img[len(rows)] += 1
        for r in rows:
            per_class[r[0]] += 1

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    header = {"type": "dataset", "task": args.task, "name": ds_name, "description": "",
              "bytes": byte_total, "url": "",
              "class_names": {str(k): v for k, v in sorted(names.items())},
              "version": "latest", "created_at": now, "updated_at": now}

    with out_ndjson.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for r in records:
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")},
                               ensure_ascii=False, separators=(",", ":")) + "\n")

    y = ["path: .", "train: images/train", "val: images/val", "names:"]
    y += [f"  {k}: {v}" for k, v in sorted(names.items())]
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("data.yaml", "\n".join(y) + "\n")
        for r in records:
            z.write(r["_src"], f"images/{r['split']}/{r['file']}")
            if not args.images_only:
                z.writestr(f"labels/{r['split']}/{Path(r['file']).stem}.txt", r["_text"])

    # ---------- 验收 ----------
    sp = Counter(r["split"] for r in records)
    bad = 0 if args.images_only else sum(
        1 for r in records for b in r["annotations"][key]
        if len(b) != ncoord + 1 or any(not (0.0 <= v <= 1.0) for v in b[1:]))
    print(f"images={len(records)} {dict(sp)}  boxes={box_total}  bytes={byte_total}")
    print(f"  标签来源={dict(src_kind)}  越界/畸形={bad}")
    if mismatch:
        print(f"  [警告] 同级 txt 与平铺目录不一致 {len(mismatch)} 处: {mismatch[:5]}")
    if not args.images_only:
        print(f"  每图框数={dict(sorted(per_img.items()))}  负样本={per_img.get(0,0)}")
        print("  类别分布: " + "  ".join(f"{k}:{names[k]}={per_class.get(k,0)}" for k in sorted(names)))
        zero = [k for k in names if per_class.get(k, 0) == 0]
        if zero:
            print(f"  [警告] 零样本类别: {[names[k] for k in zero]} -> 模型无法学习, 建议补数据或删类重映射")
    print(f"  -> {out_ndjson} ({out_ndjson.stat().st_size/1024:.1f} KB)")
    print(f"  -> {out_zip} ({out_zip.stat().st_size/1024/1024:.2f} MB)")

    if args.images_only:
        print("\n" + "=" * 72)
        print("IMAGE-ONLY 模式: 未写任何 labels/ —— 图片在平台上呈【未标注】态，可直接手工画框。")
        print(" (若写入空 txt，平台会当成【负样本/无目标】，与「待标注」语义完全不同)")
        print(f" ZIP 内容: data.yaml + images/{sorted(sp)[0]}/  共 {len(records)} 张")
        print(f" 类别名已预置 {len(names)} 个, 可导入后在平台内增删改")
        print("=" * 72)

    if not args.url_prefix:
        print("\n" + "!" * 72)
        print("!!  这份 NDJSON 的 url 是占位路径 (images/<file>)，公网不可达。")
        print("!!  直接上传到 Ultralytics Platform 会【所有图片加载失败 / 显示裂图】。")
        print("!!  平台回传请上传同一个 ZIP；NDJSON 仅作留档或版本快照。")
        print("!!  若必须用 NDJSON 导入，先把图片托管到公网，再加 --url-prefix https://...")
        print("!" * 72)


if __name__ == "__main__":
    main()
