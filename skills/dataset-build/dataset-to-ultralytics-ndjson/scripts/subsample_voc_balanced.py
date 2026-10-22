#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shrink a Pascal VOC dataset into a smaller, CLASS-BALANCED zip.

Use when a dataset exceeds a platform import limit (the Ultralytics platform
rejects a VOC zip with more than 3000 images) and you must not simply truncate:
truncation skews the class distribution and drops whole capture sessions.

Sampling rules
  * The unit of sampling is a GROUP (one physical scene / one placement burst).
    A coloured image and its grayscale twin share a group and always travel
    together, so the grayscale share of the subset is preserved.
  * Class stratified: every class gets the same target count, so the subset is
    balanced by class, not merely by total size.
  * Inside a class, quota is spread over sessions in proportion to what each
    session can give, and picks are evenly spaced inside each session (not just
    the first N frames).
  * With --keep-split, val and test are kept in full; only train is reduced.
  * Boxes are never recomputed: XML is copied byte-for-byte (only <folder> and
    <path> are rewritten in --out-flat so they point at images/).

Attribute CSV (header row required; names are case-insensitive, aliases accepted)
    file     image file name or stem                                  required
    cls      class name                       optional, else from the XML <name>
    session  capture session / burst id        optional, else "all"
    frame    frame index inside the session    optional, else row order
    group    scene / unit id; rows sharing it are kept together   optional
    is_bw    1 for the grayscale twin          optional, report only

Outputs
    --out-flat    images/ + xml/                Ultralytics platform VOC import
    --out-split   JPEGImages/ + Annotations/ + ImageSets/Main/*.txt + classes.txt

usage
  python subsample_voc_balanced.py --src-voc-zip in.zip --attr-csv attrs.csv \
      --total 2900 --out-flat flat.zip --out-split split.zip
"""
import argparse, collections, csv, io, os, re, sys, time, zipfile
import numpy as np


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", (s or "").strip().lower())


ALIAS = {
    "file": ("file", "filename", "name", "stem", "image", "img"),
    "cls": ("cls", "class", "clsname", "classname", "label", "category"),
    "session": ("session", "sess", "cam", "camera"),
    "frame": ("frame", "idx", "index"),
    "group": ("group", "scene", "unit", "burst"),
    "is_bw": ("isbw", "bw", "gray", "grayscale"),
    "split": ("split", "subset", "part"),
}


def map_cols(header):
    m = {}
    for key, alts in ALIAS.items():
        for h in header:
            if norm(h) in alts:
                m[key] = h
                break
    return m


def to_int(v, dv=0):
    try:
        return int(str(v).strip())
    except Exception:
        return dv


def clean_stem(s):
    """Normalise `cam5_000001 (661).jpg` -> `cam5_000001_661`, matching the zip entry."""
    s = os.path.splitext(os.path.basename(s))[0]
    s = re.sub(r"[^0-9A-Za-z_.\-]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def scan_zip(z):
    img_by, xml_by = {}, {}
    for n in z.namelist():
        st = os.path.splitext(os.path.basename(n))[0]
        low = n.lower()
        if low.endswith((".jpg", ".jpeg", ".png")) and low.startswith(("jpegimages/", "images/")):
            img_by[st] = n
        elif low.endswith(".xml") and low.startswith(("annotations/", "xml/")):
            xml_by[st] = n
    return img_by, xml_by


def xml_name(z, path):
    try:
        x = z.read(path).decode("utf-8", "replace")
    except Exception:
        return "?"
    m = re.search(r"<name>\s*([^<\s][^<]*?)\s*</name>", x)
    return m.group(1) if m else "?"


def pick_class(units, quota, tag=""):
    """Even quota over sessions, evenly spaced inside each session."""
    groups = collections.defaultdict(list)
    for u in units:
        groups[u["session"]].append(u)
    for s in groups:
        groups[s].sort(key=lambda u: (u["frame"], u["sid"]))
    tf = sum(len(u["files"]) for u in units)
    if tf <= quota:
        return list(units)
    order = sorted(groups, key=lambda x: (-len(groups[x]), x))
    picked, seen = [], set()
    for s in order:
        g = groups[s]
        k = min(len(g), int(round(len(g) * quota / float(tf))))
        if k <= 0:
            continue
        for i in sorted(set(int(round(v)) for v in np.linspace(0, len(g) - 1, k))):
            picked.append((s, i))
            seen.add((s, i))
    n = sum(len(groups[s][i]["files"]) for s, i in picked)
    while n > quota and picked:                       # roll back overshoot
        s, i = picked.pop()
        seen.discard((s, i))
        n -= len(groups[s][i]["files"])
    ptr = dict((s, 0) for s in groups)
    while n < quota:                                  # top up without overshoot
        moved = False
        for s in order:
            if n >= quota:
                break
            g = groups[s]
            j = ptr[s]
            while j < len(g) and (s, j) in seen:
                j += 1
            ptr[s] = j
            while j < len(g):
                if (s, j) not in seen and n + len(g[j]["files"]) <= quota:
                    picked.append((s, j))
                    seen.add((s, j))
                    n += len(g[j]["files"])
                    ptr[s] = j + 1
                    moved = True
                    break
                j += 1
            ptr[s] = j
        if not moved:
            break
    return [groups[s][i] for s, i in picked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-voc-zip", required=True)
    ap.add_argument("--attr-csv", default="")
    ap.add_argument("--total", type=int, default=2900)
    ap.add_argument("--out-flat", default="")
    ap.add_argument("--out-split", default="")
    ap.add_argument("--keep-split", action="store_true",
                    help="val and test are kept in full, only train is reduced")
    ap.add_argument("--seed", type=int, default=20260930)
    args = ap.parse_args()
    if not args.out_flat and not args.out_split:
        sys.exit("need at least one of --out-flat / --out-split")

    t0 = time.time()
    z = zipfile.ZipFile(args.src_voc_zip)
    img_by, xml_by = scan_zip(z)
    stems = sorted(img_by)
    if not stems:
        sys.exit("no images found (expected JPEGImages/ or images/)")
    print("source: %d images, %d xml" % (len(stems), len(xml_by)))

    meta = {}
    if args.attr_csv:
        rows = list(csv.DictReader(io.open(args.attr_csv, encoding="utf-8-sig", newline="")))
        cm = map_cols(rows[0].keys())
        print("attr csv columns -> %s" % cm)
        known = set(stems)
        for i, r in enumerate(rows):
            if not cm.get("file"):
                continue
            raw = r[cm["file"]]
            st = os.path.splitext(os.path.basename(raw))[0]
            if st not in known:
                st = clean_stem(raw)
            meta[st] = dict(cls=(r.get(cm.get("cls", ""), "") or "").strip(),
                            session=(r.get(cm.get("session", ""), "") or "all").strip(),
                            frame=to_int(r.get(cm.get("frame", ""), ""), i),
                            group=(r.get(cm.get("group", ""), "") or st).strip(),
                            is_bw=str(r.get(cm.get("is_bw", ""), "")).strip(),
                            split=(r.get(cm.get("split", ""), "") or "train").strip())
        hit = sum(1 for s in stems if s in meta)
        print("attr csv matched %d / %d images" % (hit, len(stems)))
    for st in stems:
        if st not in meta:
            meta[st] = dict(cls=xml_name(z, xml_by[st]), session="all", frame=0,
                            group=st, is_bw="", split="train")

    # ---- build units (group = scene; coloured + grayscale twin stay together)
    tmp = collections.defaultdict(list)
    for st in stems:
        tmp[meta[st]["group"]].append(st)
    units = []
    for gid, fl in tmp.items():
        base = next((s for s in fl if meta[s]["is_bw"] not in ("1", "true", "True")), fl[0])
        units.append(dict(sid=gid, files=sorted(fl), cls=meta[base]["cls"] or xml_name(z, xml_by[base]),
                          session=meta[base]["session"], frame=meta[base]["frame"]))
    by_cls = collections.defaultdict(list)
    for u in units:
        by_cls[u["cls"]].append(u)
    cls_list = sorted(by_cls)
    print("units (scenes) = %d  classes = %d  %s" % (len(units), len(cls_list), dict((k, len(v)) for k, v in by_cls.items())))
    quota = args.total // len(cls_list)
    print("total target %d -> per-class quota %d" % (args.total, quota))

    # ---- flat / platform build: class balanced over the whole set
    rng = np.random.default_rng(args.seed)
    selA, splitB = [], collections.defaultdict(list)
    if args.out_flat:
        for c in cls_list:
            selA += pick_class(by_cls[c], quota)
    if args.out_split:
        if args.keep_split:
            vt = [u for u in units if meta[u["files"][0]].get("split", "train") != "train"]
            vtf = collections.defaultdict(int)
            for u in vt:
                vtf[u["cls"]] += len(u["files"])
            selB = list(vt)
            for c in cls_list:
                need = quota - vtf[c]
                if need > 0:
                    selB += pick_class([u for u in by_cls[c] if meta[u["files"][0]].get("split", "train") == "train"], need)
        else:
            for c in cls_list:
                selB += pick_class(by_cls[c], quota)
        for u in selB:
            for s in u["files"]:
                splitB[meta[s].get("split", "train")].append(s)

    def write_flat(stems_, out):
        if os.path.exists(out):
            os.remove(out)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zo:
            for st in stems_:
                zo.writestr("images/%s.jpg" % st, z.read(img_by[st]))
                x = z.read(xml_by[st]).decode("utf-8")
                x = x.replace("<folder>JPEGImages</folder>", "<folder>images</folder>")
                x = x.replace("<path>JPEGImages/", "<path>images/")
                zo.writestr("xml/%s.xml" % st, x.encode("utf-8"))

    def write_split(smap, out):
        if os.path.exists(out):
            os.remove(out)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zo:
            for sp in ("train", "val", "test"):
                for st in smap.get(sp, []):
                    zo.writestr("JPEGImages/%s.jpg" % st, z.read(img_by[st]))
                    zo.writestr("Annotations/%s.xml" % st, z.read(xml_by[st]))
                if smap.get(sp):
                    zo.writestr("ImageSets/Main/%s.txt" % sp, "\n".join(smap[sp]) + "\n")
            zo.writestr("classes.txt", "\n".join(cls_list) + "\n")
            zo.writestr("README.txt", "\n".join([
                "class-balanced subsample of %s" % os.path.basename(args.src_voc_zip),
                "  JPEGImages/*.jpg + Annotations/*.xml",
                "  ImageSets/Main/*.txt  %s" % ", ".join("%s=%d" % (k, len(v)) for k, v in sorted(smap.items())),
                "  sampling unit = scene (coloured image + its grayscale twin, kept together)",
                "  class stratified (equal count per class), evenly spread over sessions",
                "  coordinates copied byte-for-byte from the source package",
            ]) + "\n")

    def report(tag, stems_, path):
        c = collections.Counter(meta[s]["cls"] or xml_name(z, xml_by[s]) for s in stems_)
        sess = collections.Counter(meta[s]["session"] for s in stems_)
        bw = sum(1 for s in stems_ if meta[s]["is_bw"] in ("1", "true", "True"))
        zz = zipfile.ZipFile(path)
        bad = zz.testzip()
        print("%s n=%d  bw=%d  per-class=%s (spread %d)  per-session=%s" % (
            tag, len(stems_), bw, dict(sorted(c.items())), max(c.values()) - min(c.values()), dict(sorted(sess.items()))))
        print("   zip %.1f MB  entries=%d  CRC=%s" % (os.path.getsize(path) / 1048576.0, len(zz.infolist()),
                                                      "OK" if bad is None else "FAIL " + str(bad)))

    print("")
    if args.out_flat:
        stems_ = sorted(s for u in selA for s in u["files"])
        write_flat(stems_, args.out_flat)
        report("FLAT ", stems_, args.out_flat)
    if args.out_split:
        nB = sum(len(v) for v in splitB.values())
        write_split(splitB, args.out_split)
        report("SPLIT", [s for v in splitB.values() for s in v], args.out_split)
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
