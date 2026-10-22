#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YOLO (images/ + labels/ + data.yaml) ZIP -> standard Pascal VOC dataset ZIP.

Output layout (standard VOC, keeps the split):
    JPEGImages/*.jpg
    Annotations/*.xml
    ImageSets/Main/{train,val,test}.txt   (stem lists, one per line)
    classes.txt
    README.txt
    CONVERT_REPORT.txt

Why this script exists: "give me a Pascal VOC dataset" is ambiguous. The Ultralytics
platform wants a FLAT images/ + xml/ pair (and rejects >3000 images); tools like
LabelImg / MMDetection / torchvision want the standard VOC layout above. Picking the
wrong one makes the platform report "all labels lost" - not a conversion bug.
Use --flat for the platform variant.

Coordinates: bndbox is 1-based inclusive pixel (original VOC devkit convention).
    xmin = round((cx - w/2) * W),  xmax = round((cx + w/2) * W),  clamped to [1, W]
Degenerate (zero area) boxes are dropped; swapped x1>x2 are fixed.
W,H are read from the real JPEG SOF header, never from metadata.

usage:
  python yolo_zip_to_voc.py --src-yolo-zip in.zip --dst-zip out.zip
  python yolo_zip_to_voc.py --src-yolo-zip in.zip --dst-zip out_flat.zip --flat --max-images 2900
"""
import argparse, io, os, re, struct, sys, time, zipfile
import xml.etree.ElementTree as ET


def jpeg_info(b):
    """(width, height, n_components) from the JPEG SOF marker, or None."""
    i, n = 2, len(b)
    while i < n - 9:
        if b[i] != 0xFF:
            i += 1
            continue
        m = b[i + 1]
        if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h, w = struct.unpack(">HH", b[i + 5:i + 9])
            return w, h, b[i + 9]
        if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        i += 2 + struct.unpack(">H", b[i + 2:i + 4])[0]
    return None


def parse_names(yaml_text):
    """Minimal parser for `names:` in an Ultralytics data.yaml (list or {id: name})."""
    names = {}
    for ln in yaml_text.splitlines():
        ln = ln.strip()
        m = re.match(r"^(\d+)\s*:\s*[\"']?([^\"'#]+?)[\"']?\s*$", ln)
        if m:
            names[int(m.group(1))] = m.group(2).strip()
    if not names:
        m = re.search(r"names\s*:\s*\[(.*?)\]", yaml_text, re.S)
        if m:
            items = [x.strip().strip("\"'") for x in m.group(1).split(",") if x.strip()]
            names = {i: v for i, v in enumerate(items)}
    if not names:
        m = re.search(r"names\s*:\s*\n((?:\s*-\s*.+\n?)+)", yaml_text)
        if m:
            items = [x.strip().lstrip("-").strip().strip("\"'") for x in m.group(1).splitlines() if x.strip()]
            names = {i: v for i, v in enumerate(items)}
    return {k: names[k] for k in sorted(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-yolo-zip", required=True)
    ap.add_argument("--dst-zip", required=True)
    ap.add_argument("--flat", action="store_true",
                    help="emit flat images/ + xml/ (Ultralytics platform import) instead of standard VOC")
    ap.add_argument("--max-images", type=int, default=0, help="0 = no limit (platform VOC import caps at 3000)")
    ap.add_argument("--verify-visual", action="store_true", help="also write a contact sheet PNG next to --dst-zip")
    args = ap.parse_args()

    zin = zipfile.ZipFile(args.src_yolo_zip)
    all_names = [i.filename for i in zin.infolist()]
    img_by_stem = {os.path.splitext(n.split("/")[-1])[0]: n for n in all_names if n.startswith("images/")}
    lab_by_stem = {os.path.splitext(n.split("/")[-1])[0]: n for n in all_names if n.startswith("labels/")}
    if not img_by_stem:
        sys.exit("no images/ entries found")
    if set(img_by_stem) != set(lab_by_stem):
        print("WARN: image/label stems differ: %d / %d" % (len(img_by_stem), len(lab_by_stem)))
    names_map = {}
    if "data.yaml" in all_names:
        names_map = parse_names(zin.read("data.yaml").decode("utf-8", "replace"))
    elif "classes.txt" in all_names:
        names_map = {i: v.strip() for i, v in enumerate(zin.read("classes.txt").decode().splitlines()) if v.strip()}
    if not names_map:
        sys.exit("cannot determine class names (no data.yaml / classes.txt)")

    splits = {}
    for sp in ("train", "val", "test"):
        s = sorted(os.path.splitext(n.split("/")[-1])[0] for n in all_names if n.startswith("images/%s/" % sp))
        if s:
            splits[sp] = s
    if not splits:
        splits = {"train": sorted(img_by_stem)}

    all_stems = sum(splits.values(), [])
    if args.max_images and len(all_stems) > args.max_images:
        print("NOTE: --max-images %d -> keeping first %d stems (platform VOC import limit)"
              % (args.max_images, args.max_images))
        kept, out = set(), {}
        for sp in splits:
            outs = []
            for st in splits[sp]:
                if len(kept) < args.max_images:
                    outs.append(st); kept.add(st)
            out[sp] = outs
        splits = out
        all_stems = sum(splits.values(), [])

    print("src: %d images  splits=%s  classes=%d" % (len(img_by_stem), {k: len(v) for k, v in splits.items()}, len(names_map)))

    R = []
    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        R.append(s)

    stat = dict(obj=0, clamped=0, swapped=0, degen=0, depth={})
    ver = dict(n=0, xml_bad=0, name_bad=0, range_bad=0, cnt_bad=0, size_bad=0,
               maxerr=0.0, sumerr=0.0, maxerr_y=0.0, sumerr_y=0.0)
    written = []
    if os.path.exists(args.dst_zip):
        os.remove(args.dst_zip)
    t0 = time.time()
    with zipfile.ZipFile(args.dst_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zo:
        for k, stem in enumerate(all_stems):
            data = zin.read(img_by_stem[stem])
            info = jpeg_info(data)
            if info is None:
                print("WARN: cannot read SOF for %s" % stem); continue
            w, h, nc = info
            depth = nc if nc in (1, 3, 4) else 3
            stat["depth"][depth] = stat["depth"].get(depth, 0) + 1
            recs = []
            for ln in zin.read(lab_by_stem[stem]).decode("utf-8", "replace").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                p = ln.split()
                if len(p) != 5:
                    print("WARN: %s not a 5-field detect label: %r" % (stem, ln)); continue
                recs.append((int(float(p[0])), *[float(x) for x in p[1:5]]))
            L = ['<annotation>', '\t<folder>JPEGImages</folder>',
                 '\t<filename>%s.jpg</filename>' % stem, '\t<path>JPEGImages/%s.jpg</path>' % stem,
                 '\t<source>', '\t\t<database>%s</database>' % os.path.basename(args.src_yolo_zip), '\t</source>',
                 '\t<size>', '\t\t<width>%d</width>' % w, '\t\t<height>%d</height>' % h,
                 '\t\t<depth>%d</depth>' % depth, '\t</size>', '\t<segmented>0</segmented>']
            for (c, cx, cy, bw, bh) in recs:
                x1 = int(round((cx - bw / 2.0) * w)); x2 = int(round((cx + bw / 2.0) * w))
                y1 = int(round((cy - bh / 2.0) * h)); y2 = int(round((cy + bh / 2.0) * h))
                if x2 < x1: x1, x2 = x2, x1; stat["swapped"] += 1
                if y2 < y1: y1, y2 = y2, y1; stat["swapped"] += 1
                a, b, cc, dd = x1, x2, y1, y2
                x1, x2 = min(max(x1, 1), w), min(max(x2, 1), w)
                y1, y2 = min(max(y1, 1), h), min(max(y2, 1), h)
                if (x1, x2, y1, y2) != (a, b, cc, dd): stat["clamped"] += 1
                if x2 <= x1 or y2 <= y1: stat["degen"] += 1; continue
                stat["obj"] += 1
                nm = names_map.get(c, str(c))
                L += ['\t<object>', '\t\t<name>%s</name>' % nm, '\t\t<pose>Unspecified</pose>',
                      '\t\t<truncated>0</truncated>', '\t\t<difficult>0</difficult>', '\t\t<bndbox>',
                      '\t\t\t<xmin>%d</xmin>' % x1, '\t\t\t<ymin>%d</ymin>' % y1,
                      '\t\t\t<xmax>%d</xmax>' % x2, '\t\t\t<ymax>%d</ymax>' % y2,
                      '\t\t</bndbox>', '\t</object>']
            L.append('</annotation>')
            xmls = "\n".join(L) + "\n"
            # --- per-box verification: re-parse the XML we just built ---
            try:
                root = ET.fromstring(xmls)
            except Exception:
                ver["xml_bad"] += 1; continue
            objs = root.findall("object")
            if len(objs) != len(recs): ver["cnt_bad"] += 1
            if int(root.findtext("size/width")) != w or int(root.findtext("size/height")) != h:
                ver["size_bad"] += 1
            if root.findtext("filename") != stem + ".jpg": ver["cnt_bad"] += 1
            for ob, (c, cx, cy, bw, bh) in zip(objs, recs):
                if ob.findtext("name") != names_map.get(c, str(c)): ver["name_bad"] += 1
                x1 = float(ob.findtext("bndbox/xmin")); y1 = float(ob.findtext("bndbox/ymin"))
                x2 = float(ob.findtext("bndbox/xmax")); y2 = float(ob.findtext("bndbox/ymax"))
                if not (1 <= x1 <= w and 1 <= x2 <= w and 1 <= y1 <= h and 1 <= y2 <= h):
                    ver["range_bad"] += 1
                ex = max(abs((x1 + x2) / 2 / w - cx) * w, abs((x2 - x1) / w - bw) * w)
                ey = max(abs((y1 + y2) / 2 / h - cy) * h, abs((y2 - y1) / h - bh) * h)
                ver["maxerr"] = max(ver["maxerr"], ex); ver["maxerr_y"] = max(ver["maxerr_y"], ey)
                ver["sumerr"] += ex; ver["sumerr_y"] += ey; ver["n"] += 1
            ip = ("images/%s.jpg" % stem) if args.flat else ("JPEGImages/%s.jpg" % stem)
            xp = ("xml/%s.xml" % stem) if args.flat else ("Annotations/%s.xml" % stem)
            zo.writestr(ip, data); written.append(ip)
            zo.writestr(xp, xmls); written.append(xp)
            if (k + 1) % 1000 == 0:
                print("  %d/%d %.0fs" % (k + 1, len(all_stems), time.time() - t0), flush=True)

        if not args.flat:
            for sp in ("train", "val", "test"):
                if sp in splits:
                    zo.writestr("ImageSets/Main/%s.txt" % sp, "\n".join(splits[sp]) + "\n")
                    written.append("ImageSets/Main/%s.txt" % sp)
            zo.writestr("classes.txt", "\n".join(names_map[i] for i in sorted(names_map)) + "\n")
            written.append("classes.txt")
            zo.writestr("README.txt", "\n".join([
                "Pascal VOC (standard layout)",
                "",
                "  JPEGImages/*.jpg                      images, original bytes",
                "  Annotations/*.xml                     same stem as the image",
                "  ImageSets/Main/{train,val,test}.txt  %s" % ", ".join(
                    "%s=%d" % (k, len(v)) for k, v in sorted(splits.items())),
                "  classes.txt                          line index = class id",
                "",
                "Coordinate convention: bndbox is 1-based inclusive pixel (VOC devkit).",
                "    xmin = round((cx - w/2)*W)  xmax = round((cx + w/2)*W)  clamped to [1,W]",
                "  cx,cy,w,h are the source YOLO normalised values; W,H come from the JPEG SOF.",
                "  Some tools (e.g. LabelImg) treat bndbox as 0-based -> 1 px width difference;",
                "  the drawn rectangle is visually identical either way.",
                "",
                "Verification results are in CONVERT_REPORT.txt.",
            ]) + "\n")
            written.append("README.txt")
        log("=== conversion ===")
        log("images written :", len(all_stems))
        log("objects written:", stat["obj"])
        log("depth histogram:", stat["depth"])
        log("clamped / swapped / degenerate-dropped:", stat["clamped"], "/", stat["swapped"], "/", stat["degen"])
        log("")
        log("=== per-box verification (XML re-parsed vs source label) ===")
        log("boxes verified         :", ver["n"])
        log("max round-trip err (px): x %.4f  y %.4f" % (ver["maxerr"], ver["maxerr_y"]))
        log("mean round-trip err(px): x %.5f  y %.5f" % (ver["sumerr"] / max(ver["n"], 1), ver["sumerr_y"] / max(ver["n"], 1)))
        log("xml parse failures / wrong name / out-of-range / count / size:",
            ver["xml_bad"], ver["name_bad"], ver["range_bad"], ver["cnt_bad"], ver["size_bad"])
        log("top-level:", sorted(set(n.split("/")[0] for n in written)))
        log("elapsed %.0fs" % (time.time() - t0))
        zo.writestr("CONVERT_REPORT.txt", "\n".join(R) + "\n")

    with zipfile.ZipFile(args.dst_zip) as z:
        bad = z.testzip()
        nentry = len(z.infolist())
    print("")
    print("ZIP %s : %.1f MB  entries=%d  CRC=%s" % (
        args.dst_zip, os.path.getsize(args.dst_zip) / 1048576.0, nentry, "OK" if bad is None else "FAIL " + str(bad)))
    ok = (ver["xml_bad"] == 0 and ver["name_bad"] == 0 and ver["range_bad"] == 0
          and ver["cnt_bad"] == 0 and ver["size_bad"] == 0 and ver["maxerr"] < 1.0
          and ver["maxerr_y"] < 1.0 and bad is None)
    print("VERDICT:", "PASS (all boxes round-trip < 1 px, no schema/name/range errors)" if ok else "FAIL - inspect above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
