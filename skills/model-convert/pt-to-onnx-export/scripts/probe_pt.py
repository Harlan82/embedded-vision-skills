# -*- coding: utf-8 -*-
"""探测 .pt 的框架家族 / 结构 / 类别表 —— 不需要 torch（纯 zip + pickle 字节流）。

用法:
    python probe_pt.py path/to/best.pt
"""
import zipfile, re, sys, os, json


def printable_runs(b, n=200):
    return [r.decode("utf-8", "replace") for r in re.findall(rb"[ -~]{4,}", b)][:n]


def _read_unicode(blob, p):
    """p 指向字符串对象开头。返回 (str, 新位置)，非字符串则 (None, p)。

    pickle 里两个坑：
      X   = BINUNICODE       -> 后面是 **4 字节小端**长度
      0x8c = SHORT_BINUNICODE -> 后面是 1 字节长度
    """
    if p >= len(blob):
        return None, p
    op = blob[p]
    if op == 0x58:                                          # 'X'
        ln = int.from_bytes(blob[p + 1:p + 5], "little")
        return blob[p + 5:p + 5 + ln].decode("utf-8", "replace"), p + 5 + ln
    if op == 0x8C:                                          # SHORT_BINUNICODE
        ln = blob[p + 1]
        return blob[p + 2:p + 2 + ln].decode("utf-8", "replace"), p + 2 + ln
    return None, p


def extract_names(blob):
    """定位 names 字典并解析其 (键, 值) 序列。

    结构形如: names <memo> } <memo> ( K 0 <memo> X <4B len> 'yuan' <memo> K 1 ...
    K = BININT1（键，0~255 的小整数）
    """
    i = blob.find(b"names")
    while i > 0:
        j = blob.find(b"(", i, i + 300)
        if j < 0:
            i = blob.find(b"names", i + 1); continue
        p, out = j + 1, []
        while p < len(blob):
            if blob[p] != 0x4B:                             # 'K' BININT1
                break
            p += 2                                          # 跳过键
            s, p = _read_unicode(blob, p)
            if s is None:
                break
            out.append(s)
            nxt = blob.find(b"\x4b", p)                     # 下一个 'K'
            if nxt < 0 or nxt - p > 24:                     # 不再是字典键了
                break
            p = nxt
        if len(out) >= 2:
            return out
        i = blob.find(b"names", i + 1)
    return None


def probe(path):
    print(f"file  : {path}")
    print(f"size  : {os.path.getsize(path):,} bytes")
    zf = zipfile.ZipFile(path, "r")
    names = zf.namelist()
    roots = sorted({n.split("/")[0] for n in names})
    print(f"entries: {len(names)}   roots: {roots}")

    for n in names:
        if n.endswith("/version"):
            print(f"[{n}] = {zf.read(n).decode('utf-8', 'replace').strip()!r}")

    pkls = [n for n in names if n.endswith(".pkl")]
    if not pkls:
        print("!! no .pkl found -- not a standard PyTorch 1.6+ checkpoint")
        return
    blob = b""
    for n in pkls:
        b = zf.read(n)
        blob += b
        print(f"[{n}] {len(b):,} bytes")

    runs = printable_runs(blob, 300)
    joined = " ".join(runs)

    print("\n== 家族判定 ==")
    is_ultralytics = ("ultralytics" in joined) or ("cultralytics" in joined)
    print(f"  ultralytics 包训练 : {is_ultralytics}")
    print(f"  原版 yolov5 repo   : {'models.common' in joined or 'anchors' in joined}")

    print("\n== 关键元信息 ==")
    for key in ("yaml_file", "pt_path", "version", "license", "date"):
        i = blob.find(key.encode())
        if i > 0:
            seg = blob[i:i + 160]
            s = "".join(chr(c) if 32 <= c < 127 else "." for c in seg)
            print(f"  {key:10s} {s}")

    print("\n== 结构线索 ==")
    for kw in ("DetectionModel", "ClassificationModel", "SegmentationModel",
               "OBBModel", "PoseModel", "detect", "segment", "obb", "pose", "classify",
               "HalfStorage", "FloatStorage", "reg_max", "end2end", "dfl"):
        print(f"  {kw:20s} {blob.count(kw.encode())}")

    print("\n== 权重存储类型 ==")
    print(f"  HalfStorage 出现 {blob.count(b'HalfStorage')} 次  (ckpt 存 fp16)")
    print(f"  FloatStorage 出现 {blob.count(b'FloatStorage')} 次 (ckpt 存 fp32)")

    print("\n== 类别表 names ==")
    seq = extract_names(blob)
    if seq:
        for i_, n in enumerate(seq):
            print(f"  {i_}  {n}")
        print(f"  -> 共 {len(seq)} 类。这就是类别 id 顺序，端侧类别映射必须按它核对")
    else:
        print("  未能自动提取，请手工查看 x/data.pkl 中 'names' 之后的字节")

    print("\n== 提示 ==")
    if is_ultralytics:
        print("  用 ultralytics 包导出：  yolo export model=... format=onnx")
    else:
        print("  可能是原版 yolov5 repo：用其自带 export.py")
    print("  HalfStorage 表示 ckpt 存 fp16 -> 导出时务必转 fp32（量化前必须）")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    probe(sys.argv[1])
