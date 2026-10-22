# -*- coding: utf-8 -*-
"""
ask_vlm.py —— 用本地 Ollama 模型复核「可疑帧」。**可选功能**，跑不通就自动跳过。

为什么需要它：标注流水线是确定性的，但它只能给出「哪些帧可疑」，
判断「框有没有咬进目标」这一步通常要人看图。本地 4B 视觉模型可以承担这一步，
拿不到模型时退化为「输出清单请人看」——不影响主流程。

用法:
  python scripts/ask_vlm.py --img "./work/f1.jpg" "./work/f2.jpg" --model qwen2.5vl:3b

作为库使用:
  import ask_vlm
  r = ask_vlm.review_zip_frames(zip_path, flagged_records, "qwen2.5vl:3b", out_dir)
"""
import os
import sys
import json
import base64
import argparse
import urllib.request

HOST = "http://127.0.0.1:11434"
PROMPT = (
    "这是一张工件检测数据集的标注预览图，红色矩形是自动标注框。"
    "请只回答一个词：如果红框基本包住了画面中间的工件，回答 GOOD；"
    "如果红框明显偏离工件、或只框住工件的一部分，回答 BAD。"
)

LF = chr(10)


def ollama_generate(model, prompt, images=None, host=HOST, timeout=90):
    """调用 Ollama /api/generate。任何异常都返回 (None, 错误信息)，绝不抛给上层。"""
    payload = {"model": model, "prompt": prompt, "stream": False}
    if images:
        payload["images"] = images
    try:
        req = urllib.request.Request(
            host.rstrip("/") + "/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8")).get("response", ""), None
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, e)


def available(host=HOST, timeout=4):
    try:
        req = urllib.request.Request(host.rstrip("/") + "/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            tags = json.loads(r.read().decode("utf-8"))
        return [m.get("name") for m in tags.get("models", [])]
    except Exception:
        return []


def review_zip_frames(zip_path, records, model, out_dir, max_frames=8, host=HOST):
    """把 zip 里的可疑帧导出成 jpg，逐张问本地模型。返回 dict(summary, items)。"""
    import zipfile
    import numpy as np
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    z = zipfile.ZipFile(zip_path)
    names = z.namelist()
    items = []
    n_ok = n_bad = n_unk = 0
    for i, r in enumerate(records[:max_frames]):
        stem = r.get("name")
        if not stem:
            continue
        member = None
        for m in names:
            if os.path.splitext(os.path.basename(m))[0] == stem and m.endswith(".jpg"):
                member = m
                break
        if member is None:
            continue
        img = cv2.imdecode(np.frombuffer(z.read(member), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        # 画上框，让模型看到的就是最终标注
        split = member.split("/")[0]
        ln = "%s/labels/%s.txt" % (split, stem)
        if ln in names:
            p = z.read(ln).decode("utf-8", "ignore").split()
            if len(p) == 5:
                H, W = img.shape[:2]
                cx, cy, bw, bh = [float(t) for t in p[1:]]
                cv2.rectangle(img, (int((cx - bw / 2) * W), int((cy - bh / 2) * H)),
                              (int((cx + bw / 2) * W), int((cy + bh / 2) * H)), (0, 0, 255), 3)
        fp = os.path.join(out_dir, "review_%02d_%s.jpg" % (i, stem[:40]))
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            continue
        with open(fp, "wb") as f:
            f.write(buf.tobytes())
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        resp, err = ollama_generate(model, PROMPT, [b64], host=host)
        if resp is None:
            items.append(dict(frame=stem, verdict="UNAVAILABLE", error=err))
            n_unk += 1
            break
        v = resp.strip().upper()
        verdict = "GOOD" if "GOOD" in v else ("BAD" if "BAD" in v else "UNCLEAR")
        items.append(dict(frame=stem, verdict=verdict, raw=resp.strip()[:120], file=fp))
        n_ok += verdict == "GOOD"
        n_bad += verdict == "BAD"
        n_unk += verdict == "UNCLEAR"

    summary = "GOOD=%d BAD=%d UNCLEAR=%d" % (n_ok, n_bad, n_unk)
    res = dict(model=model, summary=summary, items=items, out_dir=out_dir)
    with open(os.path.join(out_dir, "vlm_review.json"), "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)
    if n_bad:
        summary += " -> 有 %d 张被判 BAD，请人工复核（见 %s）" % (n_bad, out_dir)
    res["summary"] = summary
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--prompt", default=PROMPT)
    a = ap.parse_args()

    models = available(a.host)
    if not models:
        print("VLM_UNAVAILABLE: 连不上 Ollama (%s)。请先启动 ollama serve。" % a.host)
        sys.exit(0)
    print("本机模型: %s" % models)
    for p in a.img:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        resp, err = ollama_generate(a.model, a.prompt, [b64], host=a.host)
        if resp is None:
            print("VLM_UNAVAILABLE: %s" % err)
            sys.exit(0)
        print("%s -> %s" % (os.path.basename(p), resp.strip()[:200]))


if __name__ == "__main__":
    main()
