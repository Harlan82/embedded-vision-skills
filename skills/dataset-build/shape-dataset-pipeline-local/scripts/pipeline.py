# -*- coding: utf-8 -*-
"""
pipeline.py —— 一条命令跑完全流程：标注 -> 合并 -> 灰度增广 -> 平台规范打包 -> 自检。

面向本地小模型（4B 级）设计：**不需要任何需要判断的参数**，阈值全部自动标定。
你只需要给出「图片文件夹 + 类别名」，和输出 zip 的路径。

标准用法（复制粘贴即可，注意路径用正斜杠或双引号包住）:

    python scripts/pipeline.py --src "./work/batch-a" --cls class_2 --zip "./work/batch-a.zip"

多数据集合并（顺序决定类别 id 0..N-1）:

    python scripts/pipeline.py --jobs jobs.json --zip "./work/shapes-7cls.zip" --gray 0.20

常用可选参数:
    --limit 40     每个数据集只跑前 40 张（快速试跑，先确认命令能通）
    --work DIR     报告与审计图的输出目录（默认与 zip 同目录）
    --gray 0.20    每类抽 20% 做灰度增广（不加就不做）
    --vlm MODEL    用本地 Ollama 模型复核可疑帧（例 qwen2.5vl:3b）；不可用会自动跳过
    --no-verify    跳过 ultralytics 的 check_det_dataset 校验

结束时会打印最后一行 `RESULT: OK` 或 `RESULT: FAIL:<原因>`。
退出码 0 = 成功，1 = 失败。**只看这一行 + 退出码就够了。**
"""
import os
import sys
import json
import time
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sdpipe  # noqa: E402

LF = chr(10)


def _limit_src(src, n, work, cls):
    """把前 n 张复制到独立目录做快速试跑（不删任何东西）。"""
    if not n or n <= 0:
        return src
    fs = sdpipe.list_images(src)
    if len(fs) <= n:
        return src
    d = os.path.join(work, "limit_%s_%d" % (sdpipe.sanitize_name(cls)[:20], n))
    os.makedirs(d, exist_ok=True)
    for f in fs[:n]:
        tgt = os.path.join(d, os.path.basename(f))
        if not os.path.exists(tgt):
            shutil.copy2(f, tgt)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--cls")
    ap.add_argument("--jobs", help='jobs.json: [{"src":"...","cls":"..."}, ...]')
    ap.add_argument("--zip", required=True, dest="zip_out")
    ap.add_argument("--gray", type=float, default=0.0)
    ap.add_argument("--work", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--audit-n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vlm", default=None)
    ap.add_argument("--cand-s-min", type=int, default=None,
                    help="候选级饱和度下限（默认 125，用于剔手指）。"
                         "仅当 B 类回退帧 >1%% 且目标颜色偏淡时才下调到 110~118")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    if a.jobs:
        with open(a.jobs, "r", encoding="utf-8") as fp:
            jobs = json.load(fp)
    elif a.src and a.cls:
        jobs = [dict(src=a.src, cls=a.cls)]
    else:
        print("RESULT: FAIL:用法错误，需要 --jobs 或（--src + --cls）")
        sys.exit(2)

    zip_out = a.zip_out
    work = a.work or (os.path.dirname(os.path.abspath(zip_out)) or ".")
    os.makedirs(work, exist_ok=True)
    audit_dir = os.path.join(work, os.path.splitext(os.path.basename(zip_out))[0] + "_audit")

    specs = []
    for i, j in enumerate(jobs):
        src = _limit_src(j["src"], a.limit, work, j.get("cls") or ("class%d" % i))
        specs.append((src, j.get("cls") or ("class%d" % i), i))

    print("=== 开始：%d 个数据集 -> %s" % (len(specs), zip_out))
    for (src, cls, cid) in specs:
        print("   id=%d  %-14s <- %s" % (cid, cls, src))

    t0 = time.time()
    P = dict(seed=a.seed)
    if a.cand_s_min:
        P["cand_s_min"] = int(a.cand_s_min)
    try:
        rep = sdpipe.build_dataset(specs, zip_out, P=P, gray_frac=a.gray,
                                   audit_dir=audit_dir, audit_n=a.audit_n,
                                   suppress_fixed=True)
    except Exception as e:
        print("RESULT: FAIL:标注/打包异常 %s: %s" % (type(e).__name__, e))
        sys.exit(1)
    rep["seconds"] = round(time.time() - t0, 1)

    lines = []
    def w(s=""):
        lines.append(str(s))
        print(s)
    w("")
    w("=== 结果 ===")
    w("总计 %d 张，命中 %d，未命中 %d，耗时 %.1fs"
      % (rep["total"], rep["hit"], len(rep["miss"]), rep["seconds"]))
    w("彩色 %d / 灰度 %d，体积 %.2f MB" % (rep["n_color"], rep["n_gray"], rep["zip_mb"]))
    w("%-14s %6s %6s %7s %6s %6s" % ("class", "total", "hit", "train", "val", "gray"))
    for cls, d in rep["per_class"].items():
        w("%-14s %6d %6d %7d %6d %6d" % (cls, d["total"], d["hit"], d["train"], d["val"], d["gray"]))
    w("阈值路径: %s" % rep["threshold_hist"])
    fb_n = sum(v for k, v in rep["threshold_hist"].items() if str(k).startswith("fallback"))
    fb_pct = 100.0 * fb_n / max(1, rep["hit"])
    hint = ""
    if fb_pct > 1.0:
        hint = "  <-- 超过 1%%，若目标颜色偏淡，用 --cand-s-min 110 重跑"
    w("回退帧比例: %d/%d = %.2f%%%s" % (fb_n, rep["hit"], fb_pct, hint))
    w("裁剪率 clip: p50=%s p90=%s（>0.10 表示没有收紧，属正常保护）"
      % (rep["clip_p50"], rep["clip_p90"]))
    if rep["miss"]:
        w("未命中 %d 张（这些帧没有任何可标注目标）：" % len(rep["miss"]))
        for m in rep["miss"][:20]:
            w("   %s  %s  %s" % (m["file"], m["cls"], m["status"]))
    if rep.get("artifacts"):
        w("已自动抑制的固定杂物 %d 处：%s"
          % (len(rep["artifacts"]), [(x["cx"], x["cy"], x["count"]) for x in rep["artifacts"]]))

    # ---- 可疑帧清单（给不看图的小模型用：纯文本）----
    # 分两类，不要混在一起，否则会误导：
    #   A. 正常路径里 clip 偏高  -> 可能真的咬进目标，必须看
    #   B. 走了回退路径的帧      -> 手指/手掌与目标粘连，已按「目标本体」框选，
    #                               clip 天然偏高（框故意不含手指），属预期，抽看即可
    recs_path = os.path.splitext(zip_out)[0] + "_records.json"
    flagged = []
    if os.path.exists(recs_path):
        try:
            with open(recs_path, "r", encoding="utf-8") as fp:
                recs = json.load(fp)
            ok = [r for r in recs if r.get("ok")]
            strict_hi = sorted([r for r in ok
                                if (r.get("path") or "strict") == "strict"
                                and (r.get("clip") or 0) > 0.10],
                               key=lambda r: -(r.get("clip") or 0))
            fb = [r for r in ok if (r.get("path") or "").startswith("fallback")]
            flagged = strict_hi[:10] or fb[:10]
            if strict_hi:
                w("A. 正常路径但裁剪率偏高（可能咬进目标，需人工看一眼）：")
                for r in strict_hi[:10]:
                    w("   %-34s clip=%.3f area=%.4f thr=%s" %
                      (r.get("name"), r.get("clip") or 0, r.get("area_ratio") or 0, r.get("thr")))
            else:
                w("A. 正常路径裁剪率全部 ≤0.10（没有咬进目标的帧）")
            if fb:
                w("B. 回退帧 %d 张（手指/手掌与目标粘连，已按目标本体框选，clip 偏高属预期）："
                  % len(fb))
                for r in fb[:10]:
                    w("   %-34s clip=%.3f area=%.4f thr=%s" %
                      (r.get("name"), r.get("clip") or 0, r.get("area_ratio") or 0, r.get("thr")))
        except Exception as e:
            w("（读取逐图清单失败：%s）" % e)

    # ---- 可选：本地 4B 模型复核 ----
    vlm_result = None
    if a.vlm and flagged:
        try:
            import ask_vlm
            vlm_result = ask_vlm.review_zip_frames(zip_out, flagged, a.vlm,
                                                   os.path.join(work, "vlm_review"))
            w("本地模型复核(%s)：%s" % (a.vlm, vlm_result["summary"]))
        except Exception as e:
            w("本地模型复核跳过：%s" % e)

    # ---- 自检 ----
    v = None
    if not a.no_verify:
        w("")
        w("=== 自检 ===")
        v = sdpipe.verify_zip(zip_out)
        w("CRC: %s   顶层: %s" % ("OK" if v["crc_ok"] else "BAD", v["top_level"]))
        for split, st in v["stats"].items():
            w("%-5s images=%d labels=%d 缺=%d 多=%d"
              % (split, st["images"], st["labels"], st["missing_label"], st["extra_label"]))
        w("文件名违规 %d   标签异常 %d   坐标范围 %s   类别计数 %s"
          % (v["charset_violations"], v["label_anomalies"], v["coord_range"], v["class_ids"]))
        w("ultralytics: %s" % v.get("ultralytics"))

    summary_path = os.path.join(work, os.path.splitext(os.path.basename(zip_out))[0] + "_RUN_SUMMARY.txt")
    with open(summary_path, "w", encoding="utf-8") as fp:
        fp.write(LF.join(lines) + LF)
    print("汇总已写入: %s" % summary_path)

    if v is not None and not v["ok"]:
        print("")
        print("问题清单：")
        for p in v["problems"]:
            print("   - %s" % p)
        print("RESULT: FAIL:自检未通过")
        sys.exit(1)

    print("")
    print("交付物：%s" % zip_out)
    print("审计图（必须肉眼确认框贴住目标）：%s" % (rep.get("audit") or audit_dir))
    print("RESULT: OK")


if __name__ == "__main__":
    main()
