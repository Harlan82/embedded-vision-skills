# -*- coding: utf-8 -*-
"""
annotate.py —— 单（或多）数据集标注 -> 平台可导入 ZIP。

用法:
  # 单个数据集
  python scripts/annotate.py --src "./work/batch-a" --cls class_2 --zip "./work/batch-a.zip"

  # 多个数据集一次跑（等于合并，类别 id 按 --jobs 顺序 0..N-1）
  python scripts/annotate.py --jobs jobs.json --zip "./work/shapes.zip"

  # 顺带做灰度增广（每类抽 20%，train:val = 85:15）
  python scripts/annotate.py --jobs jobs.json --zip "./work/shapes.zip" --gray-frac 0.20

jobs.json 格式:
  [{"src": "./work/batch-a", "cls": "class_2"},
   {"src": "./work/batch-b",     "cls": "class_7"}]

所有阈值都有默认值（见 sdpipe.DEFAULTS），默认值是 8 个真实数据集上验证过的；
只有 probe.py 显示异常时才需要覆盖。
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sdpipe  # noqa: E402

TH_PARAMS = ["sat_lo", "val_max", "cand_s_min", "tighten_k", "tighten_hi", "clip_tol",
             "area_lo", "area_hi", "ratio_max", "fill_min", "center_box", "center_w",
             "edge_margin", "fallback_lo"]


def load_jobs(a):
    if a.jobs:
        with open(a.jobs, "r", encoding="utf-8") as fp:
            js = json.load(fp)
    else:
        js = [dict(src=a.src, cls=a.cls)]
    specs = []
    for i, j in enumerate(js):
        specs.append((j["src"], j.get("cls") or ("class%d" % i), i))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--cls")
    ap.add_argument("--jobs", help="jobs.json：多数据集（=合并）")
    ap.add_argument("--zip", required=True, dest="zip_out")
    ap.add_argument("--gray-frac", type=float, default=0.0)
    ap.add_argument("--val-ratio", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--audit-n", type=int, default=24)
    ap.add_argument("--audit-dir", default=None)
    ap.add_argument("--no-suppress-fixed", action="store_true",
                    help="关闭固定杂物自动抑制（默认开启）")
    for p in TH_PARAMS:
        ap.add_argument("--" + p.replace("_", "-"), type=float, default=None, dest=p)
    a = ap.parse_args()

    if not a.jobs and not (a.src and a.cls):
        print("需要 --src + --cls，或 --jobs")
        sys.exit(2)

    P = {}
    for p in TH_PARAMS:
        v = getattr(a, p)
        if v is not None:
            P[p] = int(v) if p in ("sat_lo", "val_max", "cand_s_min", "tighten_hi",
                                   "fallback_lo") else float(v)
    if a.val_ratio is not None:
        P["val_ratio"] = a.val_ratio
    if a.seed is not None:
        P["seed"] = a.seed

    specs = load_jobs(a)
    audit_dir = a.audit_dir or (os.path.splitext(a.zip_out)[0] + "_audit")
    r = sdpipe.build_dataset(specs, a.zip_out, P=P, gray_frac=a.gray_frac,
                             audit_dir=audit_dir, audit_n=a.audit_n,
                             suppress_fixed=not a.no_suppress_fixed)
    print("命中 %d/%d，未命中 %d，报告: %s" % (r["hit"], r["total"], len(r["miss"]), r["report"]))
    if r["miss"]:
        print("未命中样本（必须逐张确认是「真该丢」还是「被误杀」）：")
        for m in r["miss"][:30]:
            print("   %s  %s  %s" % (m["file"], m["cls"], m["status"]))
    print("审计图: %s" % r.get("audit"))


if __name__ == "__main__":
    main()
