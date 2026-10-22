# -*- coding: utf-8 -*-
"""
probe.py —— 标定：一条命令量出定阈值需要的全部实测分布。

用法:
  python scripts/probe.py --src "./work/batch-a" [--out ./work/out] [--top 12]

必须**先跑这个再动手**：先测量、后定参，不要凭直觉设阈值。
输出 probe_<name>.txt / .json，含：
  背景 S(p95) / 阴影 S(p50) / 目标 S(p05~p95)  —— 决定 sat_lo、cand_s_min 的安全区间
  面积比 / 宽高比 / 填充率的分位数            —— 决定 area_lo/hi、ratio_max、fill_min
  质心范围                                   —— 决定 center_box 该放宽到多少
  命名前缀分组、色相分布、多候选帧数、贴边帧数 —— 判断是否混了两批拍摄、是否要禁边缘余量
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sdpipe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=None, help="报告输出目录，默认写在源目录的上一级")
    ap.add_argument("--max-images", type=int, default=200)
    a = ap.parse_args()
    out = a.out or os.path.dirname(os.path.normpath(a.src))
    r = sdpipe.probe(a.src, out_dir=out, max_images=a.max_images)
    if r.get("error"):
        print("错误: %s" % r["error"])
        sys.exit(2)
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
