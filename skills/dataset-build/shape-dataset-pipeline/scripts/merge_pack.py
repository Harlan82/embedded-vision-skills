# -*- coding: utf-8 -*-
"""
merge_pack.py —— 把多个已有数据集（ZIP）合并成一个平台可导入的 ZIP。

为什么不能直接拼包（两个真实存在的坑）：
  1. **类别 id 冲突**：每个单类包的类别 id 都是 0，直接拼会退化成单类。
     本脚本按 --jobs 顺序重排为 0..N-1。
  2. **文件名撞车**：多个数据集都用 cam_*.jpg 命名，直接拼会互相覆盖。
     本脚本统一加 <cls>__ 前缀，并清洗成 [A-Za-z0-9._-]。
另外：源包布局兼容 images/train 与 train/images 两种；输出统一为平台规范的
train/images + train/labels + val/... ，data.yaml 不写 path。

用法:
  python scripts/merge_pack.py --jobs jobs.json --zip "./work/shapes-7cls.zip" --gray-frac 0.20
  python scripts/merge_pack.py --job "./work/batch-a.zip:class_2" --job "./work/batch-b.zip:class_7" --zip out.zip
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sdpipe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", default=[],
                    help='重复给出，格式 "zip路径:类别名"')
    ap.add_argument("--jobs", help="jobs.json：[{\"zip\":\"...\",\"cls\":\"...\"}]")
    ap.add_argument("--zip", required=True, dest="zip_out")
    ap.add_argument("--gray-frac", type=float, default=0.0)
    ap.add_argument("--gray-val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audit-n", type=int, default=24)
    ap.add_argument("--audit-dir", default=None)
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    jobs = []
    if a.jobs:
        with open(a.jobs, "r", encoding="utf-8") as fp:
            jobs = json.load(fp)
    for s in a.job:
        zp, _, cls = s.rpartition(":")
        jobs.append(dict(zip=zp, cls=cls))
    if not jobs:
        print("需要 --jobs 或 --job")
        sys.exit(2)

    specs = [(j["zip"], j["cls"], i) for i, j in enumerate(jobs)]
    audit_dir = a.audit_dir or (os.path.splitext(a.zip_out)[0] + "_audit")
    r = sdpipe.merge_zips(specs, a.zip_out, P=dict(seed=a.seed),
                          gray_frac=a.gray_frac, gray_val_ratio=a.gray_val_ratio,
                          audit_dir=audit_dir, audit_n=a.audit_n)
    print("合并完成: %d 张（彩色 %d / 灰度 %d）-> %s"
          % (r["total"], r["n_color"], r["n_gray"], a.zip_out))

    if not a.no_verify:
        v = sdpipe.verify_zip(a.zip_out)
        print("")
        print("=== 校验 ===")
        print("CRC: %s   顶层: %s" % ("OK" if v["crc_ok"] else "BAD", v["top_level"]))
        for split, st in v["stats"].items():
            print("%-5s images=%d labels=%d 缺=%d 多=%d" % (split, st["images"], st["labels"],
                                                            st["missing_label"], st["extra_label"]))
        print("文件名违规: %d   标签异常: %d   坐标范围: %s   类别计数: %s"
              % (v["charset_violations"], v["label_anomalies"], v["coord_range"], v["class_ids"]))
        print("ultralytics: %s" % v.get("ultralytics"))
        if not v["ok"]:
            print("")
            print("!! 存在问题:")
            for p in v["problems"]:
                print("   - %s" % p)
            sys.exit(1)
        print("校验通过")


if __name__ == "__main__":
    main()
