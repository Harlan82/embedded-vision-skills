# -*- coding: utf-8 -*-
"""
verify.py —— 平台导入前的最后一道闸：结构 / 配对 / 标签 / 坐标 / yaml / 文件名。

用法:
  python scripts/verify.py --zip "./work/shapes-7cls.zip"

退出码 0 = 通过；1 = 有问题（会逐条列出）。
还会用 ultralytics 的 check_det_dataset 真跑一遍（装了 ultralytics 时），
这是最接近平台侧解析的验证方式。
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sdpipe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, dest="zip_path")
    ap.add_argument("--json", action="store_true", help="额外打印 JSON")
    a = ap.parse_args()
    v = sdpipe.verify_zip(a.zip_path)
    print("ZIP: %s  (%.1f MB)" % (a.zip_path, os.path.getsize(a.zip_path) / 1048576.0))
    print("CRC: %s" % ("OK" if v["crc_ok"] else "BAD"))
    print("顶层: %s" % v["top_level"])
    print("data.yaml:")
    for ln in (v.get("data_yaml") or "").split(sdpipe.LF):
        print("   |%s|" % ln)
    for split, st in v["stats"].items():
        print("%-5s images=%d labels=%d 缺=%d 多=%d" % (split, st["images"], st["labels"],
                                                        st["missing_label"], st["extra_label"]))
    print("文件名违规: %d" % v["charset_violations"])
    print("标签异常: %d" % v["label_anomalies"])
    print("坐标范围: %s" % v["coord_range"])
    print("类别计数: %s" % v["class_ids"])
    print("ultralytics: %s" % v.get("ultralytics"))
    if a.json:
        print(json.dumps(v, ensure_ascii=False, indent=2))
    if not v["ok"]:
        print("")
        print("!! 未通过:")
        for p in v["problems"]:
            print("   - %s" % p)
        sys.exit(1)
    print("")
    print("通过。交付前还必须：肉眼看过审计图，确认框贴住目标。")


if __name__ == "__main__":
    main()
