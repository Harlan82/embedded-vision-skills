# -*- coding: utf-8 -*-
"""
doctor.py —— 环境自检。跑 pipeline 之前先跑这个，能省掉 90% 的"莫名失败"。

检查项：Python 版本 / 依赖包 / 中文路径读图 / 写权限 / 磁盘空间 / Ollama 可用性（可选）。
用法:
  python scripts/doctor.py
  python scripts/doctor.py --src "./work/batch-a" --zip-out "./work/batch-a.zip"
"""
import os
import sys
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="要处理的数据集目录（可选）")
    ap.add_argument("--zip-out", dest="zip_out", default=None, help="输出 zip 路径（可选）")
    a = ap.parse_args()

    problems = []
    print("=== 1. Python ===")
    print("版本: %s" % sys.version.replace(os.linesep, " "))
    print("解释器: %s" % sys.executable)
    if sys.version_info < (3, 8):
        problems.append("Python 版本过低（需 >= 3.8）")

    print("")
    print("=== 2. 依赖 ===")
    mods = {}
    for m in ("numpy", "cv2", "yaml"):
        try:
            mod = __import__(m)
            mods[m] = getattr(mod, "__version__", "?")
            print("  %-8s OK  %s" % (m, mods[m]))
        except Exception as e:
            print("  %-8s 缺失 (%s)" % (m, type(e).__name__))
            problems.append("缺少依赖 %s -> 执行: python -m pip install numpy opencv-python pyyaml" % m)
    try:
        import ultralytics  # noqa: F401
        print("  %-8s OK  （可做平台级校验）" % "ultralytics")
    except Exception:
        print("  %-8s 未安装（只影响 verify 的最强一项，可忽略）" % "ultralytics")

    print("")
    print("=== 3. 读图能力（含中文路径） ===")
    if mods.get("cv2") and mods.get("numpy"):
        try:
            import sdpipe
            test_dir = a.src or os.path.dirname(os.path.abspath(__file__))
            fs = sdpipe.list_images(test_dir)
            if fs:
                img = sdpipe.imread_u(fs[0])
                if img is None:
                    problems.append("读图失败: %s（确认文件未损坏）" % fs[0])
                    print("  读图失败: %s" % fs[0])
                else:
                    print("  读图 OK: %s  尺寸=%dx%d" % (os.path.basename(fs[0]),
                                                        img.shape[1], img.shape[0]))
                    hsv = __import__("cv2").cvtColor(img, __import__("cv2").COLOR_BGR2HSV)
                    print("  该图 S(p99)=%d  —— 若目标饱和度低于 60，需要特殊处理"
                          % int(__import__("numpy").percentile(hsv[:, :, 1], 99)))
            else:
                print("  （%s 里没有图片，跳过）" % test_dir)
            print("  中文路径读图实现: np.fromfile + cv2.imdecode （已内置）")
        except Exception as e:
            problems.append("读图自检异常: %s" % e)

    print("")
    print("=== 4. 输出路径可写性 ===")
    targets = []
    if a.zip_out:
        targets.append(os.path.dirname(os.path.abspath(a.zip_out)) or ".")
    targets.append(os.path.dirname(os.path.abspath(__file__)))
    for d in targets:
        probe = os.path.join(d, "_doctor_probe.txt")
        writable = False
        try:
            os.makedirs(d, exist_ok=True)
            with open(probe, "w", encoding="utf-8") as fp:
                fp.write("ok")
            writable = True
        except Exception as e:
            print("  %s  不可写: %s" % (d, e))
            problems.append("输出目录不可写: %s" % d)
            continue
        try:                     # 清理是尽力而为，失败不影响结论
            os.remove(probe)
        except Exception:
            pass
        if writable:
            try:
                du = shutil.disk_usage(d)
                print("  %s  可写，剩余 %.1f GB" % (d, du.free / 1073741824.0))
                if du.free < 2 * 1073741824:
                    problems.append("磁盘剩余不足 2GB: %s" % d)
            except Exception as e:
                print("  %s  可写（磁盘信息读取失败: %s）" % (d, e))

    print("")
    print("=== 5. 本地模型（可选） ===")
    try:
        import ask_vlm
        ms = ask_vlm.available()
        if ms:
            print("  Ollama 可用，模型: %s" % ms)
            print("  提示：视觉复核用 --vlm <模型名>，纯文本模型只能做文案类复核")
        else:
            print("  Ollama 不可用（不影响主流程，--vlm 会自动跳过）")
    except Exception as e:
        print("  跳过: %s" % e)

    print("")
    if problems:
        print("RESULT: FAIL")
        for p in problems:
            print("   - %s" % p)
        sys.exit(1)
    print("RESULT: OK  环境可用，可以执行: python scripts/pipeline.py --jobs jobs.json --zip out.zip")


if __name__ == "__main__":
    main()
