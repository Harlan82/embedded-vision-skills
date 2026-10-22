# -*- coding: utf-8 -*-
"""三级对拍：raw 张量 / 解码检测框 / mAP（对齐批处理口径）。

用法:
    python verify_onnx.py best.pt out/best.onnx --val-dir <images/val> \
        [--names class_6 class_2-top ...] [--dataset-root DIR] [--imgsz 640]
"""
import os, sys, glob, argparse, json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt")
    ap.add_argument("onnx")
    ap.add_argument("--val-dir", required=True, help="验证集图片目录（*.jpg/*.png）")
    ap.add_argument("--dataset-root", default=None, help="数据集根（含 images/ labels/），用于 mAP")
    ap.add_argument("--names", nargs="*", default=None, help="类别名，按 id 顺序；不给则从 pt 读")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--report", default=None, help="报告输出路径")
    a = ap.parse_args()

    import torch, cv2, onnxruntime as ort
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox

    log = []
    w = log.append
    pnet, onet = YOLO(a.pt), YOLO(a.onnx)
    names = a.names or [pnet.model.names[i] for i in sorted(pnet.model.names)]

    w("== 0. 基本信息 ==")
    w(f"  pt   : {a.pt}")
    w(f"  onnx : {a.onnx}")
    w(f"  nc={len(names)}  names={json.dumps(list(names), ensure_ascii=False)}")

    files = sorted(sum([glob.glob(os.path.join(a.val_dir, e))
                        for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp")], []))
    w(f"  验证图片 {len(files)} 张")
    if not files:
        print("!! val-dir 里没有图片"); sys.exit(1)

    # ---------- 一级 ----------
    w("\n== 1. raw 张量对拍（最大幅值 + 相对误差） ==")
    net = pnet.model.float().eval()
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    lb = LetterBox((a.imgsz, a.imgsz), auto=False, scale_fill=False)

    def prep(p):
        im0 = cv2.imread(p)
        im = lb(image=im0)
        return np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1)[None],
                                    dtype=np.float32) / 255.0

    mag = 0.0; rel = 0.0; worst = (None, 0.0, 0.0)
    for p in files:
        xn = prep(p)
        with torch.no_grad():
            t = net(torch.from_numpy(xn))
        t = (t[0] if isinstance(t, (list, tuple)) else t).numpy()
        o = sess.run(None, {iname: xn})[0]
        mm = float(np.abs(t).max()); d = float(np.abs(t - o).max())
        mag = max(mag, mm)
        r = d / max(mm, 1e-9)
        if r > rel:
            rel, worst = r, (os.path.basename(p), d, mm)
    w(f"  输出最大幅值   {mag:.3f}")
    w(f"  绝对差最大     {worst[1]:.3e}  ({worst[0]}, 该图幅值 {worst[2]:.3f})")
    w(f"  相对误差最大   {rel:.3e}   -> {'PASS' if rel < 1e-4 else 'CHECK'}")

    # 输入量程判定
    xn = prep(files[0])
    with torch.no_grad():
        t = net(torch.from_numpy(xn))
    t = (t[0] if isinstance(t, (list, tuple)) else t).numpy()
    d1 = float(np.abs(t - sess.run(None, {iname: xn})[0]).max())
    d255 = float(np.abs(t - sess.run(None, {iname: xn * 255.0})[0]).max())
    w(f"  量程判定: 输入0-1 -> {d1:.3e} | 输入0-255 -> {d255:.3e}")
    w(f"  => ONNX 期望 {'0~1 归一化' if d1 < d255 else '0~255'} 输入")

    # ---------- 二级 ----------
    w("\n== 2. 解码后检测框对拍 ==")
    for conf in (0.25, 0.001):
        n1 = n2 = 0; bad = 0; clsm = 0; bd = cd = fd = 0.0
        for p in files:
            r1 = pnet.predict(p, imgsz=a.imgsz, conf=conf, iou=0.7, max_det=300,
                              verbose=False, device="cpu")[0]
            r2 = onet.predict(p, imgsz=a.imgsz, conf=conf, iou=0.7, max_det=300,
                              verbose=False, device="cpu")[0]
            n1 += len(r1.boxes); n2 += len(r2.boxes)
            if len(r1.boxes) != len(r2.boxes):
                bad += 1; continue
            if len(r1.boxes) == 0:
                continue
            if not np.array_equal(r1.boxes.cls.numpy(), r2.boxes.cls.numpy()):
                clsm += 1
            bd = max(bd, float(np.abs(r1.boxes.xyxy.numpy() - r2.boxes.xyxy.numpy()).max()))
            fd = max(fd, float(np.abs(r1.boxes.conf.numpy() - r2.boxes.conf.numpy()).max()))
        w(f"  conf={conf:<6} 框数 pt={n1} onnx={n2}  数量不等图={bad}  类别不等图={clsm}")
        w(f"               框坐标 max|Δ|={bd:.6f} px   置信度 max|Δ|={fd:.3e}")
    w("  -> 判据：数量差 0、类别差 0、框 <0.01px、置信度 <1e-4")

    # ---------- 三级 ----------
    def _level3():
        w("\n== 3. mAP 对齐（必须对齐批处理口径） ==")
        cfg = os.path.join(os.path.dirname(os.path.abspath(a.report or a.onnx)),
                           "_val_cfg.yaml")
        with open(cfg, "w", encoding="utf-8") as f:
            f.write("path: " + a.dataset_root.replace("\\", "/") + "\n")
            f.write("train: images/train\nval: images/val\n")
            f.write(f"nc: {len(names)}\nnames:\n")
            for i, n in enumerate(names):
                f.write(f"  {i}: {n}\n")
        w(f"  生成 {cfg}")
        rows = []
        for tag, mdl, kw in (("pt   batch=16", pnet, dict(batch=16)),
                             ("pt   batch=1 ", pnet, dict(batch=1, rect=False)),
                             ("onnx batch=1 ", onet, dict(batch=1))):
            r = mdl.val(data=cfg, imgsz=a.imgsz, plots=False, verbose=False,
                        device="cpu", **kw)
            rows.append((tag, float(r.box.map50), float(r.box.map)))
        w(f"  {'口径':14s} {'mAP50':>10s} {'mAP50-95':>10s}")
        for t, m50, m95 in rows:
            w(f"  {t:14s} {m50:10.4f} {m95:10.4f}")
        w(f"  pt(b16) vs pt(b1)  d(50-95) = {abs(rows[0][2]-rows[1][2]):.4f}   <- 差值来自 batch")
        w(f"  pt(b1)  vs onnx     d(50-95) = {abs(rows[1][2]-rows[2][2]):.4f}   <- 这一项为 0 才算等价")
        w("  ⚠ 不要拿 pt(batch=16) 和 onnx 直接比，批内 rect 填充不同会凭空差 0.03")

    if a.dataset_root:
        try:
            _level3()
        except Exception as exc:
            # ★ 三级失败【绝不能】吞掉一、二级结论：报告必须照常落盘。
            #   实测踩过：ultralytics 的 check_det_dataset 会联网下载 Arial.ttf，
            #   网络 502 时直接抛 ConnectionError，把已经跑完的一、二级结果全丢掉。
            w(f"  !! 三级 mAP 未完成（{type(exc).__name__}: {exc}）")
            w("     最常见根因：ultralytics check_det_dataset 要联网下字体（Arial.ttf）")
            w("     而网络失败。修法：把 C:\\Windows\\Fonts\\arial.ttf 复制到")
            w("     ultralytics 的 USER_CONFIG_DIR（通常 %APPDATA%\\Ultralytics）后重跑。")
            w("     ★ 本 skill 的等价性判据以一、二级为准，未跑三级不影响结论。")

    txt = "\n".join(log)
    print(txt)
    if a.report:
        os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
        open(a.report, "w", encoding="utf-8").write(txt)


if __name__ == "__main__":
    main()
