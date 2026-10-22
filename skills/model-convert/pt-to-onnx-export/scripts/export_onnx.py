# -*- coding: utf-8 -*-
"""Ultralytics .pt -> ONNX 导出，并打印图结构摘要。

用法:
    python export_onnx.py best.pt [--outdir DIR] [--imgsz 640] [--opset 12] [--name best.onnx]
"""
import os, sys, json, time, shutil, argparse, hashlib
import onnx


def md5(p, bs=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt")
    ap.add_argument("--outdir", default=None, help="默认与 pt 同目录")
    ap.add_argument("--name", default=None, help="默认 <stem>.onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--dynamic", action="store_true", help="默认静态")
    ap.add_argument("--no-simplify", action="store_true", help="默认开启 onnxslim 简化")
    ap.add_argument("--half", action="store_true", help="默认 fp32（量化前必须 fp32）")
    ap.add_argument("--nms", action="store_true", help="默认不烘焙 NMS")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from ultralytics import YOLO
    y = YOLO(a.pt)
    print("== 源模型 ==")
    print(f"  task  = {y.task}")
    print(f"  names = {json.dumps(y.model.names, ensure_ascii=False)}")
    print(f"  nc    = {len(y.model.names)}")
    try:
        n_par = sum(p.numel() for p in y.model.parameters())
        print(f"  参数  = {n_par:,}")
        print(f"  dtype = {next(y.model.parameters()).dtype}")
    except Exception:
        pass

    print("\n== 导出 ==")
    print(f"  imgsz={a.imgsz} opset={a.opset} dynamic={a.dynamic} "
          f"simplify={not a.no_simplify} half={a.half} nms={a.nms} device={a.device}")
    t0 = time.time()
    out = y.export(format="onnx", imgsz=a.imgsz, opset=a.opset, simplify=not a.no_simplify,
                   dynamic=a.dynamic, half=a.half, nms=a.nms, batch=1, device=a.device)
    print(f"  完成 {time.time()-t0:.1f}s -> {out}")

    outdir = a.outdir or os.path.dirname(os.path.abspath(a.pt))
    os.makedirs(outdir, exist_ok=True)
    name = a.name or (os.path.splitext(os.path.basename(a.pt))[0] + ".onnx")
    final = os.path.join(outdir, name)
    if os.path.abspath(out) != os.path.abspath(final):
        shutil.copy2(out, final)

    print("\n== 产物 ==")
    print(f"  文件 {final}")
    print(f"  大小 {os.path.getsize(final):,} bytes")
    print(f"  MD5  {md5(final)}")

    print("\n== ONNX 图 ==")
    m = onnx.load(final)
    print(f"  ir_version {m.ir_version}   producer {m.producer_name} {m.producer_version}")
    print(f"  opset      {[(o.domain or 'ai.onnx', o.version) for o in m.opset_import]}")
    for grp, tag in ((m.graph.input, "input"), (m.graph.output, "output")):
        for t in grp:
            dims = [(("dynamic:" + d.dim_param) if d.dim_param else d.dim_value)
                    for d in t.type.tensor_type.shape.dim]
            print(f"  {tag:7s} {t.name:12s} {dims}")
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"  节点 {len(m.graph.node)} 个 / {len(ops)} 种算子")
    print(f"  {dict(sorted(ops.items(), key=lambda kv: -kv[1]))}")
    bad = [k for k in ops if k in ("NonMaxSuppression", "TopK", "RoiAlign")]
    print(f"  NMS/TopK 算子: {bad if bad else '无（纯原始输出）'}")
    print("\n  下一步：verify_onnx.py 做三级对拍；若上面输出个数为 1 且形如 "
          "[1, 4+nc, 8400]，说明是 anchor-free 头，端侧后处理不要按原版 YOLOv5 写。")


if __name__ == "__main__":
    main()
