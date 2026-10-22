---
name: pt-to-onnx-export
description: 把 Ultralytics 系（YOLOv5/YOLOv8/YOLO11/YOLO26，含 -u 变体）训练产物 .pt 导出为 ONNX，并做「零误差」级别的严格对拍校验。当用户提到「pt 转 onnx」「导出 onnx」「best.pt 转 onnx」「模型转 onnx」「yolo export」「onnx 导出」「pt 和 onnx 不一致」「量化前的 onnx」时使用。产出 静态输入 ONNX + 三级校验报告（raw 张量 / 解码检测框 / mAP 对齐），并给出下游量化工具链的注意事项。
agent_created: true
---

# .pt → ONNX 导出与严格对拍

面向嵌入式部署链路（MaixCAM / Pulsar2 / RKNN / TensorRT 等）的量化前一步。

## 0. 铁律：先判定模型家族，别猜

`.pt` 里的类名、结构决定了导出方式，**必须实测**，不要按文件名猜。

```bash
python scripts/probe_pt.py path/to/best.pt
```

它不需要 torch，纯读 zip + pickle 字节流，能直接给出：

| 探测项 | 含义 |
|---|---|
| `data.pkl` 里的 `ultralytics.nn.tasks` / `DetectionModel` | 是 **ultralytics 包**训练（用 `yolo export`） |
| 出现 `models.common` / `yolo.py` / `anchors` | 是**原版 ultralytics/yolov5 repo**（用其 `export.py`） |
| `version` 条目里的 `8.4.155` | ultralytics 版本，**导出时最好装同一个版本** |
| `yaml_file` 里的 `models/v5/yolov5s.yaml` | 结构是 YOLOv5su（anchor-free），**不是**原版 YOLOv5 |
| `names` dict | 类别 id → 类名，**这是后面和端侧类别表对齐的唯一依据** |
| 权重 storage 类型 | `HalfStorage` = ckpt 存的是 fp16，导出时转 fp32 |

### ⚠ 最容易踩的坑：yolov5su ≠ yolov5

`model: yolov5su.pt` 这种带 **u 后缀**的是 Ultralytics 重写的 YOLOv5（anchor-free、DFL）。

| | 原版 ultralytics/yolov5 | Ultralytics 的 yolov5**u** |
|---|---|---|
| 导出命令 | 仓库自带 `export.py` | `yolo export`（ultralytics 包） |
| 输出个数 | **3 个**（P3/P4/P5） | **1 个** |
| 输出形状 | `[1, na*3, h, w]` × 3 | `[1, 4+nc, 8400]` |
| anchors | 有（写死 3×3） | **无**（anchor-free） |
| 回归 | 直接 xywh + 网格解码 | **DFL**：4 行距离需 `4×reg_max` 分组 softmax 加权 |
| 分类 | sigmoid | sigmoid |

**任何按原版 YOLOv5 写的端侧后处理，套到 yolov5u 上都会错。** 拿到 ONNX 后先 `netron` 或本 skill 的 `export_onnx.py` 打印的输出形状，确认是 1 个输出还是 3 个。

## 1. 环境

隔离 venv（不要污染系统 Python）。本机可复用的环境：

```
<venv>\Scripts\python.exe
```

需要的包：

```bash
<venv>/Scripts/python.exe -m pip install matplotlib tqdm psutil py-cpuinfo pyyaml pandas \
    ultralytics-thop onnx onnxslim onnxruntime
<venv>/Scripts/python.exe -m pip install --no-deps ultralytics
```

`--no-deps` 是故意的：避免 ultralytics 把 `opencv-python` 拉进来，与已装的 `opencv-python-headless` 打架（两个包都提供 `cv2/`，互相覆盖）。headless 提供 `cv2` 就够用。

### ⚠ 坏包症状与修法（本机 venv 踩过）

被一次失败的 pip 中断后，`site-packages` 里会留下**只有空壳目录**的包：`import` 不报错，但访问 `__version__` 或子模块就炸。典型报错：

| 报错 | 坏包 | 修法 |
|---|---|---|
| `cannot import name 'AsyncFileLock' from 'filelock'` | filelock | `pip install --ignore-installed --no-deps filelock==3.32.4` |
| `module 'dateutil' has no attribute '__version__'` | python-dateutil | 同上，`python-dateutil==2.9.0.post0` |
| `module 'kiwisolver' has no attribute '__version__'` | kiwisolver | `pip install --ignore-installed --no-deps kiwisolver` |
| `cannot import name 'where' from 'certifi'` | certifi | 同上 |
| `Unable to find acceptable character detection dependency` | charset_normalizer | 同上 |

关键是 **`--ignore-installed --no-deps`**：普通 `pip install -U` 会报 `Cannot uninstall ... no RECORD file was found`，装不上。

装完先跑一次全量体检再往下做：

```bash
<venv>/Scripts/python.exe -c "
import importlib
for m in ['filelock','dateutil','kiwisolver','certifi','charset_normalizer','requests',
          'matplotlib','pandas','torch','torchvision','cv2','numpy','onnx','onnxruntime',
          'onnxslim','ultralytics','yaml','tqdm','psutil']:
    try:
        x=importlib.import_module(m); print('OK ',m, getattr(x,'__version__','-'))
    except Exception as e: print('FAIL',m,type(e).__name__,e)
"
```

## 2. 导出

```bash
python scripts/export_onnx.py best.pt --outdir ./out --imgsz 640 --opset 12
```

关键参数与理由：

| 参数 | 值 | 为什么 |
|---|---|---|
| `imgsz` | 跟训练一致（如 640） | 不一致会掉点 |
| `opset` | **12**（保守用 11） | 多数边缘量化工具链（Pulsar2/RKNN/TensorRT older）对高 opset 支持差；12 是安全区 |
| `dynamic` | **False** | 边缘工具链基本都要求全静态输入形状 |
| `simplify` | **True** | 走 onnxslim，消掉冗余 Transpose/Gather，显著减少算子种类 |
| `half`/`quantize` | **False** | 量化前必须给 fp32；fp16 会让后续 PTQ 校准失真 |
| `nms` | **False** | 除非明确要端到端；NMS 算子多数 NPU 不友好，且会让后处理不可控 |
| `batch` | 1 | 静态导出 |

导出后脚本会打印输入/输出名与形状、opset、算子统计、是否含 NMS/TopK。**必看这两项**：输出形状是否符合预期（如 `[1, 4+nc, 8400]`）、有没有混进 `NonMaxSuppression`。

## 3. 校验（三级，必须全过）

```bash
python scripts/verify_onnx.py best.pt out/best.onnx --val-dir <images/val 目录> --names a b c ...
```

### 一级：raw 张量对拍
torch fp32 vs onnxruntime，逐图算 `max|Δ|` 和**相对误差**。
输出张量的绝对幅值可能到几百（DFL 距离），所以**只看绝对差会误判** —— 判据用 `max|Δ| / max|output| < 1e-4`。

### 二级：解码后检测框对拍
两边都跑 `predict(conf=0.001 和 0.25, iou=0.7, max_det=300)`，比：
- 框数是否逐图相等
- 类别序列是否逐图相等
- 框坐标 `max|Δ|`（应 < 0.01 px）
- 置信度 `max|Δ|`（应 < 1e-4）

**这一级才是"能用"的判据**，一级只是排除导出走样。

### 三级：mAP 对齐 ⚠ 有陷阱

直接 `yolo val model=best.pt` vs `yolo val model=best.onnx` 会看到 mAP50-95 有 0.03 左右的差，**这通常不是模型问题**：

> ultralytics 的 `.pt` 验证走**批内 rect 填充**（默认 batch=16），
> 而 `.onnx` 被强制 **batch=1**，两者的 letterbox 填充尺寸不同，
> 导致边缘特征略有差异 → mAP50-95 差 0.03，但 mAP50 完全一致。

**判定方法**：把同一个 `.pt` 也用 `batch=1, rect=False` 验一遍。

```
批处理口径        mAP50     mAP50-95
pt   batch=16    0.9950      0.9665
pt   batch=1     0.9950      0.9361     ← 同一个模型自己就差 0.0304
onnx batch=1     0.9950      0.9361     ← 与 pt(batch=1) 差 0.0000  ⇒ 等价
```

只要 `pt(batch=1) vs onnx` 的差是 0，就是通过的。**别被 batch=16 的那一列吓到。**

## 4. 输入量程必须实测判定

不同导出配置下 ONNX 是否含 `/255` 归一化不一致。**不要凭经验假设**，用同一张图分别喂 0-1 和 0-255，看哪个与 torch 对得上：

```
输入 0-1   -> max|diff| 1.8e-03   ✓
输入 0-255 -> max|diff| 5.4e+02   ✗
=> 期望 0~1 归一化输入
```

`verify_onnx.py` 会自动做这个判定并打印结论。

## 5. 交付物清单

导完至少给用户这些：

1. `best.onnx`（主产物，附 MD5）
2. 一份校验报告 txt：输入输出形状 / opset / 算子统计 / 三级校验数据 / 批处理口径对照
3. 若做了验证集评估，附一份可复用的 `val_*.yaml`（`path` 指到数据集根，`names` 按 id 顺序写全）
4. 下游注意事项（输出布局、后处理步骤、opset、输入量程）

## 6. 下游衔接

- **MaixCAM2 / AX620E**：ONNX → Pulsar2 → `.mud` + `_npu.axmodel` + `_vnpu.axmodel`（见 `maixcam2-obb-int8-convert` skill）
- **MaixCAM / cv181x**：ONNX → TPU-MLIR → `.cvimodel`（见 `maixcam-obb-bf16-convert` skill）
- **类别 id 对齐**：ONNX 的 `names` 顺序就是类别 id 顺序。若端侧 app 有固定的类别语义表，**用类名映射而不是 id 映射**（模型每轮重训 id 顺序都可能变）。注意 `-`/`_` 写法差异要归一化（如 `class_2-top` ↔ `class_2_top`）。

## 7. 常见坑速查

| 症状 | 原因 / 修法 |
|---|---|
| `LetterBox.__init__() got an unexpected keyword 'scaleFill'` | 新版改名叫 `scale_fill` |
| `'half' is deprecated ... Use 'quantize' instead` | 只是警告，可忽略 |
| `DetMetrics has no attribute 'args'` | 用 `r.box.map` / `r.box.map50`，别取 `args` |
| `This model has no one-to-one head; using one-to-many outputs` | 正常提示（非 end2end 头） |
| 导出很慢或卡住 | 加 `device=cpu`；GPU 缺 CUDA 时会回退很慢 |
| 验证时 `AttributeError: module 'dateutil' has no attribute '__version__'` | matplotlib 拉 dateutil，见 §1 坏包修法 |
| `ConnectionError: Download failure for https://ultralytics.com/assets/Arial.ttf`（curl 502） | `ultralytics` 的 `check_det_dataset` **每轮 val 都联网下字体**，网络不通就抛异常，且会**连带丢掉已经跑完的一、二级结果**（报告原本只在函数末尾写盘）。两条修法同时做：① `cp C:\Windows\Fonts\arial.ttf "$APPDATA/Ultralytics/Arial.ttf"` 预置字体；② `verify_onnx.py` 已改为三级包 try/except + 报告照常落盘，三级跑不成不再影响一、二级结论 |

## 8. imgsz 必须从 ckpt 里读，不要凭文件名猜

`probe_pt.py` 能直接打印 `train_args`；**读不到时**用 venv 的 torch 兜底：

```bash
<venv>/Scripts/python.exe -c "
from ultralytics import YOLO
c = YOLO(r'X.pt').ckpt
ta = c.get('train_args') or {}
print({k: ta[k] for k in ('task','data','imgsz','epochs','batch','rect','seed') if k in ta})
"
```

**Ultralytics 平台云端训练产出的 `.pt`，`data` 会是容器内路径**（如 `/datasets/top-392dbccd/data.yaml`），本地不存在 —— 这是正常的，导出不受影响；但要做三级 mAP 就得自己指一个本地带标签的 val 集（类别集必须一致），并在报告里注明「外部验证集，非该模型自带 val」，别把交叉集 mAP 当成模型的官方指标。
