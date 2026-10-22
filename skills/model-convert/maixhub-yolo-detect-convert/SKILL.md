---
name: maixhub-yolo-detect-convert
description: 把训练好的 YOLO 检测模型（YOLO26 / YOLO11 / YOLOv8 / YOLOv5u）导出成 MaixHub 在线转换能接受的 ONNX，并诊断「Detect profile does not match this ONNX」这类报错。核心是 YOLO26 的 one-to-one / one-to-many 双检测头——导出时 nms 参数决定用哪一支，而 MaixHub 的 yolo26 profile 只认 one-to-one。当用户提到「maixhub 转换失败」「Detect profile does not match」「one2one_cv2 缺失」「MaixCAM 模型转换」「.mud/.cvimodel 转换」「YOLO26 导出 ONNX」「ONNX 传不上去」「板子上跑不了」「量化后精度掉了」时使用。
agent_created: true
---

# YOLO 检测模型 → MaixHub 在线转换

## 0. 一句话

MaixHub 的 `yolo26` profile **只认 one-to-one 检测头**（`one2one_cv2.*` / `one2one_cv3.*`）。
而 YOLO26 导出时走哪一支，由 **`nms` 参数**决定 —— **导出必须显式加 `nms=False`**。

## 1. 报错长什么样

```
maixhub_utils.exception.Param_Exception: yolo26 Detect profile does not match this ONNX;
missing ['/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0', ... 共 6 个],
available tail ['/model.8/...', '/model.9/...']
```

`available tail` 是**按名字字典序**排的最后若干节点（字典序上 `model.9` > `model.23`），
**不代表模型真的只到第 9 层**，不要被它误导。

## 2. 三步确诊（不需要跑转换）

ONNX 的节点名是明文 protobuf，**直接在文件字节里搜字符串**即可，不必装 onnx 库：

```python
import re
s = open("model.onnx", "rb").read().decode("latin-1")
print("one2one count:", s.count("one2one"))                                  # 期望 > 0
print("model.N tail :", sorted(set(int(x) for x in re.findall(r"/model\.(\d+)/", s)))[-3:])
print("head nodes   :", sorted(set(re.findall(r"/model\.\d+/(?:one2one_)?cv[23]\.\d", s))))
print("meta         :", re.findall(r"nms.{1,3}(\w+)", s)[:2], re.findall(r"end2end.{1,3}(\w+)", s)[:2])
```

判读：
- `one2one count == 0` 且 metadata `nms: None` / `end2end: False` → **就是这个坑**
- 有 `one2one_*` → 导出没问题，去查别的原因（输入形状、opset、类别数、是否分类/分割模型）

## 3. 根因（ultralytics 源码，已核实 8.4.166）

| 位置 | 代码 | 含义 |
|---|---|---|
| `engine/exporter.py:697` | `model.end2end = self.args.nms is False` | `nms=False` ⇒ end2end=True；`None`/`True` ⇒ False |
| `nn/modules/head.py:141` | `if end2end: self.one2one_cv2 = copy.deepcopy(self.cv2)` | one2one **只在 head 构造时**创建，是 one2many 的深拷贝（结构完全相同） |
| `head.py:160-167` | `end2end` setter 只设 `self._end2end`，缺层时仅打 warning | **事后改 `model.end2end = True` 无效** |
| `head.py:201` | `preds["one2one" if self.end2end else "one2many"]` | end2end=False 时 ONNX 的**死代码消除会把整条 one2one 子图删掉** |

实测（用官方 yaml 构建模型，无需下载预训练权重）：
```
head: Detect   has one2one_cv2: True        # YOLO26 的 head 默认就带 one2one
one2one_cv2[*] children: ['0','1','2']      # 正是 maixhub 要找的 .0/.1/.2
end2end=False -> output (1, 4+nc, N)        # one2many raw head
end2end=True  -> output (1, 300, 6)         # one2one + postprocess
```

## 4. 修复

```bash
# 关键：nms=False
yolo export model=best.pt format=onnx imgsz=<H,W> opset=17 simplify=True dynamic=False nms=False
```

Python 等价：
```python
from ultralytics import YOLO
YOLO("best.pt").export(format="onnx", imgsz=480, opset=17, simplify=True, dynamic=False, nms=False)
```

导出后自检（必做）：
- `one2one count > 0`
- 存在 `/model.<N>/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0` 形态的节点
  （maixhub 找的是**张量名**，带 `_output_0` 后缀）
- 导出日志**不能**出现 `This model has no one-to-one head` —— 出现了说明该 .pt 根本没有 one2one 层，只能重训

## 5. Sipeed 官方口径 vs 本机版本

官方 wiki（`wiki.sipeed.com/maixpy` → ai_model_converter/online_converter）要求：

- **`ultralytics==8.4.104`**，`opset=17`，`dynamic=False`，`simplify=True`
- 尺寸顺序是 **「高,宽」**：MaixCAM / CAM Pro 示例 `imgsz=224,320`；MaixCAM2 示例 `imgsz=240,320`
- MaixCAM2 小目标可用 `imgsz=480,640`，代价是更慢
- 网页只支持 **检测**：YOLO26 / YOLO11 / YOLOv8 / YOLOv5u；分类 / 姿态 / 分割 / 旋转框要走手动转换入口
- 训练分辨率与导出分辨率应当一致，否则小目标精度会掉

⚠️ **官方命令里没有 `nms` 参数**。若本机版本比 8.4.104 新（例如 8.4.166），`nms=None` 会让导出丢掉 one2one。
→ 要么降到 `8.4.104`，要么在官方命令上**补 `nms=False`**。

## 6. 环境坑

- 导出 ONNX 需要 `import onnx` 可用。Windows 上 `pip install onnx` 后可能报
  `ImportError: DLL load failed while importing onnx_cpp2py_export`（缺 VC++ 运行时）→ 先修运行时或改用 Linux/容器。
- 若环境挂了本地代理，pip 需清变量：`env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY python -m pip install ...`
- GitHub 下载预训练权重常因证书问题失败（`CRYPT_E_NO_REVOCATION_CHECK`）；
  **验证导出分支不需要预训练权重** —— `YOLO("yolo26s.yaml")` 用官方 yaml 构建即可。

## 7. 转换 ≠ 验收

转换只是把 ONNX 变成 `.cvimodel` / `.mud`。**INT8 量化会改变阈值行为**，
部署前必须用真实场景图重跑验收，尤其低对比度物体、以及**镂空件的中心孔**（判别特征可能被量化糊掉）。
