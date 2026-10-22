# Embedded Vision Skills

面向 **嵌入式视觉 / MaixCAM 部署 / YOLO 数据工程** 的 Agent Skill 集合。

这 17 个 skill 不是教程摘抄，而是从真实项目里一路踩坑后沉淀下来的操作手册：每条结论都对应一次实测失败，每个参数都标了为什么是这个值。核心目标只有一个——**让下一次遇到同类问题时，不用再重新踩一遍**。

---

## 这是什么

每个 skill 是一个独立目录，内含 `SKILL.md`（Agent 读取的指令手册）和可选的 `scripts/`、`references/`。Agent 在识别到匹配意图时会自动加载对应 skill，按其中的流程、判据、踩坑清单执行。

适用场景集中在四块：

| 领域 | 覆盖内容 |
|---|---|
| **模型转换与量化** | `.pt` → ONNX → MaixHub / TPU-MLIR / Pulsar2，INT8 与 BF16 两条精度路线 |
| **数据集构建** | 相机照片 → YOLO 标注 → 平台可导入 ZIP，含格式互转与数据增强 |
| **嵌入式工程** | MaixCAM 组网、离线验证、类别表改造、相机标定数据分析 |
| **方法论** | YOLO 训练调参的源码级事实、无标签真值下的增量改进闭环 |

---

## 快速开始

### 方式一：安装脚本（推荐）

```bash
git clone https://github.com/Harlan82/embedded-vision-skills.git
cd embedded-vision-skills

# 查看有哪些 skill
python install.py --list

# 安装全部到默认目录（~/.workbuddy/skills）
python install.py

# 安装到指定目录
python install.py --target ~/.claude/skills

# 只装某几个
python install.py --only pt-to-onnx-export,yolo-train-tuning
```

脚本会把嵌套目录**扁平复制**成 `skills/<skill-name>/`，因为多数 skill 加载器要求目录名与 frontmatter 的 `name` 一致。

### 方式二：手动复制

把 `skills/<领域>/<skill-name>/` 整个目录拷贝到你的 skills 目录下，**去掉领域这一层**：

```bash
cp -r skills/model-convert/pt-to-onnx-export ~/.workbuddy/skills/
```

> ⚠️ 直接使用 `skills/model-convert/pt-to-onnx-export` 这样的嵌套路径，在部分加载器下可能识别失败。若你的加载器支持递归扫描（如 Claude Code 的 `skills/` 递归查找），可忽略此提示。

---

## Skill 清单

### 模型转换与量化 · `skills/model-convert/`

| Skill | 解决什么问题 |
|---|---|
| [pt-to-onnx-export](skills/model-convert/pt-to-onnx-export/) | Ultralytics `.pt` 导出 ONNX，并做**零误差级对拍**校验。含 YOLO26 one-to-one / one-to-many 双检测头该取哪一支的判据 |
| [maixhub-yolo-detect-convert](skills/model-convert/maixhub-yolo-detect-convert/) | 导出 MaixHub 在线转换能收的 ONNX；专治 `Detect profile does not match this ONNX` |
| [maixcam-obb-bf16-convert](skills/model-convert/maixcam-obb-bf16-convert/) | 【一代 cv181x】OBB 旋转框模型 **BF16** 高精度路线。含 `dual_buff` 必关的 ION 内存实测分析 |
| [maixcam2-obb-int8-convert](skills/model-convert/maixcam2-obb-int8-convert/) | 【二代 AX620E】Pulsar2 工具链 **INT8** 量化，产出 `.mud` + `_npu.axmodel` + `_vnpu.axmodel` 双模型 |
| [quant-calib-subset-select](skills/model-convert/quant-calib-subset-select/) | 挑选 INT8/PTQ 校准子集，强化**远端小目标召回**与密集场景，避免量化后小目标掉点 |

### 数据集构建 · `skills/dataset-build/`

| Skill | 解决什么问题 |
|---|---|
| [shape-dataset-pipeline](skills/dataset-build/shape-dataset-pipeline/) | 相机照片 → YOLO 数据集全流程（标定/标注/校验/合并/灰度增广/打包），**保留人工诊断分支** |
| [shape-dataset-pipeline-local](skills/dataset-build/shape-dataset-pipeline-local/) | 同上但全自动、零人工判断，一条命令出 ZIP，4B 级本地模型也能独立跑 |
| [center-object-to-yolo-zip](skills/dataset-build/center-object-to-yolo-zip/) | 没有 SAM 权重时，用 HSV 前景分割 + 几何判据标注「单目标居中」照片 |
| [dataset-to-ultralytics-ndjson](skills/dataset-build/dataset-to-ultralytics-ndjson/) | 格式互转枢纽：YOLO/OBB 目录 ↔ NDJSON ↔ Pascal VOC，含按签名 CDN 下载图片 |
| [bbox-attr-sampling](skills/dataset-build/bbox-attr-sampling/) | bbox 内取色判属性的采样规则。镂空物体中心采样必错，改采边缘更错——正解是用像素自身属性识别背景 |
| [geometric-recolor](skills/dataset-build/geometric-recolor/) | 中性灰背景几何体批量换色，用于视觉模型训练数据增强 |

### 嵌入式工程 · `skills/embedded-engineering/`

| Skill | 解决什么问题 |
|---|---|
| [maixcam-wifi-fixed-ip](skills/embedded-engineering/maixcam-wifi-fixed-ip/) | 把 MaixCAM 无线地址固定成指定 IP，每次连热点主动请求。含 Buildroot 网络配置与 Windows 热点 ICS 双份参考 |
| [maixcam-vision-offline-verify](skills/embedded-engineering/maixcam-vision-offline-verify/) | 没有板子时对 MaixPy 工程做离线冒烟验证：判定依赖边界、注入假输入驱动状态机、解码串口帧断言 |
| [label-table-reclass](skills/embedded-engineering/label-table-reclass/) | 换模型 → 改类别表（shape/color、ID 段划分）的全链路改造与 5 项自动验收 |
| [redpoint-grid-checker](skills/embedded-engineering/redpoint-grid-checker/) | 红点检测日志 → 点云可视化 → 距离聚类去重 → 双线性最小二乘拟合 N×M 网格 → 计算物理点最优坐标 |

### 方法论 · `skills/methodology/`

| Skill | 解决什么问题 |
|---|---|
| [yolo-train-tuning](skills/methodology/yolo-train-tuning/) | YOLO 训练调参的**源码级事实**：batch 上限、nbs 梯度累积、早停与 cos_lr 耦合、负样本为何让 mAP 下降 |
| [vision-free-truth-incremental](skills/methodology/vision-free-truth-incremental/) | 样本量只有 10¹~10²、又没有标签真值时，如何挖掘链路里的「免费真值」设计增量改进闭环 |

---

## 目录结构

```
embedded-vision-skills/
├── README.md
├── README.en.md
├── LICENSE
├── install.py
├── .gitignore
└── skills/
    ├── model-convert/          # 模型转换与量化
    ├── dataset-build/          # 数据集构建与增强
    ├── embedded-engineering/   # 设备与工程
    └── methodology/            # 训练与方法论
```

---

## 使用约定

**路径占位符。** 所有脚本与文档里的本机路径已脱敏为占位符，请按自己环境替换：

| 占位符 | 含义 |
|---|---|
| `<converter_root>` | 模型转换器平台根目录 |
| `<venv>` | 你的 Python 虚拟环境目录 |
| `./work/` | 模型文件、数据集、产物的临时工作目录 |
| `%LOCALAPPDATA%` | Windows 用户本地应用数据目录 |

**平台背景。** 内容以 **Windows + Bash 沙箱** 为主要环境记录，部分踩坑（Docker Desktop 路径、盘符路径、PowerShell 编码）带明显 Windows 印记。Linux/macOS 下操作逻辑通用，路径与命令需自行换算。

**版本敏感。** 涉及具体版本号的结论（Ultralytics 8.4.x、Pulsar2 6.0、TPU-MLIR、onnxruntime 1.28）标注了实测版本。上游升级后部分行为可能变化，脚本中已尽量给出**判据**而非硬编码假设。

**不是万能模板。** 这些 skill 高度贴合「固定机位 + 白底托盘 + 单个工件」的工业分选场景。换到通用目标检测场景，阈值与几何判据需要重新标定——但踩坑清单和方法论仍然适用。

---

## 来源与免责

内容来自个人项目的实战记录，随项目演进持续修订。结论均标注实测日期与版本，但不保证在你的硬件、镜像版本、数据集下完全复现。**关键决策请以自己的实测为准。**

## License

[MIT](LICENSE)
