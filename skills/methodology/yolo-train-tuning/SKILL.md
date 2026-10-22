---
name: yolo-train-tuning
description: Ultralytics YOLO 训练调参的源码级事实与安全边界。调整 batch / epochs / patience、给数据集加背景图（负样本）、理解 nbs 梯度累积、早停与 cos_lr 的耦合、验证集含负样本时 mAP 为何下降时使用。当用户问「批次能加多大」「加大 batch 会不会影响训练效果」「早停怎么调才不误停」「补了背景图 mAP 反而掉了」「负样本放验证集对不对」「nbs 是什么」「optimizer auto 到底用了哪个」时使用。
agent_created: true
---

# Ultralytics YOLO 训练调参：源码级事实与安全边界

所有结论来自 **ultralytics 8.4.166** 源码（`engine/trainer.py`、`models/yolo/detect/val.py`）。
行号可能随版本微移，但表达式稳定 —— **用前 grep 复核一次**（见文末）。

---

## 事实 1：batch 可以放心加大 —— 但有"只能取特定值"的约束

```python
# engine/trainer.py:316
self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
# :317
weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
# :319
iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
# :323  build_optimizer(..., lr=self.args.lr0, ...)   ← lr0 不随 batch 缩放
```

**推导**：`accumulate ≈ nbs / batch_size` ⇒ `batch_size × accumulate ≈ nbs` ⇒
`weight_decay` 系数 ≈ 1.0 ⇒ **有效 batch、学习率、weight_decay、iterations 全部不变**。

| batch | accumulate | 有效 batch | wd 系数 | 判定 |
|---|---|---|---|---|
| 8 / 16 / 32 / **64** | 8 / 4 / 2 / 1 | 64 | 1.0 | ✅ **训练动力学完全不变** |
| 24 | 3 | 72 | 1.125 | ❌ |
| 48 | 1 | 48 | 0.75 | ❌ |
| 128 | 1 | 128 | 2.0 | ❌ iterations 还减半 |

**结论：只有能整除 `nbs` 的 batch 才是"免费加速"（默认 nbs=64 → 用 16/32/64）。**
加大 batch 的唯一后果是**更快 + 更吃显存**，效果不变。

**反向用法**：想真正改变训练动力学（例如显存不足必须用小 batch），
就**保持 batch 不变而调 `nbs`** —— 那才是有效 batch 的开关。

---

## 事实 2：验证集里的负样本**有效**，且会让 mAP 下降（这是对的）

```python
# models/yolo/detect/val.py:387
if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
    return {"tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)}
```

无标注图（背景图）⇒ 返回**全 False 的 tp** ⇒ 模型在该图输出的**每一个框都算 FP** ⇒
拉低 precision ⇒ **拉低 mAP**。

**必须向用户说清的两点**：

1. **mAP 下降是预期行为，不是模型变差** —— 这是指标开始"惩罚误检"了，正是加负样本的目的。
2. **不能拿新旧 mAP 直接对比**：验证集内容变了（多了背景图），分母不同。
   要做对比，必须**在同一份验证集**上跑新旧模型。

**比例协调**：训练集与验证集的负样本占比应接近
（例：train 100/4539 = 2.2%，val 10/540 = 1.9% ✓）。

---

## 事实 3：`optimizer: auto` 的真实解析（**不是 SGD**）

```python
# engine/trainer.py:1172
name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
```

- 新版本 auto → **MuSGD**（Muon+SGD 混合），不是旧版的 SGD
- `iterations` 判据依赖 epoch 数 ⇒ **改 epochs 会让 auto 静默换优化器**，两次实验不可比
- ⇒ **凡是做对比实验，一律显式写 `optimizer: SGD`**（`nesterov=True`）

---

## 事实 4：`patience` 与 `cos_lr` 强耦合

```python
# engine/trainer.py:266
self.lf = one_cycle(1, self.args.lrf, self.epochs)   # 余弦按 epochs 铺满
```

早停在第 k 轮触发时，LR 只走到余弦的 `k/epochs`：

| patience | 约在第几轮停 | LR 已衰减到 | 含义 |
|---|---|---|---|
| 太小（<30） | 很早 | ~初始值 | 平台期误停风险高 |
| 30~40 | ~70~90 | ~15% | **饱和任务（mAP 早到顶）的最优区** |
| 80 | ~110+ | ~18% | 慢，但稳 |
| >150 | 基本不停 | 走完余弦 | 才有真正的"低 LR 精修" |

**关键洞察**：`epochs` 设 450 还是 600，在"早停会先触发"的任务上**几乎无区别**
（都到不了）。`epochs` 真正的唯一作用是**决定 cos_lr 的衰减速度**。

⇒ 想要低 LR 精修就必须**让训练跑满**（大 patience 或不早停）；
想要快就**接受没有尾部精修**。两者不可兼得，按任务是否饱和来选。

---

## 事实 5：背景图（负样本）的两个折扣

1. **`mosaic` 会稀释负样本**：4 图拼 1 图时，负样本只占画面 1/4，周围是别图内容
   ⇒ 有效强度打折。**但 `close_mosaic: N` 的最后 N 轮是全额定效**。
2. **增强参数会作用于负样本**：`hsv_*` / `scale` / `degrees` 等同样施加给负样本
   ⇒ 其实**增加**了负样本多样性（好事）；但 `flipud` 若为 0.5，会造出物理上不存在的姿态。

**因此**：如果补了一批负样本后误检仍在，
**优先补"困难负样本"（那些仍然被误检的形态）**，而不是无脑加量。

---

## 判读训练结果：别盯混淆矩阵，看 F1 / PR 曲线

**共同的坑**：混淆矩阵默认用**极低阈值**画 ——
val 时 `args.conf` 若为 None 会被设成 **0.001**（`engine/validator.py:143`），
而混淆矩阵用的就是这个值（`val.py:53-54`）。它会把模型输出的**所有候选框**都统计进去，
其中大量是低置信度的"顺手报的候选"。
⇒ 矩阵看起来"对角线极弱、background 列巨大、跨类到处都是"，**但这不代表模型不行**。
（平台 UI 通常**不提供**调这个阈值的入口，所以你永远看不到"干净版"。）

### 判据：三张图，优先级从高到低

| 图 | 看什么 | 判读标准 |
|---|---|---|
| **`F1_curve.png`** | 横轴 conf、纵轴 F1 的**峰值位置与高度** | 峰值 **≥0.9 且落在 0.35~0.6** ⇒ **模型健康**，把部署 conf 设成峰值即可；峰值 <0.7 或贴最左（0.05）⇒ 模型真有问题，才该动数据 |
| **`PR_curve.png`** | 各类曲线是否贴右上角 | `recall=0.95` 时 `precision` 仍 >0.9 ⇒ 该类良好 |
| `confusion_matrix*.png` | 分类**错在哪**（不是错多少） | **必须指定 conf 重画才有意义**：`yolo val conf=<峰值> plots=True` |

训练输出里 `F1_curve.png` / `PR_curve.png` 是**自动生成**的，不需要额外跑命令 —— 
**先看这两张，再决定要不要看混淆矩阵。**

### 一条不用跑任何东西的自检逻辑

> 若报告的 `mAP50` 很高（≥0.95）而混淆矩阵"没法看"，**大概率是阈值口径问题**：
> mAP50 是按全谱 PR 曲线算的，**若模型在中高置信区大量误检，mAP50 不可能高**。
> ⇒ **高 mAP50 本身就证明了"高置信区是干净的"**，低置信区的乱框提高阈值即可滤掉。

**配套的数字核对**：未归一化矩阵里，**每一列之和 = 该类真实目标数**
（因为每个 GT 要么被正确匹配进对角、要么进 background 行）。
用它可反推"某类到底有多少个、判对了几个"。

### 更便宜的止血（不改模型、不重训）

目标数恒为 1 的场景（如单物体分拣），设 **`max_det=1`** —— 只保留最高分框：

- 从**源头**消掉多报的框，对召回**零影响**（真目标必是最高分）
- 比在业务侧"只取 1 个候选"更彻底：业务侧拒收时，多余的框已经**触发了"多目标"判定**，
  可能让流程卡住（等超时 → 发清扫包）

---

## 安全调整边界表（用户说"不要动我的准确性和召回"时）

| 参数 | 可否动 | 说明 |
|---|---|---|
| `batch` | ✅ 可动 | 但**只能取能整除 nbs 的值**（16/32/64），效果不变 |
| `epochs` | ✅ 可动 | 实际只是改 cos_lr 的衰减速度 |
| `patience` | ✅ 可动 | 改早停快慢；代价是放弃余弦尾部 |
| `save_period` | ✅ 可动 | 只影响存权重频率 |
| `imgsz` | ❌ 别动 | 直接改任务难度；且必须与部署分辨率一致 |
| `lr0/momentum/weight_decay` | ❌ 别动 | 与 batch 有隐式耦合 |
| `box/cls/dfl` | ❌ 别动 | 改召回/精度的直接旋钮 |
| 全部增强参数 | ❌ 别动 | 直接改变数据分布 |

---

## 用前复核（版本漂移时）

```bash
grep -n "self.accumulate\|weight_decay = self.args.weight_decay\|iterations = math.ceil" <ultralytics>/engine/trainer.py
grep -n "MuSGD\|AdamW" <ultralytics>/engine/trainer.py | head
grep -n -A2 "def _process_batch" <ultralytics>/models/yolo/detect/val.py
grep -n "def _process_batch" -A 14 <ultralytics>/models/yolo/obb/val.py   # OBB 任务在 obb/val.py
```

> ⚠️ **OBB 任务**的 `_process_batch` 在 `models/yolo/obb/val.py`，逻辑同构（无标注图 → 全 FP），
> 但混合框的匹配方式不同，别直接套 detect 的行号。

---

## 给用户的说明模板

> 只动 N 个参数，其余一字不动：
> `batch 16→32`（**效果不变**：源码里 `accumulate` 会自动补偿，有效 batch 恒为 64）
> `epochs 600→450`（实际只影响 cos_lr 衰减速度）
> `patience 80→35`（早停提前；代价是放弃低 LR 精修——你这任务已饱和，没损失）
>
> 另外：加了负样本后 **mAP 会下降，这是对的** ——
> 验证时无标注图的预测全部计入假阳性（`val.py` 里 tp 全 False），正是在惩罚误检。
> **别拿新旧 mAP 直接比**，验证集内容已经变了。
