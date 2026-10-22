---
name: shape-dataset-pipeline-local
description: 本地离线版「相机照片 → YOLO 标注数据集 → 平台可导入 ZIP」全流程，一条命令跑完
  （标注 + 合并 + 灰度增广 + 平台规范打包 + 自检），阈值全部自动标定，**不需要任何人工判断**，
  因此 4B 级本地小模型（含 Ollama 本地模型）也能独立执行。用于固定机位/白底托盘/单个工件的
  工业分选数据。当用户说「本地模型跑标注」「没有网络也要做数据集」「一条命令出 zip」
  「批量标注并打包」时使用。需要人工判断与诊断的版本请用 shape-dataset-pipeline。
agent_created: true
disable-model-invocation: false
---

# Shape Dataset Pipeline · 本地版（一条命令，4B 模型可执行）

## 你要做的只有三件事

```bash
# 第 1 步（可选，10 秒）：环境自检
python scripts/doctor.py --src "./work/batch-a" --zip-out "./work/batch-a.zip"

# 第 2 步：跑全流程（标注 -> 打包 -> 自检）
python scripts/pipeline.py --src "./work/batch-a" --cls class_2 --zip "./work/batch-a.zip"

# 多个数据集合并成一个包，并加 20% 灰度增广
python scripts/pipeline.py --jobs jobs.json --zip "./work/shapes-7cls.zip" --gray 0.20
```

Windows 上也可以直接：

```bat
run.bat --src "./work/batch-a" --cls class_2 --zip "./work/batch-a.zip"
```

**第 3 步：看最后一行输出。**
`RESULT: OK` = 成功（退出码 0）；`RESULT: FAIL:<原因>` = 失败（退出码 1）。
不要去读中间那一大堆统计，**只看这一行**；失败时按下面的表处理。

## 输入 / 输出

输入（你只需要提供）：

| 参数 | 含义 |
|---|---|
| `--src` | 图片文件夹（`jpg/jpeg/png/bmp/webp`） |
| `--cls` | 类别名，建议拼音小写，如 `class_2` / `class_7` / `class_9` |
| `--zip` | 输出包路径 |
| `--jobs` | 多数据集时改用它，写一个 json 文件（见 `jobs.example.json`） |
| `--gray` | 可选，灰度增广比例，如 `0.20` |

`jobs.json` —— **列表顺序就是类别 id 0,1,2…**：

```json
[{"src": "./work/batch-a", "cls": "class_2"},
 {"src": "./work/batch-b",     "cls": "class_7"}]
```

输出（自动写好，不用你管）：

```
<zip>                        平台可导入数据集
<zip名>_report.txt / .json   统计报告
<zip名>_records.json         逐图清单（clip / area / 阈值路径）
<zip名>_audit/audit.png      抽样标注图 —— 人看一眼即可
<zip名>_audit/flagged.png    clip 最高 / 面积最大的帧
<zip名>_RUN_SUMMARY.txt      本次运行汇总
```

## 哪些事**不用**你做（已自动化，但你得知道它做了）

| 自动化项 | 说明 |
|---|---|
| 阈值标定 | 按每帧**目标自身的饱和度中位数**定阈值（`0.5×S0`，夹在 60~130），高于阴影、低于本体最暗 5% |
| 阴影 / 倒影排除 | 阈值天然高于阴影（实测阴影 S 23~48） |
| 收紧后验收 | 计算 `clip`（目标完整掩膜落在框外的比例），>0.10 就放弃收紧 → **不会咬进目标** |
| 手指排除 | 候选级 `S ≥ 125`（实测手指 S 90~118） |
| 粘连帧挽救 | 回退 1：阈值 120→135→150 捞出目标本体；回退 2：55 救暗色件 |
| 固定杂物抑制 | 统计重复出现在同一位置、且远离中心的框 → 判定为设备边框/线缆并自动排除 |
| 类别 id 重排 | 按 jobs 顺序 0..N-1（否则多个单类包合并会退化成单类） |
| 文件名清洗 | 去掉空格/括号等非法字符，加 `<cls>__` 前缀防撞名 |
| 平台规范 | `train/images`+`train/labels`+`val/...`，`data.yaml` 不写 `path`，纯 LF |
| 自检 | CRC / 结构 / 配对 / 标签 5 元组 / 坐标开区间 / 文件名 / ultralytics `check_det_dataset` |

## 出错怎么办（只看 RESULT 行的原因）

| RESULT 行 | 原因 | 你要做的 |
|---|---|---|
| `FAIL:用法错误…` | 参数没给全 | 必须有 `--zip`，以及 `--jobs` 或 `--src`+`--cls` |
| `FAIL:标注/打包异常 ModuleNotFoundError` | 缺依赖 | `python -m pip install -r requirements.txt` |
| `FAIL:自检未通过` + `data.yaml` 相关 | yaml 被写坏 | 重跑一次；若仍失败，把 `_report.txt` 与问题清单发出来 |
| `失败: 文件名含非法字符` | 源文件名字有空格/括号 | 脚本本应自动清洗，若仍报错说明有异常字符，请贴出文件名 |
| `images/labels 不配对` | 输出目录被别的程序占用/残留 | 换一个**新的** `--zip` 路径重跑 |
| `ultralytics ... 失败` | 目录结构或 yaml 问题 | 贴出该行完整文字 |
| `PermissionError`（写 zip 时） | zip 正被解压软件或平台打开 | 关掉占用它的窗口，或换路径 |
| 未命中若干张 | 这些帧画面里**没有任何目标**（空托盘） | 正常，不用处理；需要当负样本再单独说 |
| `回退帧比例 > 1%` 那一行 | 这类帧的目标偏淡（绿/浅蓝），饱和中位 102~124 卡在候选阈值下 | 目标颜色确实偏淡时，加 `--cand-s-min 110` 重跑；否则不用管 |

## 已知边界（不是缺陷，但你要知道）

1. **低饱和目标会走回退路径**：候选级下限默认 `125`（用来剔手指，实测手指 S 90~118）。
   若目标的饱和中位数只有 102~124（淡绿 / 浅蓝件），严格层会拒它，转而用高阈值 120
   从粘连块里捞目标 —— 框会**略紧一点**（实测 7 个数据集四千余张里有 13 张，占 0.31%）。
   处理：看汇总里的 `回退帧比例`，超过 1% 且目标确实偏淡时加 `--cand-s-min 110`。
2. **3/4203 未命中是空托盘**：画面里没有任何工件，无目标可标，属正确排除。
3. 交付物里的 `A. 正常路径裁剪率全部 ≤0.10` 才是"没有咬进目标"的判据；
   `B. 回退帧` 的 clip 偏高是**预期**（框故意不含手指），不必惊慌。

## 快速试跑（先用小样本确认命令通不通）

```bash
python scripts/pipeline.py --src "./work/batch-a" --cls class_2 --zip "./work/test.zip" --limit 40
```
`--limit 40` 只取每个数据集前 40 张（复制到独立目录，不动原图）。

## 可选：让本地模型复核可疑帧

```bash
python scripts/pipeline.py --jobs jobs.json --zip "./work/out.zip" --vlm qwen2.5vl:3b
```
- 只对 `clip > 0.15` 的少数帧调用本地 Ollama，逐张问「红框是否包住工件」。
- **Ollama 没启动 / 模型不存在时自动跳过**，不影响主流程（会打印跳过的原因）。
- 想单独试：`python scripts/ask_vlm.py --img a.jpg b.jpg --model qwen2.5vl:3b`

## 禁止事项（会直接把数据做坏）

1. **不要自己改阈值**。默认参数是在 8 个真实数据集上验证过的；除非 `probe` 报告显示异常。
   需要诊断时用正常版 `shape-dataset-pipeline`（有 `probe.py` 与决策表）。
2. **不要手工改 `data.yaml`**，尤其不要加 `path:`（会让平台换机器后找不到图）。
3. **不要删中间文件 / 不要复用同名工作目录**（会把上次残留压进包，症状是 images 与 labels 数量不等）。
4. **不要自己算坐标**。标注框由脚本给出，坐标已夹在 (0,1) 开区间内。
5. **不要跳过审计图**。交付前必须有人看一眼 `<zip名>_audit/audit.png`：
   框应贴住工件、不含阴影、不含手指、不含线材。
