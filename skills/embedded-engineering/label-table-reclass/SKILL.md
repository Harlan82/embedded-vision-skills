---
name: label-table-reclass
description: 嵌入式视觉项目「换模型 → 改类别表」的全链路改造与验收。视觉分选/检测工程（MaixCAM、STM32 主控 + 串口屏）里 class_id 表、形状/颜色类目、ID 段划分需要重排时使用。核心是先查清数据流再动手（class_id 常常不驱动分桶），并按连锁项清单一次改全，最后用 5 项自动验收替代人工核对。当用户说「把 N 类形状改成 M 类」「换模型后类别表要改」「class_id 重排」「ID 段改一下」「适配新 ONNX 的类别」「注释说要同步电控端是真的吗」时使用。
agent_created: true
---

# 类别表改造（换模型 → 改类目 / ID 段重排）

## 为什么需要这个技能

换一个模型（类别数或类目变了）改动**远不止改一行 `SHAPE_ORDER`**。
本技能来自一次真实改造：7 类 → 多类，涉及 5 个文件、17 处，其中**有 2 处不改会在开机 import 期直接 `AssertionError`**，
还有若干处「不报错但静默失效」。同时纠正了一类**广泛存在的错误注释**。

---

## 第 0 步（最重要）：先查数据流，别信注释

**不要照着注释里的「必须同步 XXX」直接动手。** 注释是上一位作者在某个历史时点写的，
架构演进后常常已经失真。

例：某个视觉分选工程里到处写着「改 class_id 必须同步电控端」，
但实际架构是 **分桶在视觉侧算好（`bin_alloc.py`），判定包直接携带 `bin_no`（占原 angle 字段偏移 8-9），
电控只按桶号开桶** —— 电控端根本不维护 `class_id → 桶` 的映射表，
所以改 ID 表**完全不需要动电控**。

**查证方法**（按顺序，成本从低到高）：

```bash
# 1. 找分包/发送函数：协议里到底发了什么字段
grep -rn --include=*.py -E "send_judge|send_pack|struct.pack|bin_no|bucket" .
# 2. 找分桶逻辑在哪一侧实现
grep -rn --include=*.py -E "def assign|def alloc|BIN_|桶" .
# 3. 用「协议文档」定案（它比代码注释权威）
grep -n -E "class_id|bin_no|映射表|落桶" 电控协议.md 对接协议总表.md
```

**判据**：如果协议文档写着「电控**不再需要**维护 X 映射表」「X **仅供显示/记录**」，
那就是**不需要**同步。此时下游真正依赖 `class_id` 的只剩**屏显**。

> 结论要敢下：查明后**顺手把错误注释改对**。留着错误注释，下次（或下一个人）还会踩。

---

## 连锁必改项清单（漏一个就出事）

| # | 位置 | 为什么必须改 | 漏改后果 |
|---|---|---|---|
| 1 | 类目表本身（`SHAPE_ORDER` / `CLASSES`） | 顺序必须与模型 metadata 的 `names` **逐项一致** | 形状与 ID 整体错位 |
| 2 | **段基数**（`UNKNOWN_ID_BASE`） | 通常有 import 期自检 `len(CLASSES)*len(COLORS) > BASE` **或** 段重叠检查 | **开机直接 AssertionError** |
| 3 | **下一段起点**（`NEW_CLASS_ID_BASE`） | Unknown 段变长后会压住它 | 段重叠，动态新类撞进 Unknown 段 |
| 4 | 段划分注释 / 模块 docstring | 写着旧段号，误导后来人 | 静默误导 |
| 5 | 中文名表（`SHAPE_CN`） | 只补新增类，**旧类名不删会留下永不被命中的死键** | 屏/日志显示英文键 |
| 6 | 屏内图标映射（`SHAPE_DRAW` 等） | 键必须覆盖全部新类 | 切不到图（不影响逻辑） |
| 7 | 分组/隔离配置（易混组、家族隔离、VLM 分组等） | 里面常硬编码**旧类名**，改完永远匹配不上 | **规则静默失效**（最难发现） |
| 8 | 硬编码的段号数字（散落在各处注释） | 如 `class_id 0~N 段` | 静默误导 |
| 9 | 屏内**图片资源**（0.png~N.png） | 用 `cid` 当图片编号切图时，编号上限要跟着变 | 只影响显示，不影响分拣 |

**第 2、3 项是最容易漏、后果最重的**，因为它们是**唯一会"当场报错"的两项**，
其余都是静默问题 —— 看到报错反而该庆幸。

**快速估算段位**（V 个类目 × C 个颜色）：

```
彩色段  0 .. V*C-1
Unknown V*C .. V*C+V-1
动态新类 V*C+V 起（取整到好看的数，如 48）
屏图片资源需要 V*C + V 张（= 0.png ~ (V*C+V-1).png）
```

---

## 不要漏：区分「代码里生效」与「只是注释」

改造范围要**区分优先级**，别一次改爆：

- **生效代码**：类目表、段基数、中文名表、图标映射、分组配置 —— 必改
- **注释 / docstring**：改了防止误导 —— 顺手改，但要**单独成批**、改完再跑一次语法检查
- **日志文本**（`logbus.warn("...同步 XXX")`）：属于生效代码，但**只影响可读性**。
  这类文本如果内容是错的（引导别人做多余操作），**要改**。

---

## 验收：5 项自动检查（不靠肉眼）

改造完**必须**跑完这 5 项。全部通过才算完成。

```bash
export PATH="/c/Windows/System32:/c/Windows:/usr/bin:/bin"
PY="<managed python 绝对路径>"
P="<项目目录>"
cd "$P"

# ① 全项目语法（别只检查你改的那几个文件）
for f in *.py; do "$PY" -m py_compile "$f" || echo "FAIL $f"; done

# ② import 自检（很多工程把"段不重叠"写成 import 期断言，这一跑就全暴露）
"$PY" -c "
import sys; sys.path.insert(0, r'$P')
import config, labels
n=len(config.SHAPE_ORDER); m=len(config.COLOR_ORDER)
assert config.UNKNOWN_ID_BASE >= n*m
assert config.NEW_CLASS_ID_BASE >= config.UNKNOWN_ID_BASE+n
ok=0
for c in config.COLOR_ORDER:
    for s in config.SHAPE_ORDER:
        assert labels.shape_of_class_id(labels.COLOR_SHAPE_TO_ID[(c,s)]) == s
        ok+=1
print('往返一致', ok, '组')
print('SHAPE_CN 缺失:', set(config.SHAPE_ORDER)-set(labels.SHAPE_CN))
"
```

```python
# ③ AST 校验「图标映射表 / 分组配置」的键覆盖率（不 import，避免 maix 依赖）
import ast, io, sys
sys.path.insert(0, P); import config
tree = ast.parse(io.open(P+'/oled_panel.py', encoding='utf-8').read())
for node in tree.body:
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in ('SHAPE_DRAW',):
                d = ast.literal_eval(node.value)
                print(t.id, len(d), '缺', set(config.SHAPE_ORDER)-set(d), '多', set(d)-set(config.SHAPE_ORDER))

# ④ dump 开机自检表（工程一般有 dump_table()），肉眼过一次，也是给用户照抄的
# ⑤ 反查「旧类名 / 旧段号」是否还有残留
```

```bash
# ⑤ 残留扫描：旧类名 + 旧段号 + 「必须同步 XXX」类错误陈述
grep -rn --include=*.py -E "旧类名1|旧类名2|0-27|0~27|28-34|28~34|同步XXX" .
# 命中的应只剩"说明性注释"（如「旧段 0-27 已废弃」），不应有生效代码
```

**关键：改注释前先确认这 5 项在哪、能不能跑通** —— 有自动化验收，改注释才敢放心批量做。

---

## 常见坑

1. **`export_icons.py` 之类的悬空引用**：注释里引用的工具脚本可能**根本不存在**。
   动手前 `find` 验证一次，别照着不存在的脚本设计方案。
2. **改名不改内容**：屏的 `.mud` 文件里 `model = xxx.cvimodel` 是相对名，
   改文件名必须同步改 `.mud` 内部字段，否则"找到了 mud、找不到 cvimodel"。
3. **名字撞标准库**：项目里放 `inspect.py` / `types.py` 会让 `import numpy` 等莫名失败。
4. **模型的 `labels` 也要对**：`main.py` 常有开机自检比对
   `set(detector.labels) == set(config.SHAPE_ORDER)`；mud 的 `labels` 行要与类目表**逐项一致**（顺序也建议一致）。
5. **别只改代码不改屏**：屏上图片资源编号 = `class_id`，改完类目表必须重导
   `0.png ~ (V*C+V-1).png`。**但要说清后果只是显示问题**，不影响分拣 —— 免得用户误判成致命故障。
6. **量化后必须重新验收**：INT8 会改变阈值行为；模型换了 + 类别表换了，上机前用真实场景图跑一遍混淆矩阵。

---

## 交付话术模板（给用户说清"哪些真的要动"）

> 1. **代码**（已完成）：改 N 个文件 M 处，全部语法 + import 自检通过。
> 2. **真正需要你动手的只有 1 处**：屏内图片资源重导 `0.png ~ X.png`（只影响显示）。
> 3. **不需要动的**：电控端 —— 分桶在视觉侧（`bin_alloc.py`），判定包带 `bin_no`，
>    `class_id` 仅供屏显/日志。（依据：`电控协议.md` 第 N 行）
> 4. **顺带纠正**：文档里仍写着「★ 请照此改电控端」等过时陈述，列清单待确认。
