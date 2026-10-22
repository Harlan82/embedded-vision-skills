---
name: maixcam-vision-offline-verify
description: 在没有 MaixCAM 硬件的情况下，对 MaixPy 视觉工程（主摄/顶视/PC 端多模块工程）做离线冒烟验证与重构后回归——判断哪些模块不依赖 maix、如何注入假输入驱动状态机、如何解码串口协议帧断言行为。当用户要「重构 MaixCAM/嵌入式视觉工程后验证」「没有板子先跑通逻辑」「离线自测状态机/协议/ID 表」时使用。
agent_created: true
---

# MaixCAM 视觉工程：离线冒烟验证与重构后回归

## 何时用

- 重构了 MaixPy 多模块工程（`main.py` + 若干业务模块），板子不在手边，想先确认逻辑没坏。
- 新增/重排了 class_id 段表、串口协议、状态机，需要回归验证。
- 用户说「先跑通再上机」「离线自测」「不接硬件验证」。

## 核心思路

MaixPy 工程的**可测面**取决于依赖切分。先做一次依赖盘点，再决定能测多少。

### 步骤 1：找出"不依赖 maix"的模块

```bash
grep -l "from maix\|import maix" <project_dir>/*.py      # 只有这些需要硬件
```

典型结论（本类工程常见切分）：
- `main.py` 唯一依赖 `maix`（相机/显示/NPU/串口/pinmap）；
- `pipeline.py` / `dispatcher.py` / `protocol.py` / `labels.py` / `color_hsv.py` /
  `config.py` / 各 `*_link.py` **只依赖 config / logbus / numpy / requests** → 全部可离线跑。

### 步骤 2：准备隔离解释器

用 managed python 建 venv，**不要污染用户环境**：

```bash
PY="C:/Users/<user>/.workbuddy/binaries/python/versions/3.13.12/python.exe"
ENV="C:/Users/<user>/.workbuddy/binaries/python/envs/default"
[ -x "$ENV/Scripts/python.exe" ] || "$PY" -m venv "$ENV"
"$ENV/Scripts/python.exe" -m pip install --quiet numpy requests
```

`main.py` 本身永远 `py_compile` 检查即可（导入会因缺 maix 失败，这是预期的）：

```bash
"$ENV/Scripts/python.exe" -m py_compile <project_dir>/*.py && echo "语法 OK"
```

### 步骤 3：写冒烟脚本（放在工作区，不要放进被重构的工程目录）

脚本里 `sys.path.insert(0, r"<project_dir>")` 后 import 各业务模块。要点：

1. **纯函数/表先测**：ID 正反查往返、分组/门控矩阵逐项断言。
2. **注入假输入代替硬件**：
   - UDP 链路：直接设 `link._objects = [...]`、`link._received_at = now`、
     `link._calibrated = True`，绕过 socket 线程；
   - 图像：`crop_fn` 直接 `lambda box: (b"\x00"*(w*h*3), w, h)`；
   - 网络：把 `vlm_service.submit_shape` 换成 `lambda *a, **k: (captured.append(a), True)[-1]`，
     再用回调 `pipeline._on_shape_done(*captured[-1][:5], "目标形状", conf)` 模拟回包。
3. **拦发包看真相**：把唯一的写口换掉，同时**解码**校验协议字段：
   ```python
   wire = []
   dispatcher._write = lambda pkt, text, idx: (wire.append(pkt), True)[-1]
   d = protocol.decode(wire[-1])
   assert d["class_id"] == 期望 and d["flags"] & 1
   ```
   只断言"发了几包"远远不够——要看**包内容**。
4. **状态机要推帧**：盒内/静止类门控是"连续 N 帧"，脚本里要循环调用 N 次；
   注意**时间戳**也由脚本给（`now` 参数），否则"数据过期/超时"分支会被误触发。

### 步骤 4：常见坑（本类工程反复出现）

| 现象 | 原因 |
|---|---|
| 状态机停在第一步 | "连续 N 帧"的**首帧要算作第 1 帧**，否则永远差一帧 → 循环次数要 N 次而不是 N+1 次 |
| 发包为空 | 上游链路 `fresh()` 判"数据过期" → 假数据的时间戳要跟着推进 |
| 断言莫名失败 | 门控矩阵里"同族互纠放行 / 跨族硬隔离"是两组不同的规则，别把同族当跨族 |
| 配置项形同虚设 | 先跑一次"config 里定义但无人引用的参数"扫描（见下），剔除死参数 |
| 上机才炸 AttributeError | 派生的配置名（如 `USE_GRAYSCALE_* = (X == "gray")`）被外部覆盖绕过 → 保留文件末尾"派生项归一"段 |

### 步骤 5：死参数扫描（重构后必做）

```python
import io, os, re, glob
cfg_path = "<dir>/config.py"
cfg = io.open(cfg_path, encoding="utf-8").read()
names = set(re.findall(r"^([A-Z][A-Z0-9_]{2,})\s*=", cfg, re.M))
used = set()
for f in glob.glob(os.path.join(os.path.dirname(cfg_path), "*.py")):
    if os.path.abspath(f) == os.path.abspath(cfg_path):
        continue
    s = io.open(f, encoding="utf-8").read()
    used |= {n for n in names if re.search(r"\b" + n + r"\b", s)}
print(sorted(names - used))
```

`config.py` 自身内部用到的（派生式、CFG_DEFAULTS 里的键）会误报，需人工排除。

## 重构这类工程的通用做法

1. **先盘点再动手**：读全量基线，列出"实体 / 链接 / 功能"三维度，明确每个模块的职责边界。
2. **把歧义摆到桌面上问**，不要猜：形状集合、class_id 段是否可重排、协议字段语义、
   模型文件名——这四项任一猜错都要重做。
3. **被移除的功能统一 `mv` 到 `_removed/`**（不 `rm`）：工作树干净且可回滚，
   确认跑通后整目录删除即可。同时把"删了什么、为什么"写进一份 `重构说明.md`。
4. **协议变更是对外契约**：class_id 段表、flags 位、回执 status 语义必须在文档里给完整表格，
   并让开机日志把表打印出来，便于对端照抄核对。
5. **每个 config 只保留真正被读的参数**；被取代的旧参数连注释一起删，避免"改了不生效"。

## config 归属约定：只有参数，没有逻辑

`config.py` 是"人会改的参数单一事实源"。凡带逻辑/副作用的东西都不该在里面，
否则用户改参数时会看到一堆与自己无关的代码，且容易改错。

| 内容 | 该放哪 | 判据 |
|---|---|---|
| 纯参数（阈值、开关、尺寸、ID 段） | `config.py` | 值是常量、无副作用 |
| 板型探测 / 引脚表查询（要问硬件） | `board.py` 之类 | 有逻辑、要 try/except |
| 开机动作（读 tune.json 覆盖参数、初始化） | `startup.py` 之类 | 有副作用、必须是"动作" |
| 线路契约（帧头、标志位、单位、占位 class_id） | `protocol.py` 顶部 | 和帧格式同级，改错对端整包解错 |
| 某一模块的实现细节（自适应系数、容错帧数、抽帧） | 该模块顶部常量 | 只有写那个模块的人才需要动 |
| 多个中间量只为算出第 N 个值 | 合并成第 N 个值 | 中间量是噪音 |

★ 顺带删掉"缺省值兜底 + 模块 `__getattr__` 劫持"这类机制：它把"参数名写错"这种硬错误
变成静默默认值，收益远小于代价（config 与代码本来同批烧录）。参数名写错就该立刻报错。

★ 重构后必做两件事，都可用脚本自动断言：
① 死参数扫描（见步骤 5）；② "config 里函数数 == 0 / import 数 == 0"的结构断言。

## 数据流改造时：让"换算"跟着"数据来源"走

一个反复踩的经验：坐标换算、单位换算、标定参数**必须和数据的第一来源放在同一侧**。

- 若 A 侧产生原始量、B 侧只是转发，那把换算放 B 侧会产生三个问题：
  两侧都要持有标定参数（容易不一致）、B 侧多出一堆用不到的参数、
  出问题时不知道该查哪台设备。
- 正解：A 侧算出**下游可直接使用的最终量**再发出，B 侧只做"取用 + 新鲜度判断"。
- 顺带收益：B 侧能删掉整条换算链路（矩阵、标定表、补偿、区域多边形），
  config 也随之瘦身；而且"该标定的东西"全部集中在 A 侧一份文件里，现场只需看一处。

判据：如果某台设备的 config 里出现了"另一台设备的标定参数"，那基本就是放错了。

