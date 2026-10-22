---
name: maixcam2-obb-int8-convert
description: 【二代】MaixCAM2（AX620E）YOLO11/YOLO26 OBB 旋转框模型 INT8 量化转换。Pulsar2 工具链，产物 .mud + _npu.axmodel + _vnpu.axmodel（type=axmodel 双模型）。当用户要把 OBB 模型部署到二代 MaixCAM2 / "二代的"平台，或提到 axmodel、vnpu、Pulsar2、AX620E、maixcam2 时使用。一代 MaixCAM/MaixCAM-Pro（cv181x/.cvimodel/INT8+BF16）请用 skill maixcam-obb-bf16-convert，两代工具链完全不同不可混用。
agent_created: true
---

# MaixCAM2（二代）OBB INT8 量化转换

> **平台定位（勿混淆）**：本 skill 只适用于**二代 MaixCAM2**——芯片 AX620E、工具链 Pulsar2、产物 `.axmodel`（npu + vnpu 双份）、量化仅 INT8。
> **一代 MaixCAM / MaixCAM-Pro**（芯片 cv181x、工具链 TPU-MLIR、产物 `.cvimodel`、INT8/BF16）见 skill `maixcam-obb-bf16-convert`。

## 一代 vs 二代 速查表

| | 一代 MaixCAM / Pro | 二代 MaixCAM2（本 skill） |
|---|---|---|
| 芯片 | cv181x | AX620E |
| 工具链 | TPU-MLIR（model_transform.py + model_deploy.py） | **Pulsar2**（`pulsar2 build --target_hardware AX620E`） |
| docker 镜像 | maixcam-tpumlir:v3.4 / sophgo/tpuc_dev:v3.4 | **pulsar2:6.0**（一代镜像里没有 Pulsar2，必须另装） |
| 产物 | `.cvimodel` + `.mud`（type=cvimodel 单模型） | `_npu.axmodel` + `_vnpu.axmodel` + `.mud`（type=axmodel 双模型） |
| 量化 | INT8 / BF16 | **仅 INT8**（Pulsar2 无 BF16；用户要 INT8 正好匹配） |
| OBB 输出数 | 6 输出（box+angle+class+3 Concat 头），3 输出是死路 | **3 输出 canonical**（box+angle+class）或 **4 输出**（3 多尺度头+angle）；另见下方"OBB 输出形状铁律" |
| mud 关键字段 | `type = cvimodel` / `model = X.cvimodel` | `type = axmodel` / `model_npu=`+`model_vnpu=` |
| 加载 API | `nn.YOLO11(model, dual_buff=...)` | `nn.YOLO11(model=".../X.mud")` 或 `nn.NN(mud)` |

## 格式样本（本地参考，已解析）
`./work/model-318609.maixcam2\`（MaixHub 下载的 MaixCAM2 检测模型包）：
- `model_318609.mud`：`[basic] type=axmodel` + `model_npu`/`model_vnpu`；`[extra] model_type=yolov5, type=detector, input_cache/output_cache=true, input_cache_flush=false, output_cache_inval=true, mean=0,0,0, scale=1/255`
- `model_318609_npu.axmodel`（7.2MB，NPU2 全算力）/ `model_318609_vnpu.axmodel`（7.4MB，NPU1 给 AI-ISP 预留）；文件头明文 `Pulsar2 Version: 3.3 Pulsar2 Commit: 3cdead5e`
- `app.yaml`（app 元数据 id/name/version/exclude）、`main.py`（`nn.YOLOv5(model=model_path)` 示例）、`report.json`（训练报告，含 labels/输入输出 shape）
- OBB 包与之同构，只是 `type=obb`、`model_type=yolo11`、main.py 换 `nn.YOLO11`

## 硬件/运行时事实（已核实）
- MaixPy 解码器 `maix_nn_yolo11.hpp`（**源码已读，以此为准**）：OBB 有两种输出模式，按**输出个数**分派：
  - **mode 1**：`outputs.size()==4`（3 个多尺度检测头 + 1 个 angle）。
  - **mode 2**（canonical）：输出个数不等于上述值时走这里，按**形状 + 名字**把输出分配到 `dfl`(box) / `sigmoid`(class) / `obb_angle`(angle) 三个槽位。
- **OBB 输出形状铁律（决定成败）**——mode 2 的分派规则（源码 396-456 行）：
  1. **box(dfl)** 必须命中 `shape[1]==1 && shape[2]==4 && shape[3]==anchor_num`，即 **(1,1,4,8400) 且为 4 维**。**绝不能给 box 加 `dst_perm(0,2,3,1)`**——加了会变成 (1,4,8400,1)，条件不成立，`dfl` 永远是 999 → 报 `can't find DFL or sigmoid output` / `get tensor idx error: -404232217 (0xE7E7E7E7)`。
  2. **class(sigmoid)**：`shape[1] > 1`（即 9 通道）**或** 名字含 `Sigmoid_1`。
  3. **angle(obb_angle)**：兜底槽位；`_obb_need_sigmoid = (名字含 "Sigmoid") ? false : true`。
  4. 输出个数必须**恰好 3**（mode 2）——多出来的检测头（如 `Concat/Concat_1/Concat_2`）会反复覆盖 dfl/sigmoid/obb_angle 的索引，把分派彻底打乱。**"导出 6 个输出"是错的，已回滚。**
  5. OBB mode 2 要求 `_out_chw == true`（源码 1117-1120 行，否则 throw `not support output hwc layout`）。
  6. angle 解码：`angle = (need_sigmoid ? sigmoid(v) : v) - 0.25; angle_rad = angle * M_PI`（1139-1140 行）。
- **因此正确的名字约定是：`Sigmoid_1` = 类别，`Sigmoid` = 角度**（"在线平台"约定）。源码第 430 行 `name.find("Sigmoid_1")` → 分配给 **sigmoid(类别)**。原版 `build_obb_yolo11_single_canon_onnx` 把 class 改名成 `Sigmoid_1` 是**正确**的，不要改回。
- **mud 的 `type` 必须= `obb`**，解码器按它分派 OBB 模式；写成 detector 会按普通检测器解码（框/角度全错）。
- `model_npu`（NPU2 全算力）与 `model_vnpu`（NPU1，AI-ISP 预留）两个 axmodel 都要生成，用户按是否开 AI-ISP 自选。
- 部署路径 `/root/models/`，三文件同目录，代码只写 `.mud` 路径。
- 与一代相同的坑：OBB 角度语义 = Sigmoid 后 `(v-0.25)*PI`；图手术时 Sin/Cos 必须变死代码删净（一代 cv181x 不支持 Sin/Cos 算子；二代 canon 图手术复用同一套 DCE，天然满足）。

## 本地转换器链路（maix_converter_platform，已接线验证）
- 入口：`convert_cli.py --target maixcam2 --task obb --yolo-version yolo26|yolo11`
- backend：`converter/backends/maixcam2_pulsar2.py`
  - `prepare_job`：OBB 自动做 **canon 图手术**（yolo26 无 DFL 走 `build_obb26_canon_onnx`、DFL+SinCos 变体走 `build_obb26_dflsincos_canon_onnx`、yolo11 走 `build_obb_canon_onnx`；已是 canon 的直接复制）→ 生成 `.mud`（OBB 时 `type=obb`）→ 写容器脚本（内嵌 3 个输出节点）
  - `run_pulsar2_job`：`docker run --mount job目录:/data pulsar2:6.0`，容器内自动：extract 3 输出 → onnxsim（无则跳过）→ 打包校准图 tar → `pulsar2 build --target_hardware AX620E` 跑两遍（NPU2→`X_npu.axmodel`，NPU1→`X_vnpu.axmodel`）
- profile：`("yolo26","obb")` 的 model_type 写 yolo11（canon 图伪装成 YOLO11-OBB 形态，MaixPy 用 nn.YOLO11+type=obb 解码）；`("yolo11","obb")` 同理
- OBB 输出节点（**3 个**）：`/model.23/dfl/conv/Conv_output_0`(box, **无 dst_perm**) + `/model.23/Sigmoid_1_output_0`(class) + `/model.23/Sigmoid_output_0`(angle)。
- **⚠ yolo11/obb 与 yolo26/obb profile 的关键差异（踩过坑）**：`("yolo11","obb")` 的 `YOLO11_OBB_OUTPUTS` 给 box 带了 `dst_perm=(0,2,3,1)`，会把 box 拧成 (1,4,8400,1)，**OBB mode 2 找不到 dfl → 机端报 0xE7E7E7E7**；而 `("yolo26","obb")` 用 `YOLO26_OBB_CANON_OUTPUTS`（纯名字、**无 perm**），box 保持 (1,1,4,8400)，**能正常解码**。所以 OBB 优先走 yolo26 profile。

### ⚠ 输出个数铁律：**恰好 3 个**（不是 6 个）——别再搞错
用户/历史常误记「OBB 要导出 6 个输出」。**错**。6 输出（box+angle+class+`Concat`/`Concat_1`/`Concat_2` 三个检测头）是**一代 MaixCAM（cv181x）**的形态；**二代 MaixCAM2 的 mode 2 必须恰好 3 个**——多出的检测头会反复覆盖 dfl/sigmoid/obb_angle 的索引槽位，把分派彻底打乱（机端仍报 0xE7E7E7E7）。见「OBB 输出形状铁律」第 4 条。转换前请**用 onnx 实际数一遍 graph.output**，不要凭记忆。

**mode 1 / mode 2 到底怎么分（源码 291-293 行，逐字）**：
```cpp
if((outputs.size() == 5 && _type == SEG) ||
   (outputs.size() == 4 && (_type == OBB || _type == POSE)) ||
   (outputs.size() == 3 && _type == DETECT))   // <- 注意这里是 DETECT，不是 OBB
```
→ **OBB + 3 输出 = 不满足 mode 1 = 落 mode 2**（`print_outputs` 里能打出 3 个张量即走 mode 2），这是正解路径，不要因为有「3」字就以为命中了 mode 1。mode 1 才是那个「多尺度头」形态（OBB 要 4 个）。
另：`_out_chw` 默认 `true`（276 行），**只有 mode 1 分支会把它置 false**（305/389 行）→ mode 2 天然满足 1117 行的 CHW 前置条件，不必额外处理。

### 三条 OBB canon 图手术路径（按图的「原始形态」选，勿混用）
`prepare_job` 会自动分派，但**新模型接手时务必先确认落到哪条**：

| 路径 | 适用图的特征 | 函数 |
|---|---|---|
| A. 形态一 | 有 cv2 直出 box（无 DFL）、角度源 `Concat_2`、**图内无 Sin/Cos** | `build_obb26_canon_onnx` |
| B. 9030-pro 式 | 图中**已存在** `dfl/conv/Conv_output_0` + `Sigmoid_1_output_0`（DFL+SinCos 变体） | `build_obb26_dflsincos_canon_onnx` |
| C. **model-a 式（2026-09 新增）** | 单输出 `[1,14,A]`、**有 Sin/Cos**、有 cv4 角度头、**dfl=0**、且 `Sigmoid_1`/`dfl/conv` 都**不存在** | `build_obb26_raw_canon_onnx` |

判据函数 `_model_is_yolo26_obb_raw()` 即按路径 C 的特征写。**注意**：路径 B 的图上机前已被改写过（`90.onnx` 留存的就是 canon 后的版本），不要把「已有 Sigmoid_1」的图误判成原始图。

### 角度语义的两种情况（换算错了角度会做对折/整体偏移）
- **情况 1（model-a / 90，`Concat_2` = 最终弧度）**：图内旋转公式实测 `rot=[hw·cosθ−hh·sinθ, hw·sinθ+hh·cosθ]`，θ 直接取自角度头，**未除 π/2、未经 Sigmoid**，范围约 ±0.8~1.1 rad（±63°，OBB 允许超 45°）。→ 喂 `v = θ/π + 0.25`，令解码端 `(v−0.25)·π == θ`。
- **情况 2（旧 YOLO11-OBB 单输出，`Concat_2` = 原始弧度但图内先 /2）**：见下文 `build_obb_yolo11_single_canon_onnx` 注，标称 `v = theta/π + 0.25`（若实测发现有 `/2` 环节则需同步调整）。
- **如何判定属哪种**：用 ONNXRuntime 跑一遍图，取角度头张量看**值域**并反推。若角度头值域 ≈ 90 模型的 [-0.82, 0.95] 同量级（弧度 ±60°），属情况 1；若值域缩到 ±0.4（≈±π/4）则可能是情况 2。**不要只看「是弧度」就套公式**——本 skill 早期曾误判为 `θ/(π/2)+0.25`。
- **值域宽度不影响解码约定**：model-b 实测角度头值域 `[-0.83, 3.77]` rad（-47°~216°，比 model-a 宽得多），但两者同属情况 1、同一换算 `v = θ/π + 0.25` 都成立。**判定看的是「有没有 /π、/2 之类的额外缩放」，不是值域绝对值有多大**。

### 「输出头手术」不是画蛇添足，也不是精度损失源（2026-09 复核，用于回应质疑）
常有人怀疑"把输出改造成 3 个会不会弄坏"。**逐文件核实过，结论是不会，而且必须改**：
- **原始 Ultralytics YOLO26-OBB 导出只有 1 个输出**：`output0 [1,14,8400]`。不是"两个"。这个前提经常被记错。
  - ⚠️ **通道顺序更正（2026-09 实测 model-c 逐张量核对）**：`output0 = Concat_5(Mul_4[4ch], Sigmoid[9ch], Concat_2[1ch])`
    → 真实顺序是 **4(box) + 9(class) + 1(angle)**，**不是** 4+1+9。
    其中 `Mul_4 = Concat_output_0 × stride`（已烘焙 stride，值域可达 640），而 class/angle 位是原始 `Sigmoid_output_0` / `Concat_2_output_0` 本身。
    写验证脚本按 `output0[:, 0:4]=box, [:,4:5]=angle, [:,5:14]=class` 切片**必然得到 box diff≈641 / class diff≈0.95 的假失败**。
    **正确做法**：不要猜切片，用 `onnx.utils.extract_model(src, tmp, ["images"], ["output0", "/model.23/Concat_output_0", "/model.23/Sigmoid_output_0", "/model.23/Concat_2_output_0"])` 抽出中间张量，再与 canon 三输出一一比对。
- canon 的 3 个输出**合起来正好 14 通道**，本质是**把同一条张量按通道切成 3 份**，不新增任何计算。
- 手术的算子净变化是**负数**（实测 401 → 386，删除 Sin/Cos/Div/Sub/Mul/Concat/Split 的旋转与拼装分支，
  只新增 1 个 Unsqueeze）—— 烘焙角度的分支被 DCE 掉了。
- 数值等价：box/class 误差 **0.00e+00**、angle **2.38e-07**。
- **决定性反证**：float ONNX 用的就是同一套 3 输出 canon；若手术有错，float 也该挂，但它在 25–96px 处检出 97.3%。
- 另外 MaixPy mode 2 **要求恰好 3 个输出**（mode 1 要 4 个），**直接喂 14 通道单输出反而无法分派**。

### 如何验证"旋转框解码是否正确"（比角度数值更可靠）
不要拿"GT 最长边方向角"去比"模型输出的角度值"——**矩形在 θ 与 θ+90° 下把 w/h 互换后几何完全等价**，
按数值比会得出"误差 87°"的假象（实测两种写法多边形 IoU 完全相同）。
正确做法：**算几何 IoU**——按解码公式还原四角，与 GT 四边形做多边形 IoU
（Sutherland-Hodgman 裁剪，注意线线交点公式 `u = ((a−p)×(b−a)) / ((q−p)×(b−a))`，
写错会得到恒为 0 的假结果，务必先用"同一/半重叠/内含/不重叠"四例单测）。
实测本模型：**中位 IoU 0.925、IoU>0.5 占 100%** → 解码正确。
配套：召回率按**目标图像内旋转角**分箱统计（实测 0–30/30–60/60–90/90–120/120–150/150–180° 分别为
97.9/94.4/100/100/100/97.2%，总 98.3%）——**召回与角度无关**，可据此否掉"只有特定角度才能识别"的传言。
⚠️ **但仍需留意**：若下游按**角度数值**使用（如串口协议里的 angle 字段）而不做 w/h 归一，
这个 90° 二义性会实际影响结果 —— 属独立待查项，与死区无关。


### 设备端旋转方向与图内旋转的交叉验证（推荐在 canon 前做一次，最稳）
不要只验证「canon vs 原始图」，还要验证「**设备解码式 vs 图内旋转式**」是否同向——这能提前排除角度符号/方向反了的问题（真机上表现为框整体镜像或转反）：

1. 图内旋转（实测法）：`Concat_3 == [hw·cosθ − hh·sinθ, hw·sinθ + hh·cosθ]`，其中 `hw=Div[0]=(rb_x−lt_x)/2`、`hh=Div[1]`，θ 取角度头。
2. 设备解码式（`maix_nn_yolo11.hpp` 1139-1152 行）：
   `xf=(rb_x−lt_x)/2`、`yf=(rb_y−lt_y)/2`、`bbox_x=((xf·cos−yf·sin)+ax+0.5)·stride−w/2`、`bbox_y=((xf·sin+yf·cos)+ay+0.5)·stride−h/2`。
3. 两者逐项对照：`xf≡hw`、`yf≡hh`、旋转项 `(xf·cos−yf·sin)` ≡ 图的 `(hw·cosθ−hh·sinθ)` → **同向同式**，说明「喂未旋转 ltrb + `v=θ/π+0.25`」可让设备**逐项复现**图内旋转，方向与幅值均无损。
   （model-a / model-b 均已这样核过，图内旋转公式拟合误差 2.4e-7。）


### 已修复的两个关键 bug（勿回退）
1. **`converter/yolo/mud.py` `write_maixcam2_yolo_mud` 原来硬编码 `type=detector`**（2026-09 修复）：新增 `task` 参数透传，OBB 时 mud 写 `type=obb`。不修则设备端把 OBB 模型当普通检测器解码。
2. **`maixcam2_pulsar2.prepare_job` 原来不做 canon 图手术**（2026-09 修复）：原始 YOLO26-OBB onnx 里 `/model.23/dfl/conv/Conv_output_0` 不存在（由手术 Unsqueeze 创建），不手术则容器内 extract_model 直接失败。现在与一代 maixcam 路径同构：已 canon（graph.output 恰含 3 个 profile 输出名）直接复制，否则做手术。
3. **【2026-09 重大踩坑，勿重犯】真机 `get tensor idx error: -404232217 (0xE7E7E7E7)` 的真正根因不是"漏了 3 个检测头"，而是两条**：
   - **(a) 用错了 `--yolo-version`**：`90.onnx` 实际是 **YOLO26-OBB**（有 `/model.23/cv2.x.2`(4ch) / `cv3.x.2`(9ch) / **`cv4.x.2`(1ch 角度头)**，**dfl 节点数=0**），却按 `--yolo-version yolo11` 转 → 走了 yolo11 分支，而 yolo11 profile 给 box 带 `dst_perm`，box 变 (1,4,8400,1)，mode 2 找不到 dfl。
   - **(b) 中途误改（已回滚，勿再改）**：曾据一句注释推断"解码器按 6 个固定名查找"，把输出扩成 6 个（多出 `Concat/Concat_1/Concat_2`）并把名字调成 `Sigmoid_1`=angle / `Sigmoid`=class —— **全错**。多出的检测头会覆盖索引槽位；`Sigmoid_1` 在源码里是分配给**类别**的。已全部回滚到 3 输出 + `Sigmoid_1`=class / `Sigmoid`=angle。
   - **判定模型是 26 还是 11 的可靠方法**：查 `/model.23/` 下是否有 `cv4.x`（YOLO26 的角度头）以及 `dfl` 节点数是否为 0（YOLO11 必有 `/model.23/dfl/...`）。不能只看"是不是单输出 + 有没有 Sin/Cos"——`_model_is_yolo11_obb_single` 对 YOLO26 单输出也会误判为 True。

## docker 镜像获取（pulsar2:6.0）
- 本机一代只有 `maixcam-tpumlir:v3.4` / `sophgo/tpuc_dev:v3.4`，**不含 Pulsar2，不能用于二代**。
- 官方源：HuggingFace `AXERA-TECH/Pulsar2`（国内直连不通）→ 用 **ModelScope 镜像仓库**：
  ```
  https://www.modelscope.cn/models/AXERA-TECH/Pulsar2/resolve/master/6.0/ax_pulsar2_6.0.tar.gz   # 5.48GB
  ```
- 导入：`docker load -i ax_pulsar2_6.0.tar.gz`，确认 `docker images` 里名字为 `pulsar2:6.0`（convert_cli 默认值）；若 tag 不同用 `docker tag` 改名或 CLI `--docker-image` 指定。
- **沙箱坑**：Bash 沙箱拦截 `curl.exe` 的磁盘写入（网络通、dd/python 写盘正常，唯独 curl 落盘失败，报 exit 23 / write error）→ 大文件下载用 python urllib（断点续传脚本 `./work/pulsar2_download/dl.py`，约 7MiB/s，5.48GB 约 12 分钟）。
- 下载完整性校验：`gzip -t` 或 python gzip 流式解压定位坏块（曾遇"大小正确但压缩流 1.2GB 处 invalid block type"的损坏文件，只能整档重下，续传救不了）。

## 标准转换流程
1. **校准图**：20–100 张真实场景图放一个目录（`--images-num` 必须 ≤ 实际张数；量化质量取决于图片贴近现场程度）。
2. **跑转换**（在 `<converter_root>` 下、用其 venv python）：
   ```
   ./venv/Scripts/python.exe convert_cli.py \
     --model X.onnx \
     --dataset <校准图目录> \
     --target maixcam2 --yolo-version yolo26 --task obb \
     --images-num 20 --model-name X \
     --fast --job-dir jobs/X_maixcam2
   ```
   - `--fast`：跳过 precision_analysis 与 compiler check（调试快）；正式转换去掉 `--fast`。
   - 原始（非 canon）OBB onnx 可直接喂，prepare_job 自动做 canon 手术。
   - **换 yolo11 模型用 `--yolo-version yolo11`**（但见下条：OBB 优先按 yolo26 转）。
   - **⚠ 先判模型世代再选 `--yolo-version`（2026-09 血的教训）**：单输出 OBB 里，`_model_is_yolo11_obb_single` 对 **YOLO26 单输出也会误判为 True**（它只看"单输出 + 有 Sin/Cos + 有 Concat/Sigmoid/Concat_2"）。可靠判据是看 `/model.23/` 的结构：
     - 有 `cv2.x/cv3.x/**cv4.x**`（cv4=角度头）且 **`dfl` 节点数 = 0** → **YOLO26-OBB** → 用 `--yolo-version yolo26`。
     - 有 `/model.23/dfl/conv/Conv` → YOLO11-OBB。
     例：`90.onnx` 有 `cv2.x.2`(4ch)/`cv3.x.2`(9ch)/`cv4.x.2`(1ch)、dfl=0 → **是 YOLO26**，按 yolo26 转（走 `build_obb26_canon_onnx`，产出 3 输出且 box 无 perm）。
   - **新增格式（2026-09）**：Ultralytics **YOLO11-OBB 单输出导出**（output0 [1,4+1+nc,A]，图尾用 Sin/Cos 把角度烘焙进 box、角度以【原始弧度】单独输出）由 `_model_is_yolo11_obb_single` 自动识别，走 `build_obb_yolo11_single_canon_onnx`：导出 **3 个**输出——box 取未旋转 ltrb `/model.23/Concat_output_0` → Unsqueeze(axis=1) 成 [1,1,4,A] 挂到 `/model.23/dfl/conv/Conv_output_0`；**class `/model.23/Sigmoid_output_0` 改名 `Sigmoid_1_output_0`**（这是对的，解码器把 `Sigmoid_1` 当类别）；angle 取 `/model.23/Concat_2_output_0` 原始弧度经 `v=theta/π+0.25` 挂到 `/model.23/Sigmoid_output_0`。Sin/Cos/旋转/×stride 全部 DCE 删净。
3. **产物**：`out/X.mud` + `out/X_npu.axmodel` + `out/X_vnpu.axmodel` + 打包 zip。
4. **部署**：三文件同目录上传设备 `/root/models/`，main.py 用 `nn.YOLO11(model="/root/models/X.mud")`（OBB 解码由 mud `type=obb` 触发）。验证加载可先 `nn.NN(mud)` 打印。
5. **app 打包（可选）**：按样本放 `app.yaml`（id/name/version/exclude）+ `main.py` + 模型三件套；`report.json` 仅训练报告可省略。

## 踩坑清单（持续补充）
- **Docker Desktop 必须先启动**，`docker info` 正常再跑转换（同一代）。
  - 本机 Docker Desktop 装在**非默认路径**：`%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe`。启动后需 **2–4 分钟**等 backend + WSL 引擎起来；就绪判据 `docker info --format '{{.ServerVersion}}'` 返回版本号。日志：`%LOCALAPPDATA%\Docker\log\host\com.docker.backend.exe.log`（找 `unholding Engines, they will start now`）。启动期间 `hub.docker.com ... context deadline exceeded` 属无害网络超时，不阻断。
- **Bash 沙箱 PATH 间歇性损坏（2026-09 遇到）**：出现 `ls: command not found` / `grep: command not found` / `wc` / `dirname` 报错（连 `|` 管道都失效，整条命令 exit 127）。**解法：命令开头加 `export PATH="/usr/bin:/bin:$PATH"`** 立刻恢复。注意别用 `grep` 过滤 python 输出时把结果滤没了（先确认 grep 可用）。
- **Windows 路径**：venv python 不认 `/e/`，用 `./work/`；bash 内部命令用 `/e/`。
- **【删文件进回收站】`send2trash` 必须传【规范反斜杠】路径（2026-09 实测）**：
  ```python
  import os
  from send2trash import send2trash
  p = os.path.normpath(r"<converter_root>//_calib_model-c")   # 关键：先 normpath
  try:
      send2trash(p)
  except Exception:
      pass                      # 成功后仍抛伪 WinError 2，必须吞掉
  assert not os.path.exists(p)  # 唯一的成功判据
  ```
  传**正斜杠**（`./work/xxx`）时 send2trash 会拼出 `\\?\./work/xxx`，`SHCreateItemFromParsingName` 解析失败 →
  抛 `FileNotFoundError [Errno 2]` 且**文件并未删除**（`exists()` 仍为 True）——这与"删成功抛伪错误"的表现极其相似，
  **极易误判**，务必用 `exists()` 复核。另外：`send2trash` 对**已清空的目录**仍可能先抛一次错误、第二次才删掉，
  重试一次即可。pip 安装：`<venv>/Scripts/python.exe -m pip install send2trash`。
- **后处理留 CPU**：输出节点取检测头 3 张量，把 decode/NMS 留给 MaixPy；不要把带 Sin/Cos 的完整输出喂给 Pulsar2。
- **`--fast` 与正式档的代价/收益（2026-09 实测）**：
  - `--fast`：约 **8 分钟**（跳过 precision_analysis 与 compiler check）。
  - 不带 `--fast`：**29m29s**（数百张校准），但会产出**逐算子量化保真表**与输出层 cosine ——准备交付/需要证据时值得多花这 20 分钟。
  - 校准耗时随张数线性增长：**数百张 Phase 1 约 3 分 16 秒**（20 张约 21 秒）。
- **【有用发现】逐算子量化保真表的位置**（`--fast` 时不生成）：
  ```
  jobs/<job>/tmp_build/quant/debug/precision_analysis_table.txt   # 每算子 Cosin / MSE 列
  jobs/<job>/tmp_build/quant/debug/precision_analysis.mmd         # 图
  ```
  用途：**定位到底是哪几个算子在量化中损失最大**（实测 model-b 全图 305 算子：min cos=0.976、P1=0.988、中位=0.9995，仅 4 个算子 <0.99、**没有一个 <0.95**）。若某模型小目标明显掉点，可在此表里找靠前的低分算子，判断是否集中在 stride-8 的高分辨率分支。
  **坑**：这张表是 pretty-printer 输出的固定列宽表格，**长节点名会被截断成 `/model.…`**，无法按名字 grep 输出层；想核对输出层请改用 `convert.log` 里的 `check npu graph ... cosin simularity is X` 行（那里名字完整）。

- `npu_mode`：NPU2=全算力（→`_npu.axmodel`），NPU1=分一半给 AI-ISP（→`_vnpu.axmodel`）；两个都转，mud 各自指向。
- **YOLO11-OBB 单输出导出的角度单位**：此类图角度以【原始弧度】输出（范围约 [-π/4,π/4]），且 box 已被 Sin/Cos 旋转+×stride 烘焙；必须回退到未旋转 ltrb（`/model.23/Concat_output_0`）并做 `v=theta/π+0.25` 补偿交给 mode2（其 `(v-0.25)*π` 才还原原始弧度、且 box 由 mode2 自行旋转避免双重旋转）。**Div 的除数必须是 π 本身**（不是 1/π，否则变成 ×π，角度完全错）。

## 实测记录（2026-09，pulsar2:6.0 全流程跑通）
- **端到端成功**：`_exp19_canon.onnx`（YOLO26-OBB 多类 640×640）→ `convert_cli.py --target maixcam2 --task obb --fast --images-num 20` → `demo_obb26.mud` + `demo_obb26_npu.axmodel`(9.9MB) + `demo_obb26_vnpu.axmodel`(12MB) + zip，全程 **9 分 52 秒**（vnpu + npu 两遍 build）。
- **测试 A（原始 onnx 自动手术）✅**：`9030pro.onnx`（yolo26 DFL+SinCos 原始单输出）→ prepare_job 自动 `build_obb26_dflsincos_canon_onnx` → 3 输出 canon、Sin/Cos 清零 → npu(9.8MB)+vnpu(11MB) 转出。job：`jobs/testA_9030pro_raw_maixcam2`。
- **测试 B（yolo11 分支）✅ 转换成功**：`9030Pro.onnx`（yolo11-obb 原始）+ `--yolo-version yolo11` → `build_obb_canon_onnx` 产出 6 输出 canon（3 核心 + 3 Concat 头），容器内按 profile 抽 3 名 → npu(9.8MB)+vnpu(12MB)。job：`jobs/testB_9030Pro_yolo11_maixcam2`。**注**：此分支同样给 box 带 `dst_perm`，若上机 OBB 报 0xE7E7E7E7，第一嫌疑就是 box 的 perm（应无 perm、形状 (1,1,4,8400)）。
- **测试 C（正式档，去 --fast）✅**：同 exp19 模型不带 `--fast`（precision_analysis + compiler check=3）→ 20m34s 成功，产物 npu(9.9MB)+vnpu(12MB)+mud。编译期 NPU 仿真 vs float 参考 **cosine 相似度：box=0.9999999995、class=0.99999999、angle=1.0**——INT8 量化在图级别几乎零损失。job：`jobs/testC_obb26_full_maixcam2`。
- **量化确认**：Network Snapshot 293/293 算子全量化（INT8），校准 20 张 Phase 1 21 秒。
- **输出 shape 正确**：box [1,1,4,8400]、class [1,9,8400]、angle [1,1,8400]——与 MaixPy mode2 解码器预期一致。
- **axmodel 头**：`Pulsar2 Version: 6.0 Commit: 48520c11`；与 MaixHub 样本（Pulsar2 3.3）同族格式，样本 vnpu>npu 大小，本产物同样 vnpu(12MB)>npu(9.9MB)，规律一致。
- **容器环境**：pulsar2:6.0 里有 python（脚本直接跑），**无 onnxsim**（自动降级为复制 extracted ONNX，不阻断）；extract→tar 打包→两次 build 全自动。
- mud 产物：`type=obb`（修复后）+ `model_type=yolo11` + labels 从模型 metadata 自动读取。
- **90.onnx 首转（2026-09）⚠️ 转换成功但机端报 0xE7E7E7E7**：模型实为 **YOLO26-OBB**（cv2/cv3/**cv4** 头、dfl=0），却按 `--yolo-version yolo11` 转 → yolo11 profile 给 box 带 `dst_perm(0,2,3,1)` → box 变 (1,4,8400,1) → mode 2 命中不了 dfl → 真机每帧 `get tensor idx error: -404232217`。job：`jobs/90_maixcam2_calib`。**教训：转换成功 ≠ 能解码，cosine 全 1.0 也没用——形状/名字不匹配照样挂。**
- **90.onnx 第二次（错误修复，已废弃）⚠️**：误信"解码器按 6 个固定名查找"，把输出扩成 6 个并调反 `Sigmoid_1`/`Sigmoid` 语义 → 编译期 6 个张量 cosine 全 ≈1.0 但**机端仍 0xE7E7E7E7**（多出的检测头覆盖索引槽位 + 名字语义反了）。job：`jobs/20260905_210851_90_maixcam2_yolo11`。**已回滚，勿复用。**
- **90.onnx 第三次（正确，2026-09）✅**：改用 `--yolo-version yolo26` → `build_obb26_canon_onnx` 产出 3 输出且 **box 无 perm=(1,1,4,8400)**、`Sigmoid_1`=类别(1,9,8400)、`Sigmoid`=角度(1,1,8400)=theta/π+0.25 → 与解码器 shape/名字规则完全对齐。校准图与模型均用工作区相对路径（见沙箱坑）。产物待构建完成后核对。

## 实测记录（2026-09，model-a 新增路径 C，全流程跑通 ✅）
- **模型**：`./work/model-a.onnx` = **Ultralytics YOLO26s-obb**（metadata 明确 `task=obb, head=OBB26, imgsz=[640,640]`），多类，单输出 `output0 [1,14,8400]`。
- **落到路径 C**：该图有 Sin/Cos + cv4 角度头 + dfl=0，且**无** `Sigmoid_1`/`dfl/conv` → 不匹配 A/B/yolo11-single 任一条 → 新增 `build_obb26_raw_canon_onnx` 处理。
- **角度语义实测锁定（重要）**：ONNXRuntime 实跑中间张量，验证
  `/model.23/Concat_3_output_0 == [hw·cosθ − hh·sinθ, hw·sinθ + hh·cosθ]`（maxdiff 2.4e-7），
  θ 直接取 `/model.23/Concat_2_output_0`，**值域 [-0.8437, 1.1050] rad（±63°）**，未除 π/2、未经 Sigmoid。
  → 采用 `v = θ/π + 0.25`；**曾差点误用 `θ/(π/2)+0.25`，会把角度做对折，务必按值域判定**。
- **canon 手术结果**：3 输出 `box(1,1,4,8400)`（无 dst_perm）/ `class(1,9,8400)` / `angle(1,1,8400)`，Sin/Cos DCE 删净。
- **数值等价性**：canon 三输出 vs 原始图中间张量 —— box/class 误差 **0.00e+00**、angle **1.79e-07**（浮点精度内无损）。
- **构建**：`convert_cli.py --target maixcam2 --yolo-version yolo26 --task obb --images-num 40 --fast`，**7m49s**；
  **293/293 算子全 INT8 量化**（与 90 参考一致）；产物 `model-a_npu.axmodel`(9.9MB) + `model-a_vnpu.axmodel`(11.1MB) + `model-a.mud`。
- **交付**：`./work/model-a_maixcam2_obb.zip`（三文件同目录，mud `type=obb`/`model_type=yolo11`/多类，可直接放 `/root/models/`）。
- **环境**：Docker Desktop 装在非默认路径 `%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe`，启动后需 2-4 分钟等 backend+WSL 引擎就绪。
- **仍待设备实测**：角度方向、置信度分布、旋转框贴合度。

## 实测记录（2026-09，model-b —— 同属路径 C，二次验证 ✅）
- **模型**：`./work/model-b.onnx` = **YOLO26s-obb**（metadata `head=OBB26, task=obb, imgsz=[640,640], version=8.4.150`），多类，单输出 `output0 [1,14,8400]`。算子分布与 model-a 逐项一致（Conv 111 / Sigmoid 94 / Mul 100 / Sin 1 / Cos 1 / Div 1 / Sub 2）。
- **判定**：cv4 角度头 21 节点、dfl=0、有 Sin/Cos、`dfl/conv`+`Sigmoid_1` 均缺 → `_model_is_yolo26_obb_raw()` = True → **自动落路径 C**，无需改代码。
- **角度**：实测 `Concat_2` 值域 `[-0.827, 3.769]` rad（-47.4°~215.9°，比 model-a 宽），同属情况 1 → `v = θ/π + 0.25`；旋转式拟合误差 2.384e-07。
- **新增交叉验证**：设备解码式（hpp 1139-1152）与图内旋转项**逐项同向同式**（见上文「设备端旋转方向与图内旋转的交叉验证」）。
- **结果**：canon 3 输出、Sin/Cos 删净；数值等价 box/class **0.00e+00**、angle **2.38e-07**；构建 **7m56s**、**293/293 全 INT8**、Pulsar2 6.0 commit 48520c11。
- **交付**：`./work/model-b_maixcam2_obb.zip`（仅 3 文件：model-b.mud + model-b_npu.axmodel 9.8MB + model-b_vnpu.axmodel 11.1MB）；zip 完整性 OK、mud 引用自洽、mud `type=obb`/`model_type=yolo11`/多类。
- **结论**：model-a 建立的路径 C 流程**可复用**，第二个同族模型一次通过。

## 实测记录（2026-09，model-b 用「精选 数百张校准集」重转 —— 正式档，最优保真 ✅）
- **动机**：同一天先用 40 张随机图 `--fast` 转过一次；用户随后要求改用专门为「远端小目标召回」筛选的 **数百张校准子集**（见 skill `quant-calib-subset-select`）重转。
- **校准集**：`./work/calib_subset_xingzhuang3_maixcam2\images`（数百张，含 81 个小目标<50px 实例，密集图占 40%，部署域占 69.5%）。
- **命令**：`--images-num 200 --yolo-version yolo26 --task obb`，**不加 `--fast`**。
- **耗时 29m29s**（校准 数百张 Phase 1 用 3:16；NPU2+NPU1 两遍 build + precision_analysis）。
- **量化覆盖**：293/293 算子全 INT8。
- **输出层保真（convert.log 逐行）**：
  - NPU2 遍：`dfl`(1,1,4,8400) = **1.0**；`Sigmoid_1`(1,9,8400) = **1.0**；`Sigmoid`(1,1,8400) = **1.0**
  - NPU1(vnpu) 遍：box = 0.9999999953；class = 0.9999999900；angle = 0.9999999967
- **全图逐算子**：305 算子，min cos=0.976、P1=0.988、中位=0.99945；仅 4 个 <0.99，**无一个 <0.95**。
- **canon 复核**（与原始图中间张量对比）：box/class **0.00e+00**、angle **2.38e-07**；3 输出、Sin/Cos 删净、box 无 perm。
- **产物**：`model-b_npu.axmodel` 10,313,568 B + `model-b_vnpu.axmodel` 11,653,532 B + `model-b.mud` 410 B（`type=obb`/`model_type=yolo11`/多类）。
- **交付**：`./work/model-b_maixcam2_obb_calib200.zip`（**仅 3 个模型文件**，zip 完整性 OK、mud 引用自洽、axmodel 内嵌 3 个输出名）。
- **命名提示**：同目录还留着早先 40 张随机图版 `model-b_maixcam2_obb.zip`。**两者架构相同、仅校准集不同**，正式使用应选 `_calib200` 这个；保留旧的仅供 A/B 对照。
- **结论**：**校准集换成精选 数百张后，编译期保真度达到 1.0（NPU2）**，路径 C 流程在同一天内第三次通过。

## 实测记录（2026-09，model-t.onnx —— 路径 C 第 4 次验证 + 交付打包约定 ✅）
- **模型**：`./work/model-t.onnx` = Ultralytics **YOLO26s-obb**（metadata `task=obb, head=OBB26, version=8.4.151, end2end=False, imgsz=[640,640]`），多类，单输出 `output0 [1,14,8400]`。
- **【务必核对的硬约束】输入张量名必须 = `images`**：容器脚本 `convert_inside_docker.py` 里 `onnx.utils.extract_model(..., ["images"], OUTPUT_NAMES)` 是**写死**的。接新模型前先确认 `graph.input[0].name == "images"`，否则 extract 直接失败。本模型 input=`images [1,3,640,640]`，OK。
- **判定**：cv4 头有、dfl=0、Sin/Cos 各 1、`Sigmoid_1`/`dfl/conv` 均缺 → `_model_is_yolo26_obb_raw()`=True → **自动落路径 C**（`build_obb26_raw_canon_onnx`）。同族第 4 个，**零代码改动一次通过**。
- **canon 复核**（ONNX 层直读 graph.output，比看日志更硬）：3 输出 `box(1,1,4,8400)` / `class(1,9,8400)` / `angle(1,1,8400)`，**Sin/Cos 节点数 = 0**（401→386 节点），Unsqueeze/Mul/Add 齐备。
- **校准集中文路径规避**：`./work/量化\images` 数百张（部分含中文名）→ 先复制到 **ASCII 路径 + 顺序名**（`img_0001.jpg…`）再喂 `--dataset`，避免 docker bind-mount / 容器内 tar 的中文编码风险。
- **构建**：`--yolo-version yolo26 --task obb --images-num 184 --model-name 123`，**不加 `--fast`**（保留 check=3 cosine 门禁，交付件应走正式档）→ **28m42s**（12:35:14→13:03:56）成功，`job.json status=success`。
- **产物**：`123_npu.axmodel` 10,313,952 B + `123_vnpu.axmodel` 11,653,252 B + `123.mud` 408 B（`type=obb`/`model_type=yolo11`/多类）；与 model-b 参考仅差几百字节，同族一致；axmodel 头 `Pulsar2…`。
- **交付打包约定（用户 2026-09 明确）**：单个 zip 放**模型文件同目录**，内含 ①源 `X.onnx` ②`X.mud` ③`X_npu.axmodel` ④`X_vnpu.axmodel` ⑤`README.txt`（类别顺序+部署路径+MaixPy API）。**必须排除**校准图、convert.log、job.json、tmp*、中间 canon onnx——用户强调"仅保留与运行直接相关的文件"。产物：`./work/123_maixcam2_obb.zip`（49.3MB，CRC OK）。

## 实测记录（2026-09，model-c.onnx —— 路径 C 第 5 次验证 + 「剩余即删」清理约定 ✅）
- **模型**：`./work/model-c.onnx` = Ultralytics **YOLO26s-obb**（metadata `task=obb, head=OBB26, version=8.4.154, end2end=False, imgsz=[640,640]`），多类，单输出 `output0 [1,14,8400]`，input `images [1,3,640,640]`（硬约束满足）。
- **判定**：cv4 头 21 节点、dfl=0、Sin/Cos 各 1、`Sigmoid_1`/`dfl/conv`/`Concat_output_0` 均缺 → `_model_is_yolo26_obb_raw()`=True → **自动落路径 C**。同族第 5 个，**零代码改动一次通过**。
- **校准集**：`./work/量化\images` **数百张**（近两百张 640×640，2 张 416×416，1 张 320×320）。非 640 的图**不必剔除**——`calibration_format: Image` 由 Pulsar2 自行 resize；`pack_images` 取 `sorted(images)[:images_num]`。
- **canon 复核**：401 → **386** 节点，Sin/Cos = **0**；3 输出 `box(1,1,4,8400)`（无 dst_perm）/ `class(1,9,8400)` / `angle(1,1,8400)`。
- **数值等价（推荐写法，比看日志硬）**：
  ```python
  onnx.utils.extract_model(SRC, TMP, ["images"],
      ["output0", "/model.23/Concat_output_0", "/model.23/Sigmoid_output_0", "/model.23/Concat_2_output_0"])
  ```
  再逐一比对 canon 三输出 → box **0.00e+00**、class **0.00e+00**、angle **5.96e-08**。**不要按 `output0` 切片猜**（见上文通道顺序更正）。
- **角度**：`Concat_2` 值域 `[-0.7937, 0.9458]` rad（±54°）→ 情况 1，`v = θ/π + 0.25`。
- **构建**：`--yolo-version yolo26 --task obb --images-num 186 --model-name model-c`，**不加 `--fast`** → **28m58s** 成功。
  输出层 cosine —— NPU2：box 0.9999999743 / class 1.0 / angle 0.9999999774；NPU1(vnpu)：box 1.0 / class 0.9999999895 / angle 0.9999999971。
- **产物**：`model-c.mud` 412 B + `model-c_npu.axmodel` 10,314,008 B + `model-c_vnpu.axmodel` 11,653,436 B；Pulsar2 6.0 commit 48520c11；axmodel 内嵌 3 个输出名核对通过。
- **交付**：`./work/model-c_maixcam2_obb.zip`（51,363,564 B ≈ 49.0 MB，CRC OK，含源 onnx+mud+npu+vnpu+README.txt）。
- **「剩下的直接删除」清理约定**：交付后**只保留压缩包**——需删除 job 目录、`_calib_*` 临时校准副本、CLI 日志、**以及源 `model-c.onnx`**（已含在 zip 内）。删除**必须走回收站**（见踩坑清单 send2trash 条目）。`量化/` 原始图片目录**绝不可动**。

## 网页转换器（2026-09 已补齐）
- 前端 target 默认就是 MaixCam2；yolo 版本下拉原只有 "YOLO11 OBB"，**已补 "YOLO26 OBB" 选项**（`index.html` select-option，data-value=yolo26 data-meta=OBB，无需改 JS）。
- 文件名自动识别已修（`app.js applyYoloVersionFromFileName`）：文件名含 obb+yolo26 → 自动选 YOLO26 OBB；只含 obb → YOLO11 OBB。
- 注意：文件名不含 yolo26 的 yolo26-obb 模型（即文件名不含 yolo26 的 yolo26-obb 模型）自动识别仍会选成 YOLO11 OBB，需手动切下拉。

## 待上机验证（转换成功 ≠ 精度/解码正确）
- **【MaixPy 运行时 API 硬约束】`detector.detect()` 返回的是 `maix._maix.nn.Objects`，它只支持
  `len()`、**整数索引** `objs[i]`、以及迭代；`objs[:N]` / `sorted(objs)` / `min`/`max`/`list(objs)`
  一律报 `TypeError: __getitem__(): incompatible function arguments ... Invoked with: ..., slice(...)`。**
  正确写法：`for i in range(len(objs)): o = objs[i]`，或直接 `for o in objs:`（迭代可用），
  攒成自己的 list 再排序/切片。写设备端脚本时特别注意——这个错误只在真机暴露，PC 上测不出来。
- **【2026-09 重大案例】实机「距离死区」的真凶常在 app 的输入处理，不在模型/量化**：
  用户项目把**灰度图**喂模型（`img.to_format(FMT_GRAYSCALE).to_format(FMT_RGB888)`）却**显示彩色原图**，
  于是「分辨率一样但模型看到的和你看到的不一样」。灰度 luma 加权下（R .299/G .587/B .114），
  **深色台面上的饱和彩色物体对比度会崩塌**（纯蓝尤甚）→ 近处大目标还能检出、远处小目标信号塌掉
  → 表现为**一堵"距离死区"墙**；而离线拿**彩色**截图喂 float ONNX 却正常。
  **排查启示**：怀疑模型之前，先确认「屏幕显示的那张」和「送进 `detect()` 的那张」是不是同一张、
  同一色彩空间。分辨率相同**完全不等于**像素内容相同。
  最省事的对齐做法：`cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format())`，
  并让显示与推理共用同一个 image 对象；调试期可把 `disp.show()` 的图直接换成送进模型的 `dimg`，
  配合 `dual_buff=False` 排除帧错位。
- `get tensor idx error: -404232217 (0xE7E7E7E7)` 的排查顺序（2026-09 实战）：
  1. **先看 `--yolo-version` 选对没**——单输出 OBB 极易把 YOLO26 误当 YOLO11（判据：`/model.23/` 有 `cv4.x` 且 `dfl` 节点数=0 → YOLO26）。这是本次真凶。
  2. **看 box 有没有被加 `dst_perm`**——OBB mode 2 只认 **(1,1,4,8400)**，加了 (0,2,3,1) 就永远找不到 dfl。
  3. **看输出个数是不是恰好 3**（mode 2）——多出的检测头会覆盖 dfl/sigmoid/obb_angle 索引槽位；不是越多越好。
  4. **名字语义**：`Sigmoid_1`=类别、`Sigmoid`=角度，别搞反。
- 仍待实测：OBB 在真机 AX620E 的角度方向、置信度分布、旋转框贴合度——需设备实测混淆矩阵（同一代方法论：30–100 张难例对比 INT8 与参考输出）。
- INT8 量化后相似形状（形状F↔形状B、形状E↔形状A）混淆率是否可接受；不行优先升校准图质量/数量，其次更高分辨率重训（二代没有 BF16 兜底）。
- 若设备加载报 mud/模型不匹配，先核对三文件同目录、`model_npu`/`model_vnpu` 文件名与实际一致。
