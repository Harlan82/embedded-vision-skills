---
name: maixcam-obb-bf16-convert
description: 【一代】MaixCAM / MaixCAM-Pro（cv181x，TPU-MLIR，.cvimodel）YOLO11/YOLO26 OBB 旋转框模型的高精度量化转换。解决"相似形状（形状F/形状B、形状E/形状A）量化后混淆"问题，提供 INT8/BF16 精度档位、640 高分辨率、近原生精度方案。当用户要把 YOLO11 OBB 模型部署到一代 MaixCAM/MaixCAM-Pro 并要求"接近原生精度 / 不混淆相似形状 / BF16 / 混合精度 / 上 640 分辨率 / 提升识别准确率"时使用。注意：二代 MaixCAM2（AX620E，Pulsar2，.axmodel）请用 skill maixcam2-obb-int8-convert。
agent_created: true
---

# 一代 MaixCAM/Pro (cv181x) OBB 高精度量化转换

> **平台定位（勿混淆）**：本 skill 只适用于**一代** MaixCAM / MaixCAM-Pro——芯片 cv181x、工具链 TPU-MLIR（docker 镜像 `maixcam-tpumlir:v3.4` / `sophgo/tpuc_dev:v3.4`）、产物 `.cvimodel`、量化 INT8/BF16。
> **二代 MaixCAM2**（芯片 AX620E、工具链 Pulsar2、产物 `.axmodel` npu+vnpu 双份、仅 INT8）走另一条完全不同的链路，见 skill `maixcam2-obb-int8-convert`。
> 一代/二代快速判别：模型包里有 `.cvimodel` = 一代；有 `_npu.axmodel` + `_vnpu.axmodel` 两个文件 = 二代。

## 适用场景
- 目标硬件 **MAIXCAM Pro**（芯片 cv181x，NPU 1 TOPS@INT8，**官方支持 BF16**）。
- 模型类型 **YOLO11 OBB（旋转框）**，六输出（box+angle+class + 3 个检测头 Concat）。
- 核心痛点：相似几何形状（形状F↔形状B、形状D↔形状A）量化后混淆。
- 根因双杠杆：**① 320 分辨率太低**，棱角/边数细节被压没；**② INT8 把动态范围压成 8 位**，相似特征被压到一起。解决 = **640 分辨率 + BF16 量化**。

## 硬件事实（官方 sipeed wiki，已核实）
- 摄像头：最高 **5M**（GC4653 / OS04A10），640×640 square crop 完全在传感器能力内。
- NPU：1 TOPS@INT8，**支持 BF16 模型**，算子支持丰富（YOLOv5/v8/v11 等）。
- 帧率基准（detect 类 YOLO11n）：640×640 INT8 ≈ 23 FPS，320×320 ≈ 95 FPS。
- **OBB 六输出更重**：640 INT8 ≈ 12–16 FPS，640 BF16 ≈ 6–9 FPS；320 INT8 明显更快但精度差。
- 结论：**摄像头带得动 640**。精度优先选 BF16（6–9 FPS），速度优先选 640 INT8（12–16 FPS，转换器修复后 INT8 也正确）。

## 本地转换器（maix_converter_platform）已具备能力
- 精度档位：`--precision int8|bf16|mix`（CLI）及网页下拉框。
- **BF16 部署命令**（cv181x，无需校准表）：
  ```
  model_deploy.py --mlir X.mlir --quantize BF16 --processor cv181x --model out/X_bf16.cvimodel
  ```
- **混合精度 mix**：依赖 `run_qtable.py`，本机镜像（`maixcam-tpumlir:v3.4`、`sophgo/tpuc_dev:v3.4`）**均无此工具 → 暂不可用**，代码已加清晰报错提示，不要尝试。
- 校准集目录约定：每个模型 job 下的 `coco/` 放 100+ 张真实场景图；`images_num` 必须 = 实际图片数。

## 两个必须保留的转换器修复（勿回退！）
这两个 bug 比"量化精度"更致命，已修在源码中，**绝对不能改回原始写法**：

1. **拓扑排序缺失**（文件 `converter/yolo/node_profiles.py` 的 `build_obb_canon_onnx`）
   - 症状：源 ONNX 节点可能乱序（如输入 Cast 排在被它消费的 Conv 之后），`onnx.checker.check_model` 直接拒收报错。
   - 修复：在 `shape_inference.infer_shapes(m)` **之后、check_model 之前**做 Kahn 拓扑排序，且必须 `g = m.graph` 重新绑定（infer_shapes 返回新模型对象，旧 `g` 已失效）。
   - 坑：protobuf repeated 字段不支持 `g.node[:] = _order`，用 `del g.node[:]; g.node.extend(_order)`。

2. **输出回退阈值写错（严重）**（同一函数末尾）
   - 原代码 `if len(out_nodes) < 7:` → 但标准 OBB 只导 **6 输出**（故意剔除冗余的 `Concat_3` Sin/Cos 头），于是 `6 < 7` 永远为真 → **所有转换只导出 3 输出**（box+angle+class）。
   - 3 输出是死路：设备上运行期每帧 `get tensor idx error: -404232217`（0xE7E7E7E7 未初始化张量）。
   - 修复：阈值改为 `if len(out_nodes) < 6:`（仅当确有检测头缺失才回退核心 3 个）。
   - **⚠ 别把一切「3 输出」都当故障**：上面这条只针对 **YOLO11-OBB** 路径。
     **YOLO26-OBB 是另一条路**（走 `build_obb26_canon_onnx()` 专门图手术），**正常产物就是 3 个输出**，
     且 `model_type` 刻意写成 `yolo11`。见 `converter/yolo/node_profiles.py` 里
     `YOLO26_OBB_CANON_OUTPUTS` 的注释：YOLO26-OBB **无 DFL**（`cv2.x.2` 直出 4 通道 ltrb）、
     **角度直出原始弧度**（不经 Sigmoid）、图尾 Sin/Cos 在**活路径**上。
     ```
     /model.23/dfl/conv/Conv_output_0   [1,1,4,A]  box（ltrb 距离，格）
     /model.23/Sigmoid_1_output_0       [1,nc,A]   类别概率（已 sigmoid）
     /model.23/Sigmoid_output_0         [1,1,A]    角度补偿值
     ```
     **正常判据**：`convert.log` 里 `--output_names` 恰好是这 3 个，且
     `out/*_tensor_info.txt` 中能查到对应 3 个张量。
     基准：2026-10 的 model-c（YOLO26s-obb / 480 / 多类 / INT8）与历史成功 job
     `jobs/20260830_104718_t_maixcam_yolo26` 的 `--output_names` **完全一致**。
   - **`--imgsz` 对 ONNX 输入不生效**：它只用于 `.pt → .onnx` 导出（`converter/yolo/export.py`）；
     分辨率实际取自 ONNX 本身。写与 ONNX 一致的值只是为了记录清晰。

## 标准转换流程（已实测通过）
1. **准备 ONNX**
   - 320（现有）：直接用 `jobs/*/X.onnx`。
   - **640（推荐，解决混淆主因）**：需用户用训练 `.pt` **重新训练/导出 640**（仅对 320 训练的 `.pt` 做 `yolo export imgsz=640` 不够，权重学的是 320 特征）。命令：
     ```
     yolo export model=X.pt format=onnx imgsz=640
     ```
     本机通常无 `.pt`，需用户端导出后给 ONNX。

2. **跑转换**（CLI，加 `--fast` 跳过 docker 内坏掉的 onnxruntime 参考推理）：
   ```
   python convert_cli.py \
     --model X.onnx \
     --dataset <job目录>/coco \
     --model-name X_bf16 \
     --target maixcam --yolo-version yolo11 --task obb \
     --imgsz 640 640 \
     --images-num 20 \
     --precision bf16 \
     --fast \
     --job-dir jobs/test_bf16_X
   ```
   - 网页方式：转换器页选 YOLO11 + OBB + maixcam + 640×640 + 精度 BF16 + 勾 fast。
   - **若只追求提速（INT8，默认档）**：去掉 `--precision bf16`（默认即 int8，或显式 `--precision int8`），其余参数完全相同，OBB 六输出流程一致。INT8 比 BF16 快约 2 倍、模型体积小一半（640 约 12MB vs 25MB）；代价仅失 BF16 那点动态范围，640 分辨率优势保留。已实测：`convert_cli.py ... --precision int8` 从同一 640 ONNX 成功产出 `X.cvimodel`（11.7MB）+ `X.mud`（指向 `X.cvimodel`，无 `_bf16` 后缀）。

3. **产物**：`out/X_bf16.cvimodel`（约 21MB，张量全 `bf16`/`qscale 1.0` = 动态范围与浮点一致、无激活量化压缩；但 BF16 仍有 8-bit 尾数截断，**是"接近 FP32"而非比特级无损**）+ `out/X_bf16.mud`（`type=obb`，标签齐全）。

4. **上机部署**：两文件放进模型目录，改 `main.py` 的 `MODEL_FILE` 指向 `X_bf16`（与 INT8 版并存，互不覆盖），设备上对比测试混淆率与帧率。

## 踩坑清单
- **【环境·必踩】解释器与 PATH**（2026-10 实测，两个都踩了）：
  - **解释器必须用项目自带 venv**：`<converter_root>\venv\Scripts\python.exe`。
    系统 Python（Microsoft Store 3.13.13）**没有可用的 onnx** —— 包能装上，但 `import onnx` 报
    `ImportError: DLL load failed while importing onnx_cpp2py_export`；venv 内 onnx 1.22.0 /
    onnxruntime 1.28.0 / numpy 2.5.1 齐全，直接可用。
  - **PATH 必须含 docker**（Docker Desktop 是 **per-user 安装**，不在默认路径）：
    `export PATH="$PATH:/c/Users/<you>/AppData/Local/Programs/DockerDesktop/resources/bin"`
    否则转换器内部 `subprocess.Popen(["docker", ...])` 报
    `FileNotFoundError: [WinError 2] 系统找不到指定的文件`，栈顶落在
    `converter/backends/maixcam_tpumlir.py` 的 `run_and_log`。**这只与 PATH 有关，与模型无关。**
  - 完整可用前缀：
    `export PATH="/c/Windows/System32:/c/Windows:/usr/bin:/bin:/c/Users/<you>/AppData/Local/Programs/DockerDesktop/resources/bin"`
- **docker 内参考推理挂了**：`model_transform.py --test_input` 阶段 `model_runner.py` 报 `UnboundLocalError: session`（镜像 onnxruntime 健全性检查坏，与模型无关）→ 加 `--fast` 绕过，部署不受影响。
- **Windows 路径**：本机 python 不认 `/e/` POSIX 路径（会被当成 `C:/e/...`），用 `./work/` 绝对路径。
- **转换器需重启**：改 `node_profiles.py` / `maixcam_tpumlir.py` 后必须重启 `uvicorn` 才生效；报错看 `jobs/<jobid>/convert.log` + `job.json`。
- **ONNX 自身问题**：若某 `.onnx`（如 model-y）在 Step1 报 `onnxsim opt failed` / `session` 未绑定，是源文件问题，换一个能被证明转成功的 ONNX 验证管线。
- **【前置条件·转换前必查】Docker Desktop 必须已启动**：`run_tpumlir_job` 走 docker，若 daemon 未运行会报 `failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine; ... The system cannot find the file specified`，转换直接失败（Status: failed）。修复：先启动 `C:\Program Files\Docker\Docker\Docker Desktop.exe`（或系统托盘启动 Docker Desktop），等 `docker info` 正常返回再重跑转换（会新建 job 目录，旧的不受影响，可删）。这是首次转换失败最常见原因，与模型/ONNX 无关。
- **【设备端 OOM 必踩】BF16 模型上 cv181x 必关 `dual_buff`**：`nn.YOLO11(model, dual_buff=True)` 会分配**两块**输入缓冲。报错特征：`ion alloc failed, size = 4915200`（=640×640×3×4，一块 FP32 输入张量字节数）→ `alloc ion failed ERROR in cvi_device_mem.cpp 451` → Assertion failed，发生在 `nn.YOLO11(...)` 初始化。根因：cv181x 的 ION 共享内存池仅 `13107200`=12.5MB；BF16 模型张量比 INT8 占内存大，`dual_buff=True` 两块输入(共≈9.4MB)+输出/中间张量直接撑爆。**修复**：改成 `dual_buff=False`（只留一块，省~4.7MB）。INT8 模型(8bit张量)能塞下双缓冲，换 BF16 必爆——这是 BF16 部署的硬坑，凡是 cv181x 跑 BF16 OBB 都把 dual_buff 关掉。兜底（若关双缓冲仍 OOM）：降输入分辨率 或 改用 INT8 版（激活张量可1字节，ION 占用骤降）。本系统 COOLDOWN_TIME=1.5s 节拍不靠吞吐，关双缓冲无影响。

## 精度验证方法（必须实测，勿断言"无损耗"）
**BF16 转换成功 ≠ 精度一定高。** 以下任一方式才能定性/定量确认混淆是否消除：
- **方案 A（最可信，需设备上机）**：准备 30–100 张含目标形状的**标注测试图**（尤其形状F/形状B、形状D/形状A成对的难例），分别跑 INT8 版与 BF16 版，统计这两对形状的**混淆矩阵**（误判率），直接对比。
- **方案 B（无硬件也可，MLIR 级参考）**：用 TPU-MLIR 的 `model_runner.py` 跑 `_tpu.mlir`（INT8/Bf16 各一）对同一批图输出，与 float ONNX（onnxruntime CPU）的参考输出比 cosine / 比框，近似判断精度损失落在哪。
- **关键认知**：相似形状的区分主要依赖**几何细节（边数/顶点数/棱角锐度）**，320 分辨率会把这些细节压没 → **640 分辨率往往是比 BF16 更重的杠杆**；BF16 主要补偿"量化噪声导致的细粒度退化"，救不回分辨率已丢弃的信息。所以结论必须建立在"640 + BF16 实测混淆矩阵"上，单凭 BF16 转换成功不能下"不混淆"的结论。

## 理论最精准配置（cv181x 天花板）
当目标是"这台机器上理论上能达到的最高精度"时，按以下组合逼近上限：

1. **量化：BF16（已封顶）**。cv181x 上 BF16 是可用的最高精度档；mix 因缺 `run_qtable.py` 不可用。BF16 **无需校准集**（INT8 才要 100+ 张），相对 INT8 有额外工程优势。

2. **分辨率：拉到 NPU 内存与速度能承受的最大**。cv181x **没有硬编码最大输入限制**（非正方形、>320 都验证可跑），约束来自 256–512MB DDR + 1 TOPS 算力。
   - 阶梯：640（基准 6–9 FPS）→ **960（理论最精准推荐，~3–4.5 FPS OBB BF16）** → 1280（极致 ~1–2 FPS，仅当传送带节拍 >1s/物体时用）。
   - **铁律：训练 `imgsz` 必须 == 导出/推理 `imgsz`**，否则权重与尺度不匹配、精度反崩。重训时即定目标尺寸。

3. **模型规模：YOLO11s-obb 优先于 n**。COCO mAP：n=39.5 / s=47.0 / m=51.5。s 比 n 高 ~7.5 点且 cv181x 勉强可跑（21.5 vs 6.5 GFLOPs，约慢 3x）；m/l 在 1 TOPS 基本不可用。理论最精准用 s，求稳用 n。

4. **训练质量（最终杠杆）**：几何多样性（不同边数/姿态/远近/遮挡）、难例挖掘、充足 epoch（lr 余弦尾压平再停）。若 960+BF16 仍偶混，根因在"模型对边数学得不够"，需加样本而非继续调量化。

5. **后处理**：NMS IoU（当前 0.45）与置信度阈值用验证集实测调优。

**速度现实（OBB BF16, cv181x）**：640≈6–9 / 960≈3–4.5 / 1280≈1–2 FPS。先定现场节拍容许的最低 FPS，再反推最大分辨率——"理论最精准"与"可用"需按节拍权衡。

## 验证结果（本次会话实测）
- model-a OBB **BF16 转换成功**：6 输出，`qscale=1.0`，mud 正确指向 `*_bf16.cvimodel`。
- mix 混合精度：本机镜像缺 `run_qtable.py` → 暂不可用。
- 交付物示例：`./work/aaaproject\angle\model-a_bf16.cvimodel` + `model-a_bf16.mud`（可直接上机对比）。

## 决策建议（给用户的）
- 传送带慢 / 精度第一 → **640 BF16**（近原生，6–9 FPS）。
- 传送带快 / 不能漏检 → **640 INT8**（12–16 FPS，转换器修复后 INT8 已正确）。
- 不要再用 320：分辨率是混淆主因，BF16 也救不回 320 丢失的棱角信息。
