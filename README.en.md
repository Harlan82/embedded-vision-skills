# Embedded Vision Skills

A collection of **17 Agent Skills** for embedded computer vision, MaixCAM deployment, and YOLO data engineering.

These are not tutorial summaries. Each skill is an operating manual distilled from real project failures — every conclusion maps to an actual test, every parameter is annotated with why it has that value. The single goal: **you never debug the same problem twice.**

> Full documentation in Chinese: [README.md](README.md)

---

## What's inside

| Area | Coverage |
|---|---|
| **Model conversion & quantization** | `.pt` → ONNX → MaixHub / TPU-MLIR / Pulsar2, both INT8 and BF16 precision paths |
| **Dataset construction** | Camera photos → YOLO annotations → platform-importable ZIP, with format conversion and augmentation |
| **Embedded engineering** | MaixCAM networking, offline verification, class-table refactoring, camera calibration analysis |
| **Methodology** | Source-level facts on YOLO training params, incremental improvement loops without labeled ground truth |

---

## Quick start

```bash
git clone https://github.com/Harlan82/embedded-vision-skills.git
cd embedded-vision-skills

python install.py --list              # list available skills
python install.py                     # install all to ~/.workbuddy/skills
python install.py --target ~/.claude/skills
python install.py --only pt-to-onnx-export,yolo-train-tuning
```

The installer **flattens** the nested layout into `skills/<skill-name>/`, because most skill loaders require the directory name to match the frontmatter `name`.

Manual install: copy `skills/<area>/<skill-name>/` to your skills directory, dropping the area level.

> Skill loaders that scan recursively (e.g. Claude Code's `skills/`) can use the nested layout directly.

---

## Skill inventory

### `skills/model-convert/` — Model conversion & quantization

| Skill | Problem it solves |
|---|---|
| [pt-to-onnx-export](skills/model-convert/pt-to-onnx-export/) | Export Ultralytics `.pt` to ONNX with zero-error verification. Covers which head YOLO26's one-to-one / one-to-many structure actually exports |
| [maixhub-yolo-detect-convert](skills/model-convert/maixhub-yolo-detect-convert/) | Produce ONNX accepted by MaixHub online conversion; fixes `Detect profile does not match this ONNX` |
| [maixcam-obb-bf16-convert](skills/model-convert/maixcam-obb-bf16-convert/) | [Gen-1 cv181x] BF16 high-precision path for OBB models. Includes measured ION memory analysis proving `dual_buff` must be off |
| [maixcam2-obb-int8-convert](skills/model-convert/maixcam2-obb-int8-convert/) | [Gen-2 AX620E] Pulsar2 INT8 quantization producing `.mud` + `_npu.axmodel` + `_vnpu.axmodel` |
| [quant-calib-subset-select](skills/model-convert/quant-calib-subset-select/) | Select INT8/PTQ calibration subsets that preserve **distant small-object recall** and dense scenes |

### `skills/dataset-build/` — Dataset construction

| Skill | Problem it solves |
|---|---|
| [shape-dataset-pipeline](skills/dataset-build/shape-dataset-pipeline/) | Camera photos → YOLO dataset, full pipeline with a human-diagnosis branch |
| [shape-dataset-pipeline-local](skills/dataset-build/shape-dataset-pipeline-local/) | Same, fully automatic, one command to ZIP. Runs on 4B-class local models |
| [center-object-to-yolo-zip](skills/dataset-build/center-object-to-yolo-zip/) | Annotate single-centered-object photos via HSV segmentation + geometric criteria, no SAM weights needed |
| [dataset-to-ultralytics-ndjson](skills/dataset-build/dataset-to-ultralytics-ndjson/) | Format hub: YOLO/OBB ↔ NDJSON ↔ Pascal VOC, including signed-CDN image download |
| [bbox-attr-sampling](skills/dataset-build/bbox-attr-sampling/) | Sampling rules for in-bbox attribute judgment. Center sampling fails on hollow objects; edge sampling fails worse. The fix is pixel-level background rejection |
| [geometric-recolor](skills/dataset-build/geometric-recolor/) | Batch recolor geometric objects on neutral gray backgrounds for training augmentation |

### `skills/embedded-engineering/` — Embedded engineering

| Skill | Problem it solves |
|---|---|
| [maixcam-wifi-fixed-ip](skills/embedded-engineering/maixcam-wifi-fixed-ip/) | Pin MaixCAM to a fixed wireless IP. Includes Buildroot network config and Windows hotspot ICS references |
| [maixcam-vision-offline-verify](skills/embedded-engineering/maixcam-vision-offline-verify/) | Offline smoke testing of MaixPy projects without hardware: dependency boundaries, fake inputs, protocol frame assertions |
| [label-table-reclass](skills/embedded-engineering/label-table-reclass/) | Full-chain refactor when swapping models and rewriting class tables, with 5 automated acceptance checks |
| [redpoint-grid-checker](skills/embedded-engineering/redpoint-grid-checker/) | Red-dot detection logs → point cloud → distance clustering → bilinear least-squares grid fit → optimal physical coordinates |

### `skills/methodology/` — Methodology

| Skill | Problem it solves |
|---|---|
| [yolo-train-tuning](skills/methodology/yolo-train-tuning/) | Source-level facts on batch limits, `nbs` gradient accumulation, early-stopping × `cos_lr` coupling, and why negative samples can lower mAP |
| [vision-free-truth-incremental](skills/methodology/vision-free-truth-incremental/) | How to find "free ground truth" already present in your pipeline and build a self-improving loop when you have 10¹–10² samples and no labels |

---

## Conventions

**Path placeholders.** Local machine paths have been sanitized into placeholders:

| Placeholder | Meaning |
|---|---|
| `<converter_root>` | Model converter platform root |
| `<venv>` | Your Python virtual environment |
| `./work/` | Scratch directory for models, datasets, and outputs |
| `%LOCALAPPDATA%` | Windows per-user application data |

**Platform.** Recorded primarily on **Windows + Bash sandbox**. Some pitfalls are Windows-specific (Docker Desktop paths, drive letters, PowerShell encoding); the operational logic is portable, commands are not.

**Version sensitive.** Conclusions tied to specific versions (Ultralytics 8.4.x, Pulsar2 6.0, TPU-MLIR, onnxruntime 1.28) are annotated with the tested version. Scripts favor **criteria** over hardcoded assumptions so they degrade gracefully.

**Not a universal template.** These skills target industrial sorting with fixed cameras, white trays, and single workpieces. Thresholds and geometric criteria need recalibration elsewhere — the pitfall lists and methodology still apply.

## License

[MIT](LICENSE)
