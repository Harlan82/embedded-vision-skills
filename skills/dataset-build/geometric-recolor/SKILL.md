---
name: geometric-recolor
description: 将中性灰背景上单色/多色几何体的照片批量换色到指定目标色（赤橙黄绿青蓝紫粉等），用于视觉模型训练数据增强。支持多物体检测、形状B拉伸、颜色准确性归一化、自动质量校验。当用户有"把图片里的物体换颜色 / 生成多颜色标注数据 / 彩色物体换色增强"类需求时使用。
---

# 几何体批量换色（训练数据增强）

## 适用场景
- 源图：灰/白/中性背景上的彩色物体（3D打印件、积木、产品图），单物体或多物体
- 目标：把物体换到 N 种指定颜色，生成数百~数千张标注用图
- 要求：保留原图 3D 光影立体感、背景不被染色、颜色准确鲜艳、可加形状/姿态增强

## 核心 Pipeline（按顺序）
1. **前景检测** → 2. **精确换色** → 3. **质量校验** → 4. **形状增强** → 5. **分布裁剪**

---

## 1. 前景检测（关键：多物体安全）
不能用"只取最大连通域"——多物体图会漏掉其余物体。

```python
def build_foreground_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:,:,1].astype(np.float32)
    b,g,r = cv2.split(img.astype(np.float32))
    cf = np.maximum.reduce([r,g,b]) - np.minimum.reduce([r,g,b])  # colorfulness
    sat_mask = (s > 28).astype(np.uint8)*255
    cf_mask  = (cf > 22).astype(np.uint8)*255
    combined = cv2.bitwise_or(sat_mask, cf_mask)
    k_o, k_c = np.ones((3,3),np.uint8), np.ones((7,7),np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k_o)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_c)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    mask = np.zeros_like(cleaned)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= 200:   # 保留所有>200px色块
            mask[labels==i] = 255
    return mask
```
- 阈值 s>28 / cf>22 对灰色背景鲁棒；MIN_COMPONENT_AREA=200 过滤噪点
- 多物体、物体带独立小部件都能完整覆盖

## 2. 精确换色（颜色准确的关键，必看坑）
**坑**：只改 H 通道、S/V 继承原图 → 原物若是品红(V≈180)，换到黄/青(需V≥240)会发暗发闷，颜色不正。
**正解**：H/S/V 三通道按目标色设定标准值，V 用"亮度比例"归一化保光影，S 用"比例"归一化保材质质感。

```python
def recolor_accurate(img, mask, t_h, t_s, t_v, bright=1.0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h,s,v = hsv[:,:,0].copy(), hsv[:,:,1].copy(), hsv[:,:,2].copy()
    fg = mask > 128
    h[fg] = t_h                                   # 色相精确
    ov = v[fg]; mv = ov.mean()
    if mv > 5:
        v[fg] = np.clip(ov * (t_v/mv) * bright, 35, 255)   # 保光影比例
    else:
        v[fg] = t_v
    os = s[fg]; ms = os.mean()
    if ms > 5:
        s[fg] = np.clip(os * (t_s/ms), 55, 255)            # 保材质纹理比例
    else:
        s[fg] = t_s
    return cv2.cvtColor(np.stack([h,s,v],-1).astype(np.uint8), cv2.COLOR_HSV2BGR)
```

**目标色直接用 HSV 定义**（OpenCV: H 0-179, S/V 0-255）：
```python
TARGET_HSV = {
    "chi":(0,240,215), "cheng":(16,240,235), "huang":(28,240,242),
    "lv":(60,230,210), "qing":(90,235,230), "lan":(117,225,215),
    "zi":(140,205,185), "fen":(165,120,240),
}
```
- 粉(fen)用低饱和高亮度才像"粉"；紫(zi)用 H≈140 区别于品红
- 调色口诀：黄/青要亮(V高)、粉要浅(S低V高)、红绿蓝适中

## 3. 质量校验（自动删废图）
输出图重算前景，检查三项，任一不过即删：
- **目标色覆盖率**：前景 hue 与目标 hue 距离≤20 的像素占比 ≥ 0.70
- **原色残留**：非品红目标时，前景中残存品红(hue 138-172)占比 ≤ 0.15
  - ⚠ 若目标本身接近品红(H∈[130,178]，即 zi/fen)，**跳过此项**否则误删
- **背景污染**：非前景区饱和度>45 的像素占比 ≤ 0.02

```python
def hue_distance(h1,h2):
    d = np.abs(h1-h2); return np.minimum(d, 180-d)   # 必须np.minimum，min()对数组报错
```

## 4. 形状增强：形状B拉伸
对近似形状H(包围盒宽高比≈1、单连通域)的图做非均匀缩放：
```python
def stretch_image(img, sx, sy):
    h,w = img.shape[:2]
    return cv2.resize(img, (max(int(w*sx),w), max(int(h*sy),h)), cv2.INTER_LINEAR)
```
常用 STRETCH_PARAMS = [(1.35,1.00,"_hx"),(1.00,1.25,"_vy"),(1.20,1.15,"_hv")]
拉伸后**重新** build_foreground_mask（尺寸变了）。

## 5. 分布裁剪到目标总数
生成数 = 源图数 × 颜色数 × 变体数 × (1+拉伸倍率)，往往远超目标。
- 各 bucket(原图/拉伸) 随机 shuffle 后按目标占比取样
- 目标占比示例 DIST_TARGET={"orig":0.5,"stretch":0.5}
- 某类不足则从其他类补足 deficit

---

## 完整参考脚本
见同目录 `recolor_ref.py`（精确换色 + 拉伸 + 校验 + 裁剪）。

## 命名规范（建议）
`原图stem_颜色key[_b/_d].jpg` = 原图基准/微亮/微暗
`原图stem_stretch_hx/vy/hv_颜色key[_b/_d].jpg` = 拉伸形状B变体
标注时按文件名即可拿颜色标签与形状类别。

## 常见失败与排查
| 现象 | 原因 | 解决 |
|---|---|---|
| 黄/青发暗发闷 | 只改H继承原V | 用 recolor_accurate 归一化 V |
| 紫/粉被全删 | 当成原色残留 | 目标H∈[130,178]时跳过残留检查 |
| 多物体只换一个 | 只取最大连通域 | 保留所有>min_area组件 |
| 背景被染色 | 阈值过低吃进灰背景 | s>28 且 cf>22 双条件 + 形态学清理 |
| numpy bool 报错 | hue_distance 用 min() | 改用 np.minimum |
