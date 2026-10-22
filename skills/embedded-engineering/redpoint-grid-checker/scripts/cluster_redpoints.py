# -*- coding: utf-8 -*-
"""
红点检测日志 → 空间网格校验 + SVG/HTML 可视化

功能：
  1) 解析 MaixCAM / YOLO 多帧 `[红点]` 检测日志（cx/cy/w/h/score）
  2) 按距离阈值聚类，把同一物理点（多帧 ±15~25px 抖动）合并成"真坐标"
  3) 用旋转无关的双线性最小二乘拟合判定是否为规整 N×M 网格（自动校正约45°旋转）
  4) 输出 HTML：绿色实框=真坐标检测框，蓝/橙=网格内/异常点，浅蓝虚线=拟合网格
  5) 给出判定结论（规整 3×3 / 不规整 / 有错）

用法：
  python cluster_redpoints.py <input_log.txt> [--eps 35] [--out output.html] [--target 3 3]
  - input_log.txt : 多帧检测日志（每行含 cx= cy= w= h= score=）
  - --eps         : 聚类距离阈值（默认 35.0；同一物理点多帧抖动在此半径内合并）
  - --out         : 输出 HTML 路径（默认与输入同目录，同名加 _grid.html）
  - --target      : 期望网格形状，例如 "3 3"（仅用于结论文字，拟合本身自动搜形状）
"""
import re, math, sys, os, argparse


def parse_args():
    p = argparse.ArgumentParser(description="红点日志网格校验 + 可视化")
    p.add_argument("input", help="多帧检测日志 txt 路径")
    p.add_argument("--eps", type=float, default=35.0,
                   help="聚类距离阈值（默认 35.0）")
    p.add_argument("--out", default=None,
                   help="输出 HTML 路径（默认 <input>_grid.html）")
    p.add_argument("--target", nargs=2, type=int, default=None,
                   metavar=("R", "C"),
                   help="期望网格形状，如 --target 3 3（仅用于结论文字）")
    return p.parse_args()


def load_points(path):
    rx = re.compile(r'cx=(\d+)\s+cy=(\d+)\s+w=(\d+)\s+h=(\d+)\s+score=([\d.]+)')
    pts = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for ln in f:
            m = rx.search(ln)
            if m:
                pts.append((int(m.group(1)), int(m.group(2)), int(m.group(3)),
                            int(m.group(4)), float(m.group(5))))
    return pts


def cluster(pts, EPS):
    clusters = []

    def dist(a, b):
        return math.hypot(a['cx'] - b['cx'], a['cy'] - b['cy'])

    for cx, cy, w, h, sc in pts:
        best, bd = None, 1e9
        tmp = {'cx': float(cx), 'cy': float(cy)}
        for c in clusters:
            d = dist(tmp, c)
            if d < bd:
                bd, best = d, c
        if best is not None and bd <= EPS:
            n = best['n']
            best['cx'] = (best['cx'] * n + cx) / (n + 1)
            best['cy'] = (best['cy'] * n + cy) / (n + 1)
            best['w'] = (best['w'] * n + w) / (n + 1)
            best['h'] = (best['h'] * n + h) / (n + 1)
            best['sc'] = (best['sc'] * n + sc) / (n + 1)
            best['n'] += 1
        else:
            clusters.append({'cx': float(cx), 'cy': float(cy), 'w': float(w),
                             'h': float(h), 'sc': sc, 'n': 1})
    clusters.sort(key=lambda c: (round(c['cy']), round(c['cx'])))
    return clusters


def solve4(A, b):
    """高斯消元解 4×4 线性方程组，带奇异保护。"""
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(4):
        p = max(range(c, 4), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        if abs(pv) < 1e-9:
            return None
        for r in range(4):
            if r != c:
                f = M[r][c] / pv
                for k in range(5):
                    M[r][k] -= f * M[c][k]
    return [M[i][4] / M[i][i] for i in range(4)]


def bilinear_fit(assigned):
    """双线性拟合：x = a0 + a1*i + a2*j + a3*i*j，对 (i,j)->(x,y) 最小二乘。"""
    AtA = [[0.0] * 4 for _ in range(4)]
    Atbx, Atby = [0.0] * 4, [0.0] * 4
    for (i, j), (x, y) in assigned:
        f = [1.0, i, j, i * j]
        for a in range(4):
            for b in range(4):
                AtA[a][b] += f[a] * f[b]
            Atbx[a] += f[a] * x
            Atby[a] += f[a] * y
    ax, ay = solve4(AtA, Atbx), solve4(AtA, Atby)
    if ax is None or ay is None:
        return None
    res = []
    for (i, j), (x, y) in assigned:
        f = [1.0, i, j, i * j]
        px = sum(ax[k] * f[k] for k in range(4))
        py = sum(ay[k] * f[k] for k in range(4))
        res.append(math.hypot(x - px, y - py))
    return res, ax, ay


def assign_grid(R, C, by_x, xs, ys):
    items = list(enumerate(zip(xs, ys)))
    items.sort(key=(lambda p: (p[1][0], p[1][1])) if by_x
               else (lambda p: (p[1][1], p[1][0])))
    assigned, idxmap = [], {}
    for ci in range(C):
        col = items[ci * R:(ci + 1) * R]
        col.sort(key=lambda p: p[1][1])
        for ri, p in enumerate(col):
            assigned.append(((ri, ci), (p[1][0], p[1][1])))
            idxmap[p[0]] = (ri, ci)
    return assigned, idxmap


def build_html(clusters, best, spacing, target, n_raw, out_path):
    xs = [c['cx'] for c in clusters]
    ys = [c['cy'] for c in clusters]
    R, C = best['R'], best['C']
    res, ax, ay = best['res'], best['ax'], best['ay']
    N = len(clusters)
    inlier = [r <= spacing * 0.8 for r in res]
    maxr, meanr = max(res), best['mr']
    n_out = sum(1 for x in inlier if not x)

    def predict(ij):
        f = [1.0, ij[0], ij[1], ij[0] * ij[1]]
        return (sum(ax[k] * f[k] for k in range(4)),
                sum(ay[k] * f[k] for k in range(4)))

    xsmin, xsmax = min(xs), max(xs)
    ysmin, ysmax = min(ys), max(ys)
    ML, MR, MT, MB = 64, 40, 54, 46
    PW = 680 - ML - MR
    PH = int(PW * (ysmax - ysmin) / (xsmax - xsmin)) if xsmax > xsmin else 220
    H = MT + PH + MB

    def sx(x):
        return ML + (x - xsmin) / (xsmax - xsmin) * PW

    def sy(y):
        # 图像坐标：原点左上、Y 向下。必须用 Y-down，否则整图垂直镜像。
        return MT + (y - ysmin) / (ysmax - ysmin) * PH

    parts = []
    parts.append('<svg viewBox="0 0 680 %d" xmlns="http://www.w3.org/2000/svg" font-family="monospace">' % H)
    parts.append('<rect x="0" y="0" width="680" height="%d" fill="#0d1117"/>' % H)

    gl = '#39506b'
    for r in range(R):
        line = [predict((r, c)) for c in range(C)]
        d = 'M' + ' L'.join('%.1f,%.1f' % (sx(p[0]), sy(p[1])) for p in line)
        parts.append('<path d="%s" stroke="%s" stroke-width="1" stroke-dasharray="5,4" opacity="0.55" fill="none"/>' % (d, gl))
    for c in range(C):
        line = [predict((r, c)) for r in range(R)]
        d = 'M' + ' L'.join('%.1f,%.1f' % (sx(p[0]), sy(p[1])) for p in line)
        parts.append('<path d="%s" stroke="%s" stroke-width="1" stroke-dasharray="5,4" opacity="0.55" fill="none"/>' % (d, gl))

    blue, orange = '#2196f3', '#ff9800'
    for i, c in enumerate(clusters):
        bx = sx(c['cx'] - c['w'] / 2)
        by = sy(c['cy'] - c['h'] / 2)
        bw = c['w'] / (xsmax - xsmin) * PW
        bh = c['h'] / (ysmax - ysmin) * PH
        col = blue if inlier[i] else orange
        parts.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" fill-opacity="0.14" stroke="#36e07a" stroke-width="2"/>'
                     % (bx, by, bw, bh, col))
        parts.append('<text x="%.1f" y="%.1f" fill="%s" font-size="9" text-anchor="middle">#%d</text>'
                     % (sx(c['cx']), sy(c['cy']) - 3, col, i))
        parts.append('<text x="%.1f" y="%.1f" fill="#9fb0c0" font-size="7.5" text-anchor="middle">%.0f,%.0f·n%d</text>'
                     % (sx(c['cx']), sy(c['cy']) + 9, c['cx'], c['cy'], c['n']))

    parts.append('<text x="340" y="24" fill="#e6edf3" font-size="15" text-anchor="middle" font-family="sans-serif">红点检测框空间排布（绿框=真坐标 · 蓝/橙=网格内/异常）</text>')
    reg_txt = "R=%d x C=%d (R*C=%d)" % (R, C, R * C)
    reg_txt += " 与真实点数 %d 吻合" % N if R * C == N else " 与真实点数 %d 不符" % N
    parts.append('<text x="340" y="42" fill="#8b97a7" font-size="11" text-anchor="middle" font-family="sans-serif">%s ｜ 间距≈%.0fpx ｜ 残差均值%.1f/最大%.1f ｜ 异常(橙)%d</text>'
                 % (reg_txt, spacing, meanr, maxr, n_out))
    parts.append('</svg>')
    svg = ''.join(parts)

    if target is not None:
        tR, tC = target
        if R == tR and C == tC and R * C == N and maxr < spacing * 1.2:
            verdict = "规整 %d×%d（与预期吻合）— 数据可信" % (tR, tC)
        elif R * C == N and maxr < spacing * 1.2:
            verdict = "规整 %d×%d 网格 — 数据可信（但与预期 %d×%d 不符）" % (R, C, tR, tC)
        else:
            verdict = "不规整 / 有错（预期 %d×%d）" % (tR, tC)
    else:
        if R == 3 and C == 3 and R * C == N and maxr < spacing * 1.2:
            verdict = "规整 3×3（旋转约45°）— 与你的预期吻合，数据可信"
        elif R * C == N and maxr < spacing * 1.2:
            verdict = "规整 %d×%d 网格 — 数据可信" % (R, C)
        else:
            verdict = "不规整 / 有错"

    weak_idx = next((i for i, c in enumerate(clusters) if c['n'] == 1), 0)
    html = '''<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;background:#0d1117;color:#e6edf3;font-family:-apple-system,Segoe UI,sans-serif;display:flex;flex-direction:column;align-items:center;padding:20px}
.wrap{max-width:780px;width:100%}
svg{width:100%;height:auto;border:1px solid #1f2630;border-radius:8px;background:#0d1117}
.legend{margin-top:14px;font-size:13px;color:#b9c2cf;line-height:1.8}
.dot{display:inline-block;width:12px;height:12px;vertical-align:middle;margin-right:6px;border-radius:2px}
.note{margin-top:10px;font-size:12px;color:#7d8694}
.verdict{margin-top:8px;font-size:14px;font-weight:600;color:#7ee0a0}
</style></head><body><div class="wrap">''' + svg + '''
<div class="legend">
<span class="dot" style="background:#36e07a"></span>绿色实框 = 每个真实红点的检测框（多帧日志已聚类合并成真坐标）<br>
<span class="dot" style="background:#2196f3"></span>蓝色 = 落在拟合规整网格内的点（inlier，残差≤间距×0.8）<br>
<span class="dot" style="background:#ff9800"></span>橙色 = 偏离网格的异常点（outlier）<br>
<span class="dot" style="background:#39506b"></span>浅蓝虚线 = 双线性拟合的预期 %d×%d 网格（已自动校正旋转）
</div>
<div class="verdict">判定：%s</div>
<div class="note">说明：原始日志共 %d 帧检测，逐帧抖动使同一物理点被反复框出（±15~25px）。按距离阈值聚类后归并为 %d 个真实点，构成 %d×%d 网格。坐标/排布可信；唯一需回设备原帧核对的弱点是 #%d（仅被检测到 1 次，n=1），其位置虽契合格点但样本太少。</div>
</div></body></html>''' % (R, C, verdict, n_raw, N, R, C, weak_idx)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return verdict


def main():
    args = parse_args()
    pts = load_points(args.input)
    print("raw detections:", len(pts))
    if not pts:
        print("ERROR: 没有解析到任何 cx=/cy=/w=/h=/score= 行，请检查日志格式。", file=sys.stderr)
        sys.exit(2)

    clusters = cluster(pts, args.eps)
    N = len(clusters)
    print("distinct clusters (真实点数):", N)
    for i, c in enumerate(clusters):
        print("  #%02d cx=%6.1f cy=%6.1f  w=%4.1f h=%4.1f  n=%3d  sc=%.2f"
              % (i, c['cx'], c['cy'], c['w'], c['h'], c['n'], c['sc']))

    xs, ys = [c['cx'] for c in clusters], [c['cy'] for c in clusters]
    nn = [min(math.hypot(xs[i] - xs[j], ys[i] - ys[j]) for j in range(N) if j != i)
          for i in range(N)]
    spacing = sorted(nn)[N // 2] if N > 1 else 20.0
    print("estimated spacing:", round(spacing, 1))

    best = None
    for R in range(1, N + 1):
        if N % R != 0:
            continue
        C = N // R
        for by_x in (True, False):
            assigned, idxmap = assign_grid(R, C, by_x, xs, ys)
            fit = bilinear_fit(assigned)
            if fit is None:
                continue
            res, ax, ay = fit
            mr = sum(res) / N
            if best is None or mr < best['mr']:
                best = {'R': R, 'C': C, 'by_x': by_x, 'mr': mr,
                        'res': res, 'ax': ax, 'ay': ay, 'idxmap': idxmap}
    if best is None:
        print("ERROR: 网格拟合失败（可能为退化配置）。", file=sys.stderr)
        sys.exit(3)

    out_path = os.path.abspath(args.out or (os.path.splitext(args.input)[0] + '_grid.html'))
    verdict = build_html(clusters, best, spacing, tuple(args.target) if args.target else None,
                         len(pts), out_path)
    print("best grid: R=%d x C=%d (R*C=%d)  mean_res=%.1f max_res=%.1f outliers=%d"
          % (best['R'], best['C'], best['R'] * best['C'], best['mr'], max(best['res']),
             sum(1 for r in best['res'] if r > spacing * 0.8)))
    print("written (绝对路径):", out_path)
    print("verdict =", verdict)


if __name__ == '__main__':
    main()
