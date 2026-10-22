# -*- coding: utf-8 -*-
"""
红点日志 → N 个物理点的最优精确坐标（实测中心 + 理想网格位），按左上→右下排序。
同时作为可视化脚本(visualize_redpoints.py)的**共享计算核心**，保证两套输出完全一致。

算法要点（最优/无偏估计）：
  - 同一物理点多帧检测 ≈ 高斯抖动 → 样本均值(质心) 是位置的最大似然/最小方差估计。
  - 双线性拟合给出"若严格成规则网格"的理想坐标，作为规则化参照（会平滑掉真实偏差）。
  - 时间覆盖率(first..last 行 / 总行) 判定该点是否全程存在：≈0% 即单帧闪现(疑似误检/瞬态)。
  - 旋转约45°的网格在 by_x=True / by_x=False 下拟合残差完全相等（平局），(row,col)编号本就
    无物理"起点"；此处固定确定性约定（见 search_best 的 tie-break，平局取 by_x=False），
    且两脚本共用本模块 → 输出必然一致，不会一份日志跑出两套编号。
"""
import re, math, json, os, sys

RX = re.compile(r'cx=(\d+)\s+cy=(\d+)\s+w=(\d+)\s+h=(\d+)\s+score=([\d.]+)')
EPS = 35.0
DEFAULT_PATH = r'./work\redpoints_log.txt'


def load(path):
    det = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for i, ln in enumerate(f.read().splitlines()):
            m = RX.search(ln)
            if m:
                det.append((i + 1, int(m.group(1)), int(m.group(2)),
                            int(m.group(3)), int(m.group(4)), float(m.group(5))))
    return det


def cluster(pts, eps=EPS):
    clusters = []
    for ln, cx, cy, w, h, sc in pts:
        best, bd = None, 1e9
        for c in clusters:
            d = math.hypot(c['cx'] - cx, c['cy'] - cy)
            if d < bd:
                bd, best = d, c
        if best is not None and bd <= eps:
            n = best['n']
            best['cx'] = (best['cx'] * n + cx) / (n + 1)
            best['cy'] = (best['cy'] * n + cy) / (n + 1)
            best['w'] = (best['w'] * n + w) / (n + 1)
            best['h'] = (best['h'] * n + h) / (n + 1)
            best['sc'] = (best['sc'] * n + sc) / (n + 1)
            best['n'] += 1
            best['members'].append((cx, cy))
            best['lines'].append(ln)
        else:
            clusters.append({'cx': float(cx), 'cy': float(cy), 'w': float(w),
                             'h': float(h), 'sc': sc, 'n': 1,
                             'members': [(cx, cy)], 'lines': [ln]})
    # 不在此重排序：idxmap 以"首次出现顺序"为键，(row,col)编号须与可视化一致
    return clusters


def solve4(A, b):
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


def assign(R, C, by_x, xs, ys):
    items = list(enumerate(zip(xs, ys)))
    items.sort(key=(lambda p: (p[1][0], p[1][1])) if by_x
               else (lambda p: (p[1][1], p[1][0])))
    idxmap = {}
    for ci in range(C):
        col = items[ci * R:(ci + 1) * R]
        col.sort(key=lambda p: p[1][1])
        for ri, p in enumerate(col):
            idxmap[p[0]] = (ri, ci)
    return idxmap


def fit(assigned):
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


def search_best(clusters):
    """搜索最优 R×C（R*C==ncl）取双线性残差最小；平局(Δ≤1e-9)固定取 by_x=False。"""
    ncl = len(clusters)
    xs = [c['cx'] for c in clusters]
    ys = [c['cy'] for c in clusters]
    best = None
    for R in range(1, ncl + 1):
        if ncl % R != 0:
            continue
        C = ncl // R
        for by_x in (True, False):
            idxmap = assign(R, C, by_x, xs, ys)
            assigned = [((idxmap[i][0], idxmap[i][1]), (clusters[i]['cx'], clusters[i]['cy']))
                        for i in range(ncl)]
            r = fit(assigned)
            if r is None:
                continue
            res, ax, ay = r
            mr = sum(res) / ncl
            # 确定性 tie-break：明显更小则替换；平局(Δ≤1e-9)时优先 by_x=False
            if (best is None or mr < best['mr'] - 1e-9
                    or (abs(mr - best['mr']) <= 1e-9 and (not by_x) and best['by_x'])):
                best = {'R': R, 'C': C, 'by_x': by_x, 'mr': mr,
                        'res': res, 'ax': ax, 'ay': ay, 'idxmap': idxmap}
    return best


def compute_coords(clusters, best, n_raw):
    """每个簇算出：实测质心(±σ)、理想网格位、n、时间覆盖率、置信。按 (row,col) 左上→右下排序。"""
    ax, ay = best['ax'], best['ay']
    rows = []
    for i, c in enumerate(clusters):
        xs = [m[0] for m in c['members']]
        ys = [m[1] for m in c['members']]
        sx = math.sqrt(sum((x - c['cx']) ** 2 for x in xs) / c['n'])
        sy = math.sqrt(sum((y - c['cy']) ** 2 for y in ys) / c['n'])
        lines = sorted(c['lines'])
        first, last = lines[0], lines[-1]
        cov = (last - first) / n_raw if n_raw else 0.0
        ri, ci = best['idxmap'][i]
        f = [1.0, ri, ci, ri * ci]
        gx = sum(ax[k] * f[k] for k in range(4))
        gy = sum(ay[k] * f[k] for k in range(4))
        conf = 'low' if cov < 0.05 else ('medium' if c['n'] < 120 else 'high')
        rows.append({'idx': i, 'RC': (ri + 1, ci + 1), 'cx': c['cx'], 'cy': c['cy'],
                     'sx': sx, 'sy': sy, 'gx': gx, 'gy': gy, 'n': c['n'],
                     'cov': cov, 'conf': conf})
    rows.sort(key=lambda d: (d['RC'][0], d['RC'][1]))
    return rows


def spacing_estimate(clusters):
    N = len(clusters)
    if N < 2:
        return 20.0
    xs = [c['cx'] for c in clusters]
    ys = [c['cy'] for c in clusters]
    nn = [min(math.hypot(xs[i] - xs[j], ys[i] - ys[j]) for j in range(N) if j != i)
          for i in range(N)]
    return sorted(nn)[N // 2]


def run(path):
    """完整流水线：加载→聚类→网格拟合→坐标计算。返回供可视化复用的全部数据。"""
    det = load(path)
    n_raw = len(det)
    clusters = cluster(det)
    ncl = len(clusters)
    best = search_best(clusters)
    coords = compute_coords(clusters, best, n_raw)
    spacing = spacing_estimate(clusters)
    return {'path': path, 'n_raw': n_raw, 'clusters': clusters, 'ncl': ncl,
            'best': best, 'coords': coords, 'spacing': spacing}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    res = run(path)
    N, ncl = res['n_raw'], res['ncl']
    R, C = res['best']['R'], res['best']['C']
    print("检测总数=%d  聚类数=%d  最优网格=%dx%d  残差均值=%.2f" % (N, ncl, R, C, res['best']['mr']))
    print("%-3s %-7s %8s %8s %8s %8s %8s %8s %5s %7s %-8s" %
          ("序", "网格", "实测X", "实测Y", "±σx", "±σy", "理想X", "理想Y", "n", "覆盖%", "置信"))
    for k, d in enumerate(res['coords'], 1):
        print("%-3d %-7s %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %5d %6.1f%% %-8s" %
              (k, "R%dC%d" % d['RC'], d['cx'], d['cy'], d['sx'], d['sy'],
               d['gx'], d['gy'], d['n'], d['cov'] * 100, d['conf']))

    ref = os.path.join(os.path.dirname(path), 'coords_result.json')
    with open(ref, 'w', encoding='utf-8') as f:
        json.dump({'input': os.path.basename(path), 'N': N, 'grid': '%dx%d' % (R, C),
                   'residual_mean': round(res['best']['mr'], 2),
                   'points': [{'order': d['RC'], 'cluster_id': d['idx'], 'grid_RC': d['RC'],
                               'cx': round(d['cx'], 1), 'cy': round(d['cy'], 1),
                               'sx': round(d['sx'], 1), 'sy': round(d['sy'], 1),
                               'ideal_x': round(d['gx'], 1), 'ideal_y': round(d['gy'], 1),
                               'n': d['n'], 'coverage_pct': round(d['cov'] * 100, 1),
                               'confidence': d['conf']} for d in res['coords']]},
                  f, ensure_ascii=False, indent=2)
    print("saved:", ref)


if __name__ == '__main__':
    main()
