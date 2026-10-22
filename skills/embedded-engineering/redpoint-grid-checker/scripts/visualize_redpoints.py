# -*- coding: utf-8 -*-
"""
红点检测日志 → 原始分布 + 聚类真坐标 叠加可视化（透明核对用）

与 compute_coords.py 共用同一计算核心（import compute_coords.run），
保证图上标注的 (cx,cy) 与坐标表的 R/C 编号、与 coords_result.json 三者**完全一致**。

本脚本不替你下结论，而是把每一帧的原始检测点（带 ±15~25px 抖动、彼此有差异的那些点）
画成小彩点，再把聚类合并后的"真坐标"用小绿点(圈)叠上去并标注实测坐标，并画出拟合网格，
供你**自己看分布判断**。底部附按 R1C1→R3C3 排序的精确坐标表。

用法：
  python visualize_redpoints.py <input_log.txt> [--eps 35] [--out output.html]
                                [--target 3 3] [--maxpp 70]

  --maxpp : 每个簇最多画多少个原始点（子采样，省渲染；小簇全部保留以保真）
"""
import sys, os, argparse

# 与本脚本同目录的 compute_coords 作为共享计算核心
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_coords import run

PALETTE = ['#ff5252', '#40c4ff', '#ffb300', '#e040fb', '#18ffff',
           '#ff6e40', '#b2ff59', '#ff4081', '#7c4dff', '#69f0ae',
           '#ffd740', '#00e5ff']


def parse_args():
    p = argparse.ArgumentParser(description="红点原始分布 + 聚类真坐标 叠加可视化")
    p.add_argument("input")
    p.add_argument("--eps", type=float, default=35.0)
    p.add_argument("--out", default=None)
    p.add_argument("--target", nargs=2, type=int, default=None, metavar=("R", "C"))
    p.add_argument("--maxpp", type=int, default=70,
                   help="每个簇最多绘制的原始点数（子采样上限）")
    return p.parse_args()


def subsample(pts, maxp):
    n = len(pts)
    if n <= maxp:
        return pts
    stride = n / maxp
    out, i = [], 0
    while i < n:
        out.append(pts[int(i)])
        i += stride
    if out[-1] is not pts[-1]:
        out.append(pts[-1])
    return out


def build_html(clusters, best, spacing, target, n_raw, maxpp, out_path, coords):
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
    ML, MR, MT, MB = 64, 40, 70, 46
    PW = 680 - ML - MR
    PH = int(PW * (ysmax - ysmin) / (xsmax - xsmin)) if xsmax > xsmin else 220
    H = MT + PH + MB

    def sx(x):
        return ML + (x - xsmin) / (xsmax - xsmin) * PW

    def sy(y):
        # MaixCAM 检测输出为图像坐标：原点左上、Y 向下（Y 越大越靠下）。
        # 必须用 Y-down 映射，否则整图会被垂直镜像，与设备原帧上下相反。
        return MT + (y - ysmin) / (ysmax - ysmin) * PH

    parts = []
    parts.append('<svg viewBox="0 0 680 %d" xmlns="http://www.w3.org/2000/svg" font-family="monospace">' % H)
    parts.append('<rect x="0" y="0" width="680" height="%d" fill="#0d1117"/>' % H)

    # 拟合网格（浅蓝虚线，供参照是否成规整网格）
    gl = '#2f4a6b'
    for r in range(R):
        line = [predict((r, c)) for c in range(C)]
        d = 'M' + ' L'.join('%.1f,%.1f' % (sx(p[0]), sy(p[1])) for p in line)
        parts.append('<path d="%s" stroke="%s" stroke-width="1" stroke-dasharray="5,4" opacity="0.5" fill="none"/>' % (d, gl))
    for c in range(C):
        line = [predict((r, c)) for r in range(R)]
        d = 'M' + ' L'.join('%.1f,%.1f' % (sx(p[0]), sy(p[1])) for p in line)
        parts.append('<path d="%s" stroke="%s" stroke-width="1" stroke-dasharray="5,4" opacity="0.5" fill="none"/>' % (d, gl))

    # 原始检测点（小彩点，按簇着色，展示抖动/分布）
    for i, c in enumerate(clusters):
        col = PALETTE[i % len(PALETTE)]
        mem = subsample(c['members'], maxpp)
        for (mx, my) in mem:
            parts.append('<circle cx="%.1f" cy="%.1f" r="2.4" fill="%s" fill-opacity="0.5"/>'
                         % (sx(mx), sy(my), col))

    # 聚类真坐标：仅用小绿点标记中心（不画大框，避免遮挡点云；点云才是主角）
    # 绿点旁直接标实测坐标 (cx, cy)，省去来回对照
    for i, c in enumerate(clusters):
        col = PALETTE[i % len(PALETTE)]
        px, py = sx(c['cx']), sy(c['cy'])
        parts.append('<circle cx="%.1f" cy="%.1f" r="3.4" fill="none" stroke="#36e07a" stroke-width="2"/>' % (px, py))
        parts.append('<text x="%.1f" y="%.1f" fill="%s" font-size="10" text-anchor="middle" font-weight="700">#%d</text>'
                     % (px, py - 6, col, i))
        parts.append('<text x="%.1f" y="%.1f" fill="#9fb0c0" font-size="7.5" text-anchor="middle">(%d, %d)</text>'
                     % (px, py + 11, round(c['cx']), round(c['cy'])))

    parts.append('<text x="340" y="24" fill="#e6edf3" font-size="15" text-anchor="middle" font-family="sans-serif">红点原始分布 + 聚类真坐标（自行判断数据可信度）</text>')
    parts.append('<text x="340" y="44" fill="#8b97a7" font-size="11" text-anchor="middle" font-family="sans-serif">彩色小点 = 各帧原始检测（按簇着色，展示抖动分布）｜ 绿点 = 聚类真坐标（旁标实测坐标）｜ 浅蓝虚线 = 拟合 %d×%d 网格</text>' % (R, C))
    reg_txt = "R=%d x C=%d (R*C=%d)" % (R, C, R * C)
    reg_txt += " 与真实点数 %d 吻合" % N if R * C == N else " 与真实点数 %d 不符" % N
    parts.append('<text x="340" y="60" fill="#8b97a7" font-size="11" text-anchor="middle" font-family="sans-serif">%s ｜ 原始帧=%d ｜ 间距≈%.0fpx ｜ 残差均值%.1f/最大%.1f ｜ 异常(橙)%d</text>'
                 % (reg_txt, n_raw, spacing, meanr, maxr, n_out))
    parts.append('</svg>')
    svg = ''.join(parts)

    if target is not None:
        tR, tC = target
        if R == tR and C == tC and R * C == N and maxr < spacing * 1.2:
            verdict = "（拟合为规整 %d×%d，与预期吻合；但请以图中分布为准自行判断）" % (tR, tC)
        elif R * C == N and maxr < spacing * 1.2:
            verdict = "（拟合为规整 %d×%d，但与预期 %d×%d 不符；请以图中分布为准）" % (R, C, tR, tC)
        else:
            verdict = "（拟合不规整 / 有错；请以图中分布为准）"
    else:
        if R == 3 and C == 3 and R * C == N and maxr < spacing * 1.2:
            verdict = "（拟合为规整 3×3，旋转约45°；但请以图中分布为准自行判断）"
        elif R * C == N and maxr < spacing * 1.2:
            verdict = "（拟合为规整 %d×%d；但请以图中分布为准自行判断）" % (R, C)
        else:
            verdict = "（拟合不规整 / 有错；请以图中分布为准）"

    # 坐标表（按 R1C1 → R3C3 排序，与 compute_coords 输出、coords_result.json 完全一致）
    conf_cls = {'low': 'low', 'medium': 'med', 'high': 'hi'}
    rows_html = []
    for k, d in enumerate(coords, 1):
        rc = "R%dC%d" % d['RC']
        cls = conf_cls.get(d['conf'], 'med')
        rows_html.append(
            '<tr><td>%d</td><td>%s</td><td>%.1f</td><td>%.1f</td>'
            '<td>%.1f</td><td>%.1f</td><td>%.1f</td><td>%.1f</td>'
            '<td>%d</td><td>%.1f%%</td><td class="%s">%s</td></tr>'
            % (k, rc, d['cx'], d['cy'], d['sx'], d['sy'], d['gx'], d['gy'],
               d['n'], d['cov'] * 100, cls, d['conf']))
    coord_table = (
        '<div class="ctitle">9 个物理点最优坐标（按网格 R1C1 → R3C3 排序；实测=多帧质心/最大似然估计）</div>\n'
        '<table class="coords">\n'
        '<tr><th>序</th><th>网格</th><th>实测X</th><th>实测Y</th><th>±σx</th><th>±σy</th>'
        '<th>理想X</th><th>理想Y</th><th>n</th><th>覆盖%</th><th>置信</th></tr>\n'
        + '\n'.join(rows_html) +
        '\n</table>\n'
        '<div class="note">坐标说明：①「实测X/Y」=该点所有帧的均值（高斯抖动下的最优/最小方差估计），即"实际打在哪"；「理想X/Y」=假设严格成规则网格时"应在哪"，会平滑掉真实偏差，仅作参照。②「覆盖%」=首末命中行号跨度/总行数；≈0% 即<b>单帧闪现</b>（最不可信，疑似误检）。③「置信」low=覆盖%&lt;5%，medium=n&lt;120，high=其余。④ 网格相对相机约旋转45°，R1C1…R3C3 是网格内行列序号而非屏幕左上→右下（如 R1C1=(298,112) 在屏幕右上角），请按物理 3×3 用 (实测X,实测Y) 对应每个物体。⑤ 坐标绝对值是否正确仍需回 MaixCAM 原帧核对。</div>\n'
    )

    html = (
        '<!doctype html><html><head><meta charset="utf-8"><style>\n'
        'body{margin:0;background:#0d1117;color:#e6edf3;font-family:-apple-system,Segoe UI,sans-serif;display:flex;flex-direction:column;align-items:center;padding:20px}\n'
        '.wrap{max-width:820px;width:100%}\n'
        'svg{width:100%;height:auto;border:1px solid #1f2630;border-radius:8px;background:#0d1117}\n'
        '.legend{margin-top:14px;font-size:13px;color:#b9c2cf;line-height:1.9}\n'
        '.dot{display:inline-block;width:11px;height:11px;vertical-align:middle;margin-right:6px;border-radius:50%}\n'
        '.note{margin-top:10px;font-size:12px;color:#7d8694;line-height:1.7}\n'
        '.verdict{margin-top:8px;font-size:13px;font-weight:600;color:#9fb0c0}\n'
        '.ctitle{margin-top:18px;font-size:13px;font-weight:600;color:#c9d4e0}\n'
        '.coords{margin-top:8px;width:100%;border-collapse:collapse;font-size:12px;color:#c9d4e0;font-family:monospace}\n'
        '.coords th,.coords td{border:1px solid #232c38;padding:5px 6px;text-align:center}\n'
        '.coords th{background:#161d27;color:#8b97a7;font-weight:600}\n'
        '.coords tr:nth-child(even){background:#11161e}\n'
        '.coords .low{color:#ff6b6b;font-weight:700}\n'
        '.coords .med{color:#ffb300}\n'
        '.coords .hi{color:#69f0ae}\n'
        '</style></head><body><div class="wrap">' + svg + '\n'
        '<div class="legend">\n'
        '<span class="dot" style="background:#ff5252"></span><span class="dot" style="background:#40c4ff"></span><span class="dot" style="background:#ffb300"></span> 彩色小点 = 各帧原始检测（同一颜色=同一个物理点；点云散开程度 = 该点的检测抖动/不稳定范围）<br>\n'
        '<span class="dot" style="background:#36e07a;border-radius:50%"></span> 小绿点（圈）= 把同色小点聚类平均后的"真坐标"中心标记（旁标实测坐标；不画大框，点云为主角）<br>\n'
        '<span class="dot" style="background:#2f4a6b;border-radius:2px"></span> 浅蓝虚线 = 双线性拟合的预期 ' + str(R) + '×' + str(C) + ' 网格（已校正旋转，供你对照点云是否贴线）\n'
        '</div>\n'
        '<div class="verdict">拟合参考：' + verdict + '</div>\n'
        '<div class="note">读图要点：① 每个颜色的点云应当聚成一小团（抖动小=稳定）；若某团拉得很长/发散，说明该点检测不稳。② 小绿点(圈) 标记该色点云的聚类中心，旁边标注的 (cx,cy) 即该点实测坐标（与下方坐标表一致）。③ 各团应整体贴住浅蓝虚线网格、彼此等距——若某团明显偏离线或间距异常，即证明该处数据有错。④ 判定真伪看两件事：n（命中帧数）与<b>时间覆盖率</b>（首末命中行号跨度/总行数）。n 极小(如 1)且覆盖率≈0% 的是<b>单帧闪现</b>（疑似误检/瞬态，最不可信）；n 小但覆盖率≈100% 说明全程都在、只是每帧常漏检（真实但弱）。⑤ 本图只证明"点彼此是否成网格、抖动是否稳定"，坐标绝对值是否正确要看设备原帧。</div>\n'
        + coord_table +
        '</div></body></html>'
    )

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return verdict


def main():
    args = parse_args()
    # 共享计算核心：聚类 / 网格拟合 / 坐标 全部由 compute_coords.run 完成
    res = run(args.input)
    clusters = res['clusters']
    best = res['best']
    n_raw = res['n_raw']
    spacing = res['spacing']
    coords = res['coords']
    print("raw detections:", n_raw)
    print("distinct clusters (真实点数):", len(clusters))
    for i, c in enumerate(clusters):
        print("  #%02d cx=%6.1f cy=%6.1f  n=%3d  members=%d" % (i, c['cx'], c['cy'], c['n'], len(c['members'])))

    out_path = os.path.abspath(args.out or (os.path.splitext(args.input)[0] + '_overlay.html'))
    verdict = build_html(clusters, best, spacing, tuple(args.target) if args.target else None,
                         n_raw, args.maxpp, out_path, coords)
    print("written (绝对路径):", out_path)
    print("verdict =", verdict)
    print("--- 9 点最优坐标（按 R1C1→R3C3，与 coords_result.json 一致）---")
    for k, d in enumerate(coords, 1):
        print("  %d. %s  实测(%.1f, %.1f) ±(%.1f,%.1f)  n=%d  覆盖%.1f%%  %s"
              % (k, "R%dC%d" % d['RC'], d['cx'], d['cy'], d['sx'], d['sy'],
                 d['n'], d['cov'] * 100, d['conf']))


if __name__ == '__main__':
    main()
