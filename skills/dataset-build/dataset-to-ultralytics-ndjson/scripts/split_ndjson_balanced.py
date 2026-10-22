#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NDJSON -> 分层均衡 train/val/test 划分（按采集连拍段取连续块，把泄漏压到最低）

适用：采集类检测/旋转框数据集（文件名自带会话前缀与帧号），需按 类别(可选再加一个属性如颜色)
      分层抽取固定比例作 val/test，且要避免「同一段连拍被拆散」造成的指标虚高。

核心做法
  1. 解析 会话 + 帧号；按 (会话, 类别[, 属性]) 游程切分 -> 连拍段
  2. 逐 (类别[,属性]) 组合、逐段按比例分配，段内取【连续块】：
        train = 段首   val = 中段连续块   test = 段尾连续块
     （不是打散乱抽 —— 那会让每张 val 图旁边都有一张 train）
  3. 增强孪生（如 _bw / _gray / _flip）通过 --twin-csv 给出 (child,parent)，强制跟随父图同一 split
  4. 输出改写 split 的 NDJSON + 清单 CSV + burst 清单 CSV + 校验报告

用法
  python split_ndjson_balanced.py \
    --ndjson  demo-yolo.ndjson \
    --attr-csv attrs.csv            # 列: file,color  (每个 file 一行; 不传则只按类别分层)
    --twin-csv twins.csv            # 列: child,parent
    --out-ndjson demo-yolo-split.ndjson \
    --out-manifest split-manifest.csv \
    --out-burst burst-inventory.csv \
    --val 0.10 --test 0.05

不传 --out-ndjson 时只打印报告（dry run）。
"""
import argparse, collections, csv, io, json, os, re, sys

FRAME_PATTERNS = [r"^(.*?)\s*\((\d+)\)\.jpg$", r"^(cam[0-9._]*?)_(\d+)\.jpg$"]


def parse_sess_frame(name):
    for p in FRAME_PATTERNS:
        m = re.match(p, name or "")
        if m:
            return m.group(1), int(m.group(2))
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndjson", required=True)
    ap.add_argument("--out-ndjson", default=None, help="不传则 dry run，只出报告")
    ap.add_argument("--out-manifest", default=None)
    ap.add_argument("--out-burst", default=None)
    ap.add_argument("--out-report", default=None)
    ap.add_argument("--attr-csv", default=None, help="列 file,<attrname> 为每个文件给出附加分层属性")
    ap.add_argument("--attr-name", default="color")
    ap.add_argument("--attr-regex", default=None,
                    help="从文件名解析属性，捕获组1为属性值，如 '^\\w+_(red|green|blue)_\\d+\\.jpg$'")
    ap.add_argument("--twin-csv", default=None, help="列 child,parent：增强孪生强制跟随父图")
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.05)
    ap.add_argument("--task", default=None, help="detect/obb，默认取 NDJSON 表头")
    args = ap.parse_args()

    header, recs = None, []
    for line in io.open(args.ndjson, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get("type") == "dataset":
            header = o
        elif o.get("type") == "image":
            recs.append(o)
    task = args.task or (header or {}).get("task", "detect")
    ann_key = "boxes" if task == "detect" else "obb"
    CN = {int(k): v for k, v in ((header or {}).get("class_names") or {}).items()}

    attr = {}
    if args.attr_csv:
        for r in csv.DictReader(io.open(args.attr_csv, "r", encoding="utf-8", newline="")):
            attr[r["file"]] = r.get(args.attr_name)
    attr_re = re.compile(args.attr_regex) if args.attr_regex else None
    twin_parent = {}
    if args.twin_csv:
        for r in csv.DictReader(io.open(args.twin_csv, "r", encoding="utf-8", newline="")):
            twin_parent[r["child"]] = r["parent"]

    def get_attr(f):
        if f in attr and attr[f]:
            return attr[f]
        if attr_re:
            m = attr_re.match(f)
            if m:
                return m.group(1)
        return None

    # ---- scene 单元（非孪生图）----
    scenes, noann, orph = {}, [], []
    for r in recs:
        if r["file"] in twin_parent:
            continue
        ann = (r.get("annotations") or {}).get(ann_key)
        if not ann:
            noann.append(r["file"]); continue
        s, k = parse_sess_frame(r["file"])
        if s is None:
            orph.append((r["file"], "无会话/帧号")); continue
        scenes[r["file"]] = dict(file=r["file"], s=s, k=k, cls=int(ann[0][0]),
                                 attr=get_attr(r["file"]), twins=[], boxes=ann)
    for r in recs:
        p = twin_parent.get(r["file"])
        if p is None:
            continue
        if p in scenes:
            scenes[p]["twins"].append(r["file"])
        else:
            orph.append((r["file"], "找不到父图 %s" % p))

    S = sorted(scenes.values(), key=lambda x: (x["s"], x["k"]))
    bursts, prev = [], None
    for it in S:
        key = (it["s"], it["cls"], it["attr"])
        if key != prev:
            bursts.append([]); prev = key
        bursts[-1].append(it)
    bygroup = collections.defaultdict(list)
    for b in bursts:
        bygroup[(b[0]["cls"], b[0]["attr"])].append(b)

    # ---- 分配 ----
    assign = {}
    for key, bs in bygroup.items():
        n_img = sum(1 + len(it["twins"]) for b in bs for it in b)
        t_val = int(round(args.val * n_img))
        t_test = int(round(args.test * n_img))
        alloc = []
        for b in bs:
            nb = sum(1 + len(it["twins"]) for it in b)
            vb = int(round(t_val * nb / float(n_img)))
            tb = int(round(t_test * nb / float(n_img)))
            while vb + tb > 0 and nb - vb - tb < 1:
                if tb > 0: tb -= 1
                elif vb > 0: vb -= 1
            alloc.append([vb, tb, nb])
        for li in (0, 1):
            target = t_val if li == 0 else t_test
            diff = target - sum(a[li] for a in alloc)
            guard = 0
            while diff != 0 and guard < 500:
                guard += 1
                if diff > 0:
                    cand = max(range(len(alloc)), key=lambda i: alloc[i][2] - alloc[i][0] - alloc[i][1])
                    if alloc[cand][2] - alloc[cand][0] - alloc[cand][1] < 2:
                        break
                    alloc[cand][li] += 1; diff -= 1
                else:
                    cand = max(range(len(alloc)), key=lambda i: alloc[i][li])
                    if alloc[cand][li] == 0:
                        break
                    alloc[cand][li] -= 1; diff += 1
        for b, (vb, tb, nb) in zip(bs, alloc):
            n = len(b)
            idx_test = n
            acc = 0
            while idx_test > 0 and acc < tb:
                idx_test -= 1; acc += 1 + len(b[idx_test]["twins"])
            idx_val = idx_test          # 必须从 idx_test 起，否则与 test 块重叠
            acc = 0
            while idx_val > 0 and acc < vb:
                idx_val -= 1; acc += 1 + len(b[idx_val]["twins"])
            for i, it in enumerate(b):
                sp = "test" if i >= idx_test else ("val" if i >= idx_val else "train")
                assign[it["file"]] = sp
                for t in it["twins"]:
                    assign[t] = sp

    # ---- 输出 NDJSON ----
    if args.out_ndjson:
        with io.open(args.out_ndjson, "w", encoding="utf-8", newline="") as f:
            f.write(json.dumps(header, ensure_ascii=False) + "\n")
            for r in recs:
                if r["file"] not in assign:
                    continue
                o = dict(r); o["split"] = assign[r["file"]]
                f.write(json.dumps(o, ensure_ascii=False) + "\n")

    # ---- 清单 / burst 清单 ----
    if args.out_manifest:
        with io.open(args.out_manifest, "w", encoding="utf-8", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["file", "split", "cls_id", "cls_name", args.attr_name, "session", "frame",
                         "is_twin", "scene"])
            for r in recs:
                f0 = r["file"]
                if f0 not in assign:
                    continue
                p = twin_parent.get(f0, f0)
                v = scenes.get(p)
                if v is None:
                    continue
                wr.writerow([f0, assign[f0], int(v["boxes"][0][0]), CN.get(int(v["boxes"][0][0])),
                             v["attr"], v["s"], v["k"], 1 if f0 != p else 0, p])
    if args.out_burst:
        sp_of = {}
        for r in recs:
            if r["file"] in assign:
                sp_of.setdefault(r["file"], assign[r["file"]])
        rows = []
        for b in bursts:
            imgs = collections.Counter()
            for it in b:
                imgs[assign[it["file"]]] += 1
                for t in it["twins"]:
                    imgs[assign[t]] += 1
            cnt = collections.Counter(assign[it["file"]] for it in b)
            rows.append(dict(session=b[0]["s"], cls=CN.get(b[0]["cls"]), **{args.attr_name: b[0]["attr"]},
                             scenes=len(b), frame_start=b[0]["k"], frame_end=b[-1]["k"],
                             img_train=imgs["train"], img_val=imgs["val"], img_test=imgs["test"],
                             n_img=sum(imgs.values()),
                             pure=("是" if len([x for x in cnt.values() if x]) == 1 else "否")))
        with io.open(args.out_burst, "w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            for r in rows: wr.writerow(r)

    # ---- 报告 ----
    buf = io.StringIO()
    def w(*a): print(*a, file=buf)
    ci = collections.Counter(assign.values())
    n = sum(ci.values())
    w("记录=%d  已划分=%d  剔除(无标注)=%d  剔除(无法归位)=%d" % (len(recs), n, len(noann), len(orph)))
    w("连拍段=%d  组合数=%d" % (len(bursts), len(bygroup)))
    w("图片: train=%d(%.2f%%) val=%d(%.2f%%) test=%d(%.2f%%)" % (
        ci["train"], 100.0*ci["train"]/n, ci["val"], 100.0*ci["val"]/n, ci["test"], 100.0*ci["test"]/n))
    w("")
    w("%-14s %-8s %6s %6s %6s %6s %8s %8s" % ("cls", args.attr_name, "n", "train", "val", "test", "val%", "test%"))
    wv = wt = 0.0
    for k in sorted(bygroup, key=lambda x: (x[0], str(x[1]))):
        c = collections.Counter()
        for b in bygroup[k]:
            for it in b:
                c[assign[it["file"]]] += 1
                for t in it["twins"]:
                    c[assign[t]] += 1
        nn = sum(c.values())
        pv, pt = 100.0*c["val"]/nn, 100.0*c["test"]/nn
        wv = max(wv, abs(pv - args.val*100)); wt = max(wt, abs(pt - args.test*100))
        w("%-14s %-8s %6d %6d %6d %6d %7.2f%% %7.2f%%" % (
            CN.get(k[0]), k[1], nn, c["train"], c["val"], c["test"], pv, pt))
    w("val%% 最大偏离: %.2f 个百分点   test%% 最大偏离: %.2f 个百分点" % (wv, wt))
    bad = sum(1 for p, v in scenes.items()
              if p in assign and any(assign.get(t) != assign[p] for t in v["twins"]))
    w("")
    w("孪生(或增强子图)与父图不同 split 的数量 : %d  (必须为 0)" % bad)
    idx_of = {}
    for b in bursts:
        for i, it in enumerate(b): idx_of[it["file"]] = (b, i)
    f2s = {}
    for p, v in scenes.items():
        f2s[p] = p
        for t in v["twins"]: f2s[t] = p
    leak = tot = 0
    for f0, sp in assign.items():
        if sp == "train": continue
        sc = f2s.get(f0)
        if sc not in idx_of: continue
        b, i = idx_of[sc]; tot += 1
        if any(j != i and assign[b[j]["file"]] == "train"
               for j in range(max(0, i-2), min(len(b), i+3))):
            leak += 1
    w("val/test 中「±2 帧内有 train 邻帧」的比例 : %.1f%%" % (100.0*leak/max(tot, 1)))
    w("被跨 split 切开的连拍段 : %d / %d" % (sum(1 for b in bursts if len(set(assign[it["file"]] for it in b)) > 1), len(bursts)))
    w("")
    w("剔除清单(无标注) : %s" % (noann[:10] if noann else "无"))
    w("剔除清单(无法归位) : %s" % ([x[0] for x in orph[:10]] if orph else "无"))
    txt = buf.getvalue()
    if args.out_report:
        io.open(args.out_report, "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
