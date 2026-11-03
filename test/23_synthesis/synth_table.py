"""A. 把 E22 best.json 每人配置解析为方法族标签 → synth_table.csv。"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
E22 = HERE.parent / "22_subject"

TOK = [  # (正则, 族)；按顺序匹配
    (r"^ens8$", "ens8"), (r"^k\d+_w[\d.]+$", "joint"), (r"^jw[\d.]+_k\d+_C", "joint"),
    (r"^j[\d.]+_\d+$", "cross"), (r"^x\d+_C", "cross"),
    (r"^own\d+_C", "own"), (r"^sc\d+_k\d+$", "own"), (r"^wcal\d+_C", "own"),
    (r"^idx\d+$", "idx"), (r"^idx[a-z]+\d+$", "idxg"),
    (r"^sim", "sim"), (r"^b23", "b23"),
    (r"^g[a-z]+(_k\d+)?$", "g"), (r"^f[gb]", "fg"), (r"^f\d+$", "f"),
]


def fam(tok):
    for p, f in TOK:
        if re.match(p, tok):
            return f
    return "?"


def parse(cfg):
    m = re.match(r"^(r\d|s034):", cfg)
    pre = m.group(1) if m else ""
    body, thr = cfg[len(pre) + (1 if pre else 0):].rsplit("|", 1)
    members = [p.strip("()[]") for p in re.split(r"\)\+\(|\)\+\[|\]\+\(", body)] if body.startswith(("(", "[")) \
        else [body]
    toks = [t for mm in members for t in re.split(r"\+", mm.strip("()[]")) if t]
    fams = sorted({fam(t) for t in toks})
    if thr == "t0":
        tt = "t0"
    elif re.match(r"^t\d+$", thr):
        tt = "中位数"
    else:
        tt = "分位数"
    return pre, len(members), toks, fams, tt, thr


def main():
    best = json.load(open(E22 / "best.json"))
    sj = json.load(open(E22 / "subject.json"))["per_subject"]
    r2 = json.load(open(E22 / "round2.json")) if (E22 / "round2.json").exists() else {}
    rows = []
    for q, v in best.items():
        pre, nm, toks, fams, tt, thr = parse(v["cfg"])
        if pre:
            rnd = pre[1:] if pre[0] == "r" else "034专项"
        else:
            rnd = "2" if any(re.match(r"^(jw|sc|wcal)", t) for t in toks) else "1"
        main_f = "+".join(fams)
        ps = sj.get(q, {})
        rows.append(dict(subject=q, bacc=round(v["bacc"], 4), auc=round(v["auc"], 4), f1=round(v["f1"], 4),
                         round=rnd, families=main_f, n_members=nm, n_tokens=len(toks), thr_type=tt, thr=thr,
                         ens8_bacc=round(ps["ens8"]["bacc"], 4) if ps else "",
                         calsel_bacc=round(ps["calsel"]["bacc"], 4) if ps else "",
                         calsel_cfg=ps["calsel"]["cfg"] if ps else "", cfg=v["cfg"]))
    with open(HERE / "synth_table.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fc = Counter(f for r in rows for f in r["families"].split("+"))
    print("族出现次数（含组合）:", fc.most_common())
    print("组合:", Counter(r["families"] for r in rows).most_common())
    print("阈值:", Counter(r["thr_type"] for r in rows), Counter(r["thr"] for r in rows).most_common())
    print("轮次:", Counter(r["round"] for r in rows))
    print("元数:", Counter(r["n_members"] for r in rows))
    json.dump(dict(family_count=fc, combo=Counter(r["families"] for r in rows),
                   thr=Counter(r["thr_type"] for r in rows), rounds=Counter(r["round"] for r in rows)),
              open(HERE / "synth_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
