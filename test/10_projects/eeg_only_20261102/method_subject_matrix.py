"""E10：把历史每次实验的逐被试结果汇成「方法 × 57 被试」矩阵，并找规律。

用户要求：不再用整体指标盲目尝试，而是看清每种方法在每个人身上的效果。
做法：
  1. 从所有历史 JSON 抽取逐被试卷级 AUC / bACC，统一被试键为 '001'…'059'；
  2. 每列标注协议类别：
       deploy   单窗 6 s、跨被试、无位置/次序（可部署）
       history  依赖超过当前窗的历史/记忆
       diag     同人训练、oracle、诊断量（不可部署）
     split 标注是否与 SGKF(5,shuffle,rs0) 同一划分。
  3. 输出 CSV / JSON / md 宽表；
  4. 在同一划分、deploy 类方法子矩阵上做规律分析：
     被试效应 vs 方法效应 / 方法间排序一致性 / 逐被试最优方法上界 / 稳健被试与脆弱被试。

运行：py -3.11 test/10_projects/eeg_only_20261102/method_subject_matrix.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SUBS = [f"{i:03d}" for i in range(1, 60) if i not in (32, 33)]


def norm_key(k: str) -> str:
    k = str(k).replace("sub-", "").replace("sub_", "")
    return f"{int(k):03d}" if k.isdigit() else k


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def spec_sources():
    """(列名, 文件, 取值函数, 协议, 划分, 备注)"""
    S = []
    hd = load("test/03_deep/highdim_deep_results.json")
    for v, o in hd["variants"].items():
        S.append((f"highdim/{v}", "03_deep", lambda q, o=o: o["subjects"].get(q, {}).get("AUC"),
                  "deploy", "SGKF5rs0", "878维，内层选特征"))
    ns = load("test/08_diagnostics/nonstationary_results.json")
    for v, o in ns["variants"].items():
        S.append((f"nonstat/{v}", "08_diag", lambda q, o=o: o["subjects"].get(q, {}).get("AUC"),
                  "deploy", "SGKF5rs0", f"{o.get('n_features')}维特征族消融"))
    kl = load("test/03_deep/kalman_deep_results.json")
    for v, o in kl["models"].items():
        S.append((f"kalman/{v}", "03_deep",
                  lambda q, o=o: o["subject_metrics"].get(q, {}).get("AUC"),
                  "deploy", "SGKF5rs0", "每窗重置滤波器"))
    bn = load("test/08_diagnostics/bottleneck_diagnosis.json")
    for v in ("within_subject", "cross_subject"):
        S.append((f"bottleneck/{v}", "08_diag", lambda q, o=bn[v]: o["subjects"].get(q, {}).get("AUC"),
                  "diag" if v == "within_subject" else "deploy", "SGKF5rs0", "同人 vs 跨人"))
    bnt = load("test/08_diagnostics/bottleneck_diagnosis_trials.json")
    S.append(("bottleneck/within_trial", "08_diag",
              lambda q: bnt["within_subject_trial_blocked"]["subjects"].get(q, {}).get("AUC"),
              "diag", "SGKF5rs0", "按试次阻断的同人训练"))
    pl = load("test/08_diagnostics/polarity_diagnosis_window6_results.json")
    rows = {norm_key(r["subject"]): r for r in pl["baseline"]["normal"]["per_subject"]}
    S.append(("polarity/base", "08_diag", lambda q: rows.get(q, {}).get("auc"), "deploy", "SGKF5rs0", "符号压力前的基线"))
    rs = load("test/06_adaptive/reliability_shrink_results.json")
    S.append(("reliability/base", "06_adapt", lambda q: rs["subject_rows"].get(q, {}).get("auc"),
              "deploy", "SGKF5rs0", "收缩前的基线 LR"))
    hdl = load("test/07_depositioned/depositioned_history_delta_results.json")
    for v in ("current_only__lr", "current_only__hgb", "delta_k32__hgb"):
        o = hdl["subject_counts_seed0"][v]["per_subject"]
        S.append((f"depo/{v}", "07_depo", lambda q, o=o: o.get(f"sub-{q}"), 
                  "history" if v.startswith("delta") else "deploy", "depo-stream", "去位置化流式"))
    ad = load("test/07_depositioned/depositioned_anchor_direction.json")
    for v, o in ad["methods"].items():
        S.append((f"depo/{v}", "07_depo", lambda q, o=o: o["subjects"].get(f"sub-{q}"),
                  "history", "depo-stream", "锚点/方向法"))
    ds = load("test/audit/direction_scoring/direction_scoring.json")
    S.append(("audit/xs", "audit", lambda q: ds["subjects"].get(f"sub-{q}", {}).get("auc_xs"),
              "deploy", "SGKF5rs0", "全局方向"))
    S.append(("audit/own_cv", "audit", lambda q: ds["subjects"].get(f"sub-{q}", {}).get("auc_own_cv"),
              "diag", "SGKF5rs0", "本人方向(交叉拟合)"))
    od = load("test/theory/own_direction_loading_results.json")
    S.append(("theory/own_loading_xs", "theory",
              lambda q: od["subjects"].get(f"sub-{q}", {}).get("auc_xs_all"), "deploy", "SGKF5rs0",
              "特征族能量份额分析"))
    tc = load("test/07_depositioned/depositioned_transfer_ceiling.json")
    S.append(("ceiling/order_free_xs", "07_depo",
              lambda q: tc["order_free"]["subjects"].get(f"sub-{q}", {}).get("auc_xs"),
              "deploy", "depo-stream", "天花板审计"))
    S.append(("ceiling/within_rand", "07_depo",
              lambda q: tc["order_free"]["subjects"].get(f"sub-{q}", {}).get("auc_ws_rand"),
              "diag", "depo-stream", "同人随机切分"))
    ps = load("test/08_diagnostics/per_subject_ceiling_results.json")
    psm = {norm_key(r["subject"]): r for r in ps["per_subject"]}
    S.append(("ceiling/merge_best", "08_diag", lambda q: psm.get(q, {}).get("merge_best_auc"),
              "deploy", "SGKF5rs0", "历史融合最佳"))
    ea = load("test/10_projects/eeg_only_20261102/per_subject_audit.json")
    S.append(("eeg_only/lr_c001", "eeg_only", lambda q: ea["per_subject"].get(q, {}).get("auc"),
              "deploy", "SGKF5rs0", "本轮 LR 复现"))
    mb = HERE / "method_bank.json"
    if mb.exists():
        bank = json.loads(mb.read_text(encoding="utf-8"))
        for v, o in bank["per_subject"].items():
            S.append((f"bank/{v}", "bank", lambda q, o=o: (o.get(q) or {}).get("auc"),
                      "deploy", "SGKF5rs0", "本轮同协议方法银行"))
    return S


def main():
    S = spec_sources()
    rows, meta = {}, []
    for name, src, fn, proto, split, note in S:
        vals = {}
        for q in SUBS:
            try:
                v = fn(q)
            except Exception:
                v = None
            vals[q] = None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
        n = sum(v is not None for v in vals.values())
        if n < 10:
            print(f"[skip] {name}: only {n}/57")
            continue
        rows[name] = vals
        meta.append({"method": name, "src": src, "protocol": proto, "split": split, "note": note,
                     "coverage": n,
                     "mean": float(np.nanmean([v for v in vals.values() if v is not None]))})

    # ---- 矩阵 ----
    subj_ok = SUBS
    A = np.array([[rows[m][q] if rows[m][q] is not None else np.nan for q in subj_ok] for m in rows])
    names = list(rows)
    print(f"[matrix] {A.shape[0]} 方法 × {A.shape[1]} 被试，覆盖 {np.isfinite(A).sum()}/{A.size}")

    df_rows = ["| 方法 | 协议 | 覆盖 | 均值 | 中位 | 最低 | 最高 | 说明 |", "|---|---|---|---|---|---|---|---|"]
    order = sorted(range(len(names)), key=lambda i: -np.nanmean(A[i]))
    for i in order:
        r = A[i]
        df_rows.append(f"| {names[i]} | {meta[i]['protocol']} | {meta[i]['coverage']} | "
                       f"{np.nanmean(r):.3f} | {np.nanmedian(r):.3f} | {np.nanmin(r):.3f} | "
                       f"{np.nanmax(r):.3f} | {meta[i]['note']} |")
    (HERE / "method_subject_matrix.md").write_text(
        "# 方法 × 被试 AUC 矩阵（n×57）\n\n"
        "协议：deploy=单窗6s跨被试无位置；history=依赖历史；diag=同人/oracle诊断。"
        "split=SGKF5rs0 表示与 5 折(shuffle,rs=0) 同一被试划分。\n\n"
        + "\n".join(df_rows) + "\n", encoding="utf-8")

    # CSV
    with (HERE / "method_subject_matrix.csv").open("w", encoding="utf-8-sig") as f:
        f.write("method,protocol,split," + ",".join(subj_ok) + "\n")
        for i, m in enumerate(names):
            vals = ",".join("" if not np.isfinite(v) else f"{v:.4f}" for v in A[i])
            f.write(f"{m},{meta[i]['protocol']},{meta[i]['split']},{vals}\n")

    # ---- 规律分析：同划分 deploy 子矩阵 ----
    sel = [i for i, m in enumerate(meta) if m["protocol"] == "deploy" and m["split"] == "SGKF5rs0"]
    Ad = A[sel]
    nd = names_m = [names[i] for i in sel]
    print(f"\n[strict] 同划分 deploy 方法 {len(sel)} 个，覆盖 {np.isfinite(Ad).sum()}/{Ad.size}")
    keep = np.isfinite(Ad).all(0)
    print(f"[strict] 全方法都有值的被试 {keep.sum()} 人")

    # 被试效应 / 方法效应方差分解（两因素无重复，交互=残差）
    X = Ad[:, keep]
    gm = np.nanmean(X)
    sm = np.nanmean(X, axis=0) - gm     # 被试主效应
    mm = np.nanmean(X, axis=1) - gm     # 方法主效应
    res = X - gm - sm[None, :] - mm[:, None]
    tot = np.nansum((X - gm) ** 2)
    print(f"[variance] 被试={np.nansum(sm**2)*X.shape[0]/tot:.3f} "
          f"方法={np.nansum(mm**2)*X.shape[1]/tot:.3f} "
          f"被试×方法={np.nansum(res**2)/tot:.3f}")

    # 方法间排序一致性
    from itertools import combinations
    from scipy.stats import spearmanr
    rhos = []
    for i, j in combinations(range(len(sel)), 2):
        a, b = A[sel[i]], A[sel[j]]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 20:
            rhos.append(spearmanr(a[m], b[m]).statistic)
    print(f"[agreement] 方法间逐被试 AUC 的 Spearman：中位 {np.median(rhos):.3f} "
          f"范围 [{np.min(rhos):.3f}, {np.max(rhos):.3f}]")
    # 被试跨方法稳定度
    subj_mean = np.nanmean(Ad, axis=0)
    subj_std = np.nanstd(Ad, axis=0)
    print("[stable] 跨方法标准差最大(最受方法影响)的被试：",
          [f"{subj_ok[i]}({subj_std[i]:.3f})" for i in np.argsort(-subj_std)[:8]])
    print("[stable] 跨方法标准差最小(最不受方法影响)的被试：",
          [f"{subj_ok[i]}({subj_std[i]:.3f})" for i in np.argsort(subj_std)[:8]])
    # 每被试最优方法
    best_cnt = {}
    best_gain = []
    for jj, q in enumerate(subj_ok):
        col = Ad[:, jj]
        if not np.isfinite(col).any():
            continue
        bi = int(np.nanargmax(col))
        best_cnt[names_m[bi]] = best_cnt.get(names_m[bi], 0) + 1
        best_gain.append(float(np.nanmax(col) - np.nanmean(col)))
    print(f"[oracle-method] 每人取最优方法：均值 {np.mean([np.nanmax(Ad[:,j]) for j in range(Ad.shape[1]) if np.isfinite(Ad[:,j]).any()]):.3f}"
          f"，相对该人跨方法均值的平均增益 {np.mean(best_gain):+.3f}")
    print("[oracle-method] 最常获胜的方法：", sorted(best_cnt.items(), key=lambda z: -z[1])[:8])
    top = np.argsort(-subj_mean)[:10]
    bot = np.argsort(subj_mean)[:10]
    print("[top-10 被试] ", [(subj_ok[i], round(float(subj_mean[i]), 3)) for i in top])
    print("[bot-10 被试] ", [(subj_ok[i], round(float(subj_mean[i]), 3)) for i in bot])

    # 逐被试汇总（含 E12 的可分性/迁移性，若已算过）
    wg = HERE / "subject_why_good.json"
    wgm = {r["subject"]: r for r in json.loads(wg.read_text(encoding="utf-8"))["rows"]} if wg.exists() else {}
    with (HERE / "subject_summary.csv").open("w", encoding="utf-8-sig") as f:
        f.write("subject,cross_method_mean,cross_method_max,best_method,best_minus_mean,"
                "own_cv,cos,xs,transfer_eff,within_subject\n")
        for jj, q in enumerate(subj_ok):
            col = Ad[:, jj]
            bi = int(np.nanargmax(col))
            w = wgm.get(q, {})
            f.write(f"{q},{subj_mean[jj]:.4f},{col[bi]:.4f},{names_m[bi]},{col[bi]-subj_mean[jj]:.4f},"
                    f"{w.get('own_cv','')},{w.get('cos','')},{w.get('xs','')},"
                    f"{w.get('transfer_eff','')},{w.get('within','')}\n")
    md = ["# 逐被试卷级结果（按跨方法均值降序）", "",
          "cross_method_mean = 45 个同划分可部署方法的逐被试 AUC 均值；max = 其中最优（oracle 选择，不可部署）；"
          "own_cv = 本人标签交叉拟合上限；cos = 本人方向与全局方向的余弦；transfer_eff = xs/own_cv。", "",
          "| 被试 | 跨方法均值 | 最优方法 | 最优-均值 | own_cv | \\|cos\\| | xs | 迁移效率 | 同人上限 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for jj in np.argsort(-subj_mean):
        q = subj_ok[jj]
        col = Ad[:, jj]
        bi = int(np.nanargmax(col))
        w = wgm.get(q, {})
        md.append(f"| {q} | {subj_mean[jj]:.3f} | {names_m[bi]} | {col[bi]-subj_mean[jj]:+.3f} | "
                  f"{w.get('own_cv', float('nan')):.3f} | {abs(w.get('cos', float('nan'))):.3f} | "
                  f"{w.get('xs', float('nan')):.3f} | {w.get('transfer_eff', float('nan')):.2f} | "
                  f"{w.get('within', float('nan')):.3f} |")
    (HERE / "subject_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    out = {"n_methods": len(names), "subjects": subj_ok,
           "meta": meta, "matrix": {m: rows[m] for m in names},
           "strict_methods": names_m,
           "variance_share": {"subject": float(np.nansum(sm**2)*X.shape[0]/tot),
                              "method": float(np.nansum(mm**2)*X.shape[1]/tot),
                              "interaction": float(np.nansum(res**2)/tot)},
           "method_spearman_median": float(np.median(rhos)),
           "subject_mean": dict(zip(subj_ok, np.round(subj_mean, 4).tolist())),
           "subject_std": dict(zip(subj_ok, np.round(subj_std, 4).tolist())),
           "best_method_count": best_cnt}
    (HERE / "method_subject_matrix.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
    print("\n[done] 写出 method_subject_matrix.{md,csv,json}")


if __name__ == "__main__":
    main()
