"""E86 fixed-seed paired replication; independent caches, no parameter search."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("e86_paired", HERE.parent / "e86_subject_adversarial.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
METHODS = {"plain": 0.0, "adv": 0.5}
KEYS = ("auc", "f1", "bacc")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def fit_adv(Xs, ys, adv_lambda, seed, epochs=80):
    # Same operations and architecture as E86; only audit hashes are added.
    torch.manual_seed(seed)
    X = torch.cat(Xs)
    y = torch.tensor(np.concatenate(ys), device=X.device, dtype=torch.float32)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    Z = (X - mu) / sd
    domains = torch.cat([torch.full((len(x),), i, device=X.device, dtype=torch.long) for i, x in enumerate(Xs)])
    net = E.AdvNet(X.shape[1], len(Xs)).to(X.device)
    initial = hashlib.sha256()
    for name, tensor in net.state_dict().items():
        initial.update(name.encode())
        initial.update(tensor.detach().cpu().numpy().tobytes())
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    pw = (y == 0).sum() / (y == 1).sum()
    gen = torch.Generator(device=X.device)
    gen.manual_seed(seed + 17)
    order = hashlib.sha256()
    epoch_hashes = []
    bs = 2048
    for epoch in range(epochs):
        idx = torch.randperm(len(Z), device=X.device, generator=gen)
        data = idx.cpu().numpy().tobytes()
        order.update(data)
        epoch_hashes.append(hashlib.sha256(data).hexdigest())
        for start in range(0, len(Z), bs):
            j = idx[start:start + bs]
            h = net.encoder(Z[j])
            cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(net.cls(h).squeeze(1), y[j], pos_weight=pw)
            dom_loss = torch.nn.functional.cross_entropy(net.dom(E.GRL.apply(h, adv_lambda)), domains[j])
            loss = cls_loss + dom_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
    def predict(Xnew):
        with torch.no_grad():
            return net.cls(net.encoder((Xnew - mu) / sd)).squeeze(1)
    audit = dict(seed=seed, batch_seed=seed + 17, initial_state_sha256=initial.hexdigest(),
                 batch_order_sha256=order.hexdigest(), epoch_order_sha256=epoch_hashes,
                 epochs=epochs, batch_size=bs, train_frames=len(Z))
    return predict, audit


def assert_pair(audits):
    assert audits["plain"] == audits["adv"], "Paired initialization or batch-order mismatch"


def summarize(folds, config):
    complete = len(folds) == 57 and [r["target"] for r in folds] == config["targets"]
    aggregate, comparisons = {}, {}
    for method in METHODS:
        aggregate[method] = {}
        for part in E.PARTS:
            aggregate[method][part] = {}
            for key in KEYS:
                vals = [r["metrics"][method][part][key] for r in folds if r["metrics"][method][part][key] is not None]
                aggregate[method][part][key] = float(np.mean(vals)) if vals else None
    for part in E.PARTS:
        comparisons[part] = {}
        for key in KEYS:
            vals = [r["delta_adv_minus_plain"][part][key] for r in folds if r["delta_adv_minus_plain"][part][key] is not None]
            comparisons[part][key] = dict(mean_delta=float(np.mean(vals)) if vals else None,
                improved=sum(v > 0 for v in vals), regressed=sum(v < 0 for v in vals),
                equal=sum(v == 0 for v in vals), valid_subjects=len(vals))
    return dict(status="complete" if complete else "partial", completed_folds=len(folds),
                config=config, aggregate=aggregate, comparisons=comparisons, folds=folds)


def report(result):
    lines = ["# E86 同种子配对复验", "", f"状态：{result['status']}；完成 {result['completed_folds']}/57 折。", "",
             "仅比较 plain λ=0 与 adv λ=0.5；每折双方种子为 89000+fold（fold 从1开始）。",
             "复用原 E86 的 E42 特征、网络、80 epochs、batch size 2048、AdamW；标准化及输出分数尺度仅来自56名训练被试。EMA1200 + 因果前向 HMM300 固定，不扫描权重、不用目标标签调参。plain 仍训练域头，但 GRL λ=0 阻断域损失对编码器的梯度，与原 E86 一致。",
             "每折完整初始 state 和80轮排列 SHA256 已逐项断言一致；排列按2048切分即实际批次顺序。目标标签仅用于最终评估，block仅用于分层评估。", "",
             "## 被试宏平均指标", "", "|方法|阶段|AUC|F1|bACC|", "|---|---|---:|---:|---:|"]
    for method in METHODS:
        for part in E.PARTS:
            m = result["aggregate"][method][part]
            lines.append(f"|{method}|{part}|" + "|".join(f"{m[k]:.6f}" if m[k] is not None else "NA" for k in KEYS) + "|")
    lines += ["", "full=全程，early=block≤2，late=block≥4；AUC/bACC遇单类别阶段记NA，不纳入该指标宏平均；F1遵循原E86的zero_division=0。", "",
              "## 配对差值（adv − plain）", "", "|阶段|指标|平均差值|改善|退步|持平|有效人数|", "|---|---|---:|---:|---:|---:|---:|"]
    for part, metrics in result["comparisons"].items():
        for key, c in metrics.items():
            lines.append(f"|{part}|{key}|{c['mean_delta']:+.6f}|{c['improved']}|{c['regressed']}|{c['equal']}|{c['valid_subjects']}|")
    lines += ["", "## 逐被试差值", "", "|被试|阶段|ΔAUC|ΔF1|ΔbACC|", "|---|---|---:|---:|---:|"]
    for row in result["folds"]:
        for part, m in row["delta_adv_minus_plain"].items():
            lines.append(f"|{row['target']}|{part}|" + "|".join(f"{m[k]:+.6f}" if m[k] is not None else "NA" for k in KEYS) + "|")
    lines += ["", "## 解释与局限", "", "这是一次预先固定的配对种子方案，不是跨种子重复试验。相同初始化和批次顺序控制了原比较中的随机训练差异，但不能证明域不变表征或其他机制已经成立。57个LOSO模型的训练集高度重叠，逐被试差值并非57次独立训练重复；不据此作独立样本显著性宣称。仍需预先固定多组配对种子检验稳定性，不能据本轮目标标签扫描参数。",
              "E42的多尺度在线统计与EMA/HMM包含长历史状态；本实验复核原E86因果口径，不等同于已经证明满足赛题“有效感受野≤6秒”的所有解释，提交前需单独澄清。", "",
              "独立缓存 fold_*.json 保存配置哈希、训练名单、双方审计哈希与指标；fold_*.npz保存真实标签、block及两方raw/score；results.json含全量结果。原E86脚本、结果及原始数据未修改。",
              "", "复现：`.venv\\Scripts\\python.exe -B test/86_subject_adversarial/paired_seed/paired_seed.py`。短程验证：追加 `--smoke`，仅使用合成数据，不写入正式fold缓存。"]
    (HERE / "报告_E86_同种子配对复验.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(1)
    if args.smoke:
        torch.manual_seed(123)
        xs = [torch.randn(2100, 15, device="cuda") for _ in range(3)]
        ys = [np.arange(2100) % 2 for _ in xs]
        audits = {}
        for name, lamb in METHODS.items():
            model, audits[name] = fit_adv(xs, ys, lamb, 89001, epochs=2)
            assert torch.isfinite(model(xs[0])).all()
        assert_pair(audits)
        # Full 80-epoch small synthetic equivalence against the original function.
        xs = [x[:64] for x in xs]
        ys = [y[:64] for y in ys]
        for name, lamb in METHODS.items():
            original = E.fit_adv(xs, ys, lamb, 89001)
            instrumented, _ = fit_adv(xs, ys, lamb, 89001)
            torch.testing.assert_close(original(xs[0]), instrumented(xs[0]), rtol=0, atol=0)
        print("SMOKE PASS: paired hashes, finite outputs, exact E86 prediction equivalence", flush=True)
        return
    subjects, D = E.L.load()
    assert len(subjects) == len(set(subjects)) == 57
    checks = E.checks(D, subjects)
    sources = [Path(__file__), Path(E.__file__), Path(E.E42.__file__), Path(E.E61.__file__), Path(E.L.__file__)]
    config = dict(experiment="E86_paired_seed_v1", targets=subjects, epochs=80, batch_size=2048,
                  seed="89000+fold (1-based)", batch_seed="seed+17", methods=METHODS,
                  features=list(E.E42.SCALES), postprocess="EMA1200+HMM300", standardization="train-fold only",
                  source_sha256={str(p): digest(p) for p in sources},
                  data_sha256={q: v["source_sha256"] for q, v in checks.items()},
                  torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0), cpu_threads=1)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest = HERE / "protocol.json"
    if manifest.exists():
        assert json.loads(manifest.read_text(encoding="utf-8"))["config_sha256"] == config_hash, "Cache configuration changed"
    write_json(manifest, dict(status="running", config=config, config_sha256=config_hash, data_checks=checks))
    F = E.E42.build_features(subjects, D)
    folds = []
    for fold, q in enumerate(subjects, 1):
        path = HERE / f"fold_{fold:02d}_{q}.json"
        scores_path = path.with_suffix(".npz")
        tr = [p for p in subjects if p != q]
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            assert row["config_sha256"] == config_hash and row["target"] == q and row["fold"] == fold
            assert row["train_subjects"] == tr and row["audit"]["plain"]["seed"] == 89000 + fold
            assert row["scores_sha256"] == digest(scores_path)
            assert_pair(row["audit"])
            folds.append(row)
            print(f"[{fold}/57] {q} verified cache", flush=True)
            continue
        start = time.time()
        Xs, ys = [F[p] for p in tr], [D[p][1] for p in tr]
        raw, scores, metrics, audits = {}, {}, {}, {}
        for name, lamb in METHODS.items():
            model, audits[name] = fit_adv(Xs, ys, lamb, 89000 + fold)
            train_raw = [model(F[p]).detach() for p in tr]
            sd = float(torch.cat(train_raw).std())
            assert np.isfinite(sd) and sd > 0
            raw[name] = model(F[q]).detach().cpu().numpy() / sd
            scores[name] = E.E61.hmm(E.E61.ema_center(raw[name], 1200, None), 300)
            assert np.isfinite(scores[name]).all()
            metrics[name] = E.evaluate(D[q][1], D[q][2], scores[name])
            del model, train_raw
        assert_pair(audits)
        delta = {part: {key: metrics["adv"][part][key] - metrics["plain"][part][key]
                 if metrics["adv"][part][key] is not None and metrics["plain"][part][key] is not None else None
                 for key in KEYS} for part in E.PARTS}
        np.savez_compressed(scores_path, target=np.array(q), train_subjects=np.array(tr), y=D[q][1], block=D[q][2],
                            **{f"{m}_raw": raw[m] for m in METHODS}, **{f"{m}_score": scores[m] for m in METHODS})
        row = dict(fold=fold, target=q, train_subjects=tr, config_sha256=config_hash, audit=audits,
                   metrics=metrics, delta_adv_minus_plain=delta, scores_sha256=digest(scores_path), elapsed_s=time.time()-start)
        write_json(path, row)
        folds.append(row)
        write_json(HERE / "progress.json", dict(status="partial", completed_folds=len(folds), total_folds=57))
        print(f"[{fold}/57] {q} {time.time()-start:.1f}s plain={metrics['plain']['full']} adv={metrics['adv']['full']}", flush=True)
        torch.cuda.empty_cache()
    result = summarize(folds, config)
    assert result["status"] == "complete"
    write_json(HERE / "results.json", result)
    write_json(manifest, dict(status=result["status"], config=config, config_sha256=config_hash, data_checks=checks))
    write_json(HERE / "progress.json", dict(status=result["status"], completed_folds=len(folds), total_folds=57))
    report(result)
    print(json.dumps({k: result[k] for k in ("status", "completed_folds", "aggregate", "comparisons")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
