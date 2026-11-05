import json
from pathlib import Path

r = json.loads((Path(__file__).parent / "e44.json").read_text(encoding="utf-8"))
for k, v in r.items():
    parts = []
    for a in ("full", "late45"):
        s = v[a]
        x = s["all"]
        parts.append(f"{a}:{x['bacc']:.4f}/{x['auc']:.4f}/{x['f1']:.4f} n90={x['n90']} min={x['min']:.3f} "
                     f"auc<.5={x['auc_below_0.5']} H={s['high']['bacc']:.3f} L={s['low']['bacc']:.3f} R={s['rev4']['bacc']:.3f}")
    print(f"{k:22s}", " | ".join(parts))
print(r["nested_select"]["chosen_counts"])

