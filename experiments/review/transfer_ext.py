"""
Extension of transfer.py / sentiment.py for the review response (does not modify them).

(1) Head-parity control: for each CCA-type map, English in-language accuracy of the head
    refitted on the CCA-mapped English training embeddings (vs. 0.879 raw e5), and a
    refitted-head variant of centred Procrustes (head trained on centred English, target
    mapped without adding the mean back), so every map can be compared under a refitted head.
(2) Many maps per task (ridge and CCA lambda sweeps, CCA truncation sweep, Procrustes, MLP)
    so that the P@1-vs-transfer rank correlation has n well above 6.

Usage: python experiments/review/transfer_ext.py {intent|sentiment} [cuda:0]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
from tasks import load_crosslingual  # noqa: E402
import common as C  # noqa: E402
import transfer as T  # noqa: E402
import sentiment as S  # noqa: E402

CCA_LAMS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
KS = [64, 128, 256, 512, 768]


def cca_k_lam(Y, X, k, lam):
    fy, fx = C.cca_map(Y, X, lam)
    return (lambda y: fy(y)[:, :k]), (lambda x: fx(x)[:, :k])


def main():
    task = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
    if task == "intent":
        en_tr, en_te = T.massive("en-US", "train", T.SRC, device), T.massive("en-US", "test", T.SRC, device)
        langs = [(lab, flores, enc, (lambda s, code=code, enc=enc: T.massive(code, s, enc, device)))
                 for code, flores, enc, lab in T.TARGETS]
    else:
        en_tr, en_te = S.data("english", "train", S.SRC, device), S.data("english", "test", S.SRC, device)
        langs = [(lab, flores, enc, (lambda s, cfg=cfg, enc=enc: S.data(cfg, s, enc, device)))
                 for cfg, flores, enc, lab in S.T]
    h_raw = T.head(en_tr["X"], en_tr["y"])
    res = {"_english_raw": T.acc(h_raw, en_te["X"], en_te["y"])}
    for lab, flores, enc, get in langs:
        tr, te = get("train"), get("test")
        d = load_crosslingual("eng_Latn", flores, T.SRC, "cpu", encoder_tgt=enc)
        Ytr, Xtr, Yf, Xf = d.Y_train, d.X_train, d.Y_test, d.X_test
        mY, mX = Ytr.mean(0), Xtr.mean(0)
        fo, _ = C.ortho_map(Ytr, Xtr)
        maps, refit = {}, {}
        maps["ortho-uncentred"] = C.ortho_map(Ytr, Xtr)
        maps["ortho-centred"] = ((lambda y, fo=fo, mY=mY, mX=mX: fo(y - mY) + mX), (lambda x: x))
        # centred-Procrustes, head refit on centred English (no mean added back)
        maps["ortho-centred|refit"] = ((lambda y, fo=fo, mY=mY: fo(y - mY)), (lambda x, mX=mX: x - mX))
        for lam in CCA_LAMS:
            maps[f"cca|lam={lam:g}"] = C.cca_map(Ytr, Xtr, lam)
            for k in KS:
                maps[f"cca-k{k}|lam={lam:g}"] = cca_k_lam(Ytr, Xtr, k, lam)
        for lam in C.LAMBDAS_RIDGE:
            maps[f"ridge|lam={lam:g}"] = C.ridge_map(Ytr, Xtr, lam)
        maps["ridge|tuned"] = C.ridge_tuned(Ytr, Xtr)[0]
        maps["cca|tuned"] = C.cca_tuned(Ytr, Xtr)[0]
        maps["mlp-nce|tuned"] = C.mlp_tuned(Ytr, Xtr, loss="infonce", device=device)[0]
        r = {"supervised": T.acc(T.head(tr["X"], tr["y"]), te["X"], te["y"])}
        for k, (fy, fx) in maps.items():
            refitted = k.startswith("cca") or k.endswith("|refit")
            hk = T.head(fx(en_tr["X"]), en_tr["y"]) if refitted else h_raw
            r[k] = T.acc(hk, fy(te["X"]), te["y"])
            r[f"p1|{k}"] = C.p1(fy(Yf), fx(Xf))
            if refitted:   # English in-language accuracy in the mapped space
                r[f"en|{k}"] = T.acc(hk, fx(en_te["X"]), en_te["y"])
        res[lab] = r
        print(lab, {k: round(v, 3) for k, v in r.items() if "|lam=" not in k and "-k" not in k}, flush=True)
    labs = [l[0] for l in langs]
    keys = [k for k in res[labs[0]] if not k.startswith("p1|") and not k.startswith("en|") and k != "supervised"]
    avg = {k: float(np.mean([res[l][k] for l in labs])) for k in res[labs[0]]}
    res["_avg"] = avg
    # rank correlation, pooled means over languages, n = number of maps
    a = [avg[k] for k in keys]; p = [avg[f"p1|{k}"] for k in keys]
    rho, pv = spearmanr(p, a)
    per_lang = [spearmanr([res[l][f"p1|{k}"] for k in keys], [res[l][k] for k in keys])[0] for l in labs]
    res["_corr"] = {"n_maps": len(keys), "spearman_pooled": float(rho), "p": float(pv),
                    "spearman_per_language_mean": float(np.mean(per_lang)),
                    "spearman_per_language": [float(x) for x in per_lang],
                    "best_by_p1": max(keys, key=lambda k: avg[f"p1|{k}"]),
                    "best_by_transfer": max(keys, key=lambda k: avg[k])}
    print("CORR", res["_corr"], flush=True)
    out = ROOT / "results" / "review" / f"transfer_ext_{task}.json"
    out.write_text(json.dumps(res, indent=1)); print("saved", out)


if __name__ == "__main__":
    main()
