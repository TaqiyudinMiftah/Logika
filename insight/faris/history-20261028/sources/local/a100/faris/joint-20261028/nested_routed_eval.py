import os
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "4"

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

OUT_ROOT = Path("/mnt/home/faris/logika-joint-20261028")
FRZ_ROOT = Path("/mnt/home/faris/logika-frozen-shadow-20261026")
EXP_ROOT = Path("/mnt/home/faris/logika-faris-exploration")
PERIODIC = EXP_ROOT / "shopping-periodicity-fe-20261021"

sys.path.insert(0, str(EXP_ROOT / "transfer-20261025/source"))
import baseline as b_mod
import feature_head

with open(FRZ_ROOT / "frozen/fold0/MANIFEST.json") as f:
    HEAD18_COLS = json.load(f)["head_columns"]
NAMES = ["power_7d", "power_14d", "power_30d", "periodicity_strength", "dominant_cycle_days"]
EXTRA_PER = ["beh_spectral_" + channel + "_" + name for channel in ["spend", "orders"] for name in NAMES] + ["beh_spectral_window_supported"]
HEAD_PER_COLS = HEAD18_COLS + EXTRA_PER

rob_npz = np.load(EXP_ROOT / "transfer-20261025/ROBUSTNESS_OOF.npz", allow_pickle=True)
y_global = rob_npz["target"].astype(np.int8)
folds_global = rob_npz["fold"].astype(np.int8)
baseline_oof = rob_npz["maturity__original"].astype(np.float64)
per_orig_oof = rob_npz["periodicity__original"].astype(np.float64)
h016_oof = np.load(EXP_ROOT / "transfer-20261025/spline_direct_ks/OOF.npz", allow_pickle=True)["two_seed_mean"].astype(np.float64)

P_TOT = float(np.sum(y_global == 1))
N_TOT = float(np.sum(y_global == 0))
ROW_W = np.where(y_global == 1, 1.0 / P_TOT, -1.0 / N_TOT)

def ks_stat(y, p):
    order = np.argsort(p, kind="mergesort")
    ys = y[order]
    ps = p[order]
    pos = np.cumsum(ys == 1) / max(np.sum(ys == 1), 1)
    neg = np.cumsum(ys == 0) / max(np.sum(ys == 0), 1)
    diff = np.abs(pos - neg)
    end = np.r_[ps[:-1] != ps[1:], True]
    return float(np.max(diff[end]))

def smooth_opt_threshold(y_sub, p_sub, pi_global=0.13581, temp=0.15):
    # Find threshold t in [0.08, 0.22] that maximizes smooth global KS contribution on training/correction data:
    # w_pos = pi_global / P_sub_rate, or simply global Neyman-Pearson weights:
    # Every positive has weight 1 / pi_global, every negative has weight -1 / (1 - pi_global)
    z = logit(np.clip(p_sub, 1e-6, 1.0 - 1e-6))
    w = np.where(y_sub == 1, 1.0 / pi_global, -1.0 / (1.0 - pi_global))
    ts = np.linspace(0.09, 0.20, 221)
    z_ts = logit(ts)
    # Smooth step: expit((z - z_t) / temp)
    scores = [np.sum(w * expit((z - zt) / temp)) for zt in z_ts]
    return float(ts[int(np.argmax(scores))])

raw_npz = np.load(OUT_ROOT / "raw_domain_features.npz", allow_pickle=True)
raw_names = [str(x) for x in raw_npz["names"]]
raw_data = raw_npz["data"]
pop_masks = {p: raw_data[:, raw_names.index(p)] > 0.5 for p in ["pop_A", "pop_B", "pop_C1", "pop_C2"]}

orig_train_ids = pd.read_parquet(EXP_ROOT / "ranking-pipeline-20261020/input/data/train_features.parquet", columns=["application_id"])["application_id"]
spectral_df = pd.read_parquet(PERIODIC / "true_features.parquet", columns=["application_id"] + EXTRA_PER).set_index("application_id").loc[orig_train_ids].reset_index(drop=True)

roles_list = []
exp_tables_70k = []
sp_mat_val = []
sp_h16_val = []
kumo56_70k = []

for f in range(5):
    r = np.load(FRZ_ROOT / f"frozen/fold{f}/roles.npz", allow_pickle=True)
    roles_list.append(r)
    b = np.load(FRZ_ROOT / f"frozen/fold{f}/baseline.npz", allow_pickle=True)
    sp_mat_val.append(b["spline"])
    tb = pd.read_parquet(OUT_ROOT / f"fold{f}/expanded.parquet")
    for c in EXTRA_PER:
        tb[c] = spectral_df[c].to_numpy(dtype=np.float32)
    exp_tables_70k.append(tb)

    q_idx = np.r_[r["correction"], r["validation"]]
    tb_q = tb.iloc[q_idx].reset_index(drop=True)
    h29 = feature_head.fit(tb_q.iloc[:6000].assign(target=r["y_correction"]), b["raw"][:6000], HEAD_PER_COLS, 0.003, 0.5)
    c29_val = b_mod.corrected_probability(h29, tb_q.iloc[6000:], b["raw"][6000:])[0]
    sp_per_f = (per_orig_oof[r["validation"]] - 0.75 * c29_val) / 0.25
    h16_v = np.load(EXP_ROOT / f"transfer-20261025/spline_direct_ks/fold{f}/validation.npz")
    sp_h16_val.append(sp_per_f + h16_v["spline_delta"])

    meta = np.load(EXP_ROOT / f"nested-correction-20261021/heads/fold{f}/meta_inputs.npz", allow_pickle=True)
    raw5 = np.load(EXP_ROOT / f"nested-correction-20261021/runs/job{f*6+5}/raw.npz", allow_pickle=True)
    pk = np.full(70000, np.nan, dtype=np.float64)
    pk[meta["rows"]] = meta["prediction"]
    pk[raw5["rows"]] = raw5["prediction"]
    kumo56_70k.append(pk)

def load_20k(path_list, key="raw"):
    arrs = []
    for f, p in enumerate(path_list):
        z = np.load(p, allow_pickle=True)
        k = key if key in z else ("prediction" if "prediction" in z else ("raw" if "raw" in z else list(z.keys())[0]))
        v = np.asarray(z[k], dtype=np.float64).reshape(-1)
        q_want = np.r_[roles_list[f]["correction"], roles_list[f]["validation"]]
        if "rows" in z and len(z["rows"]) == 20000:
            rows_in = np.asarray(z["rows"], dtype=np.int64)
            if not np.array_equal(rows_in, q_want):
                vec = np.full(70000, np.nan, dtype=np.float64)
                vec[rows_in] = v
                v = vec[q_want]
        elif "query_rows" in z and len(z["query_rows"]) == 20000:
            rows_in = np.asarray(z["query_rows"], dtype=np.int64)
            if not np.array_equal(rows_in, q_want):
                vec = np.full(70000, np.nan, dtype=np.float64)
                vec[rows_in] = v
                v = vec[q_want]
        arrs.append(v)
    return arrs

m20k = {
    "base_raw": [np.load(FRZ_ROOT / f"frozen/fold{f}/baseline.npz")["raw"] for f in range(5)],
    "tr_seed": load_20k([EXP_ROOT / f"transfer-20261025/robustness/fold{f}/raw_seed_repeat.npz" for f in range(5)], "prediction"),
    "pop_spec": load_20k([OUT_ROOT / f"kumo_runs/kumo_pop_specialist/fold{f}/prediction.npz" for f in range(5)]),
    "surg96": load_20k([OUT_ROOT / f"kumo_runs/kumo_surgical96/fold{f}/prediction.npz" for f in range(5)]),
    "cov96": load_20k([EXP_ROOT / f"kumo-coverage-placement-20261021/conditions/fold{f}/coverage96/raw_predictions.npz" for f in range(5)], "prediction"),
    "fe_cov": load_20k([EXP_ROOT / f"coverage-information-fe-20261021/runs/fold{f}/raw.npz" for f in range(5)], "prediction"),
    "fe_mat": load_20k([EXP_ROOT / f"mature-ma[REDACTED_CREDENTIAL]/runs/fold{f}/raw.npz" for f in range(5)], "prediction"),
    "horiz_b90": load_20k([Path(f"/mnt/home/faris/logika-horizontal-20261026/runs/large8/baseline90/fold{f}/prediction.npz") for f in range(5)]),
    "champ_s7731": load_20k([FRZ_ROOT / f"runs/kumo_large_champion90_s7731/fold{f}/prediction.npz" for f in range(5)]),
    "shadow91": load_20k([FRZ_ROOT / f"runs/kumo_large_shadow91/fold{f}/prediction.npz" for f in range(5)]),
    "tr_early": load_20k([EXP_ROOT / f"transfer-20261025/early_average/fold{f}/raw_mean.npz" for f in range(5)], "prediction"),
    "tr_mat_orig": load_20k([EXP_ROOT / f"transfer-20261025/core_maturity/fold{f}/original_raw.npz" for f in range(5)], "prediction"),
    "kumo56": [kumo56_70k[f][np.r_[roles_list[f]["correction"], roles_list[f]["validation"]]] for f in range(5)],
}

lin56 = np.load(OUT_ROOT / "linear56k_runs.npz")
sp200_cor = [lin56["oof_spline200"][f][roles_list[f]["correction"]] for f in range(5)]
sp200_70k = lin56["val_spline200"]

cands_cor = {"sp200_56k": sp200_cor}
cands_70k = {"sp200_56k": sp200_70k}

for mk, arrs in m20k.items():
    cands_cor[f"raw_{mk}"] = []
    cands_cor[f"h18_{mk}"] = []
    cands_cor[f"h29_{mk}"] = []
    cands_cor[f"mat18_{mk}"] = []
    cands_cor[f"per29_{mk}"] = []
    cands_70k[f"raw_{mk}"] = np.zeros(70000)
    cands_70k[f"h18_{mk}"] = np.zeros(70000)
    cands_70k[f"h29_{mk}"] = np.zeros(70000)
    cands_70k[f"mat18_{mk}"] = np.zeros(70000)
    cands_70k[f"per29_{mk}"] = np.zeros(70000)

    for f in range(5):
        r = roles_list[f]
        q_idx = np.r_[r["correction"], r["validation"]]
        tb_q = exp_tables_70k[f].iloc[q_idx].reset_index(drop=True)
        p_f = arrs[f]
        cands_cor[f"raw_{mk}"].append(p_f[:6000])
        cands_70k[f"raw_{mk}"][r["validation"]] = p_f[6000:]

        h18 = feature_head.fit(tb_q.iloc[:6000].assign(target=r["y_correction"]), p_f[:6000], HEAD18_COLS, 0.003, 0.5)
        c18_c = b_mod.corrected_probability(h18, tb_q.iloc[:6000], p_f[:6000])[0]
        c18_v = b_mod.corrected_probability(h18, tb_q.iloc[6000:], p_f[6000:])[0]
        cands_cor[f"h18_{mk}"].append(c18_c)
        cands_70k[f"h18_{mk}"][r["validation"]] = c18_v
        cands_cor[f"mat18_{mk}"].append(0.75 * c18_c + 0.25 * sp200_cor[f])
        cands_70k[f"mat18_{mk}"][r["validation"]] = 0.75 * c18_v + 0.25 * sp_mat_val[f]

        h29 = feature_head.fit(tb_q.iloc[:6000].assign(target=r["y_correction"]), p_f[:6000], HEAD_PER_COLS, 0.003, 0.5)
        c29_c = b_mod.corrected_probability(h29, tb_q.iloc[:6000], p_f[:6000])[0]
        c29_v = b_mod.corrected_probability(h29, tb_q.iloc[6000:], p_f[6000:])[0]
        cands_cor[f"h29_{mk}"].append(c29_c)
        cands_70k[f"h29_{mk}"][r["validation"]] = c29_v
        cands_cor[f"per29_{mk}"].append(0.75 * c29_c + 0.25 * sp200_cor[f])
        cands_70k[f"per29_{mk}"][r["validation"]] = 0.75 * c29_v + 0.25 * sp_h16_val[f]


# ---- cache all candidates ----
names = sorted(cands_70k.keys())
np.savez_compressed(OUT_ROOT / "cands_cache.npz", names=np.array(names),
                    val=np.stack([cands_70k[n] for n in names]),
                    cor=np.stack([np.concatenate([cands_cor[n][f] for f in range(5)]) for n in names]))
print("cached", len(names), "candidates", flush=True)

V = np.stack([cands_70k[n] for n in names])
pop_id = np.zeros(70000, dtype=int)
for i, pn in enumerate(["pop_A", "pop_B", "pop_C1", "pop_C2"]):
    pop_id[pop_masks[pn]] = i

def submax(p, y, w):
    o = np.argsort(p, kind="mergesort"); ps = p[o]; ws = w[o]
    cum = ws.sum() - np.cumsum(ws)
    end = np.r_[ps[:-1] != ps[1:], True]
    return float(cum[end].max())

def greedy(mask, steps=10):
    w = ROW_W[mask]; y = y_global[mask]
    Vm = V[:, mask]
    sc = [submax(Vm[j], y, w) for j in range(len(names))]
    chosen = [int(np.argmax(sc))]; cur = Vm[chosen[0]].copy(); best = sc[chosen[0]]
    for s in range(1, steps):
        bj, bv = None, best
        for j in range(len(names)):
            v = submax((s * cur + Vm[j]) / (s + 1), y, w)
            if v > bv: bj, bv = j, v
        if bj is None: break
        chosen.append(bj); cur = (s * cur + Vm[bj]) / (s + 1); best = bv
    return chosen

nested = np.zeros(70000); nested_glob = np.zeros(70000)
for k in range(5):
    tr = folds_global != k; te = folds_global == k
    for g in range(4):
        sel = greedy(tr & (pop_id == g))
        m = te & (pop_id == g)
        nested[m] = V[sel][:, m].mean(0)
        print(f"fold{k} pop{g}: {[names[j] for j in sel]}", flush=True)
    selg = greedy(tr)
    nested_glob[te] = V[selg][:, te].mean(0)
    print(f"fold{k} global: {[names[j] for j in selg]}", flush=True)

def fks(p):
    return [round(ks_stat(y_global[folds_global == f], p[folds_global == f]), 5) for f in range(5)]
print(f"NESTED subgroup-routed KS = {ks_stat(y_global, nested):.6f} AUC={roc_auc_score(y_global, nested):.6f} folds={fks(nested)}")
print(f"NESTED global greedy KS   = {ks_stat(y_global, nested_glob):.6f} AUC={roc_auc_score(y_global, nested_glob):.6f} folds={fks(nested_glob)}")
print(f"H016 KS                   = {ks_stat(y_global, h016_oof):.6f} folds={fks(h016_oof)}")
print(f"Baseline KS               = {ks_stat(y_global, baseline_oof):.6f} folds={fks(baseline_oof)}")
np.savez_compressed(OUT_ROOT / "nested_routed_oof.npz", nested=nested, nested_glob=nested_glob)
