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

def ks_stat(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    ys = y[order]
    ps = p[order]
    pos = np.cumsum(ys == 1) / max(np.sum(ys == 1), 1)
    neg = np.cumsum(ys == 0) / max(np.sum(ys == 0), 1)
    diff = np.abs(pos - neg)
    end = np.r_[ps[:-1] != ps[1:], True]
    return float(np.max(diff[end]))

def ks_info(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    ys = y[order]
    ps = p[order]
    pos = np.cumsum(ys == 1) / max(np.sum(ys == 1), 1)
    neg = np.cumsum(ys == 0) / max(np.sum(ys == 0), 1)
    diff = np.abs(pos - neg)
    end = np.r_[ps[:-1] != ps[1:], True]
    idx = np.flatnonzero(end)[np.argmax(diff[end])]
    return float(diff[idx]), float(ps[idx])

def clip_prob(p):
    return np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)

rob_npz = np.load(EXP_ROOT / "transfer-20261025/ROBUSTNESS_OOF.npz", allow_pickle=True)
y_global = rob_npz["target"].astype(np.int8)
folds_global = rob_npz["fold"].astype(np.int8)
baseline_oof = rob_npz["maturity__original"].astype(np.float64)
per_orig_oof = rob_npz["periodicity__original"].astype(np.float64)
h016_oof = np.load(EXP_ROOT / "transfer-20261025/spline_direct_ks/OOF.npz", allow_pickle=True)["two_seed_mean"].astype(np.float64)

raw_npz = np.load(OUT_ROOT / "raw_domain_features.npz", allow_pickle=True)
raw_names = [str(x) for x in raw_npz["names"]]
raw_data = raw_npz["data"]
pop_masks = {p: raw_data[:, raw_names.index(p)] > 0.5 for p in ["pop_A", "pop_B", "pop_C1", "pop_C2"]}

roles_list = [np.load(FRZ_ROOT / f"frozen/fold{f}/roles.npz", allow_pickle=True) for f in range(5)]

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

base_raw = [np.load(FRZ_ROOT / f"frozen/fold{f}/baseline.npz")["raw"] for f in range(5)]
tr_seed = load_20k([EXP_ROOT / f"transfer-20261025/robustness/fold{f}/raw_seed_repeat.npz" for f in range(5)], "prediction")
pop_spec = load_20k([OUT_ROOT / f"kumo_runs/kumo_pop_specialist/fold{f}/prediction.npz" for f in range(5)])
surg96 = load_20k([OUT_ROOT / f"kumo_runs/kumo_surgical96/fold{f}/prediction.npz" for f in range(5)])
cov96 = load_20k([EXP_ROOT / f"kumo-coverage-placement-20261021/conditions/fold{f}/coverage96/raw_predictions.npz" for f in range(5)], "prediction")
fe_cov = load_20k([EXP_ROOT / f"coverage-information-fe-20261021/runs/fold{f}/raw.npz" for f in range(5)], "prediction")
fe_mat = load_20k([EXP_ROOT / f"mature-ma[REDACTED_CREDENTIAL]/runs/fold{f}/raw.npz" for f in range(5)], "prediction")
horiz_b90 = load_20k([Path(f"/mnt/home/faris/logika-horizontal-20261026/runs/large8/baseline90/fold{f}/prediction.npz") for f in range(5)])
champ_s7731 = load_20k([FRZ_ROOT / f"runs/kumo_large_champion90_s7731/fold{f}/prediction.npz" for f in range(5)])
med_lat99 = load_20k([FRZ_ROOT / f"runs/kumo_medium_latent99/fold{f}/prediction.npz" for f in range(5)])
tr_prop_orig = load_20k([EXP_ROOT / f"transfer-20261025/balanced_context/fold{f}/proportional__original/raw_predictions.npz" for f in range(5)], "prior_adjusted_probability")

print("=== SUBGROUP DIAGNOSTIC ON 30K CORRECTION & 70K VALIDATION ===", flush=True)
y_cor = np.concatenate([roles_list[f]["y_correction"] for f in range(5)])
cor_idx = np.concatenate([roles_list[f]["correction"] for f in range(5)])

for mname, arrs in [("base_raw", base_raw), ("tr_seed", tr_seed), ("pop_spec", pop_spec), ("surg96", surg96), ("cov96", cov96), ("fe_cov", fe_cov), ("fe_mat", fe_mat), ("horiz_b90", horiz_b90), ("champ_s7731", champ_s7731), ("med_lat99", med_lat99), ("tr_prop_orig", tr_prop_orig)]:
    c_all = np.concatenate([a[:6000] for a in arrs])
    v_all = np.zeros(70000)
    for f in range(5):
        v_all[roles_list[f]["validation"]] = arrs[f][6000:]
    ks_v, thr_v = ks_info(y_global, v_all)
    sub_str = []
    for pn in ["pop_A", "pop_B", "pop_C1", "pop_C2"]:
        m_c = pop_masks[pn][cor_idx]
        m_v = pop_masks[pn]
        ks_p_c, thr_p_c = ks_info(y_cor[m_c], c_all[m_c])
        ks_p_v, thr_p_v = ks_info(y_global[m_v], v_all[m_v])
        sub_str.append(f"{pn}:Cor={ks_p_c:.4f}/Val={ks_p_v:.4f}(@{thr_p_v:.3f})")
    print(f"{mname:13s} | PoolCor={ks_stat(y_cor, c_all):.5f} PoolVal={ks_v:.5f}(@{thr_v:.3f}) | " + " | ".join(sub_str), flush=True)
