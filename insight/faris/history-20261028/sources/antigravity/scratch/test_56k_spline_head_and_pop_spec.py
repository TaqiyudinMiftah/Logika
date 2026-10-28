import os
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "4"

import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import SplineTransformer, StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

t0 = time.time()
OUT_ROOT = Path("/mnt/home/faris/logika-joint-20261028")
FRZ_ROOT = Path("/mnt/home/faris/logika-frozen-shadow-20261026")
EXP_ROOT = Path("/mnt/home/faris/logika-faris-exploration")

sys.path.insert(0, str(EXP_ROOT / "transfer-20261025/source"))
import baseline as b_mod
import feature_head

with open(FRZ_ROOT / "frozen/fold0/MANIFEST.json") as f:
    FOLD0_MAN = json.load(f)
HEAD18_COLS = FOLD0_MAN["head_columns"]

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

def clip_prob(p):
    return np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)

# 1. Load global y, roles, baseline, and expanded tables
old_npz = np.load(EXP_ROOT / "transfer-20261025/ROBUSTNESS_OOF.npz", allow_pickle=True)
y_global = old_npz["target"].astype(np.int8)
folds_global = old_npz["fold"].astype(np.int8)
baseline_oof = old_npz["maturity__original"].astype(np.float64)

roles_list = []
baseline_raw_20k = []
baseline_spline_val = []
exp_tables_70k = []
kumo56_70k = []

for f in range(5):
    r = np.load(FRZ_ROOT / f"frozen/fold{f}/roles.npz", allow_pickle=True)
    roles_list.append(r)
    b = np.load(FRZ_ROOT / f"frozen/fold{f}/baseline.npz", allow_pickle=True)
    baseline_raw_20k.append(b["raw"])
    baseline_spline_val.append(b["spline"])
    exp_tables_70k.append(pd.read_parquet(OUT_ROOT / f"fold{f}/expanded.parquet"))

    meta = np.load(EXP_ROOT / f"nested-correction-20261021/heads/fold{f}/meta_inputs.npz", allow_pickle=True)
    raw5 = np.load(EXP_ROOT / f"nested-correction-20261021/runs/job{f*6+5}/raw.npz", allow_pickle=True)
    pk = np.full(70000, np.nan, dtype=np.float64)
    pk[meta["rows"]] = meta["prediction"]
    pk[raw5["rows"]] = raw5["prediction"]
    kumo56_70k.append(pk)

# 2. Load top 20k Kumo & Foundation models
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
    "base_raw": baseline_raw_20k,
    "pop_spec": load_20k([OUT_ROOT / f"kumo_runs/kumo_pop_specialist/fold{f}/prediction.npz" for f in range(5)]),
    "surg96": load_20k([OUT_ROOT / f"kumo_runs/kumo_surgical96/fold{f}/prediction.npz" for f in range(5)]),
    "champ90_s7731": load_20k([FRZ_ROOT / f"runs/kumo_large_champion90_s7731/fold{f}/prediction.npz" for f in range(5)]),
    "shadow91": load_20k([FRZ_ROOT / f"runs/kumo_large_shadow91/fold{f}/prediction.npz" for f in range(5)]),
    "shadow91_s7731": load_20k([FRZ_ROOT / f"runs/kumo_large_shadow91_s7731/fold{f}/prediction.npz" for f in range(5)]),
    "latent99": load_20k([FRZ_ROOT / f"runs/kumo_large_latent99/fold{f}/prediction.npz" for f in range(5)]),
    "latent99_s7731": load_20k([FRZ_ROOT / f"runs/kumo_large_latent99_s7731/fold{f}/prediction.npz" for f in range(5)]),
    "full100": load_20k([FRZ_ROOT / f"runs/kumo_large_full100/fold{f}/prediction.npz" for f in range(5)]),
    "full100_s7731": load_20k([FRZ_ROOT / f"runs/kumo_large_full100_s7731/fold{f}/prediction.npz" for f in range(5)]),
    "med_latent99": load_20k([FRZ_ROOT / f"runs/kumo_medium_latent99/fold{f}/prediction.npz" for f in range(5)]),
    "med_full100": load_20k([FRZ_ROOT / f"runs/kumo_medium_full100/fold{f}/prediction.npz" for f in range(5)]),
    "kumo_sup90": load_20k([EXP_ROOT / f"kumo-supervised-view-20261021/folds/fold{f}/raw.npz" for f in range(5)], "prediction"),
    "covplace96": load_20k([EXP_ROOT / f"kumo-coverage-placement-20261021/conditions/fold{f}/coverage96/raw_predictions.npz" for f in range(5)], "prediction"),
    "fe_cov_info": load_20k([EXP_ROOT / f"coverage-information-fe-20261021/runs/fold{f}/raw.npz" for f in range(5)], "prediction"),
    "fe_mat_mask": load_20k([EXP_ROOT / f"mature-ma[REDACTED_CREDENTIAL]/runs/fold{f}/raw.npz" for f in range(5)], "prediction"),
    "fe_mat_denom": load_20k([EXP_ROOT / f"mature-denominator-fe-20261021/runs/fold{f}/raw.npz" for f in range(5)], "prediction"),
    "horiz_l8_b90": load_20k([Path(f"/mnt/home/faris/logika-horizontal-20261026/runs/large8/baseline90/fold{f}/prediction.npz") for f in range(5)]),
    "horiz_l8_p18": load_20k([Path(f"/mnt/home/faris/logika-horizontal-20261026/runs/large8/p0018/fold{f}/prediction.npz") for f in range(5)]),
    "tr_early_avg": load_20k([EXP_ROOT / f"transfer-20261025/early_average/fold{f}/raw_mean.npz" for f in range(5)], "prediction"),
    "tr_seed_rep": load_20k([EXP_ROOT / f"transfer-20261025/robustness/fold{f}/raw_seed_repeat.npz" for f in range(5)], "prediction"),
    "tr_mat_orig": load_20k([EXP_ROOT / f"transfer-20261025/core_maturity/fold{f}/original_raw.npz" for f in range(5)], "prediction"),
    "tr_mat_fresh": load_20k([EXP_ROOT / f"transfer-20261025/core_maturity/fold{f}/fresh_raw.npz" for f in range(5)], "prediction"),
    "kumo56": [kumo56_70k[f][np.r_[roles_list[f]["correction"], roles_list[f]["validation"]]] for f in range(5)],
}
for pack in sorted(os.listdir("/mnt/home/faris/logika-augment90-20261026/runs/large8")):
    plist = [Path(f"/mnt/home/faris/logika-augment90-20261026/runs/large8/{pack}/fold{f}/prediction.npz") for f in range(5)]
    if all(p.exists() for p in plist):
        m20k[f"aug_l8_{pack}"] = load_20k(plist)

def eval_with_exact_head18(p_20k_list, label, sp_list=None, sp_weight=0.25, penalty=0.003, strength=0.5):
    if sp_list is None:
        sp_list = baseline_spline_val
    raw_val = np.zeros(70000, dtype=np.float64)
    head_val = np.zeros(70000, dtype=np.float64)
    final_val = np.zeros(70000, dtype=np.float64)
    cor_raw = []
    for f in range(5):
        r = roles_list[f]
        q_idx = np.r_[r["correction"], r["validation"]]
        tb = exp_tables_70k[f].iloc[q_idx].reset_index(drop=True)
        p_f = p_20k_list[f]
        cor_raw.append(p_f[:6000])
        raw_val[r["validation"]] = p_f[6000:]
        head = feature_head.fit(tb.iloc[:6000].assign(target=r["y_correction"]), p_f[:6000], HEAD18_COLS, penalty, strength)
        c_val = b_mod.corrected_probability(head, tb.iloc[6000:], p_f[6000:])[0]
        head_val[r["validation"]] = c_val
        final_val[r["validation"]] = (1.0 - sp_weight) * c_val + sp_weight * sp_list[f]
    y_cor = np.concatenate([roles_list[f]["y_correction"] for f in range(5)])
    print(f"  {label:38s}: CorRawKS={ks_stat(y_cor, np.concatenate(cor_raw)):.6f} | ValRawKS={ks_stat(y_global, raw_val):.6f} | ValHead18KS={ks_stat(y_global, head_val):.6f} | FinalKS={ks_stat(y_global, final_val):.6f} (AUC={roc_auc_score(y_global, final_val):.6f})", flush=True)
    return final_val, head_val, raw_val

print("\n=== STEP 1: SINGLE & PAIRWISE MODELS WITH EXACT HEAD18 + 0.25*SPLINE ===", flush=True)
eval_with_exact_head18(m20k["base_raw"], "01. Baseline (champion90)")
eval_with_exact_head18(m20k["pop_spec"], "02. joint_kumo_pop_specialist")
eval_with_exact_head18(m20k["surg96"], "03. joint_kumo_surgical96")
eval_with_exact_head18([0.5 * m20k["base_raw"][f] + 0.5 * m20k["champ90_s7731"][f] for f in range(5)], "04. 2-seed champion90 (Recipe A)")
eval_with_exact_head18([0.5 * m20k["pop_spec"][f] + 0.5 * m20k["base_raw"][f] for f in range(5)], "05. 50/50 pop_spec + base_raw")
eval_with_exact_head18([0.5 * m20k["surg96"][f] + 0.5 * m20k["base_raw"][f] for f in range(5)], "06. 50/50 surg96 + base_raw")
eval_with_exact_head18([0.4 * m20k["pop_spec"][f] + 0.3 * m20k["surg96"][f] + 0.3 * m20k["base_raw"][f] for f in range(5)], "07. 40/30/30 pop_spec+surg96+base")

# Top Kumo probability average
top_kumo_keys = [k for k in [
    "pop_spec", "surg96", "base_raw", "champ90_s7731",
    "shadow91", "shadow91_s7731", "latent99", "latent99_s7731", "full100", "full100_s7731",
    "kumo_sup90", "covplace96", "fe_cov_info", "fe_mat_mask", "horiz_l8_b90", "horiz_l8_p18",
    "tr_early_avg", "tr_seed_rep", "kumo56",
    "aug_l8_diverse", "aug_l8_repayment_credibility", "aug_l8_volatility", "aug_l8_liquidity_payment", "aug_l8_financial_capacity"
] if k in m20k]
core14_keys = [k for k in ["base_raw", "champ90_s7731", "shadow91", "shadow91_s7731", "kumo_sup90", "covplace96", "fe_cov_info", "fe_mat_mask", "horiz_l8_b90", "tr_early_avg", "kumo56", "aug_l8_diverse", "aug_l8_repayment_credibility", "aug_l8_volatility"] if k in m20k]
eval_with_exact_head18([np.mean([m20k[k][f] for k in top_kumo_keys], axis=0) for f in range(5)], "08. Top-Kumo Equal Prob Mean")
eval_with_exact_head18([expit(np.mean([logit(clip_prob(m20k[k][f])) for k in top_kumo_keys], axis=0)) for f in range(5)], "09. Top-Kumo Equal Logit Mean")

for w_new in [0.25, 0.35, 0.45, 0.55]:
    p_blend = [
        w_new * (0.55 * m20k["pop_spec"][f] + 0.45 * m20k["surg96"][f])
        + (1.0 - w_new) * np.mean([m20k[k][f] for k in core14_keys], axis=0)
        for f in range(5)
    ]
    eval_with_exact_head18(p_blend, f"10. Weighted New({w_new:.2f}) + Core Kumo")

# 3. Now test 56,000-row Spline Correction Head (feature_head.fit on all 56,000 training rows!) + Expanded Head Columns!
screen_df = pd.read_csv(OUT_ROOT / "inner_oof_residual_screen.csv")
top_res_cols = [c for c in screen_df[(screen_df["sign_consistency"] == 5) & (screen_df["abs_z"] >= 8.0)]["feature"].tolist() if c not in HEAD18_COLS]
print(f"\nFound {len(top_res_cols)} extra 5/5-fold consistent residual features with |z|>=8.0", flush=True)

lin56 = np.load(OUT_ROOT / "linear56k_runs.npz", allow_pickle=True)
cb_oof_npz = np.load(EXP_ROOT / "catboost-convergence-20261021/OOF.npz", allow_pickle=True)
cb_full_val = [cb_oof_npz["full"][roles_list[f]["validation"]] for f in range(5)]
sp200_val = [lin56["val_spline200"][roles_list[f]["validation"]] for f in range(5)]
cpu_lgbm_val = [np.load(OUT_ROOT / f"cpu_runs/fold{f}_crossfit56k.npz")["val_full_lgbm_56k"] for f in range(5)]
cpu_cat_val = [np.load(OUT_ROOT / f"cpu_runs/fold{f}_crossfit56k.npz")["val_full_cat_56k"] for f in range(5)]

# Best Kumo base candidate
p_kumo_best = [
    0.40 * (0.55 * m20k["pop_spec"][f] + 0.45 * m20k["surg96"][f])
    + 0.60 * np.mean([m20k[k][f] for k in core14_keys], axis=0)
    for f in range(5)
]

print("\n=== STEP 2: TESTING 6K vs 56K SPLINE FEATURE_HEAD & EXPANDED HEAD COLUMNS ===", flush=True)
for n_extra in [0, 8, 16, 28, 45]:
    cols_use = HEAD18_COLS + top_res_cols[:n_extra]
    for pen in [0.003, 0.008]:
        for strg in [0.5, 0.65]:
            # A) 6,000-row fit on correction split
            val_6k = np.zeros(70000, dtype=np.float64)
            # B) 56,000-row fit on (50k context + 6k correction) using kumo56 on context and p_kumo_best on correction!
            val_56k = np.zeros(70000, dtype=np.float64)
            val_hybrid = np.zeros(70000, dtype=np.float64)

            for f in range(5):
                r = roles_list[f]
                tb_full = exp_tables_70k[f]
                p_f = p_kumo_best[f]

                # 6k fit
                tb_cor = tb_full.iloc[r["correction"]].reset_index(drop=True).assign(target=r["y_correction"])
                tb_val = tb_full.iloc[r["validation"]].reset_index(drop=True)
                h6 = feature_head.fit(tb_cor, p_f[:6000], cols_use, pen, strg)
                c6 = b_mod.corrected_probability(h6, tb_val, p_f[6000:])[0]
                val_6k[r["validation"]] = 0.75 * c6 + 0.25 * baseline_spline_val[f]

                # 56k fit: combine 50,000 context rows (with kumo56 inner-OOF offset) + 6,000 correction rows (with p_f[:6000] offset)
                tr_idx = np.r_[r["context"], r["correction"]]
                y_tr = np.r_[r["y_context"], r["y_correction"]]
                p_tr = np.r_[kumo56_70k[f][r["context"]], p_f[:6000]]
                tb_tr = tb_full.iloc[tr_idx].reset_index(drop=True).assign(target=y_tr)
                h56 = feature_head.fit(tb_tr, p_tr, cols_use, pen, strg)
                c56 = b_mod.corrected_probability(h56, tb_val, p_f[6000:])[0]
                val_56k[r["validation"]] = 0.75 * c56 + 0.25 * baseline_spline_val[f]
                val_hybrid[r["validation"]] = 0.75 * (0.5 * c6 + 0.5 * c56) + 0.25 * (0.60 * baseline_spline_val[f] + 0.20 * sp200_val[f] + 0.10 * cb_full_val[f] + 0.10 * cpu_lgbm_val[f])

            print(f"[n_extra={n_extra:2d} pen={pen:.3f} str={strg:.2f}] 6k_KS={ks_stat(y_global, val_6k):.6f} | 56k_KS={ks_stat(y_global, val_56k):.6f} | Hybrid_KS={ks_stat(y_global, val_hybrid):.6f} (AUC={roc_auc_score(y_global, val_hybrid):.6f})", flush=True)
