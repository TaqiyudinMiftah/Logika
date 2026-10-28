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

orig_train_ids = pd.read_parquet(EXP_ROOT / "ranking-pipeline-20261020/input/data/train_features.parquet", columns=["application_id"])["application_id"]
spectral_df = pd.read_parquet(PERIODIC / "true_features.parquet", columns=["application_id"] + EXTRA_PER).set_index("application_id").loc[orig_train_ids].reset_index(drop=True)

roles_list = []
exp_tables_70k = []
sp_mat_val = []
sp_per_val = []
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
    sp_per_val.append(sp_per_f)
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
    "shadow91_s7731": load_20k([FRZ_ROOT / f"runs/kumo_large_shadow91_s7731/fold{f}/prediction.npz" for f in range(5)]),
    "med_lat99": load_20k([FRZ_ROOT / f"runs/kumo_medium_latent99/fold{f}/prediction.npz" for f in range(5)]),
    "tr_early": load_20k([EXP_ROOT / f"transfer-20261025/early_average/fold{f}/raw_mean.npz" for f in range(5)], "prediction"),
    "tr_mat_orig": load_20k([EXP_ROOT / f"transfer-20261025/core_maturity/fold{f}/original_raw.npz" for f in range(5)], "prediction"),
    "tr_mat_fresh": load_20k([EXP_ROOT / f"transfer-20261025/core_maturity/fold{f}/fresh_raw.npz" for f in range(5)], "prediction"),
    "kumo_sup90": load_20k([EXP_ROOT / f"kumo-supervised-view-20261021/folds/fold{f}/raw.npz" for f in range(5)], "prediction"),
    "kumo56": [kumo56_70k[f][np.r_[roles_list[f]["correction"], roles_list[f]["validation"]]] for f in range(5)],
}
for pack in sorted(os.listdir("/mnt/home/faris/logika-augment90-20261026/runs/large8")):
    plist = [Path(f"/mnt/home/faris/logika-augment90-20261026/runs/large8/{pack}/fold{f}/prediction.npz") for f in range(5)]
    if all(p.exists() for p in plist):
        m20k[f"aug_l8_{pack}"] = load_20k(plist)

# Compute 3 versions for every model:
# 1) raw_70k (uncorrected Kumo probability)
# 2) mat18_70k (0.75 * Head18 + 0.25 * sp_mat) -> Maturity pipeline (Row 11 style, strong on pop_A, pop_B, pop_C2!)
# 3) per29_70k (0.75 * Head29 + 0.25 * sp_h16) -> Periodicity/H016 pipeline (Row 13 style, strong on pop_C1!)
raw_70k = {}
h18_only_70k = {}
mat18_70k = {}
per29_70k = {}

for mk, arrs in m20k.items():
    r_v = np.zeros(70000)
    h18_v = np.zeros(70000)
    m18_v = np.zeros(70000)
    p29_v = np.zeros(70000)
    for f in range(5):
        r = roles_list[f]
        q_idx = np.r_[r["correction"], r["validation"]]
        tb_q = exp_tables_70k[f].iloc[q_idx].reset_index(drop=True)
        p_f = arrs[f]
        r_v[r["validation"]] = p_f[6000:]

        h18 = feature_head.fit(tb_q.iloc[:6000].assign(target=r["y_correction"]), p_f[:6000], HEAD18_COLS, 0.003, 0.5)
        c18 = b_mod.corrected_probability(h18, tb_q.iloc[6000:], p_f[6000:])[0]
        h18_v[r["validation"]] = c18
        m18_v[r["validation"]] = 0.75 * c18 + 0.25 * sp_mat_val[f]

        h29 = feature_head.fit(tb_q.iloc[:6000].assign(target=r["y_correction"]), p_f[:6000], HEAD_PER_COLS, 0.003, 0.5)
        c29 = b_mod.corrected_probability(h29, tb_q.iloc[6000:], p_f[6000:])[0]
        p29_v[r["validation"]] = 0.75 * c29 + 0.25 * sp_h16_val[f]
    raw_70k[mk] = r_v
    h18_only_70k[mk] = h18_v
    mat18_70k[mk] = m18_v
    per29_70k[mk] = p29_v

# Also load our 56k models
lin56 = np.load(OUT_ROOT / "linear56k_runs.npz")
sp200_70k = lin56["val_spline200"]
cb_full_70k = np.load(EXP_ROOT / "catboost-convergence-20261021/OOF.npz")["full"]
cb_comp_70k = np.load(EXP_ROOT / "catboost-convergence-20261021/OOF.npz")["compact"]
lgb56_70k = np.zeros(70000)
cat56_70k = np.zeros(70000)
corr_bnd_70k = np.zeros(70000)
res_lgb_70k = np.zeros(70000)
for f in range(5):
    r = roles_list[f]
    cf = np.load(OUT_ROOT / f"cpu_runs/fold{f}_crossfit56k.npz")
    lgb56_70k[r["validation"]] = cf["val_full_lgbm_56k"]
    cat56_70k[r["validation"]] = cf["val_full_cat_56k"]
    corr_bnd_70k[r["validation"]] = cf["val_corr_exp_boundary"]
    res_lgb_70k[r["validation"]] = cf["val_res_lgbm_56k"]

print("=== SUBGROUP OUTER VALIDATION KS COMPARISON: MAT18 vs PER29 ===", flush=True)
core_keys = ["base_raw", "tr_seed", "pop_spec", "champ_s7731", "horiz_b90", "tr_early", "fe_mat", "cov96", "fe_cov", "tr_mat_orig", "shadow91", "kumo56"]
ens_raw = np.mean([raw_70k[k] for k in core_keys], axis=0)
ens_h18 = np.mean([h18_only_70k[k] for k in core_keys], axis=0)
ens_mat18 = np.mean([mat18_70k[k] for k in core_keys], axis=0)
ens_per29 = np.mean([per29_70k[k] for k in core_keys], axis=0)
ens_50_50 = 0.5 * ens_mat18 + 0.5 * ens_per29

for nm, arr in [
    ("Ens-12 Raw Kumo", ens_raw),
    ("Ens-12 Head18 Only", ens_h18),
    ("Ens-12 Mat18 (Head18+SpMat)", ens_mat18),
    ("Ens-12 Per29 (Head29+SpH16)", ens_per29),
    ("Ens-12 50/50 Mat18+Per29", ens_50_50),
]:
    subs = [f"{pn}:{ks_stat(y_global[pop_masks[pn]], arr[pop_masks[pn]]):.5f}" for pn in ["pop_A", "pop_B", "pop_C1", "pop_C2"]]
    print(f"  {nm:30s} | PoolKS={ks_stat(y_global, arr):.6f} AUC={roc_auc_score(y_global, arr):.6f} | " + " | ".join(subs), flush=True)

# Now let's check what happens on each population when we build the best subgroup ranker!
print("\n=== BEST SUBGROUP RANKERS ON EACH POPULATION ===", flush=True)
# Pop A: Raw Kumo + Mat18 + lgb56 + corr_bnd
for w_raw in [0.0, 0.25, 0.50, 0.75]:
    for w_lgb in [0.0, 0.10, 0.20]:
        pA = (1.0 - w_raw - w_lgb) * ens_mat18[pop_masks["pop_A"]] + w_raw * ens_raw[pop_masks["pop_A"]] + w_lgb * lgb56_70k[pop_masks["pop_A"]]
        print(f"  pop_A [w_raw={w_raw:.2f} w_lgb={w_lgb:.2f}]: KS={ks_stat(y_global[pop_masks['pop_A']], pA):.6f} AUC={roc_auc_score(y_global[pop_masks['pop_A']], pA):.6f}", flush=True)

# Pop B: pop_spec + ens_mat18
for w_ps in [0.0, 0.25, 0.50, 0.75]:
    pB = (1.0 - w_ps) * ens_mat18[pop_masks["pop_B"]] + w_ps * mat18_70k["pop_spec"][pop_masks["pop_B"]]
    print(f"  pop_B [w_ps={w_ps:.2f}]: KS={ks_stat(y_global[pop_masks['pop_B']], pB):.6f} AUC={roc_auc_score(y_global[pop_masks['pop_B']], pB):.6f}", flush=True)

# Pop C1: ens_per29 + raw Kumo + champ_s7731 + kumo56
for w_raw in [0.0, 0.25, 0.50]:
    pC1 = (1.0 - w_raw) * ens_per29[pop_masks["pop_C1"]] + w_raw * ens_raw[pop_masks["pop_C1"]]
    print(f"  pop_C1 [w_raw={w_raw:.2f}]: KS={ks_stat(y_global[pop_masks['pop_C1']], pC1):.6f} AUC={roc_auc_score(y_global[pop_masks['pop_C1']], pC1):.6f}", flush=True)

# Pop C2: ens_mat18 + pop_spec + sp200 + lgb56
for w_ps in [0.0, 0.20, 0.35]:
    for w_tab in [0.0, 0.10, 0.20]:
        pC2 = (1.0 - w_ps - w_tab) * ens_mat18[pop_masks["pop_C2"]] + w_ps * h18_only_70k["pop_spec"][pop_masks["pop_C2"]] + w_tab * (0.5 * sp200_70k[pop_masks["pop_C2"]] + 0.5 * lgb56_70k[pop_masks["pop_C2"]])
        print(f"  pop_C2 [w_ps={w_ps:.2f} w_tab={w_tab:.2f}]: KS={ks_stat(y_global[pop_masks['pop_C2']], pC2):.6f} AUC={roc_auc_score(y_global[pop_masks['pop_C2']], pC2):.6f}", flush=True)
