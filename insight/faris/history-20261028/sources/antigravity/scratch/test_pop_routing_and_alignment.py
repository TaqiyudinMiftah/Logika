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

def match_subgroup_distribution_foldsafe(cand_cor, cand_val, ref_cor, ref_val):
    """Map cand_val to ref's marginal distribution using ONLY correction-split (or covariate-only) quantiles."""
    # Method A: Correction-fitted affine logit alignment (mean & std on correction split only)
    zc = logit(clip_prob(cand_cor))
    zr = logit(clip_prob(ref_cor))
    zv = logit(clip_prob(cand_val))
    mu_c, sd_c = zc.mean(), max(zc.std(), 1e-6)
    mu_r, sd_r = zr.mean(), max(zr.std(), 1e-6)
    p_affine = expit((zv - mu_c) / sd_c * sd_r + mu_r)

    # Method B: Correction-fitted quantile interpolation (100% training/correction-only CDF -> quantile map)
    qs = np.linspace(0.0, 1.0, 201)
    xc = np.quantile(cand_cor, qs)
    yr = np.quantile(ref_cor, qs)
    # Ensure strict monotonicity in xc
    xc_u, idx_u = np.unique(xc, return_index=True)
    yr_u = yr[idx_u]
    p_qcor = np.interp(cand_val, xc_u, yr_u)

    # Method C: Exact within-fold covariate marginal matching (uses NO labels at all: maps cand_val ranks to sorted ref_val)
    order = np.argsort(cand_val, kind="mergesort")
    p_qval = np.empty_like(cand_val)
    p_qval[order] = np.sort(ref_val)

    return p_affine, p_qcor, p_qval

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
h016_spline_val = []
kumo56_70k = []

for f in range(5):
    r = np.load(FRZ_ROOT / f"frozen/fold{f}/roles.npz", allow_pickle=True)
    roles_list.append(r)
    b = np.load(FRZ_ROOT / f"frozen/fold{f}/baseline.npz", allow_pickle=True)
    tb = pd.read_parquet(OUT_ROOT / f"fold{f}/expanded.parquet")
    for c in EXTRA_PER:
        tb[c] = spectral_df[c].to_numpy(dtype=np.float32)
    exp_tables_70k.append(tb)

    q_idx = np.r_[r["correction"], r["validation"]]
    tb_q = tb.iloc[q_idx].reset_index(drop=True)
    h29 = feature_head.fit(tb_q.iloc[:6000].assign(target=r["y_correction"]), b["raw"][:6000], HEAD_PER_COLS, 0.003, 0.5)
    c29_val = b_mod.corrected_probability(h29, tb_q.iloc[6000:], b["raw"][6000:])[0]
    sp_per_f = (per_orig_oof[r["validation"]] - 0.75 * c29_val) / 0.25
    h16_val = np.load(EXP_ROOT / f"transfer-20261025/spline_direct_ks/fold{f}/validation.npz")
    h016_spline_val.append(sp_per_f + h16_val["spline_delta"])

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
    "med_lat99": load_20k([FRZ_ROOT / f"runs/kumo_medium_latent99/fold{f}/prediction.npz" for f in range(5)]),
    "tr_early": load_20k([EXP_ROOT / f"transfer-20261025/early_average/fold{f}/raw_mean.npz" for f in range(5)], "prediction"),
    "tr_mat_orig": load_20k([EXP_ROOT / f"transfer-20261025/core_maturity/fold{f}/original_raw.npz" for f in range(5)], "prediction"),
    "kumo56": [kumo56_70k[f][np.r_[roles_list[f]["correction"], roles_list[f]["validation"]]] for f in range(5)],
}

indiv_cor_head = {}
indiv_val_final = {}
for mk in m20k:
    c_list = []
    v_arr = np.zeros(70000)
    for f in range(5):
        r = roles_list[f]
        q_idx = np.r_[r["correction"], r["validation"]]
        tb_q = exp_tables_70k[f].iloc[q_idx].reset_index(drop=True)
        p_f = m20k[mk][f]
        h29 = feature_head.fit(tb_p := tb_q.iloc[:6000].assign(target=r["y_correction"]), p_f[:6000], HEAD_PER_COLS, 0.003, 0.5)
        c29_cor = b_mod.corrected_probability(h29, tb_q.iloc[:6000], p_f[:6000])[0]
        c29_val = b_mod.corrected_probability(h29, tb_q.iloc[6000:], p_f[6000:])[0]
        c_list.append(c29_cor)
        v_arr[r["validation"]] = 0.75 * c29_val + 0.25 * h016_spline_val[f]
    indiv_cor_head[mk] = c_list
    indiv_val_final[mk] = v_arr

# Also load our 56k CPU and Linear residual models
lin56 = np.load(OUT_ROOT / "linear56k_runs.npz", allow_pickle=True)
cpu_corr_bnd_val = np.zeros(70000)
cpu_corr_bnd_cor = []
kumo56_dpop_val = np.zeros(70000)
kumo56_dpop_cor = []
for f in range(5):
    r = roles_list[f]
    cf = np.load(OUT_ROOT / f"cpu_runs/fold{f}_crossfit56k.npz")
    cpu_corr_bnd_cor.append(cf["oof_corr_exp_boundary"][50000:])
    cpu_corr_bnd_val[r["validation"]] = 0.75 * cf["val_corr_exp_boundary"] + 0.25 * h016_spline_val[f]
    kumo56_dpop_cor.append(expit(logit(clip_prob(kumo56_70k[f][r["correction"]])) + 0.5 * lin56["oof_delta56_pop"][f][r["correction"]]))
    kumo56_dpop_val[r["validation"]] = 0.75 * expit(logit(clip_prob(kumo56_70k[f][r["validation"]])) + 0.5 * lin56["val_delta56_pop"][r["validation"]]) + 0.25 * h016_spline_val[f]

# Check subgroup KS of each model's final output!
print("=== SUBGROUP OUTER VALIDATION KS OF FINAL (HEAD29 + H016 SPLINE) MODELS ===", flush=True)
for mk, v_arr in list(indiv_val_final.items()) + [("cpu_corr_bnd", cpu_corr_bnd_val), ("kumo56_dpop", kumo56_dpop_val), ("h016_oof", h016_oof)]:
    s_info = [f"{pn}:{ks_stat(y_global[pop_masks[pn]], v_arr[pop_masks[pn]]):.5f}" for pn in ["pop_A", "pop_B", "pop_C1", "pop_C2"]]
    print(f"  {mk:15s} | PoolKS={ks_stat(y_global, v_arr):.6f} AUC={roc_auc_score(y_global, v_arr):.6f} | " + " | ".join(s_info), flush=True)

# Now test Sub-Population Specialist Ensembles mapped onto Anchor Marginal Distribution per (fold, pop)!
anchor_cor = [np.mean([indiv_cor_head[k][f] for k in ["base_raw", "tr_seed", "champ_s7731", "horiz_b90", "tr_early", "fe_mat"]], axis=0) for f in range(5)]
anchor_val = np.mean([indiv_val_final[k] for k in ["base_raw", "tr_seed", "champ_s7731", "horiz_b90", "tr_early", "fe_mat"]], axis=0)
print(f"\nAnchor 6-Model KS = {ks_stat(y_global, anchor_val):.6f}", flush=True)

for w_spec in [0.20, 0.35, 0.50]:
    out_unaligned = np.zeros(70000)
    out_affine = np.zeros(70000)
    out_qcor = np.zeros(70000)
    out_qval = np.zeros(70000)

    for f in range(5):
        r = roles_list[f]
        c_idx = r["correction"]
        v_idx = r["validation"]

        for pn in ["pop_A", "pop_B", "pop_C1", "pop_C2"]:
            mc = pop_masks[pn][c_idx]
            mv = pop_masks[pn][v_idx]
            ref_c = anchor_cor[f][mc]
            ref_v = anchor_val[v_idx][mv]

            if pn == "pop_A":
                cand_c = (1.0 - w_spec) * ref_c + w_spec * np.mean([indiv_cor_head["cov96"][f][mc], indiv_cor_head["fe_cov"][f][mc], kumo56_dpop_cor[f][mc]], axis=0)
                cand_v = (1.0 - w_spec) * ref_v + w_spec * np.mean([indiv_val_final["cov96"][v_idx][mv], indiv_val_final["fe_cov"][v_idx][mv], kumo56_dpop_val[v_idx][mv]], axis=0)
            elif pn == "pop_B":
                cand_c = (1.0 - w_spec) * ref_c + w_spec * np.mean([indiv_cor_head["pop_spec"][f][mc], indiv_cor_head["horiz_b90"][f][mc], kumo56_dpop_cor[f][mc]], axis=0)
                cand_v = (1.0 - w_spec) * ref_v + w_spec * np.mean([indiv_val_final["pop_spec"][v_idx][mv], indiv_val_final["horiz_b90"][v_idx][mv], kumo56_dpop_val[v_idx][mv]], axis=0)
            elif pn == "pop_C1":
                cand_c = (1.0 - w_spec) * ref_c + w_spec * np.mean([indiv_cor_head["med_lat99"][f][mc], indiv_cor_head["champ_s7731"][f][mc], cpu_corr_bnd_cor[f][mc]], axis=0)
                cand_v = (1.0 - w_spec) * ref_v + w_spec * np.mean([indiv_val_final["med_lat99"][v_idx][mv], indiv_val_final["champ_s7731"][v_idx][mv], cpu_corr_bnd_val[v_idx][mv]], axis=0)
            else: # pop_C2
                cand_c = (1.0 - w_spec) * ref_c + w_spec * np.mean([indiv_cor_head["pop_spec"][f][mc], indiv_cor_head["surg96"][f][mc], cpu_corr_bnd_cor[f][mc], kumo56_dpop_cor[f][mc]], axis=0)
                cand_v = (1.0 - w_spec) * ref_v + w_spec * np.mean([indiv_val_final["pop_spec"][v_idx][mv], indiv_val_final["surg96"][v_idx][mv], cpu_corr_bnd_val[v_idx][mv], kumo56_dpop_val[v_idx][mv]], axis=0)

            p_aff, p_qc, p_qv = match_subgroup_distribution_foldsafe(cand_c, cand_v, ref_c, ref_v)
            out_unaligned[v_idx[mv]] = cand_v
            out_affine[v_idx[mv]] = p_aff
            out_qcor[v_idx[mv]] = p_qc
            out_qval[v_idx[mv]] = p_qv

    print(f"[w_spec={w_spec:.2f}] Unaligned_KS={ks_stat(y_global, out_unaligned):.6f} | Affine_KS={ks_stat(y_global, out_affine):.6f} | QCor_KS={ks_stat(y_global, out_qcor):.6f} | QVal_KS={ks_stat(y_global, out_qval):.6f} (AUC={roc_auc_score(y_global, out_qval):.6f})", flush=True)
