import os
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "4"

import glob
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import SplineTransformer
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

t0 = time.time()
OUT_ROOT = Path("/mnt/home/faris/logika-joint-20261028")
FRZ_ROOT = Path("/mnt/home/faris/logika-frozen-shadow-20261026")
EXP_ROOT = Path("/mnt/home/faris/logika-faris-exploration")

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

# 1. Load global y, roles, and matched baseline OOF (0.5306361042400267)
y_global = np.zeros(70000, dtype=np.int8)
roles_list = []
baseline_oof = np.zeros(70000, dtype=np.float64)
baseline_spline_val = np.zeros(70000, dtype=np.float64)

for f in range(5):
    r = np.load(FRZ_ROOT / f"frozen/fold{f}/roles.npz", allow_pickle=True)
    roles_list.append(r)
    y_global[r["context"]] = r["y_context"]
    y_global[r["correction"]] = r["y_correction"]
    b = np.load(FRZ_ROOT / f"frozen/fold{f}/baseline.npz", allow_pickle=True)
    baseline_oof[r["validation"]] = b["corrected"] * 0.75 + b["spline"] * 0.25
    baseline_spline_val[r["validation"]] = b["spline"]

print(f"Matched Baseline Pooled OOF KS = {ks_stat(y_global, baseline_oof):.16f} | AUC = {roc_auc_score(y_global, baseline_oof):.6f}", flush=True)

# 2. Load all 5-fold x 20,000-row (6,000 correction + 14,000 validation) models with exact row alignment
def load_20k_cohort():
    models = {}

    def add_model(name, paths, key="raw"):
        if len(paths) != 5 or not all(os.path.exists(p) for p in paths):
            return False
        arrs = []
        for f, p in enumerate(paths):
            z = np.load(p, allow_pickle=True)
            k = key if key in z else ("prediction" if "prediction" in z else ("raw" if "raw" in z else list(z.keys())[0]))
            v = np.asarray(z[k], dtype=np.float64).reshape(-1)
            if len(v) != 20000 or not np.isfinite(v).all():
                return False
            q_want = np.r_[roles_list[f]["correction"], roles_list[f]["validation"]]
            if "rows" in z and len(z["rows"]) == 20000:
                rows_in = np.asarray(z["rows"], dtype=np.int64)
                if not np.array_equal(rows_in, q_want):
                    vec = np.full(70000, np.nan, dtype=np.float64)
                    vec[rows_in] = v
                    v = vec[q_want]
                    if not np.isfinite(v).all():
                        return False
            elif "query_rows" in z and len(z["query_rows"]) == 20000:
                rows_in = np.asarray(z["query_rows"], dtype=np.int64)
                if not np.array_equal(rows_in, q_want):
                    vec = np.full(70000, np.nan, dtype=np.float64)
                    vec[rows_in] = v
                    v = vec[q_want]
                    if not np.isfinite(v).all():
                        return False
            arrs.append(v)
        models[name] = arrs
        return True

    # A) New Population-Specialist & Surgical96 Kumo runs
    add_model("joint_kumo_pop_specialist", [OUT_ROOT / f"kumo_runs/kumo_pop_specialist/fold{f}/prediction.npz" for f in range(5)])
    add_model("joint_kumo_surgical96", [OUT_ROOT / f"kumo_runs/kumo_surgical96/fold{f}/prediction.npz" for f in range(5)])

    # B) augment90 large8 & medium8
    for size in ["large8", "medium8"]:
        sdir = Path(f"/mnt/home/faris/logika-augment90-20261026/runs/{size}")
        if sdir.exists():
            for pack in sorted(os.listdir(sdir)):
                add_model(f"aug90_{size}_{pack}", [sdir / pack / f"fold{f}/prediction.npz" for f in range(5)])

    # C) frozen-shadow (exclude rank_*)
    fs_runs = FRZ_ROOT / "runs"
    if fs_runs.exists():
        for cfg in sorted(os.listdir(fs_runs)):
            if cfg.startswith("rank_"):
                continue
            add_model(f"fs_{cfg}", [fs_runs / cfg / f"fold{f}/prediction.npz" for f in range(5)])

    # D) architecture neural & GBDT models
    arch_runs = Path("/mnt/home/faris/logika-architecture-20261027/runs")
    if arch_runs.exists():
        for cfg in sorted(os.listdir(arch_runs)):
            add_model(f"arch_{cfg}", [arch_runs / cfg / f"fold{f}/prediction.npz" for f in range(5)])

    # E) model-arena tabular foundation models
    arena_runs = Path("/mnt/home/faris/logika-model-arena-20261025/storage/runs")
    if arena_runs.exists():
        for fam in sorted(os.listdir(arena_runs)):
            fdir = arena_runs / fam
            if fdir.is_dir():
                for cfg in sorted(os.listdir(fdir)):
                    add_model(f"arena_{fam}_{cfg}", [fdir / cfg / f"fold{f}/predictions.npz" for f in range(5)], key="prediction")

    # F) exploration 20,000-row models discovered in task-509
    add_model("kumo_supervised90", [EXP_ROOT / f"kumo-supervised-view-20261021/folds/fold{f}/raw.npz" for f in range(5)], key="prediction")
    add_model("tabicl_compact", [EXP_ROOT / f"tabicl-compact-20261021/folds/fold{f}/raw.npz" for f in range(5)], key="prediction")
    add_model("fe_coverage_info", [EXP_ROOT / f"coverage-information-fe-20261021/runs/fold{f}/raw.npz" for f in range(5)], key="prediction")
    add_model("fe_mature_denom", [EXP_ROOT / f"mature-denominator-fe-20261021/runs/fold{f}/raw.npz" for f in range(5)], key="prediction")
    add_model("fe_mature_mask_preserved", [EXP_ROOT / f"mature-ma[REDACTED_CREDENTIAL]/runs/fold{f}/raw.npz" for f in range(5)], key="prediction")
    add_model("hmf_classification", [EXP_ROOT / f"heldout-margin-fusion-20261021/imports/classification/fold{f}/raw.npz" for f in range(5)], key="prediction")
    add_model("hmf_regression", [EXP_ROOT / f"heldout-margin-fusion-20261021/imports/regression/fold{f}/raw.npz" for f in range(5)], key="prediction")
    for cond in ["control90", "coverage96", "null96"]:
        add_model(f"kumo_covplace_{cond}", [EXP_ROOT / f"kumo-coverage-placement-20261021/conditions/fold{f}/{cond}/raw_predictions.npz" for f in range(5)], key="prediction")
    add_model("nested_calib_outer_ctx", [EXP_ROOT / f"nested-calibration-20261020/folds/fold{f}/outer_context/prediction.npz" for f in range(5)], key="prediction")
    add_model("risk_query_match_raw", [EXP_ROOT / f"ri[REDACTED_CREDENTIAL]/folds/fold{f}/predictions.npz" for f in range(5)], key="raw")
    add_model("risk_repr_signal_raw", [EXP_ROOT / f"ri[REDACTED_CREDENTIAL]/folds/fold{f}/signal/prediction.npz" for f in range(5)], key="raw")
    add_model("risk_repr_signal_raw", [EXP_ROOT / f"ri[REDACTED_CREDENTIAL]/folds/fold{f}/signal/prediction.npz" for f in range(5)], key="raw")
    for t_idx in range(5):
        add_model(f"risk_repr_teacher{t_idx}", [EXP_ROOT / f"ri[REDACTED_CREDENTIAL]/folds/fold{f}/teachers/full/teacher{t_idx}/prediction.npz" for f in range(5)], key="prediction")

    # transfer-20261025 20,000-row models
    add_model("tr_core_late30_orig", [EXP_ROOT / f"transfer-20261025/core_late30/fold{f}/original_raw.npz" for f in range(5)], key="prediction")
    add_model("tr_core_late30_fresh", [EXP_ROOT / f"transfer-20261025/core_late30/fold{f}/fresh_raw.npz" for f in range(5)], key="prediction")
    add_model("tr_core_maturity_orig", [EXP_ROOT / f"transfer-20261025/core_maturity/fold{f}/original_raw.npz" for f in range(5)], key="prediction")
    add_model("tr_core_maturity_fresh", [EXP_ROOT / f"transfer-20261025/core_maturity/fold{f}/fresh_raw.npz" for f in range(5)], key="prediction")
    add_model("tr_early_avg_raw_mean", [EXP_ROOT / f"transfer-20261025/early_average/fold{f}/raw_mean.npz" for f in range(5)], key="prediction")
    add_model("tr_raw_seed_repeat", [EXP_ROOT / f"transfer-20261025/robustness/fold{f}/raw_seed_repeat.npz" for f in range(5)], key="prediction")
    add_model("tr_spline_query", [EXP_ROOT / f"transfer-20261025/joint_correction/fold{f}/spline_query.npz" for f in range(5)], key="prediction")
    for sub in ["balanced__original", "balanced__fresh_kumo_seed", "proportional__original", "proportional__fresh_kumo_seed"]:
        add_model(f"tr_{sub}", [EXP_ROOT / f"transfer-20261025/balanced_context/fold{f}/{sub}/raw_predictions.npz" for f in range(5)], key="prior_adjusted_probability")

    # horizontal-20261019 fits
    for seed in [2026, 2027]:
        for hname in ["account_dynamic", "account_spline", "bureau_dynamic", "bureau_logit", "champion0", "champion1", "champion2", "champion3", "champion4", "pricing_aux", "shopping_xgb"]:
            add_model(f"h19_s{seed}_{hname}", [EXP_ROOT / f"horizontal-20261019/campaign/fits/s{seed}_f{f}_context_{hname}/prediction.npz" for f in range(5)], key="prediction")

    # horizontal-20261026 large8/medium8
    for size in ["large8", "medium8"]:
        sdir = Path(f"/mnt/home/faris/logika-horizontal-20261026/runs/{size}")
        if sdir.exists():
            for pack in ["baseline90", "p0018"]:
                add_model(f"horiz_{size}_{pack}", [sdir / pack / f"fold{f}/prediction.npz" for f in range(5)])

    # G) nested-correction 56k Kumo
    nested_arrs = []
    for f in range(5):
        r = roles_list[f]
        meta = np.load(EXP_ROOT / f"nested-correction-20261021/heads/fold{f}/meta_inputs.npz", allow_pickle=True)
        raw5 = np.load(EXP_ROOT / f"nested-correction-20261021/runs/job{f*6+5}/raw.npz", allow_pickle=True)
        pk = np.full(70000, np.nan, dtype=np.float64)
        pk[meta["rows"]] = meta["prediction"]
        pk[raw5["rows"]] = raw5["prediction"]
        q = np.r_[r["correction"], r["validation"]]
        nested_arrs.append(pk[q])
    models["nested_kumo56"] = nested_arrs

    return models

models = load_20k_cohort()
print(f"Loaded {len(models)} complete 5-fold x 20,000-row models!", flush=True)

# Evaluate every model on 30,000 correction rows and 70,000 outer validation rows
y_cor_all = np.concatenate([roles_list[f]["y_correction"] for f in range(5)])
summary = []
for name, arrs in models.items():
    cor = np.concatenate([a[:6000] for a in arrs])
    val = np.zeros(70000, dtype=np.float64)
    for f in range(5):
        val[roles_list[f]["validation"]] = arrs[f][6000:]
    summary.append({
        "name": name,
        "cor_ks": ks_stat(y_cor_all, cor),
        "cor_auc": roc_auc_score(y_cor_all, cor),
        "val_ks": ks_stat(y_global, val),
        "val_auc": roc_auc_score(y_global, val),
    })

sum_df = pd.DataFrame(summary).sort_values("cor_auc", ascending=False)
print("\n=== TOP 35 20K MODELS SORTED BY 30,000-ROW CORRECTION AUC ===", flush=True)
print(sum_df.head(35).to_string(index=False), flush=True)

print("\n=== TOP 35 20K MODELS SORTED BY 30,000-ROW CORRECTION KS ===", flush=True)
print(sum_df.sort_values("cor_ks", ascending=False).head(35).to_string(index=False), flush=True)

# 3. Also evaluate Population-Specific Performance on the 30,000 Correction Rows!
raw_npz = np.load(OUT_ROOT / "raw_domain_features.npz", allow_pickle=True)
raw_names = [str(x) for x in raw_npz["names"]]
raw_data = raw_npz["data"]
pop_masks_global = {
    p: raw_data[:, raw_names.index(p)] > 0.5
    for p in ["pop_A", "pop_B", "pop_C1", "pop_C2"]
}
cor_indices_all = np.concatenate([roles_list[f]["correction"] for f in range(5)])
for p_name, p_mask in pop_masks_global.items():
    sub_cor = p_mask[cor_indices_all]
    y_sub = y_cor_all[sub_cor]
    p_scores = []
    for name, arrs in models.items():
        cor = np.concatenate([a[:6000] for a in arrs])[sub_cor]
        p_scores.append((ks_stat(y_sub, cor), roc_auc_score(y_sub, cor), name))
    p_scores.sort(reverse=True)
    print(f"\n=== TOP 12 MODELS ON 30K CORRECTION FOR {p_name} (n={sub_cor.sum()}) ===", flush=True)
    for ks_v, auc_v, nm in p_scores[:12]:
        print(f"  {nm:38s}: CorKS={ks_v:.6f} CorAUC={auc_v:.6f}", flush=True)

