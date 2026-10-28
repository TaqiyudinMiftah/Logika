import os
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "6"

import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import SplineTransformer
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

t0 = time.time()
OUT_ROOT = Path("/mnt/home/faris/logika-joint-20261028")
CPU_DIR = OUT_ROOT / "cpu_runs"
CPU_DIR.mkdir(parents=True, exist_ok=True)
FRZ_ROOT = Path("/mnt/home/faris/logika-frozen-shadow-20261026")

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

# Load global roles, y_global, and nested-correction 56k inner-OOF + 14k outer-val Kumo predictions
y_global = np.zeros(70000, dtype=np.int8)
roles_list = []
p_kumo56_global = []
inner_splits = []

for f in range(5):
    r = np.load(FRZ_ROOT / f"frozen/fold{f}/roles.npz", allow_pickle=True)
    roles_list.append(r)
    y_global[r["context"]] = r["y_context"]
    y_global[r["correction"]] = r["y_correction"]

    meta = np.load(f"/mnt/home/faris/logika-faris-exploration/nested-correction-20261021/heads/fold{f}/meta_inputs.npz", allow_pickle=True)
    raw5 = np.load(f"/mnt/home/faris/logika-faris-exploration/nested-correction-20261021/runs/job{f*6+5}/raw.npz", allow_pickle=True)
    pk = np.full(70000, np.nan, dtype=np.float64)
    pk[meta["rows"]] = meta["prediction"]
    pk[raw5["rows"]] = raw5["prediction"]
    assert np.isfinite(pk).all()
    p_kumo56_global.append(pk)

    f_splits = []
    for g in range(5):
        idx = np.load(f"/mnt/home/faris/logika-faris-exploration/nested-correction-20261021/data/job{f*6+g}/indices.npz", allow_pickle=True)
        f_splits.append((idx["context"], idx["validation"]))
    inner_splits.append(f_splits)

# Check baseline 56k Kumo inner-OOF and outer-val KS
val_kumo56 = np.zeros(70000, dtype=np.float64)
oof_kumo56_cor = []
y_cor_all = []
for f in range(5):
    r = roles_list[f]
    val_kumo56[r["validation"]] = p_kumo56_global[f][r["validation"]]
    oof_kumo56_cor.append(p_kumo56_global[f][r["correction"]])
    y_cor_all.append(r["y_correction"])
oof_kumo56_cor = np.concatenate(oof_kumo56_cor)
y_cor_all = np.concatenate(y_cor_all)
print(f"[Anchor Kumo56] 30k CorKS={ks_stat(y_cor_all, oof_kumo56_cor):.6f} | 70k ValKS={ks_stat(y_global, val_kumo56):.6f} ValAUC={roc_auc_score(y_global, val_kumo56):.6f}", flush=True)

# Select top 85 residual features for Expanded Correction Head + Spline
screen_df = pd.read_csv(OUT_ROOT / "inner_oof_residual_screen.csv")
frz0 = pd.read_parquet(FRZ_ROOT / "frozen/fold0/features.parquet")
head18_cols = list(frz0.columns[90:106])
shadow_latent_cols = list(frz0.columns[106:])

top_res_cols = (
    screen_df[~screen_df["in_champ_or_head"]]
    .sort_values("max_abs_z", ascending=False)["name"]
    .head(75)
    .tolist()
)
extra_fspec_head = [
    "v3_repay_180d_late_90_posterior_6", "v3_repay_180d_late_60_posterior_6",
    "v4_price_180d_between_category_std", "v4_price_all_between_category_std",
    "v4_price_90d_equal_category_mean", "v5_price_cancelled_all_equal_category_mean",
    "v5_ecom_180d_completion_posterior", "v5_price_income_residual",
    "v4_last12_late_90_posterior", "re_price_latent_multiplier",
    "re_income_log_measurement_sd", "re_income_log_sd",
    "re_late_1_posterior_sd", "re_late_30_posterior_sd", "joint_posterior_variance"
]
pop_cols = ["pop_A", "pop_B", "pop_C1", "pop_C2"]

def build_design_matrix(df, fit_rows, head_cols, spline_cols=None):
    parts = []
    # Population regime indicators
    pops = df[pop_cols].to_numpy(dtype=np.float32)
    parts.append(pops)

    num_std_list = []
    for c in head_cols:
        if c not in df.columns or c in pop_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            v = df[c].to_numpy(dtype=np.float64)
            v = np.sign(v) * np.log1p(np.abs(v))
            miss = ~np.isfinite(v)
            med = np.nanmedian(v[fit_rows])
            if not np.isfinite(med):
                med = 0.0
            vf = np.where(miss, med, v)
            mu = np.mean(vf[fit_rows])
            sd = max(np.std(vf[fit_rows]), 1e-4)
            z = np.clip((vf - mu) / sd, -6.0, 6.0).astype(np.float32)
            parts.append(z[:, None])
            num_std_list.append((c, z))
            miss_rate = np.mean(miss[fit_rows])
            if 0.005 <= miss_rate <= 0.995:
                parts.append(miss.astype(np.float32)[:, None])
        else:
            s = df[c].astype("string").fillna("__MISSING__")
            cats = sorted(s.iloc[fit_rows].unique())
            for cat in cats:
                parts.append((s == cat).to_numpy(dtype=np.float32)[:, None])

    # Add quantile splines for top continuous features if requested
    if spline_cols:
        sp_in = []
        for c, z in num_std_list:
            if c in spline_cols:
                sp_in.append(z)
        if sp_in:
            sp_mat = np.column_stack(sp_in)
            st = SplineTransformer(n_knots=4, degree=2, knots="quantile", extrapolation="constant", include_bias=False)
            st.fit(sp_mat[fit_rows])
            parts.append(st.transform(sp_mat).astype(np.float32))

    # Add explicit interactions of (pop_A, pop_B, pop_C1, pop_C2) with top 25 numerical features
    for _, z in num_std_list[:25]:
        parts.append(pops * z[:, None])

    X = np.hstack(parts).astype(np.float32)
    return X

# Storage for cross-fitted 56k inner-OOF + 14k outer-val predictions
models_56k = {
    "corr_exp_unweighted": np.zeros(70000, dtype=np.float64),
    "corr_exp_boundary": np.zeros(70000, dtype=np.float64),
    "res_lgbm_56k": np.zeros(70000, dtype=np.float64),
    "res_cat_56k": np.zeros(70000, dtype=np.float64),
    "full_lgbm_56k": np.zeros(70000, dtype=np.float64),
    "full_cat_56k": np.zeros(70000, dtype=np.float64),
    "spline_exp_56k": np.zeros(70000, dtype=np.float64),
}
oof_56k = {k: [np.full(70000, np.nan, dtype=np.float64) for _ in range(5)] for k in models_56k}

for f in range(5):
    tf = time.time()
    df = pd.read_parquet(OUT_ROOT / f"fold{f}/expanded.parquet")
    r = roles_list[f]
    tr_idx = np.concatenate([r["context"], r["correction"]])
    va_idx = r["validation"]
    pk_f = clip_prob(p_kumo56_global[f])
    zk_f = logit(pk_f)
    pops_f = df[pop_cols].to_numpy(dtype=np.float64)

    # 1. Build Expanded Correction & Spline Design Matrix (fit stats on tr_idx)
    corr_cols = list(dict.fromkeys(["employment_type", "loan_purpose", "region", "housing_status", "v2_prev_last_decision", "v2_ecom_last_payment_method"] + head18_cols + extra_fspec_head + top_res_cols))
    X_corr = build_design_matrix(df, tr_idx, corr_cols, spline_cols=set(head18_cols + top_res_cols[:25]))
    # Include base Kumo logit and population-specific base Kumo logit slopes!
    zk_std = (zk_f - np.mean(zk_f[tr_idx])) / np.std(zk_f[tr_idx])
    X_corr_with_kumo = np.hstack([
        zk_f[:, None].astype(np.float32),
        (pops_f * zk_std[:, None]).astype(np.float32),
        (zk_std[:, None]**2).astype(np.float32),
        X_corr
    ])

    # Cross-fit on the 5 inner folds g=0..4
    va_preds = {k: np.zeros(len(va_idx), dtype=np.float64) for k in models_56k}

    # Prepare LGBM & CatBoost dataframe
    lgb_df = df.copy()
    cat_feat_names = []
    for c in lgb_df.columns:
        if not pd.api.types.is_numeric_dtype(lgb_df[c]):
            lgb_df[c] = lgb_df[c].astype("string").fillna("__MISSING__").astype("category")
            cat_feat_names.append(c)

    cat_df = df.copy()
    for c in cat_feat_names:
        cat_df[c] = cat_df[c].astype("string").fillna("__MISSING__")

    for g in range(5):
        in_tr, in_va = inner_splits[f][g]
        y_in_tr = y_global[in_tr]
        y_in_va = y_global[in_va]

        # A1) Expanded Correction Head (Unweighted Logistic on [zk + X_corr])
        clf_unw = LogisticRegression(C=0.005, max_iter=500, solver="lbfgs")
        clf_unw.fit(X_corr_with_kumo[in_tr], y_in_tr)
        p_unw_in_va = clf_unw.predict_proba(X_corr_with_kumo[in_va])[:, 1]
        p_unw_va = clf_unw.predict_proba(X_corr_with_kumo[va_idx])[:, 1]
        oof_56k["corr_exp_unweighted"][f][in_va] = 0.65 * pk_f[in_va] + 0.35 * p_unw_in_va
        va_preds["corr_exp_unweighted"] += (0.65 * pk_f[va_idx] + 0.35 * p_unw_va) / 5.0

        # A2) Expanded Correction Head (KS-Boundary Weighted around p* = 0.1358)
        z_star = logit(0.135814)
        w_bound = 1.0 + 2.5 * np.exp(-0.5 * ((zk_f[in_tr] - z_star) / 0.85) ** 2)
        clf_bnd = LogisticRegression(C=0.004, max_iter=500, solver="lbfgs")
        clf_bnd.fit(X_corr_with_kumo[in_tr], y_in_tr, sample_weight=w_bound)
        p_bnd_in_va = clf_bnd.predict_proba(X_corr_with_kumo[in_va])[:, 1]
        p_bnd_va = clf_bnd.predict_proba(X_corr_with_kumo[va_idx])[:, 1]
        oof_56k["corr_exp_boundary"][f][in_va] = 0.65 * pk_f[in_va] + 0.35 * p_bnd_in_va
        va_preds["corr_exp_boundary"] += (0.65 * pk_f[va_idx] + 0.35 * p_bnd_va) / 5.0

        # B) SplineExpanded_56k (Standalone Full Tabular Spline-Logistic without Kumo input)
        clf_sp = LogisticRegression(C=0.015, max_iter=500, solver="lbfgs")
        clf_sp.fit(X_corr[in_tr], y_in_tr, sample_weight=w_bound)
        oof_56k["spline_exp_56k"][f][in_va] = clf_sp.predict_proba(X_corr[in_va])[:, 1]
        va_preds["spline_exp_56k"] += clf_sp.predict_proba(X_corr[va_idx])[:, 1] / 5.0

        # C) ResidualLGBM_56k (init_score = zk_f)
        dtrain_res = lgb.Dataset(lgb_df.iloc[in_tr], label=y_in_tr, init_score=zk_f[in_tr])
        dval_res = lgb.Dataset(lgb_df.iloc[in_va], label=y_in_va, init_score=zk_f[in_va], reference=dtrain_res)
        params_res = {
            "objective": "binary",
            "learning_rate": 0.015,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 250,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.45,
            "reg_alpha": 10.0,
            "reg_lambda": 60.0,
            "n_jobs": 6,
            "random_state": 31027 + f*10 + g,
            "verbose": -1,
        }
        bst_res = lgb.train(
            params_res,
            dtrain_res,
            num_boost_round=220,
            valid_sets=[dval_res],
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        raw_margin_in_va = bst_res.predict(lgb_df.iloc[in_va], raw_score=True)
        raw_margin_va = bst_res.predict(lgb_df.iloc[va_idx], raw_score=True)
        oof_56k["res_lgbm_56k"][f][in_va] = expit(zk_f[in_va] + 0.65 * raw_margin_in_va)
        va_preds["res_lgbm_56k"] += expit(zk_f[va_idx] + 0.65 * raw_margin_va) / 5.0

        # D) FullLGBM_Expanded_56k (Standalone from scratch on all 477 columns)
        dtrain_full = lgb.Dataset(lgb_df.iloc[in_tr], label=y_in_tr)
        dval_full = lgb.Dataset(lgb_df.iloc[in_va], label=y_in_va, reference=dtrain_full)
        params_full = {
            "objective": "binary",
            "learning_rate": 0.025,
            "num_leaves": 23,
            "max_depth": 5,
            "min_child_samples": 140,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.55,
            "reg_alpha": 5.0,
            "reg_lambda": 25.0,
            "n_jobs": 6,
            "random_state": 41027 + f*10 + g,
            "verbose": -1,
        }
        bst_full = lgb.train(
            params_full,
            dtrain_full,
            num_boost_round=750,
            valid_sets=[dval_full],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof_56k["full_lgbm_56k"][f][in_va] = bst_full.predict(lgb_df.iloc[in_va])
        va_preds["full_lgbm_56k"] += bst_full.predict(lgb_df.iloc[va_idx]) / 5.0

        # E) ResidualCatBoost_56k (baseline = zk_f) & FullCatBoost_56k
        pool_tr_res = Pool(cat_df.iloc[in_tr], label=y_in_tr, cat_features=cat_feat_names, baseline=zk_f[in_tr])
        pool_va_res = Pool(cat_df.iloc[in_va], label=y_in_va, cat_features=cat_feat_names, baseline=zk_f[in_va])
        pool_out_res = Pool(cat_df.iloc[va_idx], cat_features=cat_feat_names, baseline=zk_f[va_idx])
        cb_res = CatBoostClassifier(
            iterations=320,
            learning_rate=0.02,
            depth=4,
            l2_leaf_reg=40.0,
            random_seed=51027 + f*10 + g,
            thread_count=6,
            verbose=0,
        )
        cb_res.fit(pool_tr_res, eval_set=pool_va_res, early_stopping_rounds=40, verbose=False)
        oof_56k["res_cat_56k"][f][in_va] = cb_res.predict_proba(pool_va_res)[:, 1]
        va_preds["res_cat_56k"] += cb_res.predict_proba(pool_out_res)[:, 1] / 5.0

        pool_tr_full = Pool(cat_df.iloc[in_tr], label=y_in_tr, cat_features=cat_feat_names)
        pool_va_full = Pool(cat_df.iloc[in_va], label=y_in_va, cat_features=cat_feat_names)
        pool_out_full = Pool(cat_df.iloc[va_idx], cat_features=cat_feat_names)
        cb_full = CatBoostClassifier(
            iterations=900,
            learning_rate=0.035,
            depth=5,
            l2_leaf_reg=20.0,
            random_seed=61027 + f*10 + g,
            thread_count=6,
            verbose=0,
        )
        cb_full.fit(pool_tr_full, eval_set=pool_va_full, early_stopping_rounds=50, verbose=False)
        oof_56k["full_cat_56k"][f][in_va] = cb_full.predict_proba(pool_va_full)[:, 1]
        va_preds["full_cat_56k"] += cb_full.predict_proba(pool_out_full)[:, 1] / 5.0

    for k in models_56k:
        models_56k[k][va_idx] = va_preds[k]
        assert np.isfinite(oof_56k[k][f][tr_idx]).all(), f"NaN in {k} fold {f} inner OOF!"

    y_va = y_global[va_idx]
    print(f"\n--- Fold {f} CPU Cross-Fit Done in {time.time()-tf:.1f}s ---", flush=True)
    print(f"  Anchor Kumo56     : InnerOOF_KS={ks_stat(y_global[tr_idx], pk_f[tr_idx]):.6f} | ValKS={ks_stat(y_va, pk_f[va_idx]):.6f} ValAUC={roc_auc_score(y_va, pk_f[va_idx]):.6f}", flush=True)
    for k in models_56k:
        in_ks = ks_stat(y_global[tr_idx], oof_56k[k][f][tr_idx])
        in_auc = roc_auc_score(y_global[tr_idx], oof_56k[k][f][tr_idx])
        v_ks = ks_stat(y_va, models_56k[k][va_idx])
        v_auc = roc_auc_score(y_va, models_56k[k][va_idx])
        print(f"  {k:20s}: InnerOOF_KS={in_ks:.6f} InnerOOF_AUC={in_auc:.6f} | ValKS={v_ks:.6f} ValAUC={v_auc:.6f}", flush=True)

    # Save fold predictions
    np.savez_compressed(
        CPU_DIR / f"fold{f}_crossfit56k.npz",
        tr_idx=tr_idx,
        va_idx=va_idx,
        **{f"oof_{k}": oof_56k[k][f][tr_idx].astype(np.float32) for k in models_56k},
        **{f"val_{k}": models_56k[k][va_idx].astype(np.float32) for k in models_56k},
    )

print("\n=== POOLED 70,000-ROW OUTER VALIDATION SUMMARY (CPU 56K CROSS-FIT SUITE) ===", flush=True)
for k in models_56k:
    print(f"  {k:20s}: Pooled_ValKS={ks_stat(y_global, models_56k[k]):.6f} Pooled_ValAUC={roc_auc_score(y_global, models_56k[k]):.6f}", flush=True)

# Blend check on 56k inner-OOF vs 70k outer validation
blend_va = (
    0.30 * models_56k["corr_exp_boundary"]
    + 0.30 * models_56k["res_lgbm_56k"]
    + 0.20 * models_56k["res_cat_56k"]
    + 0.10 * models_56k["full_lgbm_56k"]
    + 0.05 * models_56k["full_cat_56k"]
    + 0.05 * models_56k["spline_exp_56k"]
)
print(f"  {'CPU_Blend_56k':20s}: Pooled_ValKS={ks_stat(y_global, blend_va):.6f} Pooled_ValAUC={roc_auc_score(y_global, blend_va):.6f}", flush=True)
