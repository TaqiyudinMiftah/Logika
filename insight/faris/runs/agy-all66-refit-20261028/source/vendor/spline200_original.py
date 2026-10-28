import os
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "4"

import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.metrics import roc_auc_score

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

def fit_logistic_irls(X_tr, y_tr, C=0.002, sample_weight=None, offset_tr=None, n_iter=4):
    """Exact L2-penalized logistic regression via fast Cholesky Newton-IRLS."""
    n, d = X_tr.shape
    lam = 1.0 / max(C, 1e-8)
    sw = np.ones(n, dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    off = np.zeros(n, dtype=np.float64) if offset_tr is None else np.asarray(offset_tr, dtype=np.float64)
    # Initialize intercept at log-odds of y_tr if no offset, else 0
    b0 = 0.0 if offset_tr is not None else float(logit(np.clip(np.average(y_tr, weights=sw), 1e-4, 1.0 - 1e-4)))
    w = np.zeros(d, dtype=np.float64)
    Xb = np.zeros(n, dtype=np.float64)

    for _ in range(n_iter):
        eta = off + b0 + Xb
        p = expit(eta)
        hess_w = np.maximum(sw * p * (1.0 - p), 1e-6)
        grad_r = sw * (y_tr - p)
        # Update intercept b0
        b0_step = np.sum(grad_r) / max(np.sum(hess_w), 1e-6)
        b0 += b0_step
        eta += b0_step
        p = expit(eta)
        hess_w = np.maximum(sw * p * (1.0 - p), 1e-6)
        grad_r = sw * (y_tr - p)

        # Solve (X^T diag(hess_w) X + lam I) dw = X^T grad_r - lam * w
        Xw = X_tr * np.sqrt(hess_w)[:, None].astype(np.float32)
        H = (Xw.T @ Xw).astype(np.float64)
        H[np.diag_indices_from(H)] += lam
        rhs = (X_tr.T @ grad_r.astype(np.float32)).astype(np.float64) - lam * w
        dw = np.linalg.solve(H, rhs)
        w += dw
        Xb = (X_tr @ w.astype(np.float32)).astype(np.float64)
        if np.max(np.abs(dw)) < 1e-5:
            break
    return b0, w.astype(np.float32)

def predict_irls(X, b0, w, offset=None):
    eta = b0 + (X @ w).astype(np.float64)
    if offset is not None:
        eta = eta + offset
    return expit(eta), eta

# 1. Load global y, roles, and matched baseline OOF (0.5306361042400267)
y_global = np.zeros(70000, dtype=np.int8)
roles_list = []
baseline_spline_val = np.zeros(70000, dtype=np.float64)
baseline_corr_val = np.zeros(70000, dtype=np.float64)
p_kumo56_global = []
inner_splits = []

for f in range(5):
    r = np.load(FRZ_ROOT / f"frozen/fold{f}/roles.npz", allow_pickle=True)
    roles_list.append(r)
    y_global[r["context"]] = r["y_context"]
    y_global[r["correction"]] = r["y_correction"]
    b = np.load(FRZ_ROOT / f"frozen/fold{f}/baseline.npz", allow_pickle=True)
    baseline_corr_val[r["validation"]] = b["corrected"]
    baseline_spline_val[r["validation"]] = b["spline"]

    meta = np.load(EXP_ROOT / f"nested-correction-20261021/heads/fold{f}/meta_inputs.npz", allow_pickle=True)
    raw5 = np.load(EXP_ROOT / f"nested-correction-20261021/runs/job{f*6+5}/raw.npz", allow_pickle=True)
    pk = np.full(70000, np.nan, dtype=np.float64)
    pk[meta["rows"]] = meta["prediction"]
    pk[raw5["rows"]] = raw5["prediction"]
    p_kumo56_global.append(pk)

    f_splits = []
    for g in range(5):
        idx = np.load(EXP_ROOT / f"nested-correction-20261021/data/job{f*6+g}/indices.npz", allow_pickle=True)
        f_splits.append((idx["context"], idx["validation"]))
    inner_splits.append(f_splits)

fs_oof = np.load(FRZ_ROOT / "evaluation/OOF.npz", allow_pickle=True)
baseline_oof = fs_oof["baseline"]
print(f"Verified Matched Baseline OOF KS = {ks_stat(y_global, baseline_oof):.16f} | AUC = {roc_auc_score(y_global, baseline_oof):.6f}", flush=True)

# 2. Build Upgraded FullSpline200_56k & Population-Specialist Residual Correction Head on all 56,000 inner-OOF rows!
screen_df = pd.read_csv(OUT_ROOT / "inner_oof_residual_screen.csv")
frz0 = pd.read_parquet(FRZ_ROOT / "frozen/fold0/features.parquet")
champ90_cols = list(frz0.columns[:90])
head18_cols = list(frz0.columns[90:106])
shadow_latent_cols = list(frz0.columns[106:])

top_res_cols = (
    screen_df[~screen_df["in_champ_or_head"]]
    .sort_values("max_abs_z", ascending=False)["name"]
    .head(90)
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

full_spline_cols = list(dict.fromkeys(champ90_cols + head18_cols + shadow_latent_cols + extra_fspec_head + top_res_cols))
corr_head_cols = list(dict.fromkeys(["employment_type", "loan_purpose", "region", "housing_status", "v2_prev_last_decision", "v2_ecom_last_payment_method"] + head18_cols + shadow_latent_cols + extra_fspec_head + top_res_cols))

def make_scaled_spline_matrix(df, fit_rows, feature_cols, n_knots=6, degree=2, top_inter_n=30):
    pops = df[pop_cols].to_numpy(dtype=np.float32)
    num_vecs = []
    cat_vecs = [pops]
    miss_vecs = []

    for c in feature_cols:
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
            num_vecs.append(z)
            m_rate = np.mean(miss[fit_rows])
            if 0.005 <= m_rate <= 0.995:
                miss_vecs.append(miss.astype(np.float32)[:, None])
        else:
            s = df[c].astype("string").fillna("__MISSING__")
            for cat in sorted(s.iloc[fit_rows].unique()):
                cat_vecs.append((s == cat).to_numpy(dtype=np.float32)[:, None])

    num_mat = np.column_stack(num_vecs)
    n_sp = min(60, num_mat.shape[1])
    st = SplineTransformer(n_knots=n_knots, degree=degree, knots="quantile", extrapolation="constant", include_bias=False)
    st.fit(num_mat[fit_rows, :n_sp])
    sp_mat = st.transform(num_mat[:, :n_sp]).astype(np.float32)

    inter_list = [pops * num_mat[:, j:j+1] for j in range(min(top_inter_n, num_mat.shape[1]))]
    X_raw = np.hstack([num_mat, sp_mat] + cat_vecs + miss_vecs + inter_list).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(X_raw[fit_rows])
    X_scaled = np.clip(scaler.transform(X_raw), -8.0, 8.0).astype(np.float32)
    return X_scaled

val_spline200 = np.zeros(70000, dtype=np.float64)
oof_spline200 = [np.full(70000, np.nan, dtype=np.float64) for _ in range(5)]

val_delta56_glob = np.zeros(70000, dtype=np.float64)
oof_delta56_glob = [np.full(70000, np.nan, dtype=np.float64) for _ in range(5)]

val_delta56_pop = np.zeros(70000, dtype=np.float64)
oof_delta56_pop = [np.full(70000, np.nan, dtype=np.float64) for _ in range(5)]

for f in range(5):
    tf = time.time()
    df = pd.read_parquet(OUT_ROOT / f"fold{f}/expanded.parquet")
    r = roles_list[f]
    tr_idx = np.concatenate([r["context"], r["correction"]])
    va_idx = r["validation"]
    pk_f = clip_prob(p_kumo56_global[f])
    zk_f = logit(pk_f)
    pops_f = df[pop_cols].to_numpy(dtype=np.float32)

    # 1) FullSpline200_56k
    X_sp200 = make_scaled_spline_matrix(df, tr_idx, full_spline_cols, n_knots=6, degree=2, top_inter_n=35)
    b0_sp, w_sp = fit_logistic_irls(X_sp200[tr_idx], y_global[tr_idx], C=0.0025)
    val_spline200[va_idx], _ = predict_irls(X_sp200[va_idx], b0_sp, w_sp)

    for g in range(5):
        in_tr, in_va = inner_splits[f][g]
        b0_in, w_in = fit_logistic_irls(X_sp200[in_tr], y_global[in_tr], C=0.0025)
        oof_spline200[f][in_va], _ = predict_irls(X_sp200[in_va], b0_in, w_in)

    # 2) Expanded Residual Correction Head (with offset = zk_f!)
    X_head = make_scaled_spline_matrix(df, tr_idx, corr_head_cols, n_knots=5, degree=2, top_inter_n=30)
    zk_std = ((zk_f - np.mean(zk_f[tr_idx])) / np.std(zk_f[tr_idx])).astype(np.float32)
    X_head_kumo = np.hstack([
        zk_std[:, None],
        pops_f * zk_std[:, None],
        (zk_std[:, None] ** 2),
        X_head
    ])

    z_star = logit(0.135814)
    w_tr = 1.0 + 1.5 * np.exp(-0.5 * ((zk_f[tr_idx] - z_star) / 0.90) ** 2)
    b0_h, w_h = fit_logistic_irls(X_head_kumo[tr_idx], y_global[tr_idx], C=0.0006, sample_weight=w_tr, offset_tr=zk_f[tr_idx])
    _, eta_va = predict_irls(X_head_kumo[va_idx], b0_h, w_h, offset=zk_f[va_idx])
    val_delta56_glob[va_idx] = eta_va - zk_f[va_idx]

    for g in range(5):
        in_tr, in_va = inner_splits[f][g]
        w_in = 1.0 + 1.5 * np.exp(-0.5 * ((zk_f[in_tr] - z_star) / 0.90) ** 2)
        b0_in, w_h_in = fit_logistic_irls(X_head_kumo[in_tr], y_global[in_tr], C=0.0006, sample_weight=w_in, offset_tr=zk_f[in_tr])
        _, eta_in_va = predict_irls(X_head_kumo[in_va], b0_in, w_h_in, offset=zk_f[in_va])
        oof_delta56_glob[f][in_va] = eta_in_va - zk_f[in_va]

    # 3) Population-Specialist Residual Correction Heads (Pop A, Pop B+C1, Pop C2)
    is_A = df["pop_A"].values > 0.5
    is_C2 = df["pop_C2"].values > 0.5
    is_BC1 = (~is_A) & (~is_C2)

    for reg_mask, C_reg in [(is_A, 0.0008), (is_BC1, 0.0006), (is_C2, 0.0008)]:
        tr_reg = tr_idx[reg_mask[tr_idx]]
        va_reg = va_idx[reg_mask[va_idx]]
        b0_r, w_r = fit_logistic_irls(X_head_kumo[tr_reg], y_global[tr_reg], C=C_reg, offset_tr=zk_f[tr_reg])
        _, eta_r_va = predict_irls(X_head_kumo[va_reg], b0_r, w_r, offset=zk_f[va_reg])
        val_delta56_pop[va_reg] = eta_r_va - zk_f[va_reg]

        for g in range(5):
            in_tr, in_va = inner_splits[f][g]
            in_tr_r = in_tr[reg_mask[in_tr]]
            in_va_r = in_va[reg_mask[in_va]]
            b0_ri, w_ri = fit_logistic_irls(X_head_kumo[in_tr_r], y_global[in_tr_r], C=C_reg, offset_tr=zk_f[in_tr_r])
            _, eta_r_in_va = predict_irls(X_head_kumo[in_va_r], b0_ri, w_ri, offset=zk_f[in_va_r])
            oof_delta56_pop[f][in_va_r] = eta_r_in_va - zk_f[in_va_r]

    print(f"[Fold {f} Linear56k ({time.time()-tf:.1f}s)] Spline200 ValKS={ks_stat(y_global[va_idx], val_spline200[va_idx]):.6f} (vs Spline77 {ks_stat(y_global[va_idx], baseline_spline_val[va_idx]):.6f}) | Kumo56+DeltaGlob ValKS={ks_stat(y_global[va_idx], expit(zk_f[va_idx] + 0.5*val_delta56_glob[va_idx])):.6f}", flush=True)

print(f"\n=== POOLED 70,000-ROW COMPARISON OF UPGRADED STAGE 2 & STAGE 3 ===", flush=True)
print(f"  Original Spline77_56k     : Pooled_KS={ks_stat(y_global, baseline_spline_val):.6f} Pooled_AUC={roc_auc_score(y_global, baseline_spline_val):.6f}", flush=True)
print(f"  Upgraded FullSpline200_56k: Pooled_KS={ks_stat(y_global, val_spline200):.6f} Pooled_AUC={roc_auc_score(y_global, val_spline200):.6f}", flush=True)
p_k56_va = np.zeros(70000)
for f in range(5):
    va_idx = roles_list[f]["validation"]
    p_k56_va[va_idx] = p_kumo56_global[f][va_idx]
z_k56_va = logit(clip_prob(p_k56_va))
for w_d in [0.35, 0.50, 0.65, 0.80]:
    p_c_glob = expit(z_k56_va + w_d * val_delta56_glob)
    p_c_pop = expit(z_k56_va + w_d * (0.6 * val_delta56_glob + 0.4 * val_delta56_pop))
    print(f"  Kumo56 + {w_d:.2f}*Delta56_Glob: Pooled_KS={ks_stat(y_global, p_c_glob):.6f} Pooled_AUC={roc_auc_score(y_global, p_c_glob):.6f}", flush=True)
    print(f"  Kumo56 + {w_d:.2f}*Delta56_Mix : Pooled_KS={ks_stat(y_global, p_c_pop):.6f} Pooled_AUC={roc_auc_score(y_global, p_c_pop):.6f}", flush=True)

np.savez_compressed(
    OUT_ROOT / "linear56k_runs.npz",
    val_spline200=val_spline200.astype(np.float32),
    val_delta56_glob=val_delta56_glob.astype(np.float32),
    val_delta56_pop=val_delta56_pop.astype(np.float32),
    oof_spline200=np.array(oof_spline200, dtype=np.float32),
    oof_delta56_glob=np.array(oof_delta56_glob, dtype=np.float32),
    oof_delta56_pop=np.array(oof_delta56_pop, dtype=np.float32),
)
