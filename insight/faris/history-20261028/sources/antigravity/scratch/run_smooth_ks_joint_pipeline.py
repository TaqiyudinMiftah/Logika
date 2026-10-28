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
PERIODIC = EXP_ROOT / "shopping-periodicity-fe-20261021"
MATURITY = EXP_ROOT / "repayment-severity-fe-20261021"

sys.path.insert(0, str(EXP_ROOT / "transfer-20261025/source"))
import baseline as b_mod
import feature_head

with open(FRZ_ROOT / "frozen/fold0/MANIFEST.json") as f:
    FOLD0_MAN = json.load(f)
HEAD18_COLS = FOLD0_MAN["head_columns"]

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

def ks_threshold(y, z):
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    order = np.argsort(z, kind="mergesort")
    ys = y[order]
    zs = z[order]
    pos = np.cumsum(ys == 1) / max(np.sum(ys == 1), 1)
    neg = np.cumsum(ys == 0) / max(np.sum(ys == 0), 1)
    diff = np.abs(pos - neg)
    end = np.r_[zs[:-1] != zs[1:], True]
    idx = np.flatnonzero(end)[np.argmax(diff[end])]
    return float(zs[idx]), float(diff[idx])

def clip_prob(p):
    return np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)

# 1. Load global y, roles, robustness OOF, H016 OOF, periodicity features
rob_npz = np.load(EXP_ROOT / "transfer-20261025/ROBUSTNESS_OOF.npz", allow_pickle=True)
y_global = rob_npz["target"].astype(np.int8)
folds_global = rob_npz["fold"].astype(np.int8)
baseline_oof = rob_npz["maturity__original"].astype(np.float64)
per_orig_oof = rob_npz["periodicity__original"].astype(np.float64)
per_fresh_oof = rob_npz["periodicity__fresh_kumo_seed"].astype(np.float64)
recipe_A_oof = 0.5 * (per_orig_oof + per_fresh_oof)

h016_npz = np.load(EXP_ROOT / "transfer-20261025/spline_direct_ks/OOF.npz", allow_pickle=True)
h016_oof = h016_npz["two_seed_mean"].astype(np.float64)

print(f"Matched Baseline (B) KS = {ks_stat(y_global, baseline_oof):.12f} | Recipe A KS = {ks_stat(y_global, recipe_A_oof):.12f} | H016 KS = {ks_stat(y_global, h016_oof):.12f}", flush=True)

orig_train_ids = pd.read_parquet(EXP_ROOT / "ranking-pipeline-20261020/input/data/train_features.parquet", columns=["application_id"])["application_id"]
spectral_df = pd.read_parquet(PERIODIC / "true_features.parquet", columns=["application_id"] + EXTRA_PER).set_index("application_id").loc[orig_train_ids].reset_index(drop=True)

roles_list = []
baseline_raw_20k = []
baseline_spline_val = []
per_spline_val = []
h016_spline_val = []
exp_tables_70k = []
kumo56_70k = []

for f in range(5):
    r = np.load(FRZ_ROOT / f"frozen/fold{f}/roles.npz", allow_pickle=True)
    roles_list.append(r)
    b = np.load(FRZ_ROOT / f"frozen/fold{f}/baseline.npz", allow_pickle=True)
    baseline_raw_20k.append(b["raw"])
    baseline_spline_val.append(b["spline"])

    tb = pd.read_parquet(OUT_ROOT / f"fold{f}/expanded.parquet")
    for c in EXTRA_PER:
        tb[c] = spectral_df[c].to_numpy(dtype=np.float32)
    exp_tables_70k.append(tb)

    # Recover exact per_spline_val[f] from per_orig_oof[r["validation"]] and Head29 on base_raw!
    q_idx = np.r_[r["correction"], r["validation"]]
    tb_q = tb.iloc[q_idx].reset_index(drop=True)
    h29 = feature_head.fit(tb_q.iloc[:6000].assign(target=r["y_correction"]), b["raw"][:6000], HEAD_PER_COLS, 0.003, 0.5)
    c29_val = b_mod.corrected_probability(h29, tb_q.iloc[6000:], b["raw"][6000:])[0]
    sp_per_f = (per_orig_oof[r["validation"]] - 0.75 * c29_val) / 0.25
    per_spline_val.append(sp_per_f)

    # Also load H016 spline_delta
    h16_val = np.load(EXP_ROOT / f"transfer-20261025/spline_direct_ks/fold{f}/validation.npz")
    h016_spline_val.append(sp_per_f + h16_val["spline_delta"])

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
    "tr_seed_rep": load_20k([EXP_ROOT / f"transfer-20261025/robustness/fold{f}/raw_seed_repeat.npz" for f in range(5)], "prediction"),
    "pop_spec": load_20k([OUT_ROOT / f"kumo_runs/kumo_pop_specialist/fold{f}/prediction.npz" for f in range(5)]),
    "surg96": load_20k([OUT_ROOT / f"kumo_runs/kumo_surgical96/fold{f}/prediction.npz" for f in range(5)]),
    "champ90_s7731": load_20k([FRZ_ROOT / f"runs/kumo_large_champion90_s7731/fold{f}/prediction.npz" for f in range(5)]),
    "shadow91": load_20k([FRZ_ROOT / f"runs/kumo_large_shadow91/fold{f}/prediction.npz" for f in range(5)]),
    "shadow91_s7731": load_20k([FRZ_ROOT / f"runs/kumo_large_shadow91_s7731/fold{f}/prediction.npz" for f in range(5)]),
    "latent99": load_20k([FRZ_ROOT / f"runs/kumo_large_latent99/fold{f}/prediction.npz" for f in range(5)]),
    "latent99_s7731": load_20k([FRZ_ROOT / f"runs/kumo_large_latent99_s7731/fold{f}/prediction.npz" for f in range(5)]),
    "full100": load_20k([FRZ_ROOT / f"runs/kumo_large_full100/fold{f}/prediction.npz" for f in range(5)]),
    "full100_s7731": load_20k([FRZ_ROOT / f"runs/kumo_large_full100_s7731/fold{f}/prediction.npz" for f in range(5)]),
    "kumo_sup90": load_20k([EXP_ROOT / f"kumo-supervised-view-20261021/folds/fold{f}/raw.npz" for f in range(5)], "prediction"),
    "covplace96": load_20k([EXP_ROOT / f"kumo-coverage-placement-20261021/conditions/fold{f}/coverage96/raw_predictions.npz" for f in range(5)], "prediction"),
    "fe_cov_info": load_20k([EXP_ROOT / f"coverage-information-fe-20261021/runs/fold{f}/raw.npz" for f in range(5)], "prediction"),
    "fe_mat_mask": load_20k([EXP_ROOT / f"mature-ma[REDACTED_CREDENTIAL]/runs/fold{f}/raw.npz" for f in range(5)], "prediction"),
    "fe_mat_denom": load_20k([EXP_ROOT / f"mature-denominator-fe-20261021/runs/fold{f}/raw.npz" for f in range(5)], "prediction"),
    "horiz_l8_b90": load_20k([Path(f"/mnt/home/faris/logika-horizontal-20261026/runs/large8/baseline90/fold{f}/prediction.npz") for f in range(5)]),
    "horiz_l8_p18": load_20k([Path(f"/mnt/home/faris/logika-horizontal-20261026/runs/large8/p0018/fold{f}/prediction.npz") for f in range(5)]),
    "tr_early_avg": load_20k([EXP_ROOT / f"transfer-20261025/early_average/fold{f}/raw_mean.npz" for f in range(5)], "prediction"),
    "tr_mat_orig": load_20k([EXP_ROOT / f"transfer-20261025/core_maturity/fold{f}/original_raw.npz" for f in range(5)], "prediction"),
    "tr_mat_fresh": load_20k([EXP_ROOT / f"transfer-20261025/core_maturity/fold{f}/fresh_raw.npz" for f in range(5)], "prediction"),
    "kumo56": [kumo56_70k[f][np.r_[roles_list[f]["correction"], roles_list[f]["validation"]]] for f in range(5)],
}
for pack in sorted(os.listdir("/mnt/home/faris/logika-augment90-20261026/runs/large8")):
    plist = [Path(f"/mnt/home/faris/logika-augment90-20261026/runs/large8/{pack}/fold{f}/prediction.npz") for f in range(5)]
    if all(p.exists() for p in plist):
        m20k[f"aug_l8_{pack}"] = load_20k(plist)

def get_per_corrected(m_key, sp_weight=0.25, sp_source="h016"):
    val_out = np.zeros(70000, dtype=np.float64)
    head_out = np.zeros(70000, dtype=np.float64)
    cor_head_out = []
    for f in range(5):
        r = roles_list[f]
        q_idx = np.r_[r["correction"], r["validation"]]
        tb_p = exp_tables_70k[f].iloc[q_idx].reset_index(drop=True)
        p_f = m20k[m_key][f]
        h_f = feature_head.fit(tb_p.iloc[:6000].assign(target=r["y_correction"]), p_f[:6000], HEAD_PER_COLS, 0.003, 0.5)
        c_cor = b_mod.corrected_probability(h_f, tb_p.iloc[:6000], p_f[:6000])[0]
        c_val = b_mod.corrected_probability(h_f, tb_p.iloc[6000:], p_f[6000:])[0]
        cor_head_out.append(c_cor)
        head_out[r["validation"]] = c_val
        sp_v = h016_spline_val[f] if sp_source == "h016" else (per_spline_val[f] if sp_source == "per" else baseline_spline_val[f])
        val_out[r["validation"]] = (1.0 - sp_weight) * c_val + sp_weight * sp_v
    return val_out, head_out, cor_head_out

print("\n=== STEP 2: PERIODICITY+MATURITY HEAD (29 COLS) + H016 SPLINE ON INDIVIDUAL KUMO MODELS ===", flush=True)
per_finals = {}
per_heads = {}
per_cors = {}
for mk in [k for k in ["base_raw", "tr_seed_rep", "pop_spec", "surg96", "champ90_s7731", "shadow91", "shadow91_s7731", "kumo_sup90", "covplace96", "fe_cov_info", "fe_mat_mask", "horiz_l8_b90", "tr_early_avg", "tr_mat_orig", "tr_mat_fresh", "kumo56", "aug_l8_diverse", "aug_l8_repayment_credibility", "aug_l8_volatility"] if k in m20k]:
    vf, vh, vc = get_per_corrected(mk, sp_weight=0.25, sp_source="h016")
    per_finals[mk] = vf
    per_heads[mk] = vh
    per_cors[mk] = vc
    print(f"  {mk:30s}: Head29_KS={ks_stat(y_global, vh):.6f} | Final(75/25_H016)_KS={ks_stat(y_global, vf):.6f} (AUC={roc_auc_score(y_global, vf):.6f})", flush=True)

# 3. First check blends of per_finals and per_heads from Step 2!
print("\n=== STEP 2B: ENSEMBLE BLENDS OF HEAD29 + H016 SPLINE MODELS ===", flush=True)
top_per_keys = ["base_raw", "tr_seed_rep", "pop_spec", "champ90_s7731", "horiz_l8_b90", "tr_early_avg", "fe_mat_mask", "tr_mat_orig", "shadow91", "fe_cov_info", "covplace96", "aug_l8_volatility"]
for k_cnt in [3, 5, 8, 12]:
    sub_k = top_per_keys[:k_cnt]
    p_eq = np.mean([per_finals[k] for k in sub_k], axis=0)
    print(f"  Equal Mean Top-{k_cnt:2d} ({sub_k[-1]:15s}): KS={ks_stat(y_global, p_eq):.6f} | AUC={roc_auc_score(y_global, p_eq):.6f}", flush=True)

for w_ps in [0.10, 0.20, 0.30, 0.40]:
    for w_tr in [0.15, 0.25, 0.35]:
        p_w = (
            (1.0 - w_ps - w_tr) * h016_oof
            + w_ps * per_finals["pop_spec"]
            + w_tr * np.mean([per_finals["tr_early_avg"], per_finals["horiz_l8_b90"], per_finals["champ90_s7731"], per_finals["fe_mat_mask"], per_finals["kumo56"]], axis=0)
        )
        print(f"  [w_ps={w_ps:.2f} w_tr={w_tr:.2f}] KS={ks_stat(y_global, p_w):.6f} | AUC={roc_auc_score(y_global, p_w):.6f}", flush=True)

# 4. Now let's build the 56,000-row Expanded Spline + Multi-Threshold Smooth-KS Optimizer!
screen_df = pd.read_csv(OUT_ROOT / "inner_oof_residual_screen.csv")
exp_cols_set = set(exp_tables_70k[0].columns)
screen_df = screen_df[(screen_df["sign_all"] == 5) & (screen_df["name"].isin(exp_cols_set)) & (~screen_df["name"].isin(HEAD_PER_COLS))].copy()
screen_df["abs_z"] = screen_df["z_all"].abs()
screen_df = screen_df.sort_values("abs_z", ascending=False)
top_res_cols = screen_df["name"].tolist()
print(f"\nFound {len(top_res_cols)} extra 5/5-fold consistent residual features in expanded.parquet: {top_res_cols[:10]}", flush=True)

def multi_tau_smooth_ks_obj(delta, x, eta0, y, strength, taus, temp, trust, ridge):
    disp = strength * (x @ delta)
    eta = eta0 + disp
    a = np.where(y == 1, -1.0 / max(int(y.sum()), 1), 1.0 / max(int((y == 0).sum()), 1))
    val_ks = 0.0
    grad_u = np.zeros_like(eta)
    for tau in taus:
        u = expit((eta - tau) / temp)
        val_ks += float(a @ u)
        grad_u += a * u * (1.0 - u) / temp
    val_ks /= len(taus)
    grad_u /= len(taus)
    val = val_ks + 0.5 * trust * float(np.mean(disp ** 2)) + 0.5 * ridge * float(delta @ delta)
    grad = strength * (x.T @ (grad_u + trust * disp / len(y))) + ridge * delta
    return val, grad

print("\n=== STEP 3: 56,000-ROW EXPANDED SPLINE HEAD + MULTI-THRESHOLD SMOOTH-KS ===", flush=True)
lin56 = np.load(OUT_ROOT / "linear56k_runs.npz", allow_pickle=True)

for n_extra in [0, 12, 24]:
    cols_56k = HEAD_PER_COLS + top_res_cols[:n_extra]
    val_56k_bce = np.zeros(70000, dtype=np.float64)
    val_56k_sks_t03 = np.zeros(70000, dtype=np.float64)
    val_56k_sks_t02 = np.zeros(70000, dtype=np.float64)

    for f in range(5):
        r = roles_list[f]
        tb = exp_tables_70k[f]
        tr_idx = np.r_[r["context"], r["correction"]]
        y_tr = np.r_[r["y_context"], r["y_correction"]]

        # Offset on 56k training rows: kumo56 on context (50k), average of base_raw+tr_seed_rep+pop_spec on correction (6k) and validation (14k)
        p_ens_q = 0.40 * m20k["pop_spec"][f] + 0.30 * m20k["base_raw"][f] + 0.30 * m20k["tr_seed_rep"][f]
        p_tr = np.r_[kumo56_70k[f][r["context"]], p_ens_q[:6000]]
        p_val = p_ens_q[6000:]

        tb_tr = tb.iloc[tr_idx].reset_index(drop=True).assign(target=y_tr)
        tb_val = tb.iloc[r["validation"]].reset_index(drop=True)

        # Fit 56k BCE spline head
        h56 = feature_head.fit(tb_tr, p_tr, cols_56k, 0.003, 0.5)
        x_tr, off_tr = b_mod.design(h56, tb_tr, p_tr)
        x_val, off_val = b_mod.design(h56, tb_val, p_val)

        # Appending cross-fitted 56k model logits (Spline200, ResLGBM, ResCat) to x_tr and x_val for Smooth-KS!
        cf = np.load(OUT_ROOT / f"cpu_runs/fold{f}_crossfit56k.npz")
        z_sp200_tr = logit(clip_prob(lin56["oof_spline200"][f][tr_idx])) - off_tr
        z_sp200_val = logit(clip_prob(lin56["val_spline200"][r["validation"]])) - off_val
        z_lgb_tr = logit(clip_prob(cf["oof_res_lgbm_56k"])) - off_tr
        z_lgb_val = logit(clip_prob(cf["val_res_lgbm_56k"])) - off_val

        x_tr_aug = np.column_stack([x_tr, z_sp200_tr, z_lgb_tr])
        x_val_aug = np.column_stack([x_val, z_sp200_val, z_lgb_val])

        eta0_tr = off_tr + 0.5 * (x_tr @ h56["coefficients"])
        eta0_val = off_val + 0.5 * (x_val @ h56["coefficients"])
        c_bce_val = expit(eta0_val)
        sp_mix_val = 0.80 * h016_spline_val[f] + 0.20 * lin56["val_spline200"][r["validation"]]
        val_56k_bce[r["validation"]] = 0.75 * c_bce_val + 0.25 * sp_mix_val

        # Find empirical KS threshold on 56k training rows
        tau_star, _ = ks_threshold(y_tr, eta0_tr)
        taus = [tau_star - 0.10, tau_star - 0.05, tau_star, tau_star + 0.05, tau_star + 0.10]

        for temp, trust, target_arr in [(0.30, 0.25, val_56k_sks_t03), (0.20, 0.15, val_56k_sks_t02)]:
            d0 = np.zeros(x_tr_aug.shape[1], dtype=np.float64)
            res = minimize(
                multi_tau_smooth_ks_obj, d0,
                args=(x_tr_aug, eta0_tr, y_tr, 0.5, taus, temp, trust, 0.002),
                method="L-BFGS-B", jac=True,
                options=dict(maxiter=200, ftol=1e-10, gtol=1e-6)
            )
            c_sks_val = expit(eta0_val + 0.5 * (x_val_aug @ res.x))
            target_arr[r["validation"]] = 0.75 * c_sks_val + 0.25 * sp_mix_val

    print(f"[n_extra={n_extra:2d}] 56k_BCE_KS={ks_stat(y_global, val_56k_bce):.6f} | 56k_SmoothKS(T=0.30)_KS={ks_stat(y_global, val_56k_sks_t03):.6f} | 56k_SmoothKS(T=0.20)_KS={ks_stat(y_global, val_56k_sks_t02):.6f}", flush=True)
    for w_bl in [0.25, 0.40, 0.55]:
        b1 = (1.0 - w_bl) * h016_oof + w_bl * val_56k_sks_t03
        b2 = (1.0 - w_bl) * h016_oof + w_bl * val_56k_sks_t02
        print(f"    Blend w={w_bl:.2f} with H016: T=0.30 KS={ks_stat(y_global, b1):.6f} | T=0.20 KS={ks_stat(y_global, b2):.6f} (AUC={roc_auc_score(y_global, b2):.6f})", flush=True)
