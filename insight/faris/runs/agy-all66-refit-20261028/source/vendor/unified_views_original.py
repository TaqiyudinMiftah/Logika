import os
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("SDM_CHUNK_MEMORY_FRACTION", "0.02")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/mnt/home/faris/logika-model-arena-20261025/storage/huggingface"

import gc
import hashlib
import importlib
import json
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

t0 = time.time()
OUT_ROOT = Path("/mnt/home/faris/logika-joint-20261028")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
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

# 1. Select the top 180 non-redundant CB columns (invariant + fold-specific) from inner_oof_residual_screen.csv + key domain columns
screen_df = pd.read_csv(OUT_ROOT / "inner_oof_residual_screen.csv")
raw_npz = np.load(OUT_ROOT / "raw_domain_features.npz", allow_pickle=True)
raw_names = [str(x) for x in raw_npz["names"]]
raw_data = raw_npz["data"].astype(np.float32)
raw_df = pd.DataFrame(raw_data, columns=raw_names)

extra_df = pd.read_parquet("/mnt/home/faris/logika-horizontal-20261026/extra.parquet")
extra_df.columns = [f"hz_{c}" for c in extra_df.columns]

frz0 = pd.read_parquet(FRZ_ROOT / "frozen/fold0/features.parquet")
frz_cols = list(frz0.columns)

cb0_tr = pd.read_parquet("/mnt/home/faris/logika-faris-exploration/catboost-convergence-20261021/data/fold0/train.parquet")
cb0_va = pd.read_parquet("/mnt/home/faris/logika-faris-exploration/catboost-convergence-20261021/data/fold0/validation.parquet")
cb_avail = [c for c in cb0_tr.columns if c in cb0_va.columns and c not in ["target", "application_id"] and c not in frz_cols]

# Rank cb_avail columns by max_abs_z in screen_df (or include if fold-specific / categorical)
screen_map = dict(zip(screen_df["name"], screen_df["max_abs_z"]))
cb_Extra_str = ["v2_prev_last_decision", "v2_ecom_last_payment_method"]
cb_Extra_fspec = [
    "v3_repay_180d_late_90_posterior_6", "v3_repay_180d_late_90_posterior_24",
    "v3_repay_180d_late_60_posterior_6", "v3_repay_180d_late_60_posterior_24",
    "v3_repay_all_late_90_posterior_6", "v3_repay_all_late_60_posterior_6",
    "v4_price_180d_between_category_std", "v4_price_all_between_category_std",
    "v4_price_90d_equal_category_mean", "v4_price_90d_equal_category_median",
    "v4_price_180d_equal_category_mean", "v4_price_180d_equal_category_median",
    "v5_price_pooled_180d_equal_category_mean", "v5_price_cancelled_all_equal_category_mean",
    "v5_price_cancelled_all_equal_order_mean", "v5_price_cancelled_all_mean",
    "v5_price_cancelled_180d_std", "v5_ecom_180d_completion_posterior",
    "v5_price_income_residual", "v4_last12_late_90_posterior", "v4_weighted_late_90_90d",
    "re_price_latent_multiplier", "re_income_log_measurement_sd", "re_income_log_sd",
    "re_late_1_posterior_sd", "re_late_30_posterior_sd", "re_bureau_score_gls"
]
cb_ranked_inv = sorted(
    [c for c in cb_avail if c in screen_map and c not in cb_Extra_fspec],
    key=lambda c: screen_map[c],
    reverse=True
)[:160]
selected_cb_cols = list(dict.fromkeys(cb_Extra_str + cb_Extra_fspec + cb_ranked_inv))
selected_cb_cols = [c for c in selected_cb_cols if c in cb_avail]
print(f"Selected {len(selected_cb_cols)} top CB columns + {len(frz_cols)} frozen + {len(raw_names)} raw + {len(extra_df.columns)} horizontal")

# Build and save per-fold expanded feature tables in global 0..69999 row order
roles_list = []
y_global = np.zeros(70000, dtype=np.int8)
for f in range(5):
    fold_dir = OUT_ROOT / f"fold{f}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    r = np.load(FRZ_ROOT / f"frozen/fold{f}/roles.npz", allow_pickle=True)
    roles_list.append(r)
    y_global[r["context"]] = r["y_context"]
    y_global[r["correction"]] = r["y_correction"]
    exp_path = fold_dir / "expanded.parquet"
    if not exp_path.exists():
        frz_f = pd.read_parquet(FRZ_ROOT / f"frozen/fold{f}/features.parquet")
        tr_idx = np.concatenate([r["context"], r["correction"]])
        va_idx = r["validation"]
        cb_tr = pd.read_parquet(f"/mnt/home/faris/logika-faris-exploration/catboost-convergence-20261021/data/fold{f}/train.parquet", columns=selected_cb_cols)
        cb_va = pd.read_parquet(f"/mnt/home/faris/logika-faris-exploration/catboost-convergence-20261021/data/fold{f}/validation.parquet", columns=selected_cb_cols)
        cb_full = pd.concat([cb_tr, cb_va], axis=0, ignore_index=True)
        inv_order = np.empty(70000, dtype=np.int32)
        inv_order[np.concatenate([tr_idx, va_idx])] = np.arange(70000, dtype=np.int32)
        cb_full = cb_full.iloc[inv_order].reset_index(drop=True)
        full_df = pd.concat([frz_f.reset_index(drop=True), raw_df, extra_df.reset_index(drop=True), cb_full], axis=1)
        # Deduplicate any overlapping column names
        full_df = full_df.loc[:, ~full_df.columns.duplicated()]
        full_df.to_parquet(exp_path, index=False)
        print(f"Saved fold {f} expanded table: {full_df.shape} in {time.time()-t0:.1f}s")

# 2. Define Kumo helper (identical checkpoint verification & deterministic inference as frozen-shadow-20261026)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def make_kumo(size="large", device="cuda"):
    import sdm, torch
    checkpoint = Path(os.environ["HF_HOME"]) / f"hub/models--nvidia--Kumo-Tabular/snapshots/b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f/{size}/classifier.pt"
    want = json.loads((FRZ_ROOT / "CHECKPOINTS.json").read_text())[size]
    assert sha(checkpoint) == want, "Checkpoint hash mismatch!"
    module = importlib.import_module(sdm.models.KumoTabular.__module__)
    def resolve(**kw):
        return str(checkpoint)
    with patch.object(module, "download_checkpoint", resolve):
        return sdm.models.KumoTabular(task="classification", size=size, device=device)

def kumo_fit_predict(x_df, trainrows, y_train, queryrows, estimators=8, seed=31027):
    import torch, sdm
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.88)
    ordered = x_df.iloc[np.r_[trainrows, queryrows]].reset_index(drop=True).copy()
    n = len(trainrows)
    assert ordered.shape[1] <= 99, f"Too many feature columns for Kumo: {ordered.shape[1]}"
    types = {}
    for c in ordered.columns:
        if pd.api.types.is_numeric_dtype(ordered[c]):
            types[c] = "numerical"
        else:
            vocab = set(ordered[c].iloc[:n].dropna().astype(str))
            ordered[c] = ordered[c].astype("string").where(ordered[c].astype("string").isin(vocab))
            types[c] = "categorical"
    table = sdm.TableTensor.from_pandas(ordered, stypes=types, device=device)
    target = sdm.TableTensor.from_pandas(pd.DataFrame({"target": y_train}), stypes={"target": "categorical"}, device=device)
    model = make_kumo("large", device=device)
    pred = []
    with torch.inference_mode(), torch.amp.autocast(device, dtype=torch.float16, enabled=True):
        model.fit(table[:n], target, num_estimators=estimators, estimator_batch_size=1, generator=torch.Generator(device=device).manual_seed(seed))
        for start in range(n, len(table), 512):
            pred.append(model.predict(table[start:start+512])[:, "1"].numerical.reshape(-1).float().cpu().numpy())
    raw = np.concatenate(pred)
    model.clear()
    del table, target, model
    gc.collect()
    torch.cuda.empty_cache()
    return raw

# 3. Define our Surgical & Population-Specialist Views (all <= 98 columns)
sample_df = pd.read_parquet(OUT_ROOT / "fold0/expanded.parquet")
avail_set = set(sample_df.columns)
def check_cols(cols):
    out = [c for c in dict.fromkeys(cols) if c in avail_set]
    assert len(out) <= 98, f"View has {len(out)} columns (>98)!"
    return out

c90 = json.loads((FRZ_ROOT / "FEATURE_FREEZE.json").read_text())["views"]["champion90"]
redundant_9 = {
    "has_account",
    "v4_bureau_180d_equal_provider_median_mean",
    "v2_bureau_180d_score_mean",
    "v4_bureau_all_equal_provider_mean_mean",
    "v4_bureau_history_weighted_90d",
    "v2_bureau_365d_score_mean",
    "v5_price_pooled_all_mean",
    "v5_price_pooled_all_equal_order_mean",
    "v5_price_pooled_180d_mean",
}
c81_core = [c for c in c90 if c not in redundant_9]

# View A: Pop A Specialist (72 active columns on Pop A — zero dead account/repay/prev columns!)
popA_active_core = [
    c for c in c81_core
    if not any(k in c for k in ["repay", "account", "previous", "late_", "existing_", "cash_deficit", "observed_income", "has_"])
]
popA_add = [
    "raw_bur_util_min", "v2_bureau_365d_utilization_min", "v2_bureau_180d_utilization_min",
    "v2_bureau_90d_utilization_min", "raw_bur_curr_util_min", "raw_bur_util_mean", "raw_bur_util_std",
    "v2_bureau_current_active_accounts_std", "raw_bur_active_acc_std", "raw_bur_active_acc_mean",
    "raw_bur_active_acc_max", "raw_bur_out_bal_std", "v2_bureau_current_outstanding_balance_std",
    "raw_bur_inq_per_acc_mean", "raw_bur_delinq_per_acc_mean", "raw_bur_log_out_bal_max",
    "raw_bur_curr_log_out_bal_max", "raw_bur_has_rev_lim_share", "raw_bur_log_rev_lim_mean",
    "raw_bur_log_rev_bal_mean", "bureau_Bureau_B_utilization", "re_bureau_score_posterior",
    "re_bureau_posterior_sd", "raw_ec_groceries_last_age", "ecom_spend90_income_ratio",
    "v3_ecom_180d_groceries_item_count", "v3_ecom_180d_groceries_spend",
    "v4_price_180d_between_category_std", "v4_price_all_between_category_std",
    "v4_price_90d_equal_category_mean", "v5_price_cancelled_all_equal_category_mean",
    "raw_ec_pm_COD_share", "raw_ec_pm_e_wallet_share", "raw_ec90_ewallet_spend_share",
    "raw_ec_st_cancelled_share", "raw_ec_log_up_std", "raw_ec_log_up_max",
    "re_price_log_posterior", "re_price_latent_multiplier", "raw_rate_times_pti",
    "raw_tenure_to_age_months", "v2_ecom_last_payment_method"
]
view_popA = check_cols(popA_active_core + popA_add)

# View B/C1: Pop B + C1 Specialist
popBC1_add = [
    "v2_account_12m_low_balance_fraction_min", "raw_acc_low_bal_days_min", "raw_acc_low_bal_days_max",
    "raw_acc_low_bal_days_std", "raw_acc_zero_low_bal_share", "raw_acc_signed_log_min_bal_std",
    "raw_acc_signed_log_close_bal_std", "v2_account_average_balance_min",
    "v2_account_balance_income_ratio_min", "account_income_cv",
    "hz_path_reapply_after_nonapproval_fraction", "raw_prv_n_rej", "raw_prv_rej_share",
    "raw_prv_rate_mean", "raw_prv_rate_diff_curr", "raw_prv_last_is_rej",
    "v2_prev_approved_amount_max", "v2_prev_last_requested_amount", "v2_prev_last_decision",
    "v3_ecom_180d_home_price_mean", "v3_ecom_180d_personal_care_price_median",
    "v2_ecom_spend90_balance_ratio", "v4_positive_dpd_std", "re_late_1_posterior",
    "re_late_30_posterior", "re_price_log_posterior", "re_bureau_score_posterior",
    "raw_acc_non_sal_share_mean", "raw_bur_out_bal_std", "raw_bur_inq_per_acc_mean"
]
view_popBC1 = check_cols(c81_core[:68] + popBC1_add)

# View C2: Pop C2 Specialist
popC2_add = [
    "raw_acc_non_sal_share_std", "raw_acc_non_sal_share_mean", "raw_acc_log_inf_inc_gap_mean",
    "raw_acc_low_bal_days_min", "v2_account_12m_low_balance_fraction_min",
    "v2_account_12m_balance_income_ratio_std", "v2_account_saving_rate_std",
    "v2_account_saving_rate_min", "re_income_log_sd", "re_income_log_measurement_sd",
    "v4_positive_dpd_std", "v2_repay_180d_dpd90_share", "v2_repay_365d_dpd_max",
    "v2_loan_repay_max_historical_dpd_mean", "raw_rep_delay_hrs_max",
    "v2_repay_90d_dpd60_share", "v3_repay_180d_late_90_posterior_6",
    "v3_repay_180d_late_60_posterior_6", "re_late_1_posterior", "re_late_30_posterior",
    "re_price_log_posterior", "re_bureau_score_posterior", "raw_bur_out_bal_std",
    "raw_bur_inq_per_acc_mean", "v4_price_180d_between_category_std"
]
view_popC2 = check_cols(c81_core[:72] + popC2_add)

# View Global Surgical 96: c81_core + top 15 orthogonal residual features across all populations
surgical15 = [
    "raw_acc_low_bal_days_min", "v2_account_12m_low_balance_fraction_min",
    "v4_positive_dpd_std", "raw_bur_inq_per_acc_mean", "raw_bur_out_bal_std",
    "raw_acc_non_sal_share_mean", "raw_acc_non_sal_share_std",
    "v2_loan_repay_max_historical_dpd_mean", "raw_bur_log_out_bal_max",
    "v2_repay_365d_dpd_max", "v2_account_12m_balance_income_ratio_std",
    "raw_rep_delay_hrs_max", "raw_bur_util_min", "v2_account_balance_income_ratio_min",
    "v4_price_180d_between_category_std"
]
view_surgical96 = check_cols(c81_core + surgical15)

print(f"Views ready: popA={len(view_popA)}, popBC1={len(view_popBC1)}, popC2={len(view_popC2)}, surgical96={len(view_surgical96)}")

# 4. Run Population-Routed Specialist Kumo (popA + popBC1 + popC2 -> 20,000 query rows per fold)
spec_dir = OUT_ROOT / "kumo_runs/kumo_pop_specialist"
spec_dir.mkdir(parents=True, exist_ok=True)

for f in range(5):
    f_out = spec_dir / f"fold{f}/prediction.npz"
    if f_out.exists():
        continue
    f_out.parent.mkdir(parents=True, exist_ok=True)
    ts = time.time()
    df = pd.read_parquet(OUT_ROOT / f"fold{f}/expanded.parquet")
    r = roles_list[f]
    ctx = r["context"]
    y_ctx = r["y_context"]
    query = np.r_[r["correction"], r["validation"]]
    pred_query = np.full(len(query), np.nan, dtype=np.float32)

    # Identify populations on ctx and query
    is_popA_all = df["pop_A"].values > 0.5
    is_popC2_all = df["pop_C2"].values > 0.5
    is_popBC1_all = (~is_popA_all) & (~is_popC2_all)

    # 1) Pop A
    ctx_A = ctx[is_popA_all[ctx]]
    y_ctx_A = y_ctx[is_popA_all[ctx]]
    q_sel_A = np.flatnonzero(is_popA_all[query])
    pred_A = kumo_fit_predict(df[view_popA], ctx_A, y_ctx_A, query[q_sel_A], estimators=8, seed=31027 + f*10 + 1)
    pred_query[q_sel_A] = pred_A

    # 2) Pop B + C1 (plus 5,000 random Pop C2 context rows with lowest installment count for smooth transition)
    ctx_BC1 = ctx[is_popBC1_all[ctx]]
    y_ctx_BC1 = y_ctx[is_popBC1_all[ctx]]
    ctx_C2_pool = ctx[is_popC2_all[ctx]]
    y_C2_pool = y_ctx[is_popC2_all[ctx]]
    # Pick 5,044 Pop C2 context rows with lowest repayment_installment_count (closest to C1 boundary 11)
    inst_cnt_C2 = df["repayment_installment_count"].values[ctx_C2_pool]
    near_C2_idx = np.argsort(inst_cnt_C2, kind="mergesort")[:5044]
    ctx_BC1_aug = np.r_[ctx_BC1, ctx_C2_pool[near_C2_idx]]
    y_BC1_aug = np.r_[y_ctx_BC1, y_C2_pool[near_C2_idx]]
    q_sel_BC1 = np.flatnonzero(is_popBC1_all[query])
    pred_BC1 = kumo_fit_predict(df[view_popBC1], ctx_BC1_aug, y_BC1_aug, query[q_sel_BC1], estimators=8, seed=31027 + f*10 + 2)
    pred_query[q_sel_BC1] = pred_BC1

    # 3) Pop C2
    ctx_C2 = ctx[is_popC2_all[ctx]]
    y_ctx_C2 = y_ctx[is_popC2_all[ctx]]
    q_sel_C2 = np.flatnonzero(is_popC2_all[query])
    pred_C2 = kumo_fit_predict(df[view_popC2], ctx_C2, y_ctx_C2, query[q_sel_C2], estimators=8, seed=31027 + f*10 + 3)
    pred_query[q_sel_C2] = pred_C2

    assert np.isfinite(pred_query).all()
    np.savez_compressed(f_out, rows=query, raw=pred_query)
    val_y = y_global[r["validation"]]
    val_p = pred_query[6000:]
    base_p = np.load(FRZ_ROOT / f"runs/kumo_large_champion90/fold{f}/prediction.npz")["raw"][6000:]
    blend_p = 0.75 * base_p + 0.25 * val_p
    print(f"[kumo_pop_specialist fold {f}] done in {time.time()-ts:.1f}s | Spec KS={ks_stat(val_y, val_p):.6f} AUC={roc_auc_score(val_y, val_p):.6f} | Base KS={ks_stat(val_y, base_p):.6f} | 75/25 Blend KS={ks_stat(val_y, blend_p):.6f}", flush=True)

# 5. Run Global Surgical 96 Kumo Large (50,000 context -> 20,000 query)
surg_dir = OUT_ROOT / "kumo_runs/kumo_surgical96"
surg_dir.mkdir(parents=True, exist_ok=True)

for f in range(5):
    f_out = surg_dir / f"fold{f}/prediction.npz"
    if f_out.exists():
        continue
    f_out.parent.mkdir(parents=True, exist_ok=True)
    ts = time.time()
    df = pd.read_parquet(OUT_ROOT / f"fold{f}/expanded.parquet", columns=view_surgical96)
    r = roles_list[f]
    ctx = r["context"]
    y_ctx = r["y_context"]
    query = np.r_[r["correction"], r["validation"]]
    pred_query = kumo_fit_predict(df, ctx, y_ctx, query, estimators=8, seed=31027 + f)
    assert np.isfinite(pred_query).all()
    np.savez_compressed(f_out, rows=query, raw=pred_query.astype(np.float32))
    val_y = y_global[r["validation"]]
    val_p = pred_query[6000:]
    base_p = np.load(FRZ_ROOT / f"runs/kumo_large_champion90/fold{f}/prediction.npz")["raw"][6000:]
    blend_p = 0.50 * base_p + 0.50 * val_p
    print(f"[kumo_surgical96 fold {f}] done in {time.time()-ts:.1f}s | Surg96 KS={ks_stat(val_y, val_p):.6f} AUC={roc_auc_score(val_y, val_p):.6f} | Base KS={ks_stat(val_y, base_p):.6f} | 50/50 Blend KS={ks_stat(val_y, blend_p):.6f}", flush=True)

print(f"All Kumo runs finished in {time.time()-t0:.1f}s!", flush=True)
