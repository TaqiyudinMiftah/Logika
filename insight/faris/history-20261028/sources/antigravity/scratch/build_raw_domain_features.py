import numpy as np
import pandas as pd
import time

t0 = time.time()
DATA_DIR = "/home/parasite/Project/Competition/logika-2026/dataset"

tr = pd.read_csv(f"{DATA_DIR}/train_applications.csv")
cp = pd.read_csv(f"{DATA_DIR}/customer_profiles.csv")
apps = tr.merge(cp, on="customer_id", how="left")
N = len(apps)
assert N == 70000
assert apps["customer_id"].nunique() == N
cust_to_idx = pd.Series(np.arange(N, dtype=np.int32), index=apps["customer_id"])
app_dt = pd.to_datetime(apps["application_datetime"]).values
dec_dt = pd.to_datetime(apps["decision_datetime"]).values

feats = {}

# 0. Application & Decision timing + nonlinear contract ratios
inc = apps["declared_monthly_income"].values.astype(np.float64)
loan = apps["loan_amount"].values.astype(np.float64)
inst = apps["proposed_monthly_installment"].values.astype(np.float64)
rate = apps["annual_interest_rate"].values.astype(np.float64)
age = apps["age"].values.astype(np.float64)
ten = apps["employment_tenure_months"].values.astype(np.float64)
dep = apps["dependents"].values.astype(np.float64)

decision_lag_hours = (dec_dt - app_dt) / np.timedelta64(1, "h")
app_ts = pd.to_datetime(apps["application_datetime"])
feats["raw_decision_lag_hours"] = decision_lag_hours
feats["raw_app_hour"] = app_ts.dt.hour.values.astype(np.float64)
feats["raw_app_dayofweek"] = app_ts.dt.dayofweek.values.astype(np.float64)
feats["raw_app_day"] = app_ts.dt.day.values.astype(np.float64)
feats["raw_tenure_to_age_months"] = ten / np.maximum((age - 18.0) * 12.0, 1.0)
feats["raw_log_income_minus_log_inst"] = np.log1p(inc) - np.log1p(inst)
feats["raw_disp_income_after_inst"] = inc - inst
feats["raw_disp_income_per_dep"] = (inc - inst) / (dep + 1.0)
feats["raw_rate_times_pti"] = rate * (inst / np.maximum(inc, 1.0))
# Amortization excess: actual proposed installment vs theoretical 12m installment
r_m = rate / 12.0
theo_inst = loan * (r_m * (1.0 + r_m)**12) / ((1.0 + r_m)**12 - 1.0)
feats["raw_inst_vs_theo_ratio"] = inst / np.maximum(theo_inst, 1.0)

def put_group(s, name, fill=np.nan):
    arr = np.full(N, fill, dtype=np.float64)
    arr[s.index.values] = s.values
    feats[name] = arr

# 1. Account Behavior
acc = pd.read_csv(f"{DATA_DIR}/account_behavior.csv")
acc = acc[acc["customer_id"].isin(cust_to_idx.index)].copy()
acc["idx"] = cust_to_idx.loc[acc["customer_id"]].values
acc_month = pd.to_datetime(acc["month"]).values
# Calculate months_back relative to application_datetime
months_back = (app_dt[acc["idx"].values] - acc_month) / np.timedelta64(30, "D")
acc["months_back"] = months_back

acc["non_sal_share"] = (acc["inflows"] - acc["income_like_credits"]) / (acc["inflows"] + 1.0)
acc["log_inf_inc_gap"] = np.log1p(np.maximum(acc["inflows"], 0)) - np.log1p(np.maximum(acc["income_like_credits"], 0))
acc["log_inf_out_gap"] = np.log1p(np.maximum(acc["inflows"], 0)) - np.log1p(np.maximum(acc["outflows"], 0))
acc["net_inflow_ratio"] = (acc["inflows"] - acc["outflows"]) / (acc["inflows"] + 1.0)
acc["close_open_diff_ratio"] = (acc["closing_balance"] - acc["opening_balance"]) / (np.abs(acc["opening_balance"]) + 1000.0)
acc["min_avg_bal_ratio"] = acc["minimum_balance"] / (np.abs(acc["average_balance"]) + 100.0)
acc["min_bal_neg"] = (acc["minimum_balance"] < 0).astype(np.float64)
acc["signed_log_min_bal"] = np.sign(acc["minimum_balance"]) * np.log1p(np.abs(acc["minimum_balance"]))
acc["signed_log_close_bal"] = np.sign(acc["closing_balance"]) * np.log1p(np.abs(acc["closing_balance"]))
acc["log_inc"] = np.log1p(np.maximum(acc["income_like_credits"], 0))
acc["log_inf"] = np.log1p(np.maximum(acc["inflows"], 0))
acc["log_out"] = np.log1p(np.maximum(acc["outflows"], 0))
acc["zero_low_bal"] = (acc["low_balance_days"] == 0).astype(np.float64)
acc["high_low_bal"] = (acc["low_balance_days"] >= 10).astype(np.float64)
acc["low_bal_frac"] = acc["low_balance_days"] / 30.0

g_acc = acc.groupby("idx")
put_group(g_acc["non_sal_share"].mean(), "raw_acc_non_sal_share_mean")
put_group(g_acc["non_sal_share"].std(), "raw_acc_non_sal_share_std")
put_group(g_acc["log_inf_inc_gap"].mean(), "raw_acc_log_inf_inc_gap_mean")
put_group(g_acc["log_inf_out_gap"].mean(), "raw_acc_log_inf_out_gap_mean")
put_group(g_acc["net_inflow_ratio"].min(), "raw_acc_net_inflow_ratio_min")
put_group(g_acc["net_inflow_ratio"].mean(), "raw_acc_net_inflow_ratio_mean")
put_group(g_acc["close_open_diff_ratio"].mean(), "raw_acc_close_open_diff_ratio_mean")
put_group(g_acc["low_balance_days"].mean(), "raw_acc_low_bal_days_mean")
put_group(g_acc["low_balance_days"].min(), "raw_acc_low_bal_days_min")
put_group(g_acc["low_balance_days"].max(), "raw_acc_low_bal_days_max")
put_group(g_acc["low_balance_days"].std(), "raw_acc_low_bal_days_std")
put_group(g_acc["zero_low_bal"].mean(), "raw_acc_zero_low_bal_share")
put_group(g_acc["high_low_bal"].mean(), "raw_acc_high_low_bal_share")
put_group(g_acc["min_avg_bal_ratio"].mean(), "raw_acc_min_avg_bal_ratio_mean")
put_group(g_acc["min_bal_neg"].mean(), "raw_acc_min_bal_neg_share")
put_group(g_acc["signed_log_min_bal"].std(), "raw_acc_signed_log_min_bal_std")
put_group(g_acc["signed_log_min_bal"].max(), "raw_acc_signed_log_min_bal_max")
put_group(g_acc["signed_log_min_bal"].min(), "raw_acc_signed_log_min_bal_min")
put_group(g_acc["signed_log_close_bal"].std(), "raw_acc_signed_log_close_bal_std")
put_group(g_acc["log_inc"].std(), "raw_acc_log_inc_std")
put_group(g_acc["log_inf"].std(), "raw_acc_log_inf_std")
put_group(g_acc["log_out"].std(), "raw_acc_log_out_std")

acc3 = acc[acc["months_back"] <= 3.5]
g_acc3 = acc3.groupby("idx")
put_group(g_acc3["non_sal_share"].mean(), "raw_acc3_non_sal_share_mean")
put_group(g_acc3["low_balance_days"].mean(), "raw_acc3_low_bal_days_mean")
put_group(g_acc3["low_balance_days"].min(), "raw_acc3_low_bal_days_min")
feats["raw_acc_recent_vs_all_low_bal"] = feats["raw_acc3_low_bal_days_mean"] - feats["raw_acc_low_bal_days_mean"]

print(f"Account done in {time.time()-t0:.1f}s")

# 2. Bureau Observations (strictly available_datetime <= decision_datetime)
bur = pd.read_csv(f"{DATA_DIR}/bureau_observations.csv")
bur = bur[bur["customer_id"].isin(cust_to_idx.index)].copy()
bur["idx"] = cust_to_idx.loc[bur["customer_id"]].values
avail_dt = pd.to_datetime(bur["available_datetime"]).values
rep_dt = pd.to_datetime(bur["report_datetime"]).values
# Point-in-time filter: available_datetime <= decision_datetime
bur_valid = avail_dt <= dec_dt[bur["idx"].values]
bur = bur[bur_valid].copy()
bur["age_days"] = (dec_dt[bur["idx"].values] - pd.to_datetime(bur["report_datetime"]).values) / np.timedelta64(1, "D")
bur["is_curr_app"] = (bur["application_id"].values == apps["application_id"].values[bur["idx"].values]).astype(np.float64)
bur["has_rev_lim"] = (bur["revolving_credit_limit"] > 0).astype(np.float64)
bur["log_rev_lim"] = np.log1p(np.maximum(bur["revolving_credit_limit"], 0))
bur["log_rev_bal"] = np.log1p(np.maximum(bur["revolving_outstanding_balance"], 0))
bur["log_out_bal"] = np.log1p(np.maximum(bur["outstanding_balance"], 0))
bur["rev_share_debt"] = bur["revolving_outstanding_balance"] / (bur["outstanding_balance"] + 1.0)
bur["inq_per_acc"] = bur["inquiries_last_6m"] / (bur["active_accounts"] + 1.0)
bur["delinq_per_acc"] = bur["delinquent_accounts_last_12m"] / (bur["active_accounts"] + 1.0)
bur["log_out_per_acc"] = np.log1p(bur["outstanding_balance"] / (bur["active_accounts"] + 1.0))

g_bur = bur.groupby("idx")
put_group(g_bur.size(), "raw_bur_n_obs", fill=0.0)
put_group(g_bur["active_accounts"].mean(), "raw_bur_active_acc_mean")
put_group(g_bur["active_accounts"].max(), "raw_bur_active_acc_max")
put_group(g_bur["active_accounts"].min(), "raw_bur_active_acc_min")
put_group(g_bur["active_accounts"].std(), "raw_bur_active_acc_std", fill=0.0)
put_group(g_bur["has_rev_lim"].mean(), "raw_bur_has_rev_lim_share")
put_group(g_bur["log_rev_lim"].mean(), "raw_bur_log_rev_lim_mean")
put_group(g_bur["log_rev_lim"].max(), "raw_bur_log_rev_lim_max")
put_group(g_bur["log_rev_bal"].mean(), "raw_bur_log_rev_bal_mean")
put_group(g_bur["log_rev_bal"].max(), "raw_bur_log_rev_bal_max")
put_group(g_bur["log_out_bal"].mean(), "raw_bur_log_out_bal_mean")
put_group(g_bur["log_out_bal"].max(), "raw_bur_log_out_bal_max")
put_group(g_bur["log_out_bal"].std(), "raw_bur_log_out_bal_std", fill=0.0)
put_group(g_bur["outstanding_balance"].std(), "raw_bur_out_bal_std", fill=0.0)
put_group(g_bur["rev_share_debt"].mean(), "raw_bur_rev_share_debt_mean")
put_group(g_bur["inq_per_acc"].mean(), "raw_bur_inq_per_acc_mean")
put_group(g_bur["delinq_per_acc"].mean(), "raw_bur_delinq_per_acc_mean")
put_group(g_bur["utilization"].min(), "raw_bur_util_min")
put_group(g_bur["utilization"].mean(), "raw_bur_util_mean")
put_group(g_bur["utilization"].std(), "raw_bur_util_std", fill=0.0)
put_group(g_bur["log_out_per_acc"].mean(), "raw_bur_log_out_per_acc_mean")

bur_curr = bur[bur["is_curr_app"] > 0]
g_bur_c = bur_curr.groupby("idx")
put_group(g_bur_c["active_accounts"].mean(), "raw_bur_curr_active_acc_mean")
put_group(g_bur_c["log_out_bal"].max(), "raw_bur_curr_log_out_bal_max")
put_group(g_bur_c["utilization"].min(), "raw_bur_curr_util_min")

bur_hist = bur[bur["is_curr_app"] == 0]
g_bur_h = bur_hist.groupby("idx")
put_group(g_bur_h["active_accounts"].std(), "raw_bur_hist_active_acc_std")
put_group(g_bur_h["log_out_bal"].std(), "raw_bur_hist_log_out_bal_std")
put_group(g_bur_h["score"].mean(), "raw_bur_hist_score_mean")

print(f"Bureau done in {time.time()-t0:.1f}s")

# 3. E-commerce Transactions
ec = pd.read_csv(f"{DATA_DIR}/ecommerce_transactions.csv")
ec = ec[ec["customer_id"].isin(cust_to_idx.index)].copy()
ec["idx"] = cust_to_idx.loc[ec["customer_id"]].values
ec_dt = pd.to_datetime(ec["transaction_datetime"]).values
ec_valid = ec_dt <= app_dt[ec["idx"].values]
ec = ec[ec_valid].copy()
ec["order_age_days"] = (app_dt[ec["idx"].values] - pd.to_datetime(ec["transaction_datetime"]).values) / np.timedelta64(1, "D")
ec["unit_price"] = ec["item_total"] / np.maximum(ec["quantity"], 1.0)
ec["log_up"] = np.log1p(np.maximum(ec["unit_price"], 0))
ec["log_item_tot"] = np.log1p(np.maximum(ec["item_total"], 0))

for pm in ["COD", "e_wallet", "debit_card", "credit_card", "bank_transfer"]:
    ec[f"pm_{pm}"] = (ec["payment_method"] == pm).astype(np.float64)
    ec[f"pm_amt_{pm}"] = ec["item_total"] * ec[f"pm_{pm}"]
ec["st_cancelled"] = (ec["order_status"] == "cancelled").astype(np.float64)
for cat in ["groceries", "personal_care", "fashion", "electronics", "education", "health", "home"]:
    ec[f"cat_{cat}"] = (ec["product_category"] == cat).astype(np.float64)

g_ec = ec.groupby("idx")
tot_amt = g_ec["item_total"].sum()
for pm in ["COD", "e_wallet", "debit_card", "credit_card", "bank_transfer"]:
    put_group(g_ec[f"pm_{pm}"].mean(), f"raw_ec_pm_{pm}_share")
    put_group(g_ec[f"pm_amt_{pm}"].sum() / (tot_amt + 1.0), f"raw_ec_pm_{pm}_spend_share")
put_group(g_ec["st_cancelled"].mean(), "raw_ec_st_cancelled_share")
for cat in ["groceries", "personal_care", "fashion", "electronics", "education", "health", "home"]:
    put_group(g_ec[f"cat_{cat}"].mean(), f"raw_ec_cat_{cat}_share")

put_group(g_ec["log_up"].mean(), "raw_ec_log_up_mean")
put_group(g_ec["log_up"].std(), "raw_ec_log_up_std")
put_group(g_ec["log_up"].max(), "raw_ec_log_up_max")
put_group(g_ec["log_up"].min(), "raw_ec_log_up_min")
put_group(g_ec["log_item_tot"].std(), "raw_ec_log_amt_std")
put_group(g_ec["log_item_tot"].max(), "raw_ec_log_amt_max")
put_group(g_ec["quantity"].mean(), "raw_ec_item_count_mean")
put_group(g_ec["quantity"].max(), "raw_ec_item_count_max")

# Between-category log price std
cat_means = ec.groupby(["idx", "product_category"])["log_up"].mean().unstack()
between_cat_std = cat_means.std(axis=1).fillna(0.0)
put_group(between_cat_std, "raw_ec_between_cat_log_up_std", fill=0.0)

# Groceries last age & 90d count
ec_groc = ec[ec["product_category"] == "groceries"]
put_group(ec_groc.groupby("idx")["order_age_days"].min(), "raw_ec_groceries_last_age", fill=999.0)
ec90 = ec[ec["order_age_days"] <= 90]
g_ec90 = ec90.groupby("idx")
tot_amt90 = g_ec90["item_total"].sum()
put_group(g_ec90["cat_groceries"].sum(), "raw_ec90_groceries_count", fill=0.0)
put_group(g_ec90["pm_amt_e_wallet"].sum() / (tot_amt90 + 1.0), "raw_ec90_ewallet_spend_share", fill=0.0)
put_group(g_ec90["pm_amt_COD"].sum() / (tot_amt90 + 1.0), "raw_ec90_cod_spend_share", fill=0.0)
ec180 = ec[ec["order_age_days"] <= 180]
put_group(ec180.groupby("idx")["st_cancelled"].sum(), "raw_ec180_cancelled_count", fill=0.0)

print(f"Ecommerce done in {time.time()-t0:.1f}s")

# 4. Previous Applications
prv = pd.read_csv(f"{DATA_DIR}/previous_applications.csv")
prv = prv[prv["customer_id"].isin(cust_to_idx.index)].copy()
prv["idx"] = cust_to_idx.loc[prv["customer_id"]].values
prv_dec_dt = pd.to_datetime(prv["decision_datetime"]).values
prv = prv[prv_dec_dt <= app_dt[prv["idx"].values]].copy()
prv["application_age_days"] = (app_dt[prv["idx"].values] - pd.to_datetime(prv["application_datetime"]).values) / np.timedelta64(1, "D")
prv["is_rej"] = (prv["decision"] == "rejected").astype(np.float64)
prv["is_app"] = (prv["decision"] == "approved").astype(np.float64)
prv["is_can"] = (prv["decision"] == "cancelled").astype(np.float64)
prv["app_ratio"] = np.where(prv["is_app"] > 0, prv["approved_amount"] / (prv["requested_amount"] + 1.0), np.nan)
prv["log_req"] = np.log1p(np.maximum(prv["requested_amount"], 0))

g_prv = prv.groupby("idx")
put_group(g_prv.size(), "raw_prv_n_total", fill=0.0)
put_group(g_prv["is_rej"].sum(), "raw_prv_n_rej", fill=0.0)
put_group(g_prv["is_app"].sum(), "raw_prv_n_app", fill=0.0)
put_group(g_prv["is_can"].sum(), "raw_prv_n_can", fill=0.0)
put_group(g_prv["is_rej"].mean(), "raw_prv_rej_share")
put_group(g_prv["is_app"].mean(), "raw_prv_app_share")
put_group(g_prv["annual_interest_rate"].mean(), "raw_prv_rate_mean")
put_group(g_prv["annual_interest_rate"].max(), "raw_prv_rate_max")
put_group(g_prv["annual_interest_rate"].min(), "raw_prv_rate_min")
put_group(g_prv["annual_interest_rate"].std(), "raw_prv_rate_std")
put_group(g_prv["app_ratio"].mean(), "raw_prv_app_ratio_mean")
put_group(g_prv["log_req"].mean(), "raw_prv_log_req_mean")
put_group(g_prv["term_months"].mean(), "raw_prv_tenor_mean")
put_group(g_prv["application_age_days"].min(), "raw_prv_age_min")

prv_sorted = prv.sort_values(["idx", "application_age_days"])
prv_last = prv_sorted.drop_duplicates("idx", keep="first").set_index("idx")
put_group(prv_last["is_rej"], "raw_prv_last_is_rej")
put_group(prv_last["annual_interest_rate"], "raw_prv_last_rate")

feats["raw_prv_rate_diff_curr"] = rate - feats["raw_prv_rate_mean"]
feats["raw_prv_req_diff_curr"] = np.log1p(loan) - feats["raw_prv_log_req_mean"]

print(f"Previous apps done in {time.time()-t0:.1f}s")

# 5. Repayment History (strictly due_date <= application_datetime)
rep = pd.read_csv(f"{DATA_DIR}/repayment_history.csv")
rep = rep[rep["customer_id"].isin(cust_to_idx.index)].copy()
rep["idx"] = cust_to_idx.loc[rep["customer_id"]].values
due_dt = pd.to_datetime(rep["due_date"]).values
rep = rep[due_dt <= app_dt[rep["idx"].values]].copy()
rep["due_age_days"] = (app_dt[rep["idx"].values] - pd.to_datetime(rep["due_date"]).values) / np.timedelta64(1, "D")
pay_dt = pd.to_datetime(rep["last_payment_datetime"]).values
due_dt_valid = pd.to_datetime(rep["due_date"]).values
delay_hrs = (pay_dt - due_dt_valid) / np.timedelta64(1, "h")
rep["payment_delay_hours"] = np.where(pd.isna(rep["last_payment_datetime"]), rep["days_past_due"] * 24.0, delay_hrs)
rep["is_late1"] = (rep["days_past_due"] >= 1).astype(np.float64)
rep["is_late7"] = (rep["days_past_due"] >= 7).astype(np.float64)
rep["is_late30"] = (rep["days_past_due"] >= 30).astype(np.float64)
rep["pos_dpd"] = np.where(rep["days_past_due"] > 0, rep["days_past_due"], np.nan)
rep["dpd_cure_gap"] = rep["max_days_past_due"] - rep["days_past_due"]
rep["pos_delay_hrs"] = (rep["payment_delay_hours"] > 0).astype(np.float64)
rep["early_pay_12h"] = (rep["payment_delay_hours"] < -12.0).astype(np.float64)
rep["underpaid"] = (rep["amount_paid"] < rep["amount_due"] * 0.99).astype(np.float64)

g_rep = rep.groupby("idx")
put_group(g_rep["pos_dpd"].std(), "raw_rep_pos_dpd_std", fill=0.0)
put_group(g_rep["pos_dpd"].mean(), "raw_rep_pos_dpd_mean", fill=0.0)
put_group(g_rep["pos_dpd"].count(), "raw_rep_pos_dpd_count", fill=0.0)
put_group(g_rep["days_past_due"].max(), "raw_rep_dpd_max")
put_group(g_rep["days_past_due"].std(), "raw_rep_dpd_std", fill=0.0)
put_group(g_rep["max_days_past_due"].max(), "raw_rep_max_hist_dpd_max")
put_group(g_rep["max_days_past_due"].mean(), "raw_rep_max_hist_dpd_mean")
put_group(g_rep["dpd_cure_gap"].max(), "raw_rep_dpd_cure_gap_max")
put_group(g_rep["dpd_cure_gap"].mean(), "raw_rep_dpd_cure_gap_mean")
put_group(g_rep["payment_delay_hours"].mean(), "raw_rep_delay_hrs_mean")
put_group(g_rep["payment_delay_hours"].max(), "raw_rep_delay_hrs_max")
put_group(g_rep["payment_delay_hours"].std(), "raw_rep_delay_hrs_std", fill=0.0)
put_group(g_rep["pos_delay_hrs"].mean(), "raw_rep_pos_delay_hrs_share")
put_group(g_rep["early_pay_12h"].mean(), "raw_rep_early_pay_12h_share")
put_group(g_rep["underpaid"].mean(), "raw_rep_underpaid_share")

rep_early = rep[rep["due_age_days"] > 180].groupby("idx")
rep_recent = rep[rep["due_age_days"] <= 180].groupby("idx")
put_group(rep_early["is_late1"].mean(), "raw_rep_older180_late1_mean")
put_group(rep_recent["is_late1"].mean(), "raw_rep_recent180_late1_mean")
feats["raw_rep_late_trajectory_diff"] = feats["raw_rep_recent180_late1_mean"] - feats["raw_rep_older180_late1_mean"]

rep90 = rep[rep["due_age_days"] <= 90].groupby("idx")
put_group(rep90["days_past_due"].max(), "raw_rep90_dpd_max")
put_group(rep_recent["days_past_due"].max(), "raw_rep180_dpd_max")

# 6. Structural Population Indicators
has_prev = (feats["raw_prv_n_total"] > 0).astype(np.float64)
has_acc = (~np.isnan(feats["raw_acc_low_bal_days_mean"])).astype(np.float64)
rep_cnt = np.zeros(N, dtype=np.float64)
s_rep_cnt = g_rep.size()
rep_cnt[s_rep_cnt.index.values] = s_rep_cnt.values
has_rep = (rep_cnt > 0).astype(np.float64)

feats["pop_A"] = ((has_prev == 0) & (has_acc == 0) & (has_rep == 0)).astype(np.float64)
feats["pop_B"] = ((has_prev == 1) & (has_acc == 1) & (has_rep == 0)).astype(np.float64)
feats["pop_C1"] = ((has_prev == 1) & (has_acc == 1) & (has_rep == 1) & (rep_cnt <= 11)).astype(np.float64)
feats["pop_C2"] = ((has_prev == 1) & (has_acc == 1) & (has_rep == 1) & (rep_cnt >= 12)).astype(np.float64)

names = sorted(feats.keys())
mat = np.column_stack([feats[k] for k in names]).astype(np.float32)
out_path = "/home/parasite/.gemini/antigravity-cli/brain/73181457-58b8-405a-bb23-a8afbd02f051/scratch/raw_domain_features.npz"
np.savez_compressed(out_path, names=np.array(names), data=mat, y=apps["target"].values.astype(np.int8))
print(f"Saved {len(names)} raw domain features {mat.shape} to {out_path} in {time.time()-t0:.1f}s")
print("Pop counts:", {p: int(feats[p].sum()) for p in ["pop_A", "pop_B", "pop_C1", "pop_C2"]})
