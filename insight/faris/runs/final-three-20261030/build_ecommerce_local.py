from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
R=Path(__file__).resolve().parent;S=Path.home()/'faris/kumo-large-node380-20261016/data/kumo_v3';f=pd.read_parquet(S/'features.parquet')
assert f.application_id.nunique()==len(f)==100000
positive=lambda c:pd.to_numeric(f[c],errors='coerce').clip(lower=0)
sp30=positive('v2_ecom_30d_spend');sp90=positive('ecom_spend_90d');n30=positive('v2_ecom_30d_orders');n90=positive('v2_ecom_90d_orders');income=positive('declared_monthly_income')
features=pd.DataFrame({
 'ec_spend_acceleration':np.log1p(sp30)-np.log1p((sp90-sp30).clip(lower=0)/2),
 'ec_order_acceleration':np.log1p(n30)-np.log1p((n90-n30).clip(lower=0)/2),
 'ec_budget_pressure':np.log1p(sp30/(income+1)),
 'ec_cancel_activity':positive('ecom_cancel_rate')*np.log1p(positive('ecom_order_count')),
 'ec_essential_discretionary_mix':positive('ecom_category_share_groceries')+positive('ecom_category_share_health')-positive('ecom_category_share_fashion')-positive('ecom_category_share_electronics'),
 'ec_COD_reliance':positive('ecom_payment_share_COD')*(1-positive('ecom_payment_share_credit_card').clip(upper=1)),
 'ec_spend_burstiness':np.log1p(positive('v3_ecom_monthly_spend_cv'))*positive('v3_ecom_monthly_spend_max_share'),
 'ec_recency_frequency':np.log1p(n90)*np.exp(-positive('ecom_last_purchase_recency_days')/30),
})
features=features.replace([np.inf,-np.inf],np.nan);assert all(features[c].nunique(dropna=True)>2 for c in features)
np.savez_compressed(R/'ECOMMERCE_FEATURES.npz',application_id=f.application_id.astype(str).to_numpy(dtype=str),values=features.to_numpy(float),columns=np.asarray(features.columns,dtype=str))
plan=dict(epoch=__import__('time').time(),name='ecommerce_features_seed2026',new_fields=list(features),placement='Append eight deterministic label-free ecommerce interactions to LocalGLM actuarial99 only; fixed seed2026; existing matching CANN/Kumo seed2026 unchanged.',comparison='Exact original seed2026 fixed pipeline; also report existing two-seed reference descriptively.',source_feature_SHA256=hashlib.sha256((S/'features.parquet').read_bytes()).hexdigest(),source_manifest_SHA256=hashlib.sha256((S/'manifest.json').read_bytes()).hexdigest(),source_definition='Existing named v3 transaction aggregates; no new global statistics or targets. All row matching uses application IDs.',prepared_before_new_outer_scores=True,features_npz_sha256=hashlib.sha256((R/'ECOMMERCE_FEATURES.npz').read_bytes()).hexdigest())
(R/'ECOMMERCE_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(dict(fields=list(features),bytes=(R/'ECOMMERCE_FEATURES.npz').stat().st_size)),flush=True)
