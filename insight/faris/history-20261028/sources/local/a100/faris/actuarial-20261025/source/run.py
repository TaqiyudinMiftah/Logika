"""Actuarial feature hypotheses and fixed-horizon binomial corrections."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '2'
from pathlib import Path
import datetime, json, sys, time
import numpy as np
import pandas as pd
from scipy.special import betaln, expit, logit
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import statsmodels.api as sm

R = Path(__file__).resolve().parents[1]
A = Path('/mnt/home/faris/logika-faris-exploration')
M = Path('/mnt/home/faris/logika-model-arena-20261025')
sys.path.insert(0, str(A/'transfer-20261025/source'))
sys.path.insert(0, str(M/'source'))
import baseline as b
import feature_head
import inputs

FAMILIES = {
    'loan_value': ['act_interest_income', 'act_balance6_income', 'act_interest_buffer', 'act_rate_shock_income'],
    'cash_stress': ['act_liquid_months', 'act_shortfall20', 'act_shortfall40', 'act_3month_uncovered20'],
    'credibility': [f'act_eb_{g}_{v}' for g in ['region', 'employment_type', 'loan_purpose', 'region_employment'] for v in ['logodds', 'sd']],
}

def financial(q):
    income = q.re_income_geometric_mean.to_numpy(float)
    income = np.where(income > 0, income, np.nan)
    principal = q.loan_amount.to_numpy(float)
    rate = q.annual_interest_rate.to_numpy(float)/12
    payment = q.proposed_monthly_installment.to_numpy(float)
    cash = np.maximum(q.v2_account_3m_average_balance_mean.to_numpy(float), 0)
    burden = q.v2_existing_total_burden_observed.to_numpy(float)
    # Assumptions are explicit stress scenarios, not forecasts of interest or income.
    shock_rate = rate + .04/12
    shock_payment = principal*shock_rate/(-np.expm1(-12*np.log1p(shock_rate)))
    balance6 = principal*(1+rate)**6-payment*np.expm1(6*np.log1p(rate))/rate
    total_interest = 12*payment-principal
    short20, short40 = np.maximum(burden-.8, 0), np.maximum(burden-.6, 0)
    return pd.DataFrame({
        'act_interest_income': total_interest/income,
        'act_balance6_income': balance6/income,
        'act_interest_buffer': total_interest/(cash+income),
        'act_rate_shock_income': (shock_payment-payment)/income,
        'act_liquid_months': cash/(np.maximum(burden, .01)*income),
        'act_shortfall20': short20,
        'act_shortfall40': short40,
        'act_3month_uncovered20': np.maximum(3*short20-cash/income, 0),
    }, index=q.index).replace([np.inf, -np.inf], np.nan)

def credibility(context, query):
    mu = float(context.target.mean())
    out, recipes = pd.DataFrame(index=query.index), {}
    for group in ['region', 'employment_type', 'loan_purpose', 'region_employment']:
        names = ['region', 'employment_type'] if group == 'region_employment' else [group]
        # Tuple keys avoid delimiter collisions between category values.
        ck = context[names].fillna('MISSING').astype(str).apply(tuple, axis=1)
        qk = query[names].fillna('MISSING').astype(str).apply(tuple, axis=1)
        grouped = pd.DataFrame({'key': ck, 'y': context.target.to_numpy()}).groupby('key').y.agg(['sum', 'count'])
        k, n = grouped['sum'].to_numpy(), grouped['count'].to_numpy()
        def objective(log_s):
            s = np.exp(log_s)
            return -float(np.sum(betaln(k+mu*s, n-k+(1-mu)*s)-betaln(mu*s, (1-mu)*s)))
        fit = minimize_scalar(objective, bounds=(np.log(.01), np.log(1e6)), method='bounded')
        assert fit.success
        s = float(np.exp(fit.x))
        counts = qk.map(grouped['count']).fillna(0).to_numpy()
        successes = qk.map(grouped['sum']).fillna(0).to_numpy()
        posterior = (successes+mu*s)/(counts+s)
        out[f'act_eb_{group}_logodds'] = logit(np.clip(posterior, 1e-6, 1-1e-6))-logit(mu)
        out[f'act_eb_{group}_sd'] = np.sqrt(posterior*(1-posterior)/(counts+s+1))
        recipes[group] = {'group_columns': names, 'concentration': s, 'context_prevalence': mu,
                          'fit_rows': len(context), 'groups': len(grouped), 'fit_uses_query_labels': False}
    return out, recipes

def main():
    assert R.is_relative_to(Path('/mnt/home/faris'))
    started = time.time()
    dest = R/'results';dest.mkdir(exist_ok=False)
    reference = np.load(A/'transfer-20261025/ROBUSTNESS_OOF.npz')
    y, folds = reference['target'], reference['fold']
    predictions, records = {}, []
    def add(name, rows, p):
        if name not in predictions: predictions[name] = np.full(len(y), np.nan)
        assert np.isfinite(p).all()
        predictions[name][rows] = p
    for fold in range(5):
        assert time.time()-started < 5400
        original, ix, q, context, yc, cols, query, head_cols, spline, raw, expected = inputs.inputs(fold)
        f = financial(query)
        cred, crecipe = credibility(context, query)
        qf = pd.concat([query, f, cred], axis=1)
        fresh = np.load(A/f'transfer-20261025/robustness/fold{fold}/raw_seed_repeat.npz')
        assert np.array_equal(fresh['rows'], np.r_[ix['correction'], ix['validation']])
        for seed, base in [('original', raw), ('repeat', fresh['prediction'])]:
            for family, extra in [('baseline', []), *FAMILIES.items()]:
                head = feature_head.fit(qf.iloc[:6000].assign(target=yc), base[:6000], head_cols+extra, .003, .5)
                pred = .75*b.corrected_probability(head, qf.iloc[6000:], base[6000:])[0]+.25*spline
                if family == 'baseline':
                    compare = expected if seed == 'original' else reference['maturity__fresh_kumo_seed'][ix['validation']]
                    assert np.max(abs(pred-compare)) < 2e-12
                add(f'{family}__{seed}', ix['validation'], pred)
                records.append({'fold': fold, 'family': family, 'seed': seed, 'features': head_cols+extra,
                                'gradient_max_abs': head['gradient_max_abs']})
        # Alternative link functions fit the same 6k correction labels only.
        # These are fixed-horizon binary models, not fitted survival-time models.
        cats = query[head_cols].select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        nums = [c for c in head_cols if c not in cats]
        encoder = ColumnTransformer([
            ('n', make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()), nums),
            ('c', make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore', sparse_output=False)), cats)], sparse_threshold=0)
        prepared = b.prepare(query, head_cols, cats)
        encoder.fit(prepared.iloc[:6000])
        X = np.column_stack([np.ones(len(query)), encoder.transform(prepared)])
        for link_name, link in [('logit', sm.families.links.Logit()), ('probit', sm.families.links.Probit()), ('cloglog', sm.families.links.CLogLog())]:
            for seed, base in [('original', raw), ('repeat', fresh['prediction'])]:
                offset = link(np.clip(base, 1e-6, 1-1e-6))
                model = sm.GLM(yc, X[:6000], family=sm.families.Binomial(link=link), offset=offset[:6000])
                fitted = model.fit_regularized(alpha=.003, L1_wt=0, maxiter=300)
                pred = .75*fitted.predict(X[6000:], offset=offset[6000:])+.25*spline
                add(f'binomial_{link_name}__{seed}', ix['validation'], pred)
        b.dump({'fold': fold, 'credibility_fit_summary': crecipe, 'query_ids_sha256': b.ids_hash(q, np.arange(len(q))),
                'source_sha256': b.sha(__file__), 'packet_manifest_sha256': b.sha(inputs.PACKETS/f'fold{fold}/manifest.json'),
                'fitted_weights_saved': False}, dest/f'fold{fold}_recipe.json')
        print(json.dumps({'fold': fold, 'status': 'complete', 'seconds': time.time()-started}), flush=True)
    families = ['baseline', *FAMILIES, 'binomial_logit', 'binomial_probit', 'binomial_cloglog']
    metrics = {}
    for family in families:
        predictions[f'{family}__mean'] = .5*(predictions[f'{family}__original']+predictions[f'{family}__repeat'])
    for name, pred in predictions.items():
        assert np.isfinite(pred).all()
        seed = name.rsplit('__', 1)[1]
        base = predictions[f'baseline__{seed}']
        metrics[name] = {**b.scores(y, pred), 'delta_KS': b.scores(y, pred)['KS']-b.scores(y, base)['KS'],
                         'fold_deltas': [b.scores(y[folds==f], pred[folds==f])['KS']-b.scores(y[folds==f], base[folds==f])['KS'] for f in range(5)]}
    np.savez_compressed(dest/'OOF.npz', target=y, fold=folds, **predictions)
    b.dump({'status': 'complete', 'seconds': time.time()-started, 'metrics': metrics, 'recipes': records,
            'fitted_weights_saved': False, 'independent_confirmation': False,
            'source_sha256': b.sha(__file__), 'plan_sha256': b.sha(R/'PLAN.json')}, dest/'RESULT.json')
    print(json.dumps({k:v for k,v in metrics.items() if k.endswith('__mean')}, indent=2), flush=True)

if __name__ == '__main__':
    main()
