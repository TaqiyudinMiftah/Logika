"""Source-backed actuarial representations from existing, cutoff-safe inputs.

No current target labels enter this module. Context rows determine the historical
event-rate priors and denominator scale. These are proposed credit-risk features,
not claims that an insurance model transfers unchanged to lending. All fitted
quantities can be regenerated from the recorded recipe; no weights are saved.
"""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from scipy.special import betainc, betaincinv

BOOK = 'https://doi.org/10.1007/978-3-031-12409-9'
SOURCES = {
    'bernoulli_link': {
        'url': BOOK,
        'location': 'Chapter2(single-parameter Bernoulli exponential family);Chapter5(GLMs)',
        'use': 'Canonical log-odds and nonlinear feature maps are representations, not additional raw information.',
    },
    'credibility': {
        'url': BOOK,
        'location': 'Section5.2(mixed models and credibility);Chapter6(Bayesian estimation and regularization)',
        'use': 'Exposure-aware shrinkage motivates beta-binomial historical-rate features. Beta conjugacy formulas below are an explicit Bernoulli adaptation.',
    },
    'risk_functionals': {
        'url': BOOK,
        'location': 'Chapter3(decision theory);Chapter4(prediction);Section5.8(regression of risk functionals)',
        'credit_mechanism_url': 'https://www.bis.org/publications/qr-201509/how-much-income-used-debt-payments-new-database-debt-service-ratios',
        'use': 'Liquidity stress features are deterministic borrower cash-flow scenarios, not calibrated ruin or survival probabilities.',
    },
    'regularization': {
        'url': BOOK,
        'location': 'Section6.2(regularization, ridge,LASSO,elastic net)',
        'use': 'Motivates context-only selection/reduction and matched comparisons against unreduced inputs.',
    },
}

HISTORIES = (
    ('all_t1', 'm_repayment_all_t1_m1_k', 'm_repayment_all_t1_m1_n'),
    ('365_t1', 'm_repayment_365_t1_m1_k', 'm_repayment_365_t1_m1_n'),
    ('180_t1', 'm_repayment_180_t1_m1_k', 'm_repayment_180_t1_m1_n'),
    ('365_t30', 'm_repayment_365_t30_m1_k', 'm_repayment_365_t30_m1_n'),
)
I = 'v2_account_3m_income_like_credits_mean'
ISD = 'v2_account_3m_income_like_credits_std'
E = 'v2_account_3m_outflows_mean'
ESD = 'v2_account_3m_outflows_std'
C = 'v2_account_3m_average_balance_mean'
CMIN = 'v2_account_3m_minimum_balance_min'
P = 'proposed_monthly_installment'
Q = 'v2_existing_active_installment'
T = 'term_months'
U = 'employment_tenure_months'
L = 'loan_amount'
I12 = 'v2_account_12m_income_like_credits_mean'
E12 = 'v2_account_12m_outflows_mean'


def _div(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan),
                     where=np.isfinite(a) & np.isfinite(b) & (b > 0))


def _logit(value):
    value = np.clip(value, 1e-6, 1 - 1e-6)
    return np.log(value) - np.log1p(-value)


def definitions():
    """JSON-compatible exact feature definitions; no data-dependent proposals."""
    specs = []
    formulas = {
        'logodds': 'logit((k+24*mu_context)/(n+24))',
        'lower05': 'BetaQuantile(.05,k+24*mu_context,n-k+24*(1-mu_context))',
        'upper95': 'BetaQuantile(.95,k+24*mu_context,n-k+24*(1-mu_context))',
        'above_cohort': '1-BetaCDF(mu_context,k+24*mu_context,n-k+24*(1-mu_context))',
    }
    for history, k, n in HISTORIES:
        for quantity, formula in formulas.items():
            specs.append(dict(name=f'a6_{history}_{quantity}', op='beta_' + quantity,
                              inputs=[k, n], group='historical_credibility', source='credibility',
                              formula=formula, units='dimensionless',
                              assumption='Historical late-event counts use their recorded maturity-adjusted eligible opportunity counts; not current default labels.'))
    recent = [HISTORIES[2][1], HISTORIES[2][2], HISTORIES[0][1], HISTORIES[0][2]]
    specs += [
        dict(name='a6_recent_older_logodds', op='recent_logodds', inputs=recent,
             formula='logit(p_recent180)-logit(p_older);older=(all-180),common context-all prior strength24',
             group='historical_drift', source='credibility', units='dimensionless',
             assumption='180day and all-history counts must be nested with the same1day maturity definition.'),
        dict(name='a6_recent_older_trust_logodds', op='recent_trust_logodds', inputs=recent,
             formula='recent_older_logodds*(n180/(n180+24))*(nolder/(nolder+24))',
             group='historical_drift', source='credibility', units='dimensionless',
             assumption='Reliability attenuates a rate change supported by little history.'),
        dict(name='a6_recent_opportunity_share', op='recent_share', inputs=recent,
             formula='n180/nall', group='historical_drift', source='credibility', units='dimensionless',
             assumption='Opportunity concentration proxy; zero all-history exposures remains missing.'),
        dict(name='a6_severity_opportunity_coverage', op='severity_coverage',
             inputs=[HISTORIES[3][2], HISTORIES[1][2]], formula='n_365_t30_mature/n_365_t1_mature',
             group='historical_credibility', source='credibility', units='dimensionless',
             assumption='Coverage of sufficiently old repayments; not a severe-given-late probability.'),
    ]
    # Joint cash-flow mechanisms are deliberate, not an all-pairs expansion.
    cash = [I, E, P, Q]
    scenarios = [
        ('net_margin', 'net_margin', cash, '(income-outflow-proposed-active)/income', 'dimensionless'),
        ('runway_deficit_months', 'runway', cash + [C], 'max(cash,0)/max(outflow+proposed+active-income,0),defined only for positive deficit', 'months'),
        ('runway_term_share', 'runway_term', cash + [C, T], 'runway_deficit_months/term_months', 'dimensionless'),
        ('cash_floor_due_shortfall', 'cash_floor_shortfall', [P, Q, CMIN], 'max(proposed+active-minimum_cash,0)/(proposed+active)', 'dimensionless'),
        ('cash_floor_income_buffer', 'cash_floor_buffer', [CMIN, I], 'minimum_cash/income', 'months'),
        ('cash_drawdown_fraction', 'drawdown', [C, CMIN], '(mean_cash-minimum_cash)/max(abs(mean_cash),context_cash_floor)', 'dimensionless'),
        ('surplus_to_buffer', 'surplus_buffer', cash + [C], '(income-outflow-proposed-active)/max(abs(mean_cash),context_cash_floor)', 'per_month'),
        ('one_sd_income_service', 'stress_income', [I, ISD, P, Q], '(proposed+active)/(income-income_sd),defined only for positive stressed income', 'dimensionless'),
        ('one_sd_outflow_margin', 'stress_outflow', [I, E, ESD, P, Q], '(income-outflow-outflow_sd-proposed-active)/income', 'dimensionless'),
        ('cashflow_noise_margin', 'noise_margin', [I, ISD, E, ESD, P, Q], '(income-outflow-proposed-active)/sqrt(income_sd^2+outflow_sd^2)', 'dimensionless'),
        ('income_volatility_leverage', 'vol_leverage', [I, ISD, P, Q], '(income_sd/income)*((proposed+active)/income)', 'dimensionless'),
        ('overdraft_service_pressure', 'overdraft_service', [CMIN, I, P, Q], 'max(-minimum_cash,0)*(proposed+active)/income^2', 'dimensionless'),
        ('term_liquid_obligation_load', 'term_load', [I, P, Q, C, T], 'term*(proposed+active)/(max(cash,0)+term*income)', 'dimensionless'),
        ('tenure_term_coverage', 'tenure_term', [U, T], 'employment_tenure_months/term_months', 'dimensionless'),
        ('contractual_cost_ratio', 'contract_cost', [P, T, L], '(proposed_installment*term-principal)/principal', 'dimensionless'),
        ('principal_resource_share', 'principal_resources', [L, C, I, T], 'principal/(max(cash,0)+term*income)', 'dimensionless'),
        ('income_outflow_trend_gap', 'trend_gap', [I, I12, E, E12], 'log(income3m/income12m)-log(outflow3m/outflow12m)', 'dimensionless'),
        ('surplus_3m12m_shift', 'surplus_shift', [I, I12, E, E12], '((income3m-outflow3m)-(income12m-outflow12m))/income12m', 'dimensionless'),
    ]
    for name, op, inputs, formula, units in scenarios:
        specs.append(dict(name='a6_' + name, op=op, inputs=inputs, formula=formula,
                          group='cashflow_stress', source='risk_functionals', units=units,
                          assumption='Monthly amounts are aligned as recorded. Constant-flow term scenarios are proxies, not forecasts. One-SD scenarios do not assume or claim a Gaussian tail probability; noise-margin covariance is unavailable.'))
    specs.append(dict(name='a6_cashflow_joint_missing', op='missing_fraction',
                      inputs=[I, ISD, E, ESD, C, CMIN], formula='fraction of missing financial inputs',
                      group='evidence_quality', source='regularization', units='dimensionless',
                      assumption='Missingness is observable at inference and does not replace absent values with zero.'))
    return specs


def required_inputs():
    return sorted({name for spec in definitions() for name in spec['inputs']})


def expand(frame: pd.DataFrame, schema: dict, ctx: np.ndarray):
    """Return only new a6_* columns and auditable, context-only recipes.

    Resolve plaintext inputs through identity specs' original hashed IDs. Callers
    may also supply additional plaintext base columns or schema.extra_input_ids.
    Missing inputs cause an explicit skipped definition, not fabricated values.
    The caller must re-run this function with inner-training rows when an inner
    stopping experiment requires its own independently fitted prior quantities.
    """
    ctx = np.asarray(ctx, dtype=np.int64)
    assert ctx.ndim == 1 and len(ctx) > 0 and len(np.unique(ctx)) == len(ctx)
    assert ctx.min() >= 0 and ctx.max() < len(frame)
    mapping = {s['inputs'][0]: s['id'] for s in schema['specs']
               if s['op'] == 'identity' and len(s['inputs']) == 1}
    mapping.update(schema.get('extra_input_ids', {}))
    resolved = {}
    for name in required_inputs():
        choices = [mapping.get(name), name]
        column = next((c for c in choices if c is not None and c in frame.columns), None)
        if column is not None:
            resolved[name] = column
    cache = {name: pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=float, na_value=np.nan)
             for name, column in resolved.items()}
    for name in [ISD, ESD]:
        if name in cache:
            cache[name] = np.where(cache[name] >= 0, cache[name], np.nan)
    beta_cache, fitted = {}, {}

    def beta(kname, nname, prior=None):
        key = (kname, nname, prior)
        if key in beta_cache:
            return beta_cache[key]
        k, n = cache[kname], cache[nname]
        valid = np.isfinite(k) & np.isfinite(n) & (n >= 0) & (k >= 0) & (k <= n)
        training = ctx[valid[ctx]]
        denominator = np.sum(n[training])
        if denominator <= 0:
            raise ValueError('No positive historical context opportunities for ' + nname)
        mu = float(np.sum(k[training]) / denominator) if prior is None else float(prior)
        mu = float(np.clip(mu, 1e-6, 1 - 1e-6))
        alpha = np.where(valid, k + 24 * mu, np.nan)
        beta_value = np.where(valid, n - k + 24 * (1 - mu), np.nan)
        p = _div(alpha, alpha + beta_value)
        fitted[kname + '|' + nname] = dict(mu_context=mu, strength=24,
                                           context_valid_rows=len(training),
                                           context_opportunities=float(denominator),
                                           invalid_input_rows=int((~valid).sum()))
        beta_cache[key] = (p, alpha, beta_value, mu)
        return beta_cache[key]

    def cash_floor(values):
        a = np.abs(values[ctx]);a = a[np.isfinite(a) & (a > 0)]
        floor = max(float(np.median(a)) * .01, np.finfo(np.float32).tiny) if len(a) else 1.
        fitted['cash_denominator_floor'] = dict(value=floor, rule='.01*context_median_positive_absolute_cash; fallback1 when none')
        return floor

    built, kept, skipped = {}, [], []
    # Exact-context dedupe avoids counting aliases as discoveries. Category columns
    # cannot be algebraic duplicates of these numeric financial functionals.
    def fingerprint(values):
        values = np.asarray(values[ctx], dtype='<f4').copy()
        values[~np.isfinite(values)] = np.nan
        values[values == 0] = 0  # Normalize signed zero for alias detection.
        return hashlib.sha256(values.tobytes()).hexdigest()
    seen = {}
    for column in frame:
        if pd.api.types.is_numeric_dtype(frame[column]):
            seen.setdefault(fingerprint(frame[column].to_numpy(dtype=float, na_value=np.nan)), column)
    for spec in definitions():
        absent = [name for name in spec['inputs'] if name not in cache]
        if absent:
            skipped.append(dict(name=spec['name'], reason='missing_required_inputs', inputs=absent))
            continue
        op = spec['op'];x = [cache[n] for n in spec['inputs']]
        try:
            with np.errstate(all='ignore'):
                if op.startswith('beta_'):
                    p, alpha, bv, mu = beta(*spec['inputs'])
                    if op == 'beta_logodds':value = _logit(p)
                    elif op == 'beta_lower05':value = betaincinv(alpha, bv, .05)
                    elif op == 'beta_upper95':value = betaincinv(alpha, bv, .95)
                    else:value = 1 - betainc(alpha, bv, mu)
                elif op.startswith('recent_'):
                    kr, nr, ka, na = x
                    valid = np.isfinite(kr) & np.isfinite(nr) & np.isfinite(ka) & np.isfinite(na)
                    valid &= (kr >= 0) & (nr >= 0) & (kr <= nr) & (ka >= kr) & (na >= nr) & ((ka-kr) <= (na-nr))
                    _, _, _, mu = beta(spec['inputs'][2], spec['inputs'][3])
                    no, ko = na - nr, ka - kr
                    delta = _logit(_div(kr+24*mu, nr+24)) - _logit(_div(ko+24*mu, no+24))
                    if op == 'recent_share':value = _div(nr, na)
                    elif op == 'recent_trust_logodds':value = delta * _div(nr,nr+24) * _div(no,no+24)
                    else:value = delta
                    value = np.where(valid, value, np.nan)
                elif op == 'severity_coverage':
                    value = np.where((x[0] >= 0) & (x[0] <= x[1]), _div(x[0],x[1]), np.nan)
                elif op == 'net_margin':value = _div(x[0]-x[1]-x[2]-x[3], x[0])
                elif op in ['runway','runway_term']:
                    value = _div(np.maximum(x[4],0), x[1]+x[2]+x[3]-x[0])
                    if op == 'runway_term':value = _div(value,x[5])
                elif op == 'cash_floor_shortfall':value = _div(np.maximum(x[0]+x[1]-x[2],0), x[0]+x[1])
                elif op == 'cash_floor_buffer':value = _div(x[0],x[1])
                elif op == 'drawdown':value = _div(x[0]-x[1],np.maximum(np.abs(x[0]),cash_floor(x[0])))
                elif op == 'surplus_buffer':value = _div(x[0]-x[1]-x[2]-x[3],np.maximum(np.abs(x[4]),cash_floor(x[4])))
                elif op == 'stress_income':value = _div(x[2]+x[3],x[0]-x[1])
                elif op == 'stress_outflow':value = _div(x[0]-x[1]-x[2]-x[3]-x[4],x[0])
                elif op == 'noise_margin':value = _div(x[0]-x[2]-x[4]-x[5],np.sqrt(x[1]**2+x[3]**2))
                elif op == 'vol_leverage':value = _div(x[1],x[0])*_div(x[2]+x[3],x[0])
                elif op == 'overdraft_service':value = _div(np.maximum(-x[0],0)*(x[2]+x[3]),np.where(x[1]>0,x[1]**2,np.nan))
                elif op == 'term_load':value = _div(np.where(x[4]>0,x[4]*(x[1]+x[2]),np.nan),np.maximum(x[3],0)+x[4]*x[0])
                elif op == 'tenure_term':value = _div(np.where(x[0]>=0,x[0],np.nan),x[1])
                elif op == 'contract_cost':value = _div(np.where(x[1]>0,x[0]*x[1]-x[2],np.nan),x[2])
                elif op == 'principal_resources':value = _div(x[0],np.maximum(x[1],0)+x[3]*x[2])
                elif op == 'trend_gap':value = np.log(_div(x[0],x[1]))-np.log(_div(x[2],x[3]))
                elif op == 'surplus_shift':value = _div((x[0]-x[2])-(x[1]-x[3]),x[1])
                elif op == 'missing_fraction':value = np.mean(~np.isfinite(np.column_stack(x)),axis=1)
                else:raise RuntimeError('Unknown operation:' + op)
        except ValueError as error:
            skipped.append(dict(name=spec['name'], reason=str(error)))
            continue
        value = np.asarray(value, dtype=np.float32)
        value[~np.isfinite(value)] = np.nan
        finite = value[ctx][np.isfinite(value[ctx])]
        if len(finite) == 0 or (len(np.unique(value[ctx])) <= 1):
            skipped.append(dict(name=spec['name'], reason='constant_or_all_missing_in_context'))
            continue
        fp = fingerprint(value)
        if fp in seen:
            skipped.append(dict(name=spec['name'], reason='exact_context_alias', alias_of=seen[fp]))
            continue
        seen[fp] = spec['name'];built[spec['name']] = value
        kept.append(dict(**spec, input_columns={n:resolved[n] for n in spec['inputs']},
                         context_missing_fraction=float(np.mean(~np.isfinite(value[ctx])))))
    audit = dict(candidate_definitions=len(definitions()), retained_derived_features=len(built),
                 source_raw_fields_added=0, input_resolution=resolved,
                 context_rows=len(ctx), context_rows_sha256=hashlib.sha256(ctx.astype('<i8').tobytes()).hexdigest(),
                 current_target_labels_used=False, query_rows_used_for_fitting=False,
                 learned_recipes=fitted, definitions=kept, skipped=skipped, sources=SOURCES,
                 scope='Feature hypotheses only; no predictive improvement established.',
                 missingness='Unknown inputs stay missing; observed zero exposure receives a historical context prior where identifiable.',
                 preprocessing_boundary='Recompute with inner-training ctx if using these learned priors/floors during an inner stopping pilot.')
    return pd.DataFrame(built, index=frame.index), audit


def synthetic_checks():
    """Pure feature arithmetic and heldout-mutation checks; no model training."""
    rng = np.random.default_rng(20261028);size = 48;ctx = np.arange(32)
    data = {name:rng.uniform(1,10,size) for name in required_inputs()}
    for _, kname, nname in HISTORIES:
        data[nname] = rng.integers(2,20,size).astype(float)
        data[kname] = np.array([rng.integers(0,int(n)+1) for n in data[nname]],float)
    # Make the recent and all windows genuinely nested.
    rk,rn = HISTORIES[2][1:];ak,an = HISTORIES[0][1:]
    data[an] = data[rn] + 20;data[ak] = data[rk] + 2
    data[I] = rng.uniform(50,80,size);data[E] = rng.uniform(55,100,size)
    data[P] = rng.uniform(5,10,size);data[Q] = rng.uniform(2,4,size)
    data[ISD] = rng.uniform(2,8,size);data[ESD] = rng.uniform(3,6,size)
    data[C] = rng.uniform(20,40,size);data[CMIN] = rng.uniform(-10,8,size)
    data[T] = rng.uniform(6,18,size);data[U] = rng.uniform(2,30,size)
    frame = pd.DataFrame(data)
    schema = dict(specs=[])
    first,audit = expand(frame,schema,ctx)
    changed = frame.copy();changed.iloc[32:] = changed.iloc[32:]*1000
    second,other = expand(changed,schema,ctx)
    assert audit['learned_recipes'] == other['learned_recipes']
    assert np.allclose(first.iloc[ctx],second.iloc[ctx],equal_nan=True)
    assert len(first.columns) >= 20
    missing = frame.copy();missing.loc[0,HISTORIES[0][2]] = np.nan
    result,_ = expand(missing,schema,ctx)
    assert np.isnan(result.loc[0,'a6_all_t1_logodds'])
    return dict(status='PASS', candidate_definitions=len(definitions()),
                synthetic_retained_features=first.shape[1], heldout_mutation_preserves_fitted_recipes=True,
                heldout_mutation_preserves_context_values=True, missing_exposure_remains_missing=True,
                actual_competition_data_used=False, models_trained=0)
