"""Source-constrained feature grammar; no unconstrained pairwise products."""
from pathlib import Path
import collections, hashlib, itertools, json, re

R = Path(__file__).resolve().parents[1]
SOURCES = {
    'debt_service': {'url': 'https://www.federalreserve.gov/releases/dsr/about.htm',
        'title': 'Household Debt Service Ratios', 'institution': 'Federal Reserve',
        'supports': 'Required debt payments divided by disposable income.',
        'extension': 'Observed credits and declared income are imperfect individual-income proxies; historical installments are historical burden proxies, not current aggregate obligations.'},
    'liquidity': {'url': 'https://www.rba.gov.au/publications/bulletin/2022/jun/household-liquidity-buffers-and-financial-stress.html',
        'title': 'Household Liquidity Buffers and Financial Stress', 'author': 'Lydia Wang', 'year': 2022,
        'supports': 'Liquidity relative to income, debt net of liquid assets, and buffers against adverse cash-flow shocks.',
        'extension': 'Window-specific bank balances approximate liquid assets. Stress sizes and historical-window combinations are hypotheses, not calibrated forecasts.'},
    'amortization': {'url': 'https://openstax.org/books/principles-finance-2e/pages/8-3-loan-amortization',
        'title': 'Principles of Finance, 8.3 Loan Amortization',
        'supports': 'Level-payment annuity and outstanding principal under fixed monthly interest.',
        'extension': 'Normalize scheduled remaining principal and interest by observed income; hypothetical rate shifts do not imply the fixed loan reprices.'},
    'credibility': {'url': 'https://consensus.app/papers/credibility-theory-norberg/ed0a431b4423584f8b8d5bd02159f1c5/?utm_source=chatgpt',
        'title': 'Credibility Theory', 'author': 'R. Norberg', 'year': 2015, 'citations_at_retrieval': 57,
        'supports': 'Combine individual experience and collective risk with exposure-dependent credibility.',
        'extension': 'Beta-binomial late-payment posterior, uncertainty and exposure weight; concentration grid is predeclared. Cohort rates use context rows only.'},
    'volatility': {'url': 'https://consensus.app/papers/explainable-ai-for-credit-assessment-in-banks-lange-melsom/a63842785f4e506d8834a644131e80a6/?utm_source=chatgpt',
        'title': 'Explainable AI for Credit Assessment in Banks',
        'authors': 'P. D. de Lange; B. Melsom; Christian Bakke Vennerod; Sjur Westgaard', 'year': 2022,
        'journal': 'Journal of Risk and Financial Management', 'citations_at_retrieval': 100,
        'supports': 'Credit-balance volatility and available credit contributed to default prediction in one unsecured-consumer-loan dataset.',
        'extension': 'Temporal changes and normalized dispersion of account and bureau measures are proposed extensions, not results established for this competition.'},
    'subspaces': {'url': 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html',
        'title': 'BaggingClassifier documentation', 'supports': 'Feature-subset ensembles (random subspaces).',
        'extension': 'Semantic blocks with shared anchors and inner-only ensemble selection for Kumo; improvement is unproven.'},
}

def main():
    m = json.loads((R/'BANK_MANIFEST.json').read_text())
    boundary = json.loads((R/'ANCHOR_SOURCE.json').read_text())
    extras = json.loads((R/'EXTRA_SOURCES.json').read_text())
    extra_names = [c for names in extras.values() for c in names if c not in ['application_id','customer_id']]
    existing = list(dict.fromkeys(extra_names+m['columns']+m['categorical_columns']+boundary['feature_columns']+boundary['head_columns']))
    forbidden = {'target', 'customer_id', 'application_id', 'split', 'decision_datetime', 'application_datetime'}
    assert not forbidden.intersection(existing)
    specs = []; signatures = set()
    def add(op, inputs, family, source=None, **params):
        signature = json.dumps([op, inputs, params], sort_keys=True)
        if signature in signatures: return
        signatures.add(signature)
        specs.append({'id': 'f'+hashlib.sha256(signature.encode()).hexdigest()[:16], 'op': op,
                      'inputs': inputs, 'family': family, 'source': source, **params})
    for c in existing:
        domain = m['info'].get(c, {}).get('sources', ['current_selected'])
        add('identity', [c], 'existing_'+'_'.join(domain))
    def exists(names): return [c for c in names if c in existing]
    # Actual monthly income measurements, not arbitrary monetary denominators.
    incomes = exists(['declared_monthly_income', 'account_income_mean', 'account_income_mean_3m', 're_income_geometric_mean'])
    balances = exists(['account_last_closing_balance', 'account_average_balance_mean', 'account_minimum_balance_min'])
    debt = exists(['loan_amount', 'bureau_debt_mean', 'bureau_debt_max'])
    payments = exists(['proposed_monthly_installment', 'v2_existing_active_installment', 'previous_installment_mean'])
    expenses = exists(['account_outflow_mean', 'account_outflow_mean_3m'])
    # Limit aggregates to recognizable measures with known units.
    for c in existing:
        if re.fullmatch(r'm_account_(30|60|90|180|365|all)_income_like_credits_(mean|median|min|max)', c): incomes.append(c)
        if re.fullmatch(r'm_account_(30|60|90|180|365|all)_(average_balance|minimum_balance)_(mean|median|min)', c): balances.append(c)
        if re.fullmatch(r'm_bureau_(30|60|90|180|365|all)_outstanding_balance_(mean|median|max)', c): debt.append(c)
        if re.fullmatch(r'm_previous_(30|60|90|180|365|all)_scheduled_monthly_installment_(mean|median|max)', c): payments.append(c)
        if re.fullmatch(r'm_account_(30|60|90|180|365|all)_outflows_(mean|median|max)', c): expenses.append(c)
    def window(c):
        found = re.match(r'm_(?:account|bureau|previous)_(30|60|90|180|365|all)_', c)
        return found[1] if found else 'anchor'
    def compatible(*names):
        windows = {window(c) for c in names}-{ 'anchor', 'all' }
        return len(windows) <= 1
    for p in payments:
        for i in incomes:
            if compatible(p, i): add('ratio', [p, i], 'debt_service', 'debt_service')
    for d in debt:
        for i in incomes:
            if compatible(d, i): add('ratio', [d, i], 'debt_to_income', 'liquidity')
    for cash in balances:
        for i in incomes:
            if compatible(cash, i): add('ratio', [cash, i], 'liquidity_income', 'liquidity')
        for p in payments:
            if compatible(cash, p): add('ratio', [cash, p], 'liquidity_payment', 'liquidity')
        for expense in expenses:
            if compatible(cash, expense): add('ratio', [cash, expense], 'liquidity_expense', 'liquidity')
    # Net indebtedness: matched-window debt and cash; use only central income estimates.
    central_income = [i for i in incomes if not i.endswith(('_min', '_max'))]
    for d, cash, i in itertools.product(debt, balances, central_income):
        if compatible(d, cash, i): add('net_ratio', [d, cash, i], 'net_debt_income', 'liquidity')
    # Cash-flow stress has fixed semantic operands. Do not multiply arbitrary columns.
    # Recorded outflows may include historical repayments; add only the new proposed payment.
    for i, cash, expense in itertools.product(central_income, balances, expenses):
        if not compatible(i, cash, expense): continue
        if window(i) == window(cash) == window(expense) == 'anchor': continue
        # Each window uses mean/median balances for normal scenarios and minimum as a downside proxy.
        for income_loss, expense_increase, months in [(0., 0., 1), (.2, 0., 3), (.4, .1, 3)]:
            add('stress', [i, cash, expense, 'proposed_monthly_installment'], 'cashflow_stress', 'liquidity',
                income_loss=income_loss, expense_increase=expense_increase, months=months)
    for i in central_income:
        for months in [3, 6, 9]:
            add('amort_balance', ['loan_amount', 'annual_interest_rate', 'proposed_monthly_installment', i], 'amortization', 'amortization', months=months)
        add('interest_income', ['loan_amount', 'proposed_monthly_installment', i], 'amortization', 'amortization')
        for rate_shift in [.02, .04]:
            add('payment_shock', ['loan_amount', 'annual_interest_rate', 'proposed_monthly_installment', i], 'amortization', 'amortization', rate_shift=rate_shift)
    for prior in m['priors']:
        if prior.get('legacy'): continue
        for strength in [2, 8, 24, 96]:
            for quantity in ['posterior', 'sd', 'credibility']:
                add('beta', [prior['k'], prior['n']], 'repayment_credibility', 'credibility', strength=strength, quantity=quantity)
    # Differences between windows of the SAME measured quantity, with direction preserved.
    stems = collections.defaultdict(list)
    for c in existing:
        match = re.match(r'm_(account|bureau)_(30|60|90|180|365|all)_(.+)', c)
        if match and match[3].endswith(('mean', 'median', 'cv', 'std', 'share')):
            stems[(match[1], match[3])].append((3660 if match[2]=='all' else int(match[2]), c))
    for names in stems.values():
        for (_, recent), (_, old) in itertools.combinations(sorted(names), 2):
            add('relative_change', [recent, old], 'temporal_change', 'volatility')
    for c in existing:
        if c.endswith('_std') and c[:-4]+'_mean' in existing and ('account' in c or 'bureau' in c):
            add('dispersion', [c, c[:-4]+'_mean'], 'volatility', 'volatility')
    anchors = ['age', 'employment_tenure_months', 'declared_monthly_income', 'dependents', 'loan_amount',
        'proposed_monthly_installment', 'annual_interest_rate', 'installment_observed_income_ratio',
        'bureau_score_mean', 'bureau_debt_income_ratio', 'v2_existing_total_burden_observed',
        'v2_account_3m_income_like_credits_mean', 'v2_account_3m_minimum_balance_min',
        'v2_account_3m_outflows_mean', 'ecom_last_purchase_recency_days', 'loan_purpose', 'region',
        'employment_type', 'housing_status', 're_late_30_posterior']
    by_identity = {s['inputs'][0]:s['id'] for s in specs if s['op']=='identity'}
    anchor_ids = [by_identity[c] for c in anchors]
    groups = collections.defaultdict(list)
    for s in specs:
        if s['id'] not in anchor_ids: groups[s['family']].append(s['id'])
    packs = [{'id':'baseline90', 'family':'control', 'features':[by_identity[c] for c in boundary['feature_columns']]},
             {'id':'anchors20', 'family':'control', 'features':anchor_ids}]
    # Interleave families so partial compute coverage is not just one source family.
    chunks = {family:[features[i:i+80] for i in range(0,len(features),80)] for family,features in sorted(groups.items())}
    for part in range(max(map(len,chunks.values()))):
        for family, family_chunks in chunks.items():
            if part < len(family_chunks):
                packs.append({'id':f'p{len(packs):04d}', 'family':family, 'features':anchor_ids+family_chunks[part]})
    covered = set().union(*(set(p['features']) for p in packs))
    assert covered == {s['id'] for s in specs}
    assert all(len(p['features'])<=100 and len(set(p['features']))==len(p['features']) for p in packs)
    payload = {'existing_features':len(existing), 'generated_features':len(specs)-len(existing), 'total_features':len(specs),
        'family_counts':dict(collections.Counter(s['family'] for s in specs)), 'feature_definitions':specs,
        'packs':packs, 'pack_count':len(packs), 'anchor_columns':anchors, 'sources':SOURCES,
        'interpretation':'Sources support mechanisms. Window combinations, stress grids and their usefulness here are unproven extensions.',
        'retention':'Lazy materialization per 100-feature pack; no giant dense derived matrix.',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (R/'CATALOG.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k in ['existing_features','generated_features','total_features','family_counts','pack_count']}, indent=2))

if __name__=='__main__': main()
