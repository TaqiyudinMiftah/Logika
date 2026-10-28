"""Resolve the semantic feature bank without consulting any outcomes."""
import hashlib
import json
from common import R, dump, sha


def main():
    manifest = json.loads((R / 'BANK_MANIFEST.json').read_text())
    catalog = json.loads((R / 'CATALOG.json').read_text())
    anchor = json.loads((R / 'ANCHOR_SOURCE.json').read_text())
    old = set(anchor['feature_columns'])
    allowed = set(anchor['feature_columns'] + anchor['head_columns'] + catalog['anchor_columns'])
    allowed.update(['account_income_mean', 'account_income_mean_3m', 'account_outflow_mean',
                    'account_outflow_mean_3m', 'account_last_closing_balance', 'account_average_balance_mean',
                    'account_minimum_balance_min', 'account_net_inflow_mean', 'account_income_std',
                    'account_income_cv', 'bureau_debt_max', 'bureau_debt_mean', 'ecom_spend_sum'])
    for prior in manifest['priors']:
        if prior.get('mature') and prior.get('window') in [90, 180, 'all']:
            allowed.update([prior['k'], prior['n']])
    specs = []
    for spec in catalog['feature_definitions']:
        if spec['op'] == 'identity':
            name = spec['inputs'][0]
            if ('repay' in name or 'late_' in name) and name not in old.union(catalog['anchor_columns']) and '_m1_' not in name:
                continue  # No new unmatured delinquency summaries.
            specs.append(spec)
        elif set(spec['inputs']) <= allowed:
            if spec['op'] == 'beta' and spec['strength'] != 24:
                continue
            if spec['op'] in {'amort_balance', 'payment_shock', 'interest_income'}:
                continue  # No tenor grid: the proposed loan tenor is fixed.
            if spec['op'] == 'stress':
                continue
            specs.append(spec)
    for income in ['declared_monthly_income', 'v2_account_3m_income_like_credits_mean']:
        for cash in ['account_last_closing_balance', 'v2_account_3m_minimum_balance_min',
                     'v2_account_3m_average_balance_mean']:
            spec = dict(op='stress', inputs=[income, cash, 'v2_account_3m_outflows_mean',
                        'proposed_monthly_installment'], family='cashflow_stress', source='liquidity',
                        months=3, income_loss=.2, expense_increase=0.)
            spec['id'] = 'new_' + hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
            specs.append(spec)
    for window, spend in [(30, 'v2_ecom_30d_spend'), (60, 'v2_ecom_60d_spend')]:
        for income in ['declared_monthly_income', 'v2_account_3m_income_like_credits_mean']:
            spec = dict(op='monthly_spend_ratio', inputs=[spend, income], family='commerce_capacity',
                        source='debt_service', window_days=window,
                        rationale='Monthly purchase-spend proxy divided by monthly income; not a debt-service obligation.')
            spec['id'] = 'new_' + hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
            specs.append(spec)
    identities = {s['inputs'][0]: s['id'] for s in catalog['feature_definitions'] if s['op'] == 'identity'}
    core = [identities[c] for c in catalog['anchor_columns']]

    def domains(spec):
        text = ' '.join(spec['inputs']).lower()
        family = spec['family']
        names = set()
        if any(w in text for w in ['account', 'income', 'cash', 'balance', 'outflow']):
            names.add('cashflow')
        if any(w in text for w in ['bureau', 'repay', 'late_', 'previous', 'prev_', 'loan_age']):
            names.add('credit')
        if any(w in text for w in ['ecom', 'shopping', 'price', 'spend', 'orders']):
            names.add('commerce')
        if (spec['op'] != 'identity' or any(w in text for w in ['burden', 'ratio', 'deficit', 'dependent'])
                or 'application_profile' in family and any(w in family for w in ['account', 'bureau', 'shopping'])):
            names.add('cross_domain')
        return names

    groups = {d: [] for d in ['cashflow', 'credit', 'commerce', 'cross_domain']}
    for spec in specs:
        for domain in domains(spec):
            groups[domain].append(spec['id'])
    existing = {d: list(dict.fromkeys(core + [s['id'] for s in specs
                if s['op'] == 'identity' and s['inputs'][0] in old and d in domains(s)])) for d in groups}
    assert all(0 < len(v) <= 99 for v in existing.values())
    dump(dict(specs=specs, domains=groups, existing_views=existing, core=core,
              predictor_limit=99, label_used_for_schema=False,
              source_catalog_sha256=sha(R / 'CATALOG.json'),
              source_bank_manifest_sha256=sha(R / 'BANK_MANIFEST.json'),
              new_formula_note='Shock magnitudes are fixed hypotheses. Window-matched spending is a proxy, not debt.',
              selection='Expanded views retain the fixed20 core and context-only LightGBM gain-ranked domain candidates; no correction or outer labels.'),
         R / 'FEATURE_SCHEMA.json')
    print(json.dumps(dict(candidates=len(specs), existing={k:len(v) for k,v in existing.items()},
                         expanded_candidates={k:len(v) for k,v in groups.items()})))


if __name__ == '__main__':
    main()
