"""Finite coverage pools, conditional parameters, and comparable pair tokens."""
import itertools, math, random
from .common import digest

def active(spec, params):
    return all(params.get(k, object()) == v for k, v in spec.get('when', {}).items())

def canonical(family, params):
    out = {}
    for key, spec in family.get('parameters', {}).items():
        if not active(spec, out): continue
        if key not in params: raise ValueError(f'missing active parameter {key}')
        value = params[key]
        if spec['type'] == 'categorical':
            if value not in spec['choices']: raise ValueError(f'invalid choice {key}')
        else:
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not spec['low'] <= value <= spec['high']: raise ValueError(f'invalid range {key}')
            if spec['type'] == 'int' and type(value) is not int: raise ValueError(f'expected int {key}')
        out[key] = value
    if set(params) - set(family.get('parameters', {})): raise ValueError('unknown parameters')
    for exclusion in family.get('exclude', []):
        if all(out.get(k, object()) == v for k, v in exclusion.items()): raise ValueError('excluded combination')
    return out

def suggest(trial, family):
    params = {}
    for key, s in family.get('parameters', {}).items():
        if not active(s, params): continue
        if s['type'] == 'categorical': params[key] = trial.suggest_categorical(key, s['choices'])
        elif s['type'] == 'int': params[key] = trial.suggest_int(key, s['low'], s['high'], log=s.get('log', False))
        else: params[key] = trial.suggest_float(key, s['low'], s['high'], log=s.get('log', False))
    return canonical(family, params)

def pool(family, limit=10000, seed=42):
    """Exact finite enumeration when bounded; sampled pool otherwise, explicitly labeled."""
    domains = {}
    exact = True
    for k, s in family.get('parameters', {}).items():
        if s['type'] == 'categorical': domains[k] = s['choices']
        elif s['type'] == 'int' and s['high'] - s['low'] < limit: domains[k] = range(s['low'], s['high'] + 1)
        else: exact = False; domains[k] = [s['low'], s['high']]
    total = math.prod(len(v) for v in domains.values())
    exact = exact and total <= limit
    rows = {}
    if exact:
        choices = (dict(zip(domains, vals)) for vals in itertools.product(*domains.values()))
    else:
        rng = random.Random(seed)
        def samples():
            for _ in range(limit * 3):
                row = {}
                for k, s in family.get('parameters', {}).items():
                    if not active(s, row): continue
                    if s['type'] == 'categorical': row[k] = rng.choice(s['choices'])
                    elif s['type'] == 'int': row[k] = rng.randint(s['low'], s['high'])
                    elif s.get('log'): row[k] = math.exp(rng.uniform(math.log(s['low']), math.log(s['high'])))
                    else: row[k] = rng.uniform(s['low'], s['high'])
                yield row
        choices = samples()
    for p in choices:
        try: p = canonical(family, p)
        except ValueError: continue
        rows[digest(p)] = p
        if len(rows) >= limit: break
    anchors = []
    for a in family.get('anchors', []):
        p = canonical(family, a['params']); rows[digest(p)] = p; anchors.append(p)
    return anchors + [p for p in rows.values() if p not in anchors], exact

def tokens(family, params):
    # Coverage denominators deliberately exclude unbounded numeric ranges.
    factors = [(k, digest(v)) for k, v in params.items() if family.get('parameters', {})[k]['type'] == 'categorical']
    return {('family', family['name'])} | {('factor', family['name'], k, v) for k, v in factors} | {
        ('pair', family['name'], *a, *b) for a, b in itertools.combinations(factors, 2)}

def coverage(config, records):
    result = {}
    for family in config['families']:
        rows, exact = pool(family, config.get('coverage_pool_limit', 10000), config.get('search_seed', 42))
        denominator = set().union(*(tokens(family, p) for p in rows)) if rows else set()
        completed = [r for r in records if r['family'] == family['name'] and r['protocol'] == 'screen' and r['status'] == 'complete']
        observed = set().union(*(tokens(family, r['params']) for r in completed)) if completed else set()
        result[family['name']] = {
            'denominator_kind': 'exact_finite_space' if exact else 'sampled_reference_pool',
            'candidate_count': len(rows), 'unique_completed_configs': len({digest(r['params']) for r in completed}),
            'reference_pool_configs_completed': len({digest(r['params']) for r in completed} & {digest(p) for p in rows}),
            'factors_covered': len([t for t in observed & denominator if t[0] == 'factor']),
            'factors_total': len([t for t in denominator if t[0] == 'factor']),
            'pairs_covered': len([t for t in observed & denominator if t[0] == 'pair']),
            'pairs_total': len([t for t in denominator if t[0] == 'pair']),
            'family_completed': bool(completed),
            'unresolved_hypothesis': None if completed else family['hypothesis']}
    return result
