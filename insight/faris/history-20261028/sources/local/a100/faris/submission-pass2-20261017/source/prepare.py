"""Collect completed, aligned experiment evidence and freeze a compact search."""
import json
import shutil
from pathlib import Path
import numpy as np
from metaexp.features import Bank
from metaexp.common import sha, write, read

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT.parent
BANK = BASE / 'logika-meta-experiments/meta-20261016-v1'


def main():
    bank = Bank(BANK)
    prior = read(BASE / 'submission-research-20261017/frozen_recipe.json')
    assert bank.manifest['bank_id'] == prior['components']['bank_id']
    for name, checksum in bank.manifest['files'].items():
        assert sha(bank.root / name) == checksum
    pool = []
    for suffix in ['', '-wave2']:
        root = BASE / ('logika-meta-experiments/meta-20261016-v1' + suffix)
        for row in read(root / 'reports/scoreboard_pipeline.json'):
            if row['state'] == 'completed':
                pool.append((row['pooled_oof']['ks'], root, row))
    pool.sort(key=lambda item: item[0], reverse=True)
    groups = {}
    selected = {}
    for model in ['lgb', 'linear', 'cat', 'xgb']:
        seen = set()
        groups[model] = []
        for _, root, row in pool:
            if row['model'] != model or row['representation'] in seen:
                continue
            seen.add(row['representation'])
            selected[row['id']] = root
            groups[model].append(row['id'])
            if len(groups[model]) == 2:
                break
        assert len(groups[model]) == 2
    original = prior['components']['ensemble']['parents']
    for key in original:
        selected[key] = Path(prior['components']['components'][key]['source_root'])
    recipes = {}
    provenance = {}
    evidence = ROOT / 'evidence'
    evidence.mkdir(exist_ok=True)
    for key, root in selected.items():
        folder = root / 'runs' / key
        complete = read(folder / 'complete.json')
        for name in ['prediction.npy', 'task.json', 'oof.npz']:
            assert sha(folder / name) == complete['files'][name], (key, name)
        z = np.load(folder / 'oof.npz')
        assert np.array_equal(z['target'], bank.y)
        assert np.array_equal(z['fold'], bank.folds)
        assert np.array_equal(z['application_id'].astype(str), bank.metadata.application_id.iloc[bank.dev].to_numpy(dtype=str))
        p = np.load(folder / 'prediction.npy')
        assert np.array_equal(p, z['prediction']) and np.isfinite(p).all()
        np.save(evidence / (key + '.npy'), p)
        recipes[key] = read(folder / 'task.json')['recipe']
        provenance[key] = {'root': str(root), 'oof_sha256': sha(folder / 'oof.npz'), 'prediction_sha256': sha(folder / 'prediction.npy'), 'scores': read(folder / 'scores.json')}
    sets = {
        'original': original,
        'family_leaders': [groups[m][0] for m in groups],
        'family_pairs': [key for values in groups.values() for key in values],
        'tree_leaders': [groups[m][0] for m in ['lgb', 'cat', 'xgb']],
        'linear_lgb_cat': [groups[m][0] for m in ['linear', 'lgb', 'cat']],
        'original_plus_cat': original + [groups['cat'][0]],
        'original_plus_xgb': original + [groups['xgb'][0]],
        'original_plus_both': original + [groups['cat'][0], groups['xgb'][0]],
    }
    candidates = []
    seen = set()
    for name, keys in sets.items():
        keys = sorted(set(keys))
        for method in ['trimmed', 'mean', 'logit', 'geometric']:
            signature = (method, tuple(keys))
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append({'id': name + '_' + method, 'parents': keys, 'method': method})
    plan = {
        'bank_root': str(BANK), 'bank_id': bank.manifest['bank_id'],
        'original': {'id': 'original_trimmed', 'parents': sorted(original), 'method': 'trimmed'},
        'recipes': recipes, 'candidates': candidates, 'additional_seeds': [4917, 9173],
        'selection_rule': 'Highest pooled OOF KS with at least two nonnegative fold deltas and no fold loss exceeding 0.003 versus delivered candidate; otherwise keep original.',
        'seed_rule': 'Evaluate the single selected structure using two additional seeds. Choose pooled mean across the three seeds only if the same development gate passes and KS exceeds the selected single-seed structure.',
        'promotion_rule': 'Freeze one final recipe before confirmation; promote only if its confirmation KS exceeds the existing delivered candidate and development gate passes. Reused confirmation is exploratory.',
        'limits': {'workers': 8, 'worker_threads': 1, 'max_seconds': 3600},
        'target_ks': 0.55,
        'caveat': 'Historical adaptive development and confirmation reuse. No untouched holdout, unbiased selection estimate, public-score guarantee, learned OOF stacking, or test-label access.'
    }
    write(ROOT / 'plan.json', plan)
    write(ROOT / 'evidence/provenance.json', provenance)
    write(ROOT / 'source_manifest.json', {str(p.relative_to(ROOT / 'source')): sha(p) for p in (ROOT / 'source').rglob('*.py')})
    print(json.dumps({'completed_recipe_inventory': len(pool), 'selected_components': len(recipes), 'fixed_combinations': len(candidates), 'plan_sha256': sha(ROOT / 'plan.json')}))


if __name__ == '__main__':
    main()
