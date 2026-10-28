"""Exhaustive inventory and descriptive champion-addition audit, never confirmation."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import concurrent.futures as cf
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent / 'exploration-12h-20261017'
HIST = Path('/home/faris/faris/logika-meta-experiments')
sys.path[:0] = [str(CAMPAIGN), str(CAMPAIGN/'source')]
from utils import ks, read, write, sha
Y = FOLD = CHAMP = IDS = None


def init():
    global Y, FOLD, CHAMP, IDS
    dep = read(CAMPAIGN/'deployment.json')
    bank = Path(dep['hosts']['pc2']['bank'])
    Y = np.load(bank/'development_y.npy')
    FOLD = np.load(bank/'folds.npy')
    CHAMP = np.load(CAMPAIGN/'evidence/champion_oof.npz')['existing_feature_ensemble']
    rows = np.load(bank/'development_rows.npy')
    IDS = pd.read_parquet(bank/'metadata.parquet').application_id.iloc[rows].to_numpy()


def assess(item):
    path = Path(item['prediction_path'])
    if not path.exists():
        return {**item, 'audit_state': 'missing_saved_predictions'}
    try:
        z = np.load(path)
        p = z['prediction']
        if 'application_id' in z:
            assert np.array_equal(z['application_id'], IDS)
            assert np.array_equal(z['target'], Y)
        assert np.array_equal(z['fold'], FOLD)
        assert p.shape == Y.shape and np.isfinite(p).all()
        score = ks(Y, p)
        assert abs(score-item['reported_ks']) < 1e-10
        folds = [ks(Y[FOLD==f], p[FOLD==f]) for f in range(3)]
        stats = []
        for weight in [.05, .10, .20]:
            v = (1-weight)*CHAMP+weight*p
            stats.append({'weight': weight, 'delta': ks(Y,v)-ks(Y,CHAMP),
                'fold_delta': [ks(Y[FOLD==f],v[FOLD==f])-ks(Y[FOLD==f],CHAMP[FOLD==f]) for f in range(3)]})
        warnings = []
        folder = path.parent
        for f in range(3):
            log = folder/f'fold_{f}'/'parameters.json'
            if log.exists() and item['model'] in ['act_logit','act_cloglog']:
                act = folder/f'fold_{f}'/'actuarial.json'
                if act.exists() and read(act).get('optimizer_success') is False:
                    warnings.append('glm_nonconvergence')
        return {**item, 'audit_state': 'verified', 'ks': score, 'fold_ks': folds,
            'degenerate_fold': min(folds)<.1, 'quality_warnings': sorted(set(warnings)),
            'prediction_sha256': sha(path), 'additions': stats,
            'descriptive_only_not_independent_validation': True}
    except Exception as e:
        return {**item, 'audit_state': 'invalid', 'reason': repr(e)}


def main():
    ROOT.joinpath('evidence').mkdir(exist_ok=True)
    items = []; inventory = []; seen = set()
    for name in ['meta-20261016-v1', 'meta-20261016-v1-wave2']:
        r = HIST/name
        for kind in ['pipeline','fixed_ensemble','nested_ensemble']:
            path = r/'reports'/('scoreboard_'+kind+'.json')
            rows = read(path)
            inventory.append({'source':str(path), 'sha256':sha(path), 'kind':kind,
                'records':len(rows), 'unique_ids':len({x['id'] for x in rows})})
            if kind != 'pipeline': continue
            for row in rows:
                key = name+'/'+row['id']
                if key in seen: continue
                seen.add(key)
                folder = r/'runs'/row['id']
                items.append({'id':key,'original_id':row['id'], 'origin':'historical',
                    'model':row.get('model'), 'family':row.get('family'),
                    'reported_ks':row['pooled_oof']['ks'], 'folder':str(folder),
                    'prediction_path':str(folder/'oof.npz')})
    from supervisor import records
    rows = records()
    cfg = read(CAMPAIGN/'source/configurations.json')
    known = {r['id'] for r in rows}
    for r in read(CAMPAIGN/'evidence/node380-final-outcomes.json'):
        if r['id'] not in known:
            rows.append({'id':r['id'],'host':'node380','job':r['job'],
                'evaluation':r['outcome']['evaluation'], 'recovered':True})
    inventory.append({'source':str(CAMPAIGN),'records':len(rows), 'kind':'completed_training_records'})
    for row in rows:
        if row['job']['protocol']!='screen': continue
        if row.get('recovered'):
            prediction = CAMPAIGN/'evidence/node380-final-predictions'/(row['id']+'.npz')
        else: prediction = Path(row['folder'])/'validation.npz'
        cid = row['job']['params']['config']
        items.append({'id':row['id'],'original_id':row['id'],'origin':'latest',
            'model':cfg.get(cid,{}).get('model',row['job']['family']),
            'family':row['job']['family'], 'config_id':cid,
            'reported_ks':row['evaluation']['score'], 'folder':str(prediction.parent),
            'prediction_path':str(prediction)})
    write(ROOT/'evidence/inventory_sources.json',inventory)
    write(ROOT/'evidence/review_design.json',{'scope':'All saved completed single-pipeline screening predictions, including recovered foundations; inventory all historical fixed/nested ensemble records.',
        'items':len(items), 'weights':[.05,.10,.20], 'confirmation_access':False,
        'use':'Descriptive shortlist discovery only; not an unbiased promotion benchmark',
        'future_benchmark':'Freeze small shortlist before any additional training; actual fold-local refits; no weights fitted on outer-validation labels.'})
    results=[]; started=time.time()
    with cf.ProcessPoolExecutor(max_workers=8, initializer=init) as pool:
        for result in pool.map(assess,items,chunksize=8):
            results.append(result)
            if len(results)%250==0:
                write(ROOT/'review_status.json',{'done':len(results),'total':len(items),'seconds':time.time()-started})
                print(json.dumps({'done':len(results),'total':len(items)}),flush=True)
    write(ROOT/'evidence/all_prediction_audits.json',results)
    good=[r for r in results if r['audit_state']=='verified' and not r['degenerate_fold'] and not r['quality_warnings']]
    candidates=[]
    for r in good:
        for s in r['additions']:
            candidates.append({'id':r['id'],'model':r['model'],'family':r['family'], **s,
                'mean_fold_delta':float(np.mean(s['fold_delta'])), 'worst_fold_delta':min(s['fold_delta'])})
    candidates.sort(key=lambda s:(min(s['fold_delta']),s['delta']),reverse=True)
    write(ROOT/'evidence/descriptive_additions.json',candidates)
    summary={'audited':len(results),'verified':len(good),'states':{state:sum(r['audit_state']==state for r in results) for state in sorted({r['audit_state'] for r in results})},
        'top_worst_fold':candidates[:30], 'not_independent_selection':True,'seconds':time.time()-started}
    write(ROOT/'review_summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
