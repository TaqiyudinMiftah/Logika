"""Freeze a small nested procedure after the exhaustive descriptive inventory."""
import json,shutil,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CAMPAIGN=ROOT.parent/'exploration-12h-20261017'
sys.path[:0]=[str(CAMPAIGN/'source')]
from utils import read,write,sha


def main():
    assert not (ROOT/'benchmark_plan.json').exists(),'Do not overwrite a frozen plan'
    candidates=read(ROOT/'evidence/descriptive_additions.json')
    audited={r['id']:r for r in read(ROOT/'evidence/all_prediction_audits.json')}
    chosen=[];models=set();excluded=[]
    # Availability constraints are fixed here, before any new benchmark score.
    supported={'lgb','cat','xgb','linear','elastic','ebm','cat_native'}
    manifest=read(CAMPAIGN/'evidence/champion.json')['components']
    champion={key:manifest['components'][key] for key in manifest['ensemble']['parents']}
    assert len(champion)==5
    for r in candidates:
        if r['weight'] not in [.05,.10]:continue
        if r['model'] not in supported:
            excluded.append({'id':r['id'],'model':r['model'],'reason':'Not supported by this bounded CPU nested-refit implementation; still audited descriptively.'});continue
        item=audited[r['id']]
        if item['original_id'] in champion or r['model']=='champion':continue
        model_group='regularized_logistic' if r['model'] in ['elastic','linear'] else r['model']
        if model_group in models:continue
        if r['id'] in {c['original_id'] for c in chosen}:continue
        folder=Path(item['folder'])
        if item['origin']=='depth':
            resolved=read(folder/'fold_0/resolved.json')
            component={'id':'addition'+str(len(chosen)),'original_id':r['id'],'kind':'resolved',
                'representation':resolved['representation'],'parameters':resolved['model'],
                'model':r['model'],'weight':resolved['representation']['config']['weight']}
        elif item['origin']=='historical':
            recipe=read(folder/'task.json')['recipe']
            component={'id':'addition'+str(len(chosen)),'original_id':r['id'],'kind':'historical','recipe':recipe}
        else:
            cid=item['config_id'];p=read(CAMPAIGN/'source/configurations.json')[cid]
            component={'id':'addition'+str(len(chosen)),'original_id':r['id'],'kind':'latest','configuration':p,'representation':read(CAMPAIGN/'source/views.json')[p['view']]}
        component['discovery_diagnostic']=r
        chosen.append(component);models.add(model_group)
        if len(chosen)==3:break
    assert len(chosen)>=1
    final=ROOT.parent/'submission-final-20261018'
    shutil.copytree(final/'research_source',ROOT/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name,h in read(final/'research_source_manifest.json').items():assert sha(ROOT/'source'/name)==h
    champion_csv=final/'evidence/previous_submission.csv'
    if not champion_csv.exists():champion_csv=Path('/home/faris/faris/submission-research-20261017/artifacts/submission.csv')
    assert sha(champion_csv)==read(CAMPAIGN/'evidence/champion_identity.json')['submission_sha256']
    shutil.copy2(champion_csv,ROOT/'evidence/champion_submission.csv')
    shutil.copy2(final/'artifacts/submission.csv',ROOT/'evidence/regressed_submission.csv')
    p=read(final/'plan.json')
    plan={'id':'submission-robust-20261018','frozen_at':time.time(),
        'bank':p['bank'],'extras':p['extras'],'train':p['train'],'test':p['test'],'sample':p['sample'],
        'input_sha256':p['input_sha256'],
        'champion_components':[{'id':'champion'+str(i),'original_id':key,'kind':'historical','recipe':v['recipe']} for i,(key,v) in enumerate(champion.items())],
        'candidate_components':chosen,'candidate_weights':[.05,.10],
        'outer_folds':3,'outer_seed':20261019,'inner_folds':3,
        'primary_metric':'Mean matched within-outer-fold KS; pooled KS separately reported to avoid confusing inter-model scale differences with within-fit ranking changes.',
        'inner_selection':'2/3 inner fold wins, minimum fold delta>=-0.002, pooled delta>=-0.003; maximize positive mean fold delta minus half fold std; no eligible option means champion.',
        'promotion_gate':'Positive mean matched outer-fold KS delta, pooled delta>=-0.003, 2/3 outer fold wins, worst outer fold delta>=-0.003, nonzero final addition.',
        'no_untouched_labels_claim':True,
        'final_choice':'Same inner procedure applied to full-population OOF; final outer scores do not retune weights or features.',
        'final_champion':'Preserve exact previous all-label models and CSV; do not replace their seed or refit realization.',
        'workers':8,'safety_seconds':7200,
        'supported_nested_models':sorted(supported),
        'source_manifest':read(final/'research_source_manifest.json'),
        'code_sha256':{name:sha(ROOT/name) for name in ['benchmark.py','prepare_benchmark.py']},
        'discovery_is_historically_adaptive':True,
        'no_upload':True}
    write(ROOT/'evidence/shortlist_exclusions.json',excluded)
    write(ROOT/'benchmark_plan.json',plan)
    print(json.dumps({'components':[(c['id'],c['original_id'],c['discovery_diagnostic']) for c in chosen], 'plan_sha256':sha(ROOT/'benchmark_plan.json')},indent=2))


if __name__=='__main__':main()
