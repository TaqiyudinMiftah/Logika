"""Track audited numerical and Pareto results; never promote to public winner."""
from pathlib import Path
import hashlib,json,time
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'v7'


def update():
    records=[]
    for path in (V/'runs').glob('V*/RESULT.json'):
        ident=path.parent.name
        if int(ident[1:])<120:continue
        audit_path=path.parent/'REQUIREMENTS_AUDIT.json'
        if not audit_path.exists():continue
        r=json.loads(path.read_text());a=json.loads(audit_path.read_text())
        if r.get('status')!='COMPLETE' or a.get('status')!='PASS':continue
        if r.get('frozen_Kumo_fraction')!=1. or not r.get('all_outer_indices_identical'):continue
        assert abs(a['measured_OOF']['KS']-r['OOF']['KS'])<1e-12 and a['submission_sha256']==r['submission_sha256']
        records.append(dict(id=ident,pooled_KS=r['OOF']['KS'],AUC=r['OOF']['AUC'],submission=r['submission'],submission_sha256=r['submission_sha256'],comparison_to_V193=r.get('comparison_to_V193'),comparison_to_V185=r.get('comparison_to_V185'),confidence_interval=r['confidence_interval'],overfitting_assessment=r['overfitting_assessment'],public_score=None,public_promotion=False))
    assert records
    best=max(records,key=lambda r:r['pooled_KS']);assert hashlib.sha256((ROOT/best['submission']).read_bytes()).hexdigest()==best['submission_sha256']
    pareto=[a for a in records if not any(b['pooled_KS']>=a['pooled_KS'] and b['AUC']>=a['AUC'] and (b['pooled_KS']>a['pooled_KS'] or b['AUC']>a['AUC']) for b in records)]
    current=V/'reports/BEST_DEVELOPMENT_MODEL.json';old=json.loads(current.read_text()) if current.exists() else {}
    if old.get('id')!=best['id']:
        with (V/'reports/DEVELOPMENT_BEST_HISTORY.jsonl').open('a') as f:f.write(json.dumps(dict(UTC=time.time(),previous=old.get('id'),new=best['id'],KS=best['pooled_KS'],AUC=best['AUC'],public_promotion=False))+'\n')
    current.write_text(json.dumps(best,indent=2))
    report=dict(UTC=time.time(),scope='Audited frozen-Kumo model experiments V120+ on the locked outer folds',numerical_best=best,pareto=sorted(pareto,key=lambda r:r['pooled_KS'],reverse=True),public_incumbent=dict(id='V25',KS=.56822),selection_warning='Pareto status and highest pooled KS are descriptive after repeated development search; confidence intervals do not correct selection or establish hidden-test improvement.')
    (V/'reports/DEVELOPMENT_PARETO_FRONTIER.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    r=update();print(json.dumps(dict(numerical_best=r['numerical_best']['id'],KS=r['numerical_best']['pooled_KS'],AUC=r['numerical_best']['AUC'],public_promotion=False)))
